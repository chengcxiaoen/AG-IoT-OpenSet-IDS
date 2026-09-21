"""Frozen paper protocol, streaming decisions, and deployment utilities.

This module does not import TensorFlow, so audits and metric tests run on CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import BENIGN_LABEL, UNKNOWN_LABEL, DEFAULT_COLUMNS_TO_DROP
from .context import context_anomaly_score, context_surprisal_matrix
from .data import normalize_label_name, transform_features
from .utils import to_builtin

METHODS = (
    "closed_dnn", "context_only", "global_msp", "classwise_msp",
    "global_msp_context", "classwise_msp_context",
)
PROPOSED = "classwise_msp_context"
CSV_STRING_COLUMNS = {name: str for name in (
    "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "proto", "service",
    "conn_state", "history", "local_orig", "local_resp", "tunnel_parents")}


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + str(message), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_builtin(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def protocol_digest(config, dataset_hash, source_hash):
    text = json.dumps([config, dataset_hash, source_hash], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    paths = list((Path(root) / "src/iotids").glob("*.py"))
    paths += list((Path(root) / "experiments").glob("*paper*.py"))
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_frame(path, label):
    # Full CSV, no head/sample/cap. Chunking bounds parsing temporaries.
    parts = []
    for index, part in enumerate(pd.read_csv(path, chunksize=50000, low_memory=False,
                                           dtype=CSV_STRING_COLUMNS)):
        parts.append(part)
        log(f"CSV read: {sum(len(p) for p in parts):,} rows")
    frame = pd.concat(parts, ignore_index=True)
    del parts
    if label not in frame or frame[label].isna().any():
        raise ValueError("Missing label column or missing labels; refusing silent filtering")
    frame[label] = frame[label].map(normalize_label_name)
    # Keep float64 input values; repair +/- infinity with training-fitted imputation.
    excluded = set(DEFAULT_COLUMNS_TO_DROP) | {label, "is_attack", "proto", "service", "conn_state"}
    for col in frame.columns:
        if col not in excluded:
            frame[col] = pd.to_numeric(frame[col].replace("-", np.nan), errors="coerce")
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    if len(frame) == 0:
        raise ValueError("Empty dataset")
    return frame


def split_indices(labels, unknown_classes, seed, config):
    labels = np.asarray(labels, dtype=str)
    unknown_classes = list(unknown_classes)
    if BENIGN_LABEL in unknown_classes or not unknown_classes:
        raise ValueError("Benign must stay known and unknown classes cannot be empty")
    missing = set(unknown_classes) - set(labels)
    if missing:
        raise ValueError(f"Unknown class absent from dataset: {missing}")
    excluded_classes = config.get("excluded_classes", [])
    if BENIGN_LABEL in excluded_classes or set(unknown_classes) & set(excluded_classes):
        raise ValueError("Exclusions must not contain Benign or held-out unknown classes")
    excluded = np.flatnonzero(np.isin(labels, excluded_classes))
    zero = np.flatnonzero(np.isin(labels, unknown_classes))
    known = np.flatnonzero(~np.isin(labels, unknown_classes + list(excluded_classes)))
    pool, known_test = train_test_split(
        known, test_size=config["test_fraction_of_known"], random_state=seed,
        stratify=labels[known])
    val_fraction = config["validation_fraction_of_known_pool"]
    cal_fraction = config["calibration_fraction_of_known_pool"]
    train, tuning = train_test_split(
        pool, test_size=val_fraction + cal_fraction, random_state=seed,
        stratify=labels[pool])
    validation, calibration = train_test_split(
        tuning, test_size=cal_fraction / (val_fraction + cal_fraction),
        random_state=seed + 1, stratify=labels[tuning])
    test = np.concatenate([known_test, zero])
    np.random.default_rng(seed).shuffle(test)
    splits = dict(train=train, validation=validation, calibration=calibration, test=test, excluded=excluded)
    combined = np.concatenate(list(splits.values()))
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise AssertionError("Rows are missing or shared between partitions")
    for name in ("train", "validation", "calibration"):
        if np.isin(labels[splits[name]], unknown_classes).any():
            raise AssertionError("Unknown class leaked before test")
    return splits


def clean_transform(frame, preprocessor, clip):
    frame = frame.copy()
    for col in preprocessor.numeric_columns:
        frame[col] = pd.to_numeric(frame[col].replace("-", np.nan), errors="coerce")
        frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    matrix = transform_features(frame, preprocessor)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite preprocessed features; refusing misleading metrics")
    # Fixed a priori, not estimated from test or unknown data.
    return np.clip(matrix, -clip, clip).astype(np.float32, copy=False)


def decision_arrays(probabilities, context_scores, metadata):
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Invalid probabilities")
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    tau = np.asarray(metadata["class_thresholds"])
    global_mask = confidence < metadata["global_threshold"]
    class_mask = confidence < tau[pred]
    context_mask = ((pred == metadata["benign_index"]) &
                    (context_scores > metadata["context_threshold"]))
    masks = [np.zeros(len(pred), dtype=bool), context_mask, global_mask, class_mask,
             global_mask | context_mask, class_mask | context_mask]
    unknown_id = len(metadata["known_classes"])
    return {name: np.where(mask, unknown_id, pred) for name, mask in zip(METHODS, masks)}


def score_context(frame, profile, scales):
    matrix, _ = context_surprisal_matrix(profile, frame)
    return context_anomaly_score(matrix, scales)


def evaluate_arrays(truth, predictions, known_classes, zero_day_classes):
    truth = np.asarray(truth, dtype=str)
    known_classes = list(known_classes)
    names = known_classes + [UNKNOWN_LABEL]
    mapping = {name: i for i, name in enumerate(names)}
    uid = len(known_classes)
    y = np.asarray([uid if t in zero_day_classes else mapping[t] for t in truth])
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(cm, (y, np.asarray(predictions, dtype=int)), 1)
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, pred_support, out=np.zeros(len(names)), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(names)), where=precision + recall > 0)
    benign = known_classes.index(BENIGN_LABEL)
    if support[benign] == 0 or support[uid] == 0:
        raise ValueError("Test must contain Benign and Unknown samples")
    breakdown = {}
    for label in zero_day_classes:
        mask = truth == label
        breakdown[label] = {
            "support": int(mask.sum()),
            "unknown_recall": float(np.mean(np.asarray(predictions)[mask] == uid)),
            "miss_as_benign": float(np.mean(np.asarray(predictions)[mask] == benign)),
        }
    return {
        "B_benign_false_alarm_rate": float(1 - recall[benign]),
        "F_open_set_macro_f1": float(f1.mean()),
        "U_unknown_recall": float(recall[uid]),
        "known_accuracy_after_rejection": float(tp[:uid].sum() / support[:uid].sum()),
        "known_rejection_rate": float(cm[:uid, uid].sum() / support[:uid].sum()),
        "unknown_precision": float(precision[uid]), "unknown_f1": float(f1[uid]),
        "unknown_class_macro_recall": float(np.mean([x["unknown_recall"] for x in breakdown.values()])),
        "open_set_accuracy": float(tp.sum() / cm.sum()),
        "confusion_matrix": cm, "labels": names,
        "per_class": {name: {"precision": precision[i], "recall": recall[i],
                              "f1": f1[i], "support": support[i]} for i, name in enumerate(names)},
        "zero_day_breakdown": breakdown,
    }


def system_info():
    import psutil
    versions = {}
    from importlib.metadata import version, PackageNotFoundError
    for name in ("tensorflow", "numpy", "pandas", "scikit-learn", "psutil"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_model": cpu, "logical_cpus": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total, "packages": versions}


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if sha256(bundle / name) != expected:
            raise ValueError(f"Deployment artifact integrity failure: {name}")
    return manifest
