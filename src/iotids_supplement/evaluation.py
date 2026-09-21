"""Known-only calibration and matched empirical benign-FAR evaluation.

The shape/operating calibration split is fixed before examining test outcomes.
All scores are oriented so that larger means more anomalous. Test thresholds
are never selected from a test ROC curve.
"""
from __future__ import annotations

import gc
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from iotids.context import (CONTEXT_ABLATIONS, RelationSpec, fit_context_profile,
                           context_surprisal_matrix, fit_surprisal_scales)
from iotids.paper_protocol import (atomic_json, clean_transform, evaluate_arrays,
                                  load_frame, log, sha256, verify_bundle)
from .baselines import energy_score, fit_openmax, score_openmax


METHODS = ("global_msp", "classwise_msp", "global_msp_context",
           "classwise_msp_context", "classwise_msp_context_no_source",
           "energy", "openmax")


def upper_threshold(scores, allowed):
    """Conservative strict-'>' threshold allowing at most `allowed` errors.

    None disables this rejection branch when calibration has no eligible rows.
    Ties can make achieved FAR smaller than its budget; no random tie breaking.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or allowed < 0:
        raise ValueError("Finite scores and a nonnegative integer budget required")
    if not len(scores):
        return None
    if allowed >= len(scores):
        return float(np.nextafter(scores.min(), -np.inf))
    return float(np.partition(scores, len(scores) - int(allowed) - 1)[len(scores) - int(allowed) - 1])


def above(scores, threshold):
    return np.zeros(len(scores), dtype=bool) if threshold is None else np.asarray(scores) > threshold


def fit_operating_point(pred, score, benign_index, target, context=None):
    """Inputs contain ONLY truth-benign operating-calibration records."""
    pred, score = np.asarray(pred), np.asarray(score)
    if len(pred) == 0 or len(pred) != len(score):
        raise ValueError("Empty or inconsistent benign operating calibration")
    gate = pred == benign_index
    base_errors = int((~gate).sum())
    allowed = int(np.floor(float(target) * len(pred) + 1e-12))
    result = {"target_benign_far": float(target), "benign_calibration_n": len(pred),
              "allowed_errors": allowed, "base_errors": base_errors,
              "base_benign_far": base_errors / len(pred), "feasible": base_errors <= allowed}
    if not result["feasible"]:
        result["reason"] = "Known classifier's benign-to-known-attack errors already exceed budget"
        return result
    residual = allowed - base_errors
    confidence_budget = residual if context is None else residual // 2
    context_budget = 0 if context is None else residual - confidence_budget
    threshold = upper_threshold(score[gate], confidence_budget)
    eta = None if context is None else upper_threshold(np.asarray(context)[gate], context_budget)
    rejected = above(score, threshold)
    if context is not None:
        rejected |= gate & above(context, eta)
    errors = int(((~gate) | rejected).sum())
    if errors > allowed:
        raise AssertionError("Empirical calibration FAR budget exceeded")
    result.update(confidence_threshold=threshold, context_threshold=eta,
                  confidence_extra_error_budget=confidence_budget,
                  context_extra_error_budget=context_budget,
                  achieved_calibration_errors=errors,
                  achieved_calibration_benign_far=errors / len(pred))
    return result


def apply_operating_point(pred, score, point, benign_index, unknown_id, context=None):
    if not point["feasible"]:
        raise ValueError("Do not evaluate an infeasible operating point as feasible")
    pred = np.asarray(pred)
    rejected = above(score, point["confidence_threshold"])
    if context is not None:
        rejected |= (pred == benign_index) & above(context, point["context_threshold"])
    return np.where(rejected, unknown_id, pred)


def no_source_relations():
    seen, specs = set(), []
    for spec in CONTEXT_ABLATIONS["full_role_context"]:
        columns = tuple(c for c in spec.columns if c != "id.orig_h")
        if columns and columns not in seen:
            seen.add(columns)
            specs.append(RelationSpec("no_source_" + str(len(specs)), columns))
    return tuple(specs)


def _predict_both(model, probe, frame, rows, preprocessor, config, clip):
    logits = np.empty((len(rows), model.output_shape[-1]), dtype=np.float32)
    probabilities = np.empty_like(logits)
    chunk = int(config["transform_chunk_rows"])
    batch = int(config["inference_batch_size"])
    for start in range(0, len(rows), chunk):
        x = clean_transform(frame.iloc[rows[start:start + chunk]], preprocessor, clip)
        for j in range(0, len(x), batch):
            p, z = probe(x[j:j + batch], training=False)
            probabilities[start + j:start + j + len(p)] = p.numpy()
            logits[start + j:start + j + len(z)] = z.numpy()
        if start % (chunk * 20) == 0:
            log(f"Supplement scoring: {start:,}/{len(rows):,}")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite network scores")
    return probabilities, logits


def evaluate_run(config, root, run_dir, output_dir):
    import tensorflow as tf
    root, run_dir, output_dir = Path(root), Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_bundle(run_dir / "bundle")
    meta = json.loads((run_dir / "bundle/metadata.json").read_text(encoding="utf-8"))
    done = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    if sha256(run_dir / "metrics.json") != done["metrics_sha256"]:
        raise ValueError("Source metrics checksum mismatch")
    dataset = Path(config.get("dataset") or (root / meta["config"]["dataset"]))
    if sha256(dataset) != meta["dataset_sha256"]:
        raise ValueError("Dataset does not match the trained model")
    frame = load_frame(dataset, meta["config"]["label_column"])
    truth = frame[meta["config"]["label_column"]].to_numpy(dtype=str)
    splits = dict(np.load(run_dir / "split_indices.npz", allow_pickle=False))
    known, unknown = meta["known_classes"], meta["unknown_classes"]
    benign, uid = int(meta["benign_index"]), len(known)
    mapping = {v: i for i, v in enumerate(known)}
    for part in ("train", "validation", "calibration"):
        if not set(truth[splits[part]]).issubset(known):
            raise ValueError("Unknown/excluded class in a fitting partition")
    for i, part in enumerate(("train", "validation", "calibration", "test")):
        for other in ("train", "validation", "calibration", "test")[i + 1:]:
            if np.intersect1d(splits[part], splits[other]).size:
                raise ValueError("Overlapping row IDs")
    cal_rows = splits["calibration"]
    cal_y = np.asarray([mapping[t] for t in truth[cal_rows]], dtype=np.int64)
    shape, operating = train_test_split(np.arange(len(cal_rows)), test_size=0.5,
                                        stratify=cal_y, random_state=int(meta["seed"]) + 1000)
    operating_benign = operating[cal_y[operating] == benign]
    shape_benign = shape[cal_y[shape] == benign]
    if not len(operating_benign) or not len(shape_benign):
        raise ValueError("Both calibration halves require benign traffic")
    np.savez_compressed(output_dir / "calibration_partition.npz",
                        shape_row_ids=cal_rows[shape], operating_row_ids=cal_rows[operating])
    preprocessor = joblib.load(run_dir / "bundle/preprocessor.joblib")
    model = tf.keras.models.load_model(run_dir / "bundle/classifier.h5", compile=False)
    probe = tf.keras.Model(model.input, [model.output, model.get_layer("logits").output])
    clip = meta["standardized_feature_clip"]
    cal_p, cal_z = _predict_both(model, probe, frame, cal_rows, preprocessor, config, clip)
    test_p, test_z = _predict_both(model, probe, frame, splits["test"], preprocessor, config, clip)
    cal_pred, test_pred = cal_p.argmax(1), test_p.argmax(1)
    tau, counts, sources = [], [], []
    for k in range(uid):
        correct = shape[(cal_y[shape] == k) & (cal_pred[shape] == k)]
        selected = correct if len(correct) >= config["minimum_class_calibration_samples"] else shape[cal_y[shape] == k]
        if not len(selected):
            raise ValueError("Class absent from shape calibration")
        tau.append(float(np.quantile(cal_p[selected, k].astype(np.float64), 1 - config["shape_acceptance"])))
        counts.append(len(selected))
        sources.append("correct_only" if selected is correct else "all_true_class_fallback")
    tau = np.asarray(tau, dtype=np.float64)
    contexts = {}
    benign_train = splits["train"][truth[splits["train"]] == known[benign]]
    for name, specs in (("full", CONTEXT_ABLATIONS["full_role_context"]),
                        ("no_source", no_source_relations())):
        profile = fit_context_profile(frame.iloc[benign_train], specs)
        matrix, relation_names = context_surprisal_matrix(profile, frame.iloc[cal_rows[shape_benign]])
        scales = fit_surprisal_scales(matrix, config["context_scale_percentile"])
        values = []
        for rows, pred in ((cal_rows, cal_pred), (splits["test"], test_pred)):
            scores = np.zeros(len(rows), dtype=np.float32)
            gate = np.flatnonzero(pred == benign)
            for start in range(0, len(gate), config["transform_chunk_rows"]):
                ids = gate[start:start + config["transform_chunk_rows"]]
                m, _ = context_surprisal_matrix(profile, frame.iloc[rows[ids]])
                scores[ids] = np.max(m / scales, axis=1)
            values.append(scores)
        contexts[name] = values
        joblib.dump({"profile": profile, "scales": scales}, output_dir / ("context_" + name + ".joblib"), compress=3)
    log("Fitting OpenMax to all correctly classified known training logits")
    train_p, train_z = _predict_both(model, probe, frame, splits["train"], preprocessor, config, clip)
    train_y = np.asarray([mapping[t] for t in truth[splits["train"]]], dtype=np.int64)
    baseline_failures = {}
    try:
        om_state = fit_openmax(train_z, train_y, tail_size=config["openmax_tail_size"], alpha=config["openmax_alpha"])
        atomic_json(output_dir / "openmax_state.json", om_state)
        om_cal_pred, om_cal_score = score_openmax(cal_z, om_state)
        om_test_pred, om_test_score = score_openmax(test_z, om_state)
    except ValueError as exc:
        # A class with a degenerate tail is a real baseline failure, not a
        # reason to jitter observations, tune the tail on test data, or lose
        # every other experiment. Dependency ImportError still fails loudly.
        baseline_failures["openmax"] = str(exc)
        atomic_json(output_dir / "openmax_failure.json", {"error": str(exc), "no_fallback_used": True})
        om_cal_pred, om_test_pred = cal_pred, test_pred
        om_cal_score, om_test_score = None, None
        log(f"OPENMAX UNAVAILABLE for this run: {exc}")
    del frame, train_p, train_z, train_y, model, probe
    gc.collect()
    cal_class = tau[cal_pred] - cal_p.max(1).astype(np.float64)
    test_class = tau[test_pred] - test_p.max(1).astype(np.float64)
    scores = {
        "global_msp": (-cal_p.max(1).astype(np.float64), -test_p.max(1).astype(np.float64)),
        "classwise_msp": (cal_class, test_class),
        "global_msp_context": (-cal_p.max(1).astype(np.float64), -test_p.max(1).astype(np.float64)),
        "classwise_msp_context": (cal_class, test_class),
        "classwise_msp_context_no_source": (cal_class, test_class),
        "energy": (energy_score(cal_z), energy_score(test_z)),
        "openmax": (om_cal_score, om_test_score)}
    result = {"schema": "paper-supplement-v2-evaluation", "scenario": meta["scenario"], "seed": meta["seed"],
              "known_classes": known, "unknown_classes": unknown,
              "source_metrics_sha256": sha256(run_dir / "metrics.json"),
              "source_bundle_manifest": json.loads((run_dir / "bundle/manifest.json").read_text()),
              "dataset_sha256": meta["dataset_sha256"], "config": config,
              "class_shape_thresholds": tau, "class_shape_counts": counts,
              "class_shape_sources": sources, "baseline_failures": baseline_failures, "calibration_shape_n": len(shape),
              "calibration_operating_n": len(operating), "targets": {},
              "note": "Recalibrated known-only study, not frozen legacy 87.14% results. FAR targets are empirical calibration budgets, not test guarantees."}
    prediction_record = {"row_id": splits["test"], "true_label": truth[splits["test"]], "dnn_class_id": test_pred}
    for target in config["far_targets"]:
        key, entries = f"{target:.6f}", {}
        for name in METHODS:
            if name in baseline_failures:
                entries[name] = {"calibration": {"feasible": False, "target_benign_far": target,
                                                  "reason": "Baseline fit failed: " + baseline_failures[name]},
                                 "metrics": None}
                continue
            cal_s, test_s = scores[name]
            cp, tp = (om_cal_pred, om_test_pred) if name == "openmax" else (cal_pred, test_pred)
            context_name = "no_source" if name.endswith("no_source") else "full"
            cc, tc = contexts[context_name] if "context" in name else (None, None)
            point = fit_operating_point(cp[operating_benign], cal_s[operating_benign], benign, target,
                                        None if cc is None else cc[operating_benign])
            entry = {"calibration": point, "metrics": None}
            if point["feasible"]:
                pred = apply_operating_point(tp, test_s, point, benign, uid, tc)
                entry["metrics"] = evaluate_arrays(truth[splits["test"]], pred, known, unknown)
                prediction_record[name + "__" + key] = pred.astype(np.int16)
            entries[name] = entry
        result["targets"][key] = entries
    with gzip.open(output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(prediction_record).to_csv(handle, index=False)
    np.savez_compressed(output_dir / "calibration_scores.npz", row_ids=cal_rows,
                        labels=cal_y, probabilities=cal_p, logits=cal_z,
                        context_full=contexts["full"][0], context_no_source=contexts["no_source"][0])
    atomic_json(output_dir / "metrics.json", result)
    log(f"Matched-FAR evaluation saved: {output_dir}")
