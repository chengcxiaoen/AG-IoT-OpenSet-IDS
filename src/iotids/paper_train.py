"""One full-data paper run. Called in a fresh process for each seed/scenario."""
from __future__ import annotations

import gc
import gzip
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .class_conditional import calibrate_class_confidence_thresholds
from .constants import BENIGN_LABEL, DEFAULT_COLUMNS_TO_DROP
from .context import CONTEXT_ABLATIONS, fit_context_profile, fit_surprisal_scales, context_surprisal_matrix
from .data import fit_feature_preprocessor
from .paper_protocol import (
    METHODS, PROPOSED, atomic_json, clean_transform, decision_arrays, evaluate_arrays,
    load_frame, log, protocol_digest, score_context, sha256, source_hash, split_indices, system_info,
)


def _memmap_features(frame, rows, preprocessor, config, path):
    size = config["transform_chunk_rows"]
    probe = clean_transform(frame.iloc[rows[:1]], preprocessor, config["standardized_feature_clip"])
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                     shape=(len(rows), probe.shape[1]))
    for start in range(0, len(rows), size):
        selected = rows[start:start + size]
        matrix[start:start + len(selected)] = clean_transform(
            frame.iloc[selected], preprocessor, config["standardized_feature_clip"])
        if start % (size * 20) == 0:
            log(f"Features {Path(path).name}: {start:,}/{len(rows):,}")
    matrix.flush()
    return matrix


def _predict(model, matrix, batch_size):
    # Direct batched forward passes: no full-array tensor or redundant embeddings.
    output = np.empty((len(matrix), model.output_shape[-1]), dtype=np.float32)
    for start in range(0, len(matrix), batch_size):
        output[start:start + batch_size] = model(
            np.asarray(matrix[start:start + batch_size]), training=False).numpy()
    if not np.isfinite(output).all():
        raise ValueError("Non-finite model output")
    return output


def run(config, root, output, scenario, seed, allow_cpu=False):
    import tensorflow as tf
    from .models import build_lightweight_dnn
    from .utils import configure_tensorflow, set_global_seed
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    dataset = root / config["dataset"]
    if config["sampling"] != "none":
        raise ValueError("Paper protocol forbids row caps and resampling")
    if config["context"] != "full_role_context":
        raise ValueError("Paper v1 freezes the original Full-Role context")
    for name in ("bundle", "cache"):
        (output / name).mkdir(parents=True, exist_ok=True)
    digest = protocol_digest(config, sha256(dataset), source_hash(root))
    identity = {"protocol_digest": digest, "scenario": scenario, "seed": seed}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run directory belongs to a different protocol; use a NEW output directory")
    atomic_json(identity_path, identity)
    if (output / "COMPLETE.json").exists():
        from .paper_protocol import verify_bundle
        verify_bundle(output / "bundle")
        done = json.loads((output / "COMPLETE.json").read_text())
        if sha256(output / "metrics.json") != done["metrics_sha256"]:
            raise ValueError("Completed metrics checksum mismatch")
        log(f"Verified complete run: {scenario}, seed={seed}; skipping")
        return
    configure_tensorflow()
    devices = tf.config.list_physical_devices("GPU")
    if not devices and not allow_cpu:
        raise RuntimeError("TensorFlow sees no GPU. Fix CUDA or explicitly pass --allow-cpu")
    set_global_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    with tf.device("/GPU:0" if devices else "/CPU:0"):
        tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32))).numpy()
    log(f"BEGIN {scenario} seed={seed}; GPU={devices}")
    frame = load_frame(dataset, config["label_column"])
    labels = frame[config["label_column"]].to_numpy(dtype=str)
    unknown = config["scenarios"][scenario]
    excluded = set(config.get("excluded_classes", []))
    known = [BENIGN_LABEL] + sorted(set(labels) - set(unknown) - excluded - {BENIGN_LABEL})
    if BENIGN_LABEL not in labels or len(known) < 2:
        raise ValueError("Dataset must contain Benign and known attacks")
    mapping = {name: i for i, name in enumerate(known)}
    splits = split_indices(labels, unknown, seed, config)
    np.savez_compressed(output / "split_indices.npz", **splits)
    def distribution(rows):
        return pd.Series(labels[rows]).value_counts().to_dict()
    log("Partition sizes: " + str({name: len(rows) for name, rows in splits.items()}))
    # Audit duplicate observable flows separately from row-ID disjointness.
    obs_columns = [col for col in frame if col not in (config["label_column"], "is_attack")]
    hashes = pd.util.hash_pandas_object(frame[obs_columns], index=False).to_numpy()
    shared = np.isin(hashes[splits["test"]], np.unique(hashes[splits["train"]]))
    audit = {"source_rows": len(frame), "row_cap": None, "resampling": "none",
             "protocol_rows": len(frame) - len(splits["excluded"]),
             "explicitly_excluded_classes": list(excluded),
             "explicitly_excluded_rows": len(splits["excluded"]),
             "unknown_excluded_before_test": True, "partitions_disjoint_by_row_id": True,
             "all_rows_accounted_for": True,
             "distribution": {name: distribution(rows) for name, rows in splits.items()},
             "test_rows_with_observable_duplicate_in_train": int(shared.sum()),
             "test_duplicate_fraction": float(shared.mean()),
             "split_type": "stratified_random_rows; NOT device/time-independent",
             "known_partition_fractions": "64% train, 8% validation, 8% calibration, 20% test for default config",
             "unknown_partition": "100% held-out test; never balanced or sampled",
             "warning": "Do not claim unseen-device or temporal generalization from this random-row protocol."}
    atomic_json(output / "data_audit.json", audit)
    del hashes, shared
    gc.collect()
    log("Fitting training-only feature preprocessor")
    preprocessor = fit_feature_preprocessor(frame.iloc[splits["train"]], config["label_column"],
                                           columns_to_drop=DEFAULT_COLUMNS_TO_DROP,
                                           allowed_categorical_columns=("proto", "service", "conn_state"))
    train_x = _memmap_features(frame, splits["train"], preprocessor, config, output / "cache/train.npy")
    val_x = _memmap_features(frame, splits["validation"], preprocessor, config, output / "cache/validation.npy")
    cal_x = _memmap_features(frame, splits["calibration"], preprocessor, config, output / "cache/calibration.npy")
    train_y = np.asarray([mapping[x] for x in labels[splits["train"]]], dtype=np.int64)
    val_y = np.asarray([mapping[x] for x in labels[splits["validation"]]], dtype=np.int64)
    cal_y = np.asarray([mapping[x] for x in labels[splits["calibration"]]], dtype=np.int64)
    batch_size = config["batch_size"]

    class FullSequence(tf.keras.utils.Sequence):
        def __init__(self, x, y, shuffle):
            self.x, self.y, self.shuffle = x, y, shuffle
            self.order = np.arange(len(y))
            self.rng = np.random.default_rng(seed)
            self.on_epoch_end()
        def __len__(self):
            return (len(self.y) + batch_size - 1) // batch_size
        def __getitem__(self, index):
            rows = self.order[index * batch_size:(index + 1) * batch_size]
            return np.asarray(self.x[rows]), self.y[rows]
        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

    class Guard(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if not all(np.isfinite(v) for v in logs.values()):
                raise FloatingPointError("Training/validation metric is non-finite")
            log(f"Epoch {epoch + 1}/{config['epochs']}: {logs}")

    model, _, _ = build_lightweight_dnn(train_x.shape[1], len(known),
                                      config["learning_rate"], config["dropout"])
    weights = None
    if config["class_weight"] == "balanced":
        counts = np.bincount(train_y, minlength=len(known))
        weights = {i: len(train_y) / (len(known) * int(n)) for i, n in enumerate(counts)}
    elif config["class_weight"] != "none":
        raise ValueError("Unsupported class weight")
    callbacks = [Guard(), tf.keras.callbacks.CSVLogger(str(output / "training_history.csv")),
                 tf.keras.callbacks.ModelCheckpoint(str(output / "bundle/classifier.h5"),
                                                     save_best_only=True, monitor="val_loss"),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"],
                                                  restore_best_weights=True)]
    train_start = time.perf_counter()
    log(f"Training on ALL {len(train_y):,} training rows, no cap/resampling; weights={weights}")
    history = model.fit(FullSequence(train_x, train_y, True),
                        validation_data=FullSequence(val_x, val_y, False),
                        epochs=config["epochs"], class_weight=weights,
                        callbacks=callbacks, verbose=2, workers=1,
                        use_multiprocessing=False, max_queue_size=2)
    training_seconds = time.perf_counter() - train_start
    # Always reload the exact artifact to be shipped to the benchmark process.
    model = tf.keras.models.load_model(output / "bundle/classifier.h5", compile=False)
    log("Calibrating thresholds using known calibration data only")
    cal_probs = _predict(model, cal_x, config["inference_batch_size"])
    class_cal = calibrate_class_confidence_thresholds(
        cal_probs, cal_y, config["known_acceptance_rate"],
        config["minimum_class_calibration_samples"])
    class_thresholds = (1.0 - class_cal.confidence_anomaly_max).astype(np.float32)
    global_threshold = float(np.percentile(cal_probs.max(axis=1),
                                           100 * (1 - config["known_acceptance_rate"])))
    benign_train = splits["train"][labels[splits["train"]] == BENIGN_LABEL]
    benign_cal = splits["calibration"][labels[splits["calibration"]] == BENIGN_LABEL]
    profile = fit_context_profile(frame.iloc[benign_train], CONTEXT_ABLATIONS[config["context"]])
    cal_matrix, relation_names = context_surprisal_matrix(profile, frame.iloc[benign_cal])
    scales = fit_surprisal_scales(cal_matrix, config["context_scale_percentile"])
    context_cal_scores = np.max(cal_matrix / scales, axis=1)
    context_threshold = float(np.percentile(context_cal_scores,
                                            config["context_benign_acceptance_rate"] * 100))
    metadata = {**identity, "known_classes": known, "unknown_classes": unknown,
                "benign_index": known.index(BENIGN_LABEL), "global_threshold": global_threshold,
                "class_thresholds": class_thresholds, "context_threshold": context_threshold,
                "standardized_feature_clip": config["standardized_feature_clip"],
                "relation_names": relation_names, "input_dim": int(train_x.shape[1]),
                "classifier_parameters": model.count_params(), "source_sha256": source_hash(root),
                "dataset_sha256": sha256(dataset), "feature_names": preprocessor.feature_names,
                "class_calibration_counts": class_cal.calibration_counts,
                "class_calibration_sources": class_cal.calibration_sources,
                "config": config, "context_gate": "DNN predicted Benign; no extra model or online update"}
    joblib.dump(preprocessor, output / "bundle/preprocessor.joblib", compress=3)
    joblib.dump({"profile": profile, "scales": scales}, output / "bundle/context.joblib", compress=3)
    atomic_json(output / "bundle/metadata.json", metadata)
    artifact_names = ("classifier.h5", "preprocessor.joblib", "context.joblib", "metadata.json")
    atomic_json(output / "bundle/manifest.json", {
        "sha256": {name: sha256(output / "bundle" / name) for name in artifact_names}})
    # Save calibration scores for audit without needing to train again.
    np.savez_compressed(output / "calibration_scores.npz", probabilities=cal_probs,
                        labels=cal_y, row_ids=splits["calibration"],
                        benign_context_scores=context_cal_scores, benign_context_row_ids=benign_cal)
    del train_x, val_x, cal_x, cal_probs, cal_matrix
    gc.collect()
    test_rows = splits["test"]
    predictions = {name: np.empty(len(test_rows), dtype=np.int16) for name in METHODS}
    log(f"Evaluating ALL {len(test_rows):,} test rows; six paired ablations share one DNN")
    with gzip.open(output / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        first = True
        for start in range(0, len(test_rows), config["transform_chunk_rows"]):
            rows = test_rows[start:start + config["transform_chunk_rows"]]
            chunk = frame.iloc[rows]
            x = clean_transform(chunk, preprocessor, config["standardized_feature_clip"])
            probs = _predict(model, x, config["inference_batch_size"])
            pred = probs.argmax(axis=1)
            context = np.zeros(len(rows), dtype=np.float32)
            mask = pred == metadata["benign_index"]
            if mask.any():
                context[mask] = score_context(chunk.iloc[np.flatnonzero(mask)], profile, scales)
            decisions = decision_arrays(probs, context, metadata)
            record = {"row_id": rows, "true_label": labels[rows], "dnn_class_id": pred,
                      "max_softmax": probs.max(axis=1), "class_threshold": class_thresholds[pred],
                      "context_evaluated": mask, "context_score": context}
            for name, decision in decisions.items():
                predictions[name][start:start + len(rows)] = decision
                record[name] = decision
            pd.DataFrame(record).to_csv(handle, index=False, header=first)
            first = False
            if start % (config["transform_chunk_rows"] * 10) == 0:
                log(f"Testing: {start:,}/{len(test_rows):,}")
    results = {name: evaluate_arrays(labels[test_rows], pred, known, unknown)
               for name, pred in predictions.items()}
    metrics = {**identity, "method": PROPOSED, "config": config, "data_audit": audit,
               "system": system_info(), "tensorflow_gpus": [str(d) for d in devices],
               "training_seconds": training_seconds, "epochs_completed": len(history.history["loss"]),
               "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
               "class_weights": weights, "classifier_parameters": model.count_params(),
               "methods": results, "total_run_seconds": time.perf_counter() - started,
               "note": "No latency claim from training time. No external-baseline/SOTA claim from ablations."}
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "COMPLETE.json", {**identity, "metrics_sha256": sha256(output / "metrics.json"),
                                          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"SUCCESS {scenario} seed={seed}: {results[PROPOSED]}")
