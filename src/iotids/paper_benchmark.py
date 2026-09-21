"""CPU batch-one whole-test replay. Excludes CSV IO and model loading from latency."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .paper_protocol import (
    CSV_STRING_COLUMNS, PROPOSED, atomic_json, clean_transform, decision_arrays,
    log, score_context, sha256, system_info, verify_bundle,
)


def run(root, run_dir, output, config, device_label=None):
    # Caller sets CUDA_VISIBLE_DEVICES and thread environment before TF import.
    import tensorflow as tf
    import psutil
    run_dir, output, root = Path(run_dir), Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    bcfg = config["benchmark"]
    if bcfg["batch_size"] != 1:
        raise ValueError("This latency protocol requires batch_size=1")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU benchmark process must hide GPUs before TensorFlow import")
    tf.config.threading.set_intra_op_parallelism_threads(bcfg["threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    bundle = run_dir / "bundle"
    manifest = verify_bundle(bundle)
    meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    expected_identity = {"scenario": bcfg["scenario"], "seed": bcfg["seed"]}
    if any(meta[key] != value for key, value in expected_identity.items()):
        raise ValueError("Benchmark must use the preselected seed/scenario, not the best test run")
    dataset = root / config["dataset"]
    if sha256(dataset) != meta["dataset_sha256"]:
        raise ValueError("Benchmark dataset differs from training dataset")
    benchmark_identity = {"bundle_manifest": manifest, "benchmark_config": bcfg,
                          "system": system_info(), "device_label": device_label}
    if (output / "COMPLETE.json").exists():
        old = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
        if old["identity"] != benchmark_identity:
            raise ValueError("Benchmark output belongs to another hardware/config; use a new directory")
        if sha256(output / "benchmark.json") != json.loads((output / "COMPLETE.json").read_text())["sha256"]:
            raise ValueError("Benchmark checksum mismatch")
        log("Verified existing full benchmark; skipping")
        return
    model = tf.keras.models.load_model(bundle / "classifier.h5", compile=False)
    preprocessor = joblib.load(bundle / "preprocessor.joblib")
    context = joblib.load(bundle / "context.joblib")

    @tf.function(input_signature=[tf.TensorSpec([1, meta["input_dim"]], tf.float32)])
    def forward(x):
        return model(x, training=False)

    def infer(one_row):
        features = clean_transform(one_row, preprocessor, meta["standardized_feature_clip"])
        probabilities = forward(tf.convert_to_tensor(features)).numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite deployment model output")
        pred = int(np.argmax(probabilities[0]))
        # Time ONLY the proposed deployment path, not the other five ablations.
        if probabilities[0, pred] < meta["class_thresholds"][pred]:
            return len(meta["known_classes"])
        if pred == meta["benign_index"]:
            scores = score_context(one_row, context["profile"], context["scales"])
            if scores[0] > meta["context_threshold"]:
                return len(meta["known_classes"])
        return pred

    splits = np.load(run_dir / "split_indices.npz")
    test_rows = splits["test"]
    audit = json.loads((run_dir / "data_audit.json").read_text())
    expected = np.full(audit["source_rows"], -1, dtype=np.int16)
    for part in pd.read_csv(run_dir / "test_predictions.csv.gz", chunksize=50000,
                            usecols=["row_id", PROPOSED]):
        expected[part["row_id"].to_numpy(dtype=int)] = part[PROPOSED].to_numpy(dtype=np.int16)
    if int((expected >= 0).sum()) != len(test_rows):
        raise ValueError("Saved test prediction coverage is incomplete")
    selected = np.zeros(len(expected), dtype=bool)
    selected[test_rows] = True
    preview = pd.read_csv(dataset, nrows=1, dtype=CSV_STRING_COLUMNS)
    for _ in range(bcfg["warmup"]):
        infer(preview)
    process = psutil.Process()
    loaded_rss = process.memory_info().rss
    observed_rss = loaded_rss
    result_passes = []
    log(f"CPU benchmark: batch=1, {len(test_rows):,} test rows x {bcfg['repeats']} passes; not throughput/n")
    for repeat in range(bcfg["repeats"]):
        pass_start = time.perf_counter()
        last_progress = pass_start
        path = output / f"latency_pass{repeat + 1}.npy"
        latencies = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                             shape=(len(test_rows),))
        position, offset, mismatches = 0, 0, 0
        for chunk in pd.read_csv(dataset, chunksize=2048, low_memory=False, dtype=CSV_STRING_COLUMNS):
            local = np.flatnonzero(selected[offset:offset + len(chunk)])
            for local_row in local:
                row_id = offset + int(local_row)
                start_ns = time.perf_counter_ns()
                prediction = infer(chunk.iloc[[local_row]])
                latencies[position] = (time.perf_counter_ns() - start_ns) / 1e6
                mismatches += int(prediction != expected[row_id])
                position += 1
                if position % 100 == 0 and time.perf_counter() - last_progress >= 30:
                    elapsed = time.perf_counter() - pass_start
                    remaining = elapsed / position * (len(test_rows) - position)
                    log(f"CPU pass {repeat + 1}: {position:,}/{len(test_rows):,}; elapsed={elapsed/60:.1f} min; rough remaining={remaining/60:.1f} min")
                    last_progress = time.perf_counter()
            offset += len(chunk)
            observed_rss = max(observed_rss, process.memory_info().rss)
        if position != len(test_rows) or offset != len(expected):
            raise ValueError("Benchmark did not replay exactly the complete test set")
        latencies.flush()
        stats = {"pass": repeat + 1, "rows": position, "mean_ms": float(np.mean(latencies)),
                 "p50_ms": float(np.percentile(latencies, 50)),
                 "p95_ms": float(np.percentile(latencies, 95)),
                 "p99_ms": float(np.percentile(latencies, 99)),
                 "prediction_mismatches_vs_evaluation": mismatches,
                 "mismatch_fraction": mismatches / position}
        result_passes.append(stats)
        atomic_json(output / "progress.json", {"completed_passes": result_passes})
        log(f"CPU replay finished: {stats}")
        del latencies
    means = [row["mean_ms"] for row in result_passes]
    report = {
        "identity": benchmark_identity, "device_type": bcfg["device_type"],
        "device_label": device_label or benchmark_identity["system"]["cpu_model"],
        "scenario": bcfg["scenario"], "preselected_model_seed": bcfg["seed"],
        "L_mean_single_flow_ms": float(np.mean(means)),
        "L_std_across_pass_means_ms": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        "batch_size": 1, "threads": bcfg["threads"], "full_test_replayed_each_pass": True,
        "passes": result_passes, "process_rss_after_warmup_bytes": loaded_rss,
        "observed_process_rss_max_bytes": observed_rss,
        "artifact_sizes_bytes": {name: (bundle / name).stat().st_size for name in manifest["sha256"]},
        "timing_boundary": "one ready flow record -> preprocessing -> DNN -> class MSP -> conditional context -> final label",
        "excluded": "CSV IO, model load, training, packet capture, flow aggregation/waiting, logging",
        "not_edge_hardware_proof": bcfg["device_type"] != "edge_cpu",
        "warning": "Process RSS includes Python/TF and bounded replay buffers; artifact file size is not runtime RAM. CPU/GPU numerical boundary differences are explicitly reported."}
    atomic_json(output / "benchmark.json", report)
    atomic_json(output / "COMPLETE.json", {"sha256": sha256(output / "benchmark.json")})
    log(f"SUCCESS CPU benchmark: mean single-flow latency = {report['L_mean_single_flow_ms']:.4f} ms")
