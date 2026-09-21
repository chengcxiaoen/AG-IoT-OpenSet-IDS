"""Bounded, batch-one CPU measurement of the complete frozen IDS pipeline."""
from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from iotids.paper_protocol import (CSV_STRING_COLUMNS, atomic_json, clean_transform,
                                  decision_arrays, log, score_context, system_info, verify_bundle)


def selected_csv_rows(dataset, selected):
    """Read only a bounded selected subset into memory; no label-based selection."""
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    parts, offset = [], 0
    for chunk in pd.read_csv(dataset, chunksize=50000, low_memory=False, dtype=CSV_STRING_COLUMNS):
        ids = selected[(selected >= offset) & (selected < offset + len(chunk))] - offset
        if len(ids):
            parts.append(chunk.iloc[ids].copy())
        offset += len(chunk)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != len(selected):
        raise ValueError("Selected benchmark rows missing from dataset")
    return frame, selected


def run_benchmark(config, dataset, run_dir, output):
    import tensorflow as tf
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Benchmark must start with CUDA_VISIBLE_DEVICES=-1")
    threads = int(config["threads"])
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    verify_bundle(run_dir / "bundle")
    meta = json.loads((run_dir / "bundle/metadata.json").read_text(encoding="utf-8"))
    rows = np.load(run_dir / "split_indices.npz", allow_pickle=False)["test"]
    rng = np.random.default_rng(config["sample_seed"])
    ids = rng.choice(rows, min(len(rows), int(config["samples"])), replace=False)
    frame, ids = selected_csv_rows(dataset, ids)
    preprocessor = joblib.load(run_dir / "bundle/preprocessor.joblib")
    ctx = joblib.load(run_dir / "bundle/context.joblib")
    model = tf.keras.models.load_model(run_dir / "bundle/classifier.h5", compile=False)
    gc.collect()
    process = psutil.Process()
    memory_ready = process.memory_info().rss
    memory_peak = [memory_ready]
    stop = threading.Event()

    def watch():
        while not stop.wait(0.02):
            memory_peak[0] = max(memory_peak[0], process.memory_info().rss)

    def infer(one):
        x = clean_transform(one, preprocessor, meta["standardized_feature_clip"])
        p = model(x, training=False).numpy()
        gate = int(p.argmax(1)[0]) == meta["benign_index"]
        s = score_context(one, ctx["profile"], ctx["scales"]) if gate else np.zeros(1, dtype=np.float32)
        pred = int(decision_arrays(p, s, meta)["classwise_msp_context"][0])
        return pred, gate

    log(f"CPU benchmark: {len(ids)} fixed random test flows x {config['repeats']} passes; batch=1")
    for i in range(int(config["warmup"])):
        infer(frame.iloc[[i % len(frame)]])
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    elapsed = np.empty((int(config["repeats"]), len(frame)), dtype=np.float64)
    decisions = np.empty(elapsed.shape, dtype=np.int16)
    gates = np.empty(elapsed.shape, dtype=bool)
    try:
        for repeat in range(len(elapsed)):
            order = np.random.default_rng(config["sample_seed"] + repeat + 1).permutation(len(frame))
            for count, i in enumerate(order):
                start = time.perf_counter_ns()
                pred, gate = infer(frame.iloc[[i]])
                elapsed[repeat, i] = (time.perf_counter_ns() - start) / 1e6
                decisions[repeat, i], gates[repeat, i] = pred, gate
                if count % 200 == 0:
                    log(f"CPU pass {repeat + 1}/{len(elapsed)}: {count}/{len(frame)}")
    finally:
        stop.set()
        watcher.join()
    if not np.isfinite(elapsed).all() or not (elapsed > 0).all():
        raise ValueError("Incomplete or invalid latency samples")
    # Verify decision agreement against the stored reference prediction artifact.
    reference = {}
    selected = set(ids.tolist())
    for part in pd.read_csv(run_dir / "test_predictions.csv.gz", chunksize=50000,
                            usecols=["row_id", "classwise_msp_context"]):
        take = part[part.row_id.isin(selected)]
        reference.update(zip(take.row_id, take.classwise_msp_context))
    reference_pred = np.asarray([reference[int(i)] for i in ids])
    disagreement = (decisions != reference_pred[None, :])
    def describe(values):
        if not values.size:
            return None
        return {"n": int(values.size), "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
                "p99_ms": float(np.quantile(values, .99))}
    np.savez_compressed(output / "latency_samples.npz", row_ids=ids, milliseconds=elapsed,
                        decisions=decisions, context_evaluated=gates, reference_predictions=reference_pred)
    result = {"config": config, "system": system_info(), "model_parameters": model.count_params(),
              "model_file_bytes": (run_dir / "bundle/classifier.h5").stat().st_size,
              "preprocessor_file_bytes": (run_dir / "bundle/preprocessor.joblib").stat().st_size,
              "context_file_bytes": (run_dir / "bundle/context.joblib").stat().st_size,
              "full_pipeline": describe(elapsed), "context_gate_true": describe(elapsed[gates]),
              "context_gate_false": describe(elapsed[~gates]),
              "per_pass_mean_ms": elapsed.mean(axis=1), "context_gate_fraction": float(gates.mean()),
              "rss_ready_bytes": memory_ready, "sampled_peak_rss_bytes": memory_peak[0],
              "sampled_incremental_peak_bytes": max(0, memory_peak[0] - memory_ready),
              "reference_decision_mismatches": int(disagreement.sum()),
              "repeat_decision_mismatches": int((decisions != decisions[0]).sum()),
              "cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
              "timing_scope": "Batch=1: row extraction, trained preprocessing, MLP forward and host materialization, MSP and benign-gated context, final decision. CSV input parsing, model loading, and warmup excluded.",
              "limitations": "Server CPU only, not an agricultural edge-device measurement. RSS includes Python/TensorFlow and is sampled at 20 ms. Random subset limits precision of rare-branch latency estimates. CPU/GPU floating-point threshold disagreement is reported, not hidden."}
    atomic_json(output / "benchmark.json", result)
    log(f"CPU full-pipeline mean: {result['full_pipeline']['mean_ms']:.3f} ms")
