"""Small end-to-end software check, never a scientific experiment result.

Requires the real TensorFlow and LibMR dependencies. Writes only a caller-
specified scratch directory, and exercises group training, all seven detectors,
three calibration budgets, provenance, and the separate batch-one CPU worker.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import pandas as pd
    from iotids.paper_protocol import atomic_json, sha256
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--without-native-openmax", action="store_true",
                        help="Test all other real stages and explicit unavailable-baseline handling; NOT an OpenMax test")
    parser.add_argument("--internal-unavailable-evaluation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if args.internal_unavailable_evaluation:
        from unittest.mock import patch
        from iotids_supplement.suite import worker
        cfg = json.loads((scratch / "supplement.json").read_text())
        with patch("iotids_supplement.evaluation.fit_openmax", side_effect=ValueError(
                "SOFTWARE TEST ONLY: native OpenMax intentionally unavailable; no synthetic replacement scores")):
            worker(ROOT, cfg, "evaluate", "grouped", "primary_arp_port", 42,
                   scratch / "synthetic.csv", scratch / "old/results", scratch / "results", allow_cpu=True)
        return
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError("Use a new empty scratch directory; no existing files are deleted")
    scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(730)
    records = []
    for label, center, number in (("Benign", 0, 300), ("TCP Flood", 20, 300),
                                  ("UDP Flood", -20, 300), ("Arp Spoofing", 40, 60),
                                  ("Port Scanning", -40, 60)):
        for i in range(number):
            record = {"traffic": label, "is_attack": int(label != "Benign"),
                      "id.orig_h": "device" + str(i % 3), "id.resp_h": "server",
                      "id.orig_p": str(1024 + i % 10), "id.resp_p": "80",
                      "history": "ShADadfF", "proto": "tcp", "service": "http", "conn_state": "SF",
                      "local_orig": "-", "local_resp": "-", "tunnel_parents": "-",
                      "feature_x": center + rng.normal(0, .5), "feature_y": center * 2 + rng.normal(0, .5)}
            records.extend([record.copy(), record.copy()])
    dataset = scratch / "synthetic.csv"
    pd.DataFrame(records).to_csv(dataset, index=False)
    original = json.loads((ROOT / "configs/paper_final_legacy8.json").read_text())
    original.update(dataset=str(dataset), epochs=15, patience=4, learning_rate=.001, batch_size=64,
                    transform_chunk_rows=256, inference_batch_size=64, seeds=[42],
                    scenarios={"primary_arp_port": ["Arp Spoofing", "Port Scanning"]})
    train_config = scratch / "training.json"
    atomic_json(train_config, original)
    cfg = json.loads((ROOT / "configs/paper_supplement_v2.json").read_text())
    cfg.update(legacy_config=str(train_config), expected_dataset_sha256=sha256(dataset),
               transform_chunk_rows=256, inference_batch_size=64,
               benchmark={"scenario": "primary_arp_port", "seed": 42, "threads": 1,
                          "warmup": 2, "samples": 12, "repeats": 2, "sample_seed": 91})
    config = scratch / "supplement.json"
    atomic_json(config, cfg)
    output = scratch / "results"
    base = [sys.executable, "-u", str(ROOT / "experiments/run_paper_supplement.py"), "_worker",
            "--config", str(config), "--dataset", str(dataset), "--output", str(output),
            "--old-root", str(scratch / "old"), "--legacy-results", str(scratch / "old/results"),
            "--protocol", "grouped", "--scenario", "primary_arp_port", "--seed", "42", "--allow-cpu"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
    for stage in ("train", "evaluate", "benchmark"):
        command = base + ["--stage", stage]
        if stage == "evaluate" and args.without_native_openmax:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--scratch", str(scratch),
                       "--internal-unavailable-evaluation"]
        subprocess.run(command, check=True, env=env)
    # Exact same completed invocation must skip, not retrain/recalibrate.
    repeat = base + ["--stage", "evaluate"]
    if args.without_native_openmax:
        repeat = [sys.executable, "-u", str(Path(__file__).resolve()), "--scratch", str(scratch),
                  "--internal-unavailable-evaluation"]
    subprocess.run(repeat, check=True, env=env)
    value = json.loads((output / "matched/grouped/primary_arp_port/seed42/metrics.json").read_text())
    assert len(value["targets"]) == 3
    assert all(len(methods) == 7 for methods in value["targets"].values())
    audit = json.loads((output / "grouped_runs/primary_arp_port/seed42/data_audit.json").read_text())
    assert audit["test_rows_with_observable_duplicate_in_train"] == 0
    assert audit["group_audit"]["partitions_disjoint_by_group"]
    bench = json.loads((output / "benchmark_cpu/benchmark.json").read_text())
    assert bench["full_pipeline"]["n"] == 24
    assert bench["repeat_decision_mismatches"] == 0
    if args.without_native_openmax:
        assert "openmax" in value["baseline_failures"]
        print("SYNTHETIC INTEGRATION PASSED: train/6 real detectors/explicit OpenMax-unavailable handling/benchmark/resume. Native OpenMax NOT TESTED. NOT PAPER RESULTS.")
    else:
        assert not value["baseline_failures"]
        print("SYNTHETIC INTEGRATION PASSED: train/evaluate/OpenMax/benchmark/resume. NOT PAPER RESULTS.")


if __name__ == "__main__":
    main()
