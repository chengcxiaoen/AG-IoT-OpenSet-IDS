"""Isolated workers, immutable run identities, checksums and complete summaries."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

from iotids.paper_protocol import atomic_json, log, sha256, verify_bundle


def code_hash(root):
    digest = hashlib.sha256()
    paths = list((root / "src/iotids").glob("*.py"))
    paths += list((root / "src/iotids_supplement").glob("*.py"))
    paths += [root / "experiments/run_paper_supplement.py"]
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_complete(directory):
    directory = Path(directory)
    marker = directory / "SUPPLEMENT_COMPLETE.json"
    if not marker.exists():
        return False
    done = json.loads(marker.read_text(encoding="utf-8"))
    for name, expected in done["sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"Completed output integrity failure: {path}")
    return True


def mark_complete(directory):
    paths = [p for p in directory.rglob("*") if p.is_file() and "cache" not in p.relative_to(directory).parts
             and p.name not in ("SUPPLEMENT_COMPLETE.json",) and not p.name.endswith(".tmp")]
    atomic_json(directory / "SUPPLEMENT_COMPLETE.json", {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": {p.relative_to(directory).as_posix(): sha256(p) for p in paths}})


def establish_identity(directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "supplement_identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Protocol/input changed. Do not mix runs; use a NEW output root: {directory}")
    atomic_json(path, identity)


@contextmanager
def suite_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "RUNNING.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another supplement process is using this output directory") from exc
    try:
        yield
    finally:
        handle.close()  # Kernel releases the advisory lock even after a crash.


def preflight(root, config, old_root, dataset, legacy_results, allow_cpu=False):
    import importlib.metadata
    import numpy as np
    import tensorflow as tf
    from .baselines import fit_openmax, score_openmax
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use the existing iotids310 Python 3.10 environment, not system python")
    required = {"tensorflow": "2.10.1", "numpy": "1.23.5", "pandas": "1.5.3",
                "scipy": "1.10.1", "scikit-learn": "1.2.2", "libmr": "0.1.9"}
    versions = {name: importlib.metadata.version(name) for name in required}
    if versions != required:
        raise RuntimeError(f"Use SETUP_SERVER.sh for the frozen dependencies: {versions}")
    if sha256(dataset) != config["expected_dataset_sha256"]:
        raise ValueError("Full Farm-Flow CSV checksum differs from the audited original")
    train_cfg = json.loads((root / config["legacy_config"]).read_text(encoding="utf-8"))
    for scenario in train_cfg["scenarios"]:
        for seed in train_cfg["seeds"]:
            run = legacy_results / "runs" / scenario / f"seed{seed}"
            done = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
            verify_bundle(run / "bundle")
            meta = json.loads((run / "bundle/metadata.json").read_text(encoding="utf-8"))
            if meta["dataset_sha256"] != config["expected_dataset_sha256"] or sha256(run / "metrics.json") != done["metrics_sha256"]:
                raise ValueError(f"Original run verification failed: {run}")
    gpus = tf.config.list_physical_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    if not gpus and not allow_cpu:
        raise RuntimeError("No TensorFlow GPU; check CUDA. CPU requires explicit --allow-cpu")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32))).numpy()
    rng = np.random.default_rng(1701)
    y = np.repeat(np.arange(3), 40)
    z = rng.normal(0, .3, (len(y), 3))
    z[np.arange(len(y)), y] += np.linspace(3., 7., len(y))
    state = fit_openmax(z, y)
    _, scores = score_openmax(z[:10], state)
    if not np.isfinite(scores).all():
        raise ValueError("LibMR smoke test failed")
    log(f"PREFLIGHT OK: all 10 legacy runs, full dataset, LibMR, versions={versions}; GPU={gpus}")


def worker(root, config, stage, protocol, scenario, seed, dataset, legacy_results, output, allow_cpu=False):
    config = dict(config, dataset=str(dataset))
    train_cfg = json.loads((root / config["legacy_config"]).read_text(encoding="utf-8"))
    group_dir = output / "grouped_runs" / scenario / f"seed{seed}"
    source_run = legacy_results / "runs" / scenario / f"seed{seed}" if protocol == "legacy" else group_dir
    destination = group_dir if stage == "train" else output / ("benchmark_cpu" if stage == "benchmark" else "matched")
    if stage == "evaluate":
        destination = destination / protocol / scenario / f"seed{seed}"
    identity = {"code_sha256": code_hash(root), "config": config, "stage": stage,
                "protocol": protocol, "scenario": scenario, "seed": seed,
                "dataset_sha256": sha256(dataset)}
    if identity["dataset_sha256"] != config["expected_dataset_sha256"]:
        raise ValueError("Dataset changed since protocol freeze; refusing mixed-data results")
    if stage != "train":
        identity["source_bundle_manifest_sha256"] = sha256(source_run / "bundle/manifest.json")
        identity["source_split_sha256"] = sha256(source_run / "split_indices.npz")
    establish_identity(destination, identity)
    if verify_complete(destination):
        log(f"Verified completed supplement, skipping: {destination}")
        return
    if stage == "train":
        from .grouped_train import run_grouped
        train_cfg.update(dataset=str(dataset), protocol_version="paper-supplement-v2-grouped",
                         split_strategy="exact_observable_groups", supplement_code_sha256=identity["code_sha256"])
        run_grouped(train_cfg, root, destination, scenario, seed, allow_cpu=allow_cpu)
    elif stage == "evaluate":
        import tensorflow as tf
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
        from .evaluation import evaluate_run
        evaluate_run(config, root, source_run, destination)
    elif stage == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(config["benchmark"], dataset, source_run, destination)
    else:
        raise ValueError(stage)
    mark_complete(destination)


def execute(root, args, config, stages):
    train_cfg = json.loads((root / config["legacy_config"]).read_text(encoding="utf-8"))
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    if "matched" in stages:
        jobs += [("evaluate", "legacy", s, seed) for s in train_cfg["scenarios"] for seed in train_cfg["seeds"]]
    if "grouped" in stages:
        for scenario in train_cfg["scenarios"]:
            for seed in train_cfg["seeds"]:
                jobs += [("train", "grouped", scenario, seed), ("evaluate", "grouped", scenario, seed)]
    if "benchmark" in stages:
        b = config["benchmark"]
        jobs.append(("benchmark", "legacy", b["scenario"], b["seed"]))
    for stage, protocol, scenario, seed in jobs:
        name = f"{stage}_{protocol}_{scenario}_seed{seed}"
        log(f"DISPATCH {name}")
        command = [sys.executable, "-u", str(root / "experiments/run_paper_supplement.py"), "_worker",
                   "--config", str(args.config), "--old-root", str(args.old_root), "--dataset", str(args.dataset),
                   "--legacy-results", str(args.legacy_results), "--output", str(args.output),
                   "--stage", stage, "--protocol", protocol, "--scenario", scenario, "--seed", str(seed)]
        if args.allow_cpu:
            command.append("--allow-cpu")
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONHASHSEED=str(seed), TF_CPP_MIN_LOG_LEVEL="1")
        if stage == "benchmark":
            env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
        with (logs / (name + ".log")).open("a", encoding="utf-8") as handle:
            handle.write(f"\n--- Invocation {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Stage failed with exit {code}: {name}. Saved runs remain intact; see {logs / (name + '.log')}")
        summarize(root, config, args.output)


def summarize(root, config, output):
    import numpy as np
    import pandas as pd
    from .evaluation import METHODS
    train_cfg = json.loads((root / config["legacy_config"]).read_text(encoding="utf-8"))
    missing, failures, records, groups, frozen = [], [], [], {}, []
    for protocol in ("legacy", "grouped"):
        for scenario in train_cfg["scenarios"]:
            for seed in train_cfg["seeds"]:
                directory = output / "matched" / protocol / scenario / f"seed{seed}"
                if not verify_complete(directory):
                    missing.append(str(directory.relative_to(output)))
                    continue
                value = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                for method, reason in value.get("baseline_failures", {}).items():
                    failures.append({"run": str(directory.relative_to(output)), "method": method, "reason": reason})
                for target, methods in value["targets"].items():
                    for method, entry in methods.items():
                        row = {"protocol": protocol, "scenario": scenario, "seed": seed,
                               "target_calibration_far_percent": float(target) * 100, "method": method,
                               "feasible": entry["calibration"]["feasible"]}
                        if entry["metrics"]:
                            row.update({k: v * 100 for k, v in entry["metrics"].items() if isinstance(v, (int, float))})
                        records.append(row)
                        groups.setdefault((protocol, scenario, target, method), []).append(row)
            if protocol == "grouped":
                for seed in train_cfg["seeds"]:
                    d = output / "grouped_runs" / scenario / f"seed{seed}"
                    if not verify_complete(d):
                        missing.append(str(d.relative_to(output)))
                    else:
                        data = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                        for method, metrics in data["methods"].items():
                            frozen.append({"scenario": scenario, "seed": seed, "method": method,
                                           **{k: v * 100 for k, v in metrics.items() if isinstance(v, (int, float))}})
    aggregate = []
    for key, rows in groups.items():
        feasible = [r for r in rows if r["feasible"]]
        complete = len(rows) == len(train_cfg["seeds"]) and len(feasible) == len(rows)
        row = dict(zip(("protocol", "scenario", "target_calibration_far", "method"), key))
        row.update(n_finished=len(rows), n_feasible=len(feasible),
                   valid_predefined_seed_mean=complete, valid_five_seed_mean=complete and len(rows) == 5)
        # Never publish a selective average over only feasible/favorable seeds.
        if complete:
            for metric in ("B_benign_false_alarm_rate", "F_open_set_macro_f1", "U_unknown_recall", "unknown_class_macro_recall"):
                values = [r[metric] for r in rows]
                row[metric + "_mean_percent"] = float(np.mean(values))
                row[metric + "_sample_sd_pp"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "per_seed_results.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "aggregate_results.csv", index=False)
    pd.DataFrame(frozen).to_csv(output / "grouped_frozen_ablation_per_seed.csv", index=False)
    benchmark_done = verify_complete(output / "benchmark_cpu")
    if not benchmark_done:
        missing.append("benchmark_cpu")
    status = {"complete": not missing and not failures, "execution_complete": not missing,
              "missing": missing, "baseline_failures": failures,
              "note": "All percentages are test results at calibration-derived budgets; infeasible/missing seeds are not averaged away."}
    atomic_json(output / "status.json", status)
    lines = ["# Supplement v2 / 补充实验结果", "", f"Complete: {status['complete']}", "",
             "详细数据：aggregate_results.csv（均值/样本标准差），per_seed_results.csv（每种子）。",
             "legacy = 旧模型重新校准；grouped = 重复特征分组隔离后重新训练。不可与旧摘要数值混为一组。",
             "阈值仅用已知校准流量；目标误报率不等于测试集实际误报率。不满足预算的种子标为 infeasible，不删除。", ""]
    if missing:
        lines += ["尚缺："] + ["- " + name for name in missing]
    if failures:
        lines += ["", "以下基线拟合失败，未伪造或替代结果：", json.dumps(failures, ensure_ascii=False, indent=2)]
    if benchmark_done:
        bench = json.loads((output / "benchmark_cpu/benchmark.json").read_text())
        lines += ["", "CPU完整流程测速（不是农业边缘设备）：", json.dumps(bench["full_pipeline"], ensure_ascii=False)]
    (output / "SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Summary complete={status['complete']}; missing={len(missing)}")
    return status


def pack_results(root, config, output):
    summarize(root, config, output)
    destination = output.parent / "paper_supplement_v2_results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    files = [p for p in output.rglob("*") if p.is_file() and "cache" not in p.relative_to(output).parts
             and p.name != "RUNNING.lock" and not p.name.endswith(".tmp")]
    manifest = {p.relative_to(output).as_posix(): sha256(p) for p in files}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, "paper_supplement_v2/" + path.relative_to(output).as_posix())
        archive.writestr("paper_supplement_v2/RESULTS_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError("Result ZIP CRC failure")
    os.replace(temporary, destination)
    log(f"DOWNLOAD: {destination} ({destination.stat().st_size:,} bytes)")
    return destination
