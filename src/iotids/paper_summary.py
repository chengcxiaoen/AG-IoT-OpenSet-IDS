"""Summarize frozen full-data runs; never cherry-pick seeds or mix scenarios."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .paper_protocol import METHODS, PROPOSED, atomic_json, sha256

FIELDS = ("B_benign_false_alarm_rate", "F_open_set_macro_f1", "U_unknown_recall",
          "known_accuracy_after_rejection", "known_rejection_rate", "unknown_precision",
          "unknown_f1", "unknown_class_macro_recall")


def aggregate(config, output):
    output = Path(output)
    missing, scenarios, run_digests = [], {}, set()
    for scenario in config["scenarios"]:
        runs = []
        for seed in config["seeds"]:
            folder = output / "runs" / scenario / f"seed{seed}"
            if not (folder / "COMPLETE.json").exists():
                missing.append(f"{scenario}/seed{seed}")
                continue
            done = json.loads((folder / "COMPLETE.json").read_text())
            if sha256(folder / "metrics.json") != done["metrics_sha256"]:
                raise ValueError(f"Metrics integrity error: {folder}")
            value = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            if value["config"] != config or value["seed"] != seed or value["scenario"] != scenario:
                raise ValueError(f"Mixed configuration/run identity: {folder}")
            run_digests.add(value["protocol_digest"])
            runs.append(value)
        if not runs:
            continue
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for field in FIELDS:
                values = [r["methods"][method][field] for r in runs]
                methods[method][field] = {"mean": float(np.mean(values)),
                                          "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                          "values_in_seed_order": values}
        paired = {}
        for comparison in ("global_msp", "classwise_msp", "global_msp_context"):
            paired[comparison] = {}
            for field in FIELDS:
                diffs = [r["methods"][PROPOSED][field] - r["methods"][comparison][field] for r in runs]
                paired[comparison][field] = {"mean_difference_pp": float(np.mean(diffs) * 100),
                                             "differences_pp": [v * 100 for v in diffs]}
        scenarios[scenario] = {"seeds": [r["seed"] for r in runs], "methods": methods,
                                "paired_ablations": paired,
                                "per_seed_proposed": [{"seed": r["seed"], **r["methods"][PROPOSED]} for r in runs],
                                "data_audits": [r["data_audit"] for r in runs]}
    if len(run_digests) > 1:
        raise ValueError("Mixed code or dataset hashes; refusing aggregation")
    benchmark = None
    bdir = output / "benchmark"
    if (bdir / "COMPLETE.json").exists():
        done = json.loads((bdir / "COMPLETE.json").read_text())
        if sha256(bdir / "benchmark.json") != done["sha256"]:
            raise ValueError("Benchmark report integrity failure")
        benchmark = json.loads((bdir / "benchmark.json").read_text(encoding="utf-8"))
        if benchmark["identity"]["benchmark_config"] != config["benchmark"]:
            raise ValueError("Benchmark configuration mismatch")
        bundle = output / "runs" / config["benchmark"]["scenario"] / f"seed{config['benchmark']['seed']}" / "bundle"
        manifest = json.loads((bundle / "manifest.json").read_text())
        if benchmark["identity"]["bundle_manifest"] != manifest:
            raise ValueError("Benchmark did not measure the final exported model")
    complete = not missing and benchmark is not None
    summary = {"complete": complete, "missing_runs": missing, "benchmark_missing": benchmark is None,
               "protocol_digest": next(iter(run_digests), None), "scenarios": scenarios, "benchmark": benchmark,
               "scope": "Paired ablations, not external baselines; random-row evaluation, not new-device generalization."}
    atomic_json(output / "summary.json", summary)
    lines = ["# Paper final results / 最终实验结果", "",
             f"Complete: {complete}; missing runs: {missing}; benchmark missing: {benchmark is None}", "",
             "All detection values below are percent, mean +/- SAMPLE standard deviation across predefined seeds.",
             "MSP variants are ablations, not external baselines. Do not select the best seed.", ""]
    for scenario, values in scenarios.items():
        lines += [f"## {scenario}", "", f"Seeds: {values['seeds']}", "",
                  "| Method | B benign FAR | F open macro-F1 | U unknown recall | Known accuracy | Unknown class-macro recall |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, fields in values["methods"].items():
            row = []
            for name in (FIELDS[0], FIELDS[1], FIELDS[2], FIELDS[3], FIELDS[-1]):
                stat = fields[name]
                row.append(f"{stat['mean']*100:.2f} +/- {stat['std']*100:.2f}" if stat["std"] is not None
                           else f"{stat['mean']*100:.2f} (n=1)")
            lines.append("| " + method + " | " + " | ".join(row) + " |")
        lines += ["", "### Per-seed proposed / 各种子完整方法", "",
                  "| Seed | B | F | U | Per-unknown-class recall |", "|---|---:|---:|---:|---|"]
        for result in values["per_seed_proposed"]:
            details = "; ".join(f"{k}: {v['unknown_recall']*100:.2f}%" for k, v in result["zero_day_breakdown"].items())
            lines.append(f"| {result['seed']} | {result[FIELDS[0]]*100:.2f} | {result[FIELDS[1]]*100:.2f} | {result[FIELDS[2]]*100:.2f} | {details} |")
        lines.append("")
    (output / "SUMMARY_CN.md").write_text("\n".join(lines), encoding="utf-8")
    abstract = ["# Abstract numbers / 摘要数据", "",
                "Do not claim SOTA: this package does not reproduce external baselines.",
                "Do not combine primary and stress scenarios. Inspect per-seed spread and duplicate audit.", ""]
    if missing:
        abstract += [f"INCOMPLETE: missing {missing}; no final abstract sentence generated."]
    else:
        primary = scenarios["primary_arp_port"]["methods"][PROPOSED]
        b, f, u = [primary[key]["mean"] * 100 for key in FIELDS[:3]]
        abstract += [f"{len(config['seeds'])} 个预定义随机种子均值：在 Farm-Flow 的 ARP 与 Port Scanning 双未知类别设置下，正常流量误报率为 {b:.2f}%，开放集 Macro-F1 为 {f:.2f}%，未知攻击召回率为 {u:.2f}%。",
                     "", "Above values are observed test performance, NOT a guaranteed false-alarm bound."]
    if benchmark is None:
        abstract += ["", "L/device not yet measured. Do not fill latency from training/runtime logs."]
    else:
        abstract += ["", f"Device: {benchmark['device_label']}; type: {benchmark['device_type']}",
                     f"L = {benchmark['L_mean_single_flow_ms']:.4f} ms: CPU batch=1, full-test replay, mean across passes.",
                     f"Benchmark model was preselected seed {benchmark['preselected_model_seed']}, NOT best seed.",
                     "Latency starts at a ready flow record; excludes packet capture/flow aggregation."]
        if benchmark["not_edge_hardware_proof"]:
            abstract += ["SERVER CPU ONLY: this does not substantiate a claim of actual agricultural edge-device deployment."]
        mismatch = sum(x["prediction_mismatches_vs_evaluation"] for x in benchmark["passes"])
        if mismatch:
            abstract += [f"WARNING: {mismatch} CPU-vs-evaluation decision differences across passes; inspect threshold-boundary portability before publication."]
    (output / "ABSTRACT_NUMBERS.md").write_text("\n".join(abstract), encoding="utf-8")
    return complete
