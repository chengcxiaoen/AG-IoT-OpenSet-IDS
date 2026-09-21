"""Run the manuscript's duplicate-group-disjoint protocol from a fresh checkout.

This driver intentionally does not require earlier row-random runs. It trains
the grouped protocol for each predeclared scenario and seed, then performs the
post-hoc evaluation used for the reported comparison tables. It never retries
another seed after a failure and never treats a partial five-seed average as a
paper result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "train", "evaluate", "summarize"))
    parser.add_argument("--dataset", type=Path, required=True, help="Farm-Flows.csv downloaded from the original provider")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/grouped_reproduction")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/paper_final_legacy8.json")
    parser.add_argument("--evaluation-config", type=Path, default=ROOT / "configs/paper_supplement_v2.json")
    parser.add_argument("--allow-cpu", action="store_true", help="Use CPU only when GPU execution is unavailable")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def output_rows(output: Path, train_config: dict) -> list[dict]:
    rows = []
    for scenario in train_config["scenarios"]:
        for seed in train_config["seeds"]:
            path = output / "grouped_runs" / scenario / f"seed{seed}" / "metrics.json"
            if not path.exists():
                continue
            methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
            for method, metrics in methods.items():
                rows.append({"scenario": scenario, "seed": seed, "method": method, **metrics})
    return rows


def summarize(output: Path, train_config: dict) -> None:
    rows = output_rows(output, train_config)
    expected = len(train_config["seeds"])
    columns = ["B_benign_false_alarm_rate", "F_open_set_macro_f1", "U_unknown_recall",
               "unknown_class_macro_recall"]
    aggregate = []
    for (scenario, method), part in pd.DataFrame(rows).groupby(["scenario", "method"], sort=True):
        item = {"scenario": scenario, "method": method, "n_finished": len(part),
                "valid_five_seed_mean": len(part) == expected}
        if len(part) == expected:
            for metric in columns:
                item[metric + "_mean_percent"] = float(part[metric].mean() * 100)
                item[metric + "_sample_sd_pp"] = float(part[metric].std(ddof=1) * 100)
        aggregate.append(item)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "grouped_frozen_ablation_per_seed.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "grouped_frozen_ablation_summary.csv", index=False)
    complete = all(
        (output / "grouped_runs" / scenario / f"seed{seed}" / "COMPLETE.json").exists()
        for scenario in train_config["scenarios"] for seed in train_config["seeds"]
    )
    status = {"complete_grouped_training": complete, "predefined_seeds": train_config["seeds"],
              "note": "No partial five-seed mean is marked valid. Post-hoc evaluation outputs are stored separately."}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (output / "SUMMARY_CN.md").write_text(
        "# Grouped reproduction summary\n\n"
        f"Grouped training complete: {complete}\n\n"
        "`grouped_frozen_ablation_per_seed.csv` contains one row per predefined seed; "
        "`grouped_frozen_ablation_summary.csv` contains a mean only when all five seeds completed.\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    train_config = load_config(args.train_config)
    evaluation_config = load_config(args.evaluation_config)
    train_config["dataset"] = str(dataset)
    evaluation_config["dataset"] = str(dataset)
    from iotids.paper_protocol import sha256
    from iotids_supplement.grouped_train import run_grouped
    from iotids_supplement.evaluation import evaluate_run

    if args.command in ("all", "train"):
        for scenario in train_config["scenarios"]:
            for seed in train_config["seeds"]:
                run_grouped(train_config, ROOT, output / "grouped_runs" / scenario / f"seed{seed}",
                            scenario, seed, allow_cpu=args.allow_cpu)
    if args.command in ("all", "evaluate"):
        for scenario in train_config["scenarios"]:
            for seed in train_config["seeds"]:
                run_dir = output / "grouped_runs" / scenario / f"seed{seed}"
                if not (run_dir / "COMPLETE.json").exists():
                    raise RuntimeError(f"Grouped training is incomplete: {run_dir}")
                evaluate_run(evaluation_config, ROOT, run_dir,
                             output / "matched" / "grouped" / scenario / f"seed{seed}")
    if args.command in ("all", "train", "evaluate", "summarize"):
        summarize(output, train_config)
    print(json.dumps({"output": str(output), "dataset_sha256": sha256(dataset)}, indent=2))


if __name__ == "__main__":
    main()
