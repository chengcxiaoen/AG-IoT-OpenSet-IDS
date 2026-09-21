"""EXP-04E controlled-randomness diagnosis for Port Scanning instability."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .class_conditional_experiment import (
    ClassConditionalExperimentConfig,
    run_class_conditional_experiment,
)
from .constants import DEFAULT_ZERO_DAY_CLASSES, FARM_FLOW_KNOWN_CLASSES
from .utils import ensure_dir, save_json


@dataclass
class RandomnessDiagnosticConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/randomness_diagnostic"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    base_seed: int = 42
    varied_seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
    max_rows_per_class: int | None = None
    balance_strategy: str = "median"
    max_train_per_class: int | None = 50000
    global_known_acceptance_rate: float = 0.95
    attack_acceptance_rate: float = 0.95
    context_benign_acceptance_rate: float = 0.999
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    reuse_completed: bool = True


def _method_metrics(result: dict) -> dict[str, float]:
    arp = result["zero_day_breakdown"]["Arp Spoofing"]
    port = result["zero_day_breakdown"]["Port Scanning"]
    return {
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "known_accuracy": result["known_closed_label_accuracy"],
        "arp_reject": arp["unknown_rate"],
        "port_reject": port["unknown_rate"],
    }


def _summarize_run(metrics: dict, axis: str, varied_seed: int) -> dict:
    proposed_name = "asymmetric_class_msp_full_role__a0.95"
    port_diagnostics = metrics["zero_day_confidence_diagnostics"]["Port Scanning"]
    tcp_diagnostics = port_diagnostics["by_predicted_class"].get("TCP Flood", {})
    tcp_index = list(metrics["config"]["known_classes"]).index("TCP Flood")
    attack_thresholds = metrics["asymmetric_thresholds"]["0.95"]
    tcp_confidence_min = 1.0 - float(
        attack_thresholds["confidence_anomaly_max"][tcp_index]
    )
    return {
        "axis": axis,
        "varied_seed": int(varied_seed),
        "resolved_seeds": metrics["resolved_seeds"],
        "baseline": _method_metrics(metrics["strategies"]["global_msp_full_role"]),
        "proposed": _method_metrics(metrics["strategies"][proposed_name]),
        "port_predicted_tcp_rate": tcp_diagnostics.get("rate_within_attack", 0.0),
        "port_predicted_tcp_support": tcp_diagnostics.get("support", 0),
        "port_tcp_max_softmax_quantiles": tcp_diagnostics.get(
            "max_softmax_quantiles", {}
        ),
        "port_tcp_margin_quantiles": tcp_diagnostics.get(
            "top1_margin_quantiles", {}
        ),
        "tcp_class_min_confidence": tcp_confidence_min,
        "global_msp_min_confidence": metrics["global_msp_thresholds"][
            "msp_min_confidence"
        ],
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _axis_summary(rows: list[dict]) -> dict:
    fields = {
        "baseline_port_reject": [row["baseline"]["port_reject"] for row in rows],
        "proposed_port_reject": [row["proposed"]["port_reject"] for row in rows],
        "proposed_utdr": [row["proposed"]["utdr"] for row in rows],
        "proposed_macro_f1": [row["proposed"]["macro_f1"] for row in rows],
        "proposed_known_accuracy": [row["proposed"]["known_accuracy"] for row in rows],
        "port_predicted_tcp_rate": [row["port_predicted_tcp_rate"] for row in rows],
        "tcp_class_min_confidence": [row["tcp_class_min_confidence"] for row in rows],
    }
    return {name: _mean_std(values) for name, values in fields.items()}


def _write_report(aggregate: dict, output_path: Path) -> None:
    lines = [
        "# EXP-04E Controlled Randomness Diagnostic",
        "",
        "## Purpose",
        "",
        "Identify whether Port Scanning rejection instability is mainly caused by known-data splitting/calibration, RUS-SMOTE resampling, or DNN initialization/training randomness.",
        "",
        "## Controlled Axes",
        "",
        "- `model_training`: fixed split and resampling; vary model initialization, Dropout, and training shuffle.",
        "- `split_calibration`: fixed model and resampling; vary known train/validation/calibration/holdout split.",
        "- `resampling`: fixed split and model; vary RUS and SMOTE samples.",
        "- Zero-day samples remain blind-test-only in every run.",
        "",
        "## Per-Run Port Results",
        "",
        "| Axis | Seed | Baseline Port | Proposed Port | Port -> TCP | TCP threshold | Port-TCP confidence p50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        p50 = row["port_tcp_max_softmax_quantiles"].get("p50")
        p50_text = "-" if p50 is None else f"{p50:.4f}"
        lines.append(
            f"| {row['axis']} | {row['varied_seed']} | "
            f"{row['baseline']['port_reject']:.4f} | "
            f"{row['proposed']['port_reject']:.4f} | "
            f"{row['port_predicted_tcp_rate']:.4f} | "
            f"{row['tcp_class_min_confidence']:.4f} | {p50_text} |"
        )
    lines.extend(
        [
            "",
            "## Variation by Random Source",
            "",
            "| Axis | Proposed Port mean +/- std | Port -> TCP mean +/- std | TCP threshold mean +/- std |",
            "|---|---:|---:|---:|",
        ]
    )
    for axis, summary in aggregate["axis_summaries"].items():
        port = summary["proposed_port_reject"]
        tcp_rate = summary["port_predicted_tcp_rate"]
        threshold = summary["tcp_class_min_confidence"]
        lines.append(
            f"| {axis} | {port['mean']:.4f} +/- {port['std']:.4f} | "
            f"{tcp_rate['mean']:.4f} +/- {tcp_rate['std']:.4f} | "
            f"{threshold['mean']:.4f} +/- {threshold['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The axis with the largest Port-rejection standard deviation is the strongest empirical source of instability. Compare it with Port-to-TCP rate and TCP confidence/threshold changes before choosing a remedy.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _clear_tensorflow() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def run_randomness_diagnostic(config: RandomnessDiagnosticConfig) -> dict:
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")

    combinations: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
    for axis in ("model_training", "split_calibration", "resampling"):
        for varied_seed in config.varied_seeds:
            split_seed = varied_seed if axis == "split_calibration" else config.base_seed
            model_seed = varied_seed if axis == "model_training" else config.base_seed
            resampling_seed = varied_seed if axis == "resampling" else config.base_seed
            key = (split_seed, model_seed, resampling_seed)
            combinations.setdefault(key, []).append((axis, varied_seed))

    metrics_by_seed_tuple: dict[tuple[int, int, int], dict] = {}
    for split_seed, model_seed, resampling_seed in combinations:
        run_name = f"split{split_seed}_model{model_seed}_resample{resampling_seed}"
        run_dir = output_dir / "runs" / run_name
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed diagnostic run: {run_name}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting diagnostic run: {run_name}")
            metrics = run_class_conditional_experiment(
                ClassConditionalExperimentConfig(
                    experiment_id="EXP-04E",
                    report_filename="run_report.md",
                    farm_flow_path=config.farm_flow_path,
                    output_dir=str(run_dir),
                    known_classes=config.known_classes,
                    zero_day_classes=config.zero_day_classes,
                    seed=config.base_seed,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    resampling_seed=resampling_seed,
                    max_rows_per_class=config.max_rows_per_class,
                    balance_strategy=config.balance_strategy,
                    max_train_per_class=config.max_train_per_class,
                    known_acceptance_rate=config.global_known_acceptance_rate,
                    attack_acceptance_rates=(config.attack_acceptance_rate,),
                    context_benign_acceptance_rate=config.context_benign_acceptance_rate,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    dropout=config.dropout,
                    patience=config.patience,
                    save_model=False,
                )
            )
            _clear_tensorflow()
            print(f"Finished diagnostic run: {run_name}")
        metrics_by_seed_tuple[(split_seed, model_seed, resampling_seed)] = metrics

    rows = []
    for seed_tuple, memberships in combinations.items():
        metrics = metrics_by_seed_tuple[seed_tuple]
        for axis, varied_seed in memberships:
            rows.append(_summarize_run(metrics, axis, varied_seed))
    rows.sort(key=lambda row: (row["axis"], row["varied_seed"]))

    axis_summaries = {
        axis: _axis_summary([row for row in rows if row["axis"] == axis])
        for axis in ("model_training", "split_calibration", "resampling")
    }
    aggregate = {
        "experiment": "EXP-04E",
        "config": asdict(config),
        "unique_training_runs": len(combinations),
        "runs": rows,
        "axis_summaries": axis_summaries,
    }
    save_json(aggregate, output_dir / "diagnostic_summary.json")
    _write_report(aggregate, reports_dir / "randomness_diagnostic_report.md")
    return aggregate
