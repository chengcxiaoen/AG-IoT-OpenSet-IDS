"""EXP-05C multi-seed stability for refined device-role context."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_ZERO_DAY_CLASSES, FARM_FLOW_KNOWN_CLASSES
from .device_role_experiment import DeviceRoleExperimentConfig, run_device_role_experiment
from .utils import ensure_dir, save_json


@dataclass
class DeviceRoleStabilityConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/device_role_stability"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
    max_rows_per_class: int | None = None
    balance_strategy: str = "median"
    max_train_per_class: int | None = 50000
    known_acceptance_rate: float = 0.95
    context_benign_acceptance_rate: float = 0.999
    context_scale_percentile: float = 99.0
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    reuse_completed: bool = True


KEY_METHODS = {
    "msp": "msp",
    "full_role": "msp_context__full_role_context__a0.999",
    "device_role_core": "msp_context__device_role_core__a0.999",
    "port_pair": "msp_context__port_pair__a0.999",
    "source_role": "msp_context__source_role__a0.999",
    "source_destination_port": "msp_context__source_destination_port__a0.999",
}


def _method_metrics(result: dict) -> dict[str, float]:
    arp = result["zero_day_breakdown"].get("Arp Spoofing", {})
    port = result["zero_day_breakdown"].get("Port Scanning", {})
    return {
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "known_accuracy": result["known_closed_label_accuracy"],
        "auroc": result.get("unknown_score_auc"),
        "arp_reject": arp.get("unknown_rate"),
        "port_reject": port.get("unknown_rate"),
    }


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return {"mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(filtered, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summarize_methods(rows: list[dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    methods = sorted({method for row in rows for method in row["methods"]})
    fields = ("utdr", "far", "macro_f1", "known_accuracy", "auroc", "arp_reject", "port_reject")
    output = {}
    for method in methods:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        output[method] = {
            field: _mean_std([row[field] for row in method_rows])
            for field in fields
        }
    return output


def _summarize_run(metrics: dict, seed: int) -> dict:
    methods = {}
    strategies = metrics["strategies"]
    for short_name, strategy_name in KEY_METHODS.items():
        if strategy_name in strategies:
            methods[short_name] = _method_metrics(strategies[strategy_name])
    return {
        "seed": int(seed),
        "methods": methods,
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_mean_std(summary: dict[str, float | None]) -> str:
    if summary["mean"] is None:
        return "-"
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _write_report(aggregate: dict, output_path: Path) -> None:
    method_order = [method for method in KEY_METHODS if method in aggregate["method_summaries"]]
    lines = [
        "# EXP-05C Device-Role Context Multi-Seed Stability Report",
        "",
        "## Purpose",
        "",
        (
            "Verify whether the refined agriculture device-role context remains stable "
            "across multiple train/validation/calibration/test splits. This converts "
            "the seed-42 refinement result into a more defensible method claim."
        ),
        "",
        "## Protocol",
        "",
        "- Each seed reruns the full DNN training, MSP calibration, and context calibration.",
        "- Zero-day attacks remain blind-test-only in every seed.",
        "- The selected context threshold is Benign acceptance rate 0.999.",
        "- Only the key candidates from EXP-05B are summarized.",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Method | UTDR | FAR | Macro-F1 | Known acc. | AUROC | Arp reject | Port reject |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        for method in method_order:
            if method not in row["methods"]:
                continue
            values = row["methods"][method]
            lines.append(
                f"| {row['seed']} | {method} | {_fmt(values['utdr'])} | "
                f"{_fmt(values['far'])} | {_fmt(values['macro_f1'])} | "
                f"{_fmt(values['known_accuracy'])} | {_fmt(values['auroc'])} | "
                f"{_fmt(values['arp_reject'])} | {_fmt(values['port_reject'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean +/- Std",
            "",
            "| Method | UTDR | FAR | Macro-F1 | Known acc. | AUROC | Arp reject | Port reject |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        summary = aggregate["method_summaries"][method]
        lines.append(
            f"| {method} | {_fmt_mean_std(summary['utdr'])} | "
            f"{_fmt_mean_std(summary['far'])} | {_fmt_mean_std(summary['macro_f1'])} | "
            f"{_fmt_mean_std(summary['known_accuracy'])} | {_fmt_mean_std(summary['auroc'])} | "
            f"{_fmt_mean_std(summary['arp_reject'])} | {_fmt_mean_std(summary['port_reject'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "The final method should be the smallest context variant whose multi-seed "
                "Arp rejection is close to full-role context, whose FAR remains below 1%, "
                "and whose known-class accuracy is not materially worse than MSP."
            ),
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


def run_device_role_stability(config: DeviceRoleStabilityConfig) -> dict:
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    rows = []

    for seed in config.seeds:
        run_name = f"seed{seed}"
        run_dir = output_dir / "runs" / run_name
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed device-role run: {run_name}", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting device-role stability run: {run_name}", flush=True)
            metrics = run_device_role_experiment(
                DeviceRoleExperimentConfig(
                    farm_flow_path=config.farm_flow_path,
                    output_dir=str(run_dir),
                    known_classes=config.known_classes,
                    zero_day_classes=config.zero_day_classes,
                    seed=seed,
                    max_rows_per_class=config.max_rows_per_class,
                    balance_strategy=config.balance_strategy,
                    max_train_per_class=config.max_train_per_class,
                    known_acceptance_rate=config.known_acceptance_rate,
                    context_benign_acceptance_rates=(config.context_benign_acceptance_rate,),
                    context_scale_percentile=config.context_scale_percentile,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    dropout=config.dropout,
                    patience=config.patience,
                )
            )
            _clear_tensorflow()
            print(f"Finished device-role stability run: {run_name}", flush=True)
        rows.append(_summarize_run(metrics, seed))

    aggregate = {
        "experiment": "EXP-05C",
        "config": asdict(config),
        "method_order": list(KEY_METHODS),
        "runs": rows,
        "method_summaries": _summarize_methods(rows),
    }
    save_json(aggregate, output_dir / "device_role_stability_summary.json")
    _write_report(aggregate, reports_dir / "device_role_stability_report.md")
    return aggregate
