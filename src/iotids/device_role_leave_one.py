"""EXP-06 leave-one-attack-out validation for device-role context."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import BENIGN_LABEL
from .device_role_experiment import DeviceRoleExperimentConfig, run_device_role_experiment
from .utils import ensure_dir, save_json


ATTACK_CLASSES = (
    "HTTP Flood",
    "ICMP Flood",
    "MQTT Flood",
    "TCP Flood",
    "UDP Flood",
    "Arp Spoofing",
    "Port Scanning",
)

KEY_METHODS = {
    "msp": "msp",
    "full_role": "msp_context__full_role_context__a0.999",
    "port_pair": "msp_context__port_pair__a0.999",
    "source_role": "msp_context__source_role__a0.999",
    "source_destination_port": "msp_context__source_destination_port__a0.999",
}


@dataclass
class DeviceRoleLeaveOneConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/device_role_leave_one"
    held_out_attacks: tuple[str, ...] = ATTACK_CLASSES
    seed: int = 42
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


def _method_metrics(result: dict, held_out_attack: str) -> dict[str, float | None]:
    zero_day = result.get("zero_day_breakdown", {}).get(held_out_attack, {})
    return {
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "known_accuracy": result["known_closed_label_accuracy"],
        "auroc": result.get("unknown_score_auc"),
        "held_out_reject": zero_day.get("unknown_rate"),
        "held_out_benign_miss": zero_day.get("benign_miss_rate"),
    }


def _summarize_run(metrics: dict, held_out_attack: str) -> dict:
    methods = {}
    strategies = metrics["strategies"]
    for short_name, strategy_name in KEY_METHODS.items():
        if strategy_name in strategies:
            methods[short_name] = _method_metrics(strategies[strategy_name], held_out_attack)
    return {
        "held_out_attack": held_out_attack,
        "known_classes": metrics["config"]["known_classes"],
        "methods": methods,
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
    fields = (
        "utdr",
        "far",
        "macro_f1",
        "known_accuracy",
        "auroc",
        "held_out_reject",
        "held_out_benign_miss",
    )
    output = {}
    for method in methods:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        output[method] = {
            field: _mean_std([row[field] for row in method_rows])
            for field in fields
        }
    return output


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_mean_std(summary: dict[str, float | None]) -> str:
    if summary["mean"] is None:
        return "-"
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _write_report(aggregate: dict, output_path: Path) -> None:
    method_order = [method for method in KEY_METHODS if method in aggregate["method_summaries"]]
    lines = [
        "# EXP-06 Leave-One-Attack-Out Device-Role Context Report",
        "",
        "## Purpose",
        "",
        (
            "Test whether the refined device-role context is only tuned for the original "
            "Arp Spoofing / Port Scanning zero-day split, or whether it remains meaningful "
            "when different attacks are moved into the zero-day blind test."
        ),
        "",
        "## Protocol",
        "",
        "- One attack class is removed from training at a time and used as the zero-day class.",
        "- All other attack classes are treated as known classes together with Benign.",
        "- Zero-day labels are used only for the final blind-test evaluation.",
        "- The selected context threshold is Benign acceptance rate 0.999.",
        "- BotNet DDoS is excluded because Farm-Flow contains too few rows for stable evaluation.",
        "",
        "## Per-Held-Out Results",
        "",
        "| Held-out attack | Method | UTDR | FAR | Macro-F1 | Known acc. | AUROC | Reject | Benign miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        for method in method_order:
            if method not in row["methods"]:
                continue
            values = row["methods"][method]
            lines.append(
                f"| {row['held_out_attack']} | {method} | {_fmt(values['utdr'])} | "
                f"{_fmt(values['far'])} | {_fmt(values['macro_f1'])} | "
                f"{_fmt(values['known_accuracy'])} | {_fmt(values['auroc'])} | "
                f"{_fmt(values['held_out_reject'])} | {_fmt(values['held_out_benign_miss'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean +/- Std Across Held-Out Attacks",
            "",
            "| Method | UTDR | FAR | Macro-F1 | Known acc. | AUROC | Reject | Benign miss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        summary = aggregate["method_summaries"][method]
        lines.append(
            f"| {method} | {_fmt_mean_std(summary['utdr'])} | "
            f"{_fmt_mean_std(summary['far'])} | {_fmt_mean_std(summary['macro_f1'])} | "
            f"{_fmt_mean_std(summary['known_accuracy'])} | {_fmt_mean_std(summary['auroc'])} | "
            f"{_fmt_mean_std(summary['held_out_reject'])} | "
            f"{_fmt_mean_std(summary['held_out_benign_miss'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "This experiment should not be used to claim that one context relation solves "
                "all zero-day attacks. Its role is to show whether the proposed agriculture "
                "device-role context is a general validation mechanism or only a special-case "
                "fix for the original Arp/Port split."
            ),
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_dir_name(held_out_attack: str) -> str:
    return held_out_attack.lower().replace(" ", "_").replace("/", "_")


def _clear_tensorflow() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def run_device_role_leave_one(config: DeviceRoleLeaveOneConfig) -> dict:
    invalid = sorted(set(config.held_out_attacks) - set(ATTACK_CLASSES))
    if invalid:
        raise ValueError(f"Unknown held-out attacks: {invalid}")

    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    rows = []
    for held_out_attack in config.held_out_attacks:
        known_classes = tuple(
            [BENIGN_LABEL] + [label for label in ATTACK_CLASSES if label != held_out_attack]
        )
        run_name = _run_dir_name(held_out_attack)
        run_dir = output_dir / "runs" / run_name
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed leave-one run: {held_out_attack}", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting leave-one run: {held_out_attack}", flush=True)
            metrics = run_device_role_experiment(
                DeviceRoleExperimentConfig(
                    farm_flow_path=config.farm_flow_path,
                    output_dir=str(run_dir),
                    known_classes=known_classes,
                    zero_day_classes=(held_out_attack,),
                    seed=config.seed,
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
            print(f"Finished leave-one run: {held_out_attack}", flush=True)
        rows.append(_summarize_run(metrics, held_out_attack))

    aggregate = {
        "experiment": "EXP-06",
        "config": asdict(config),
        "method_order": list(KEY_METHODS),
        "runs": rows,
        "method_summaries": _summarize_methods(rows),
    }
    save_json(aggregate, output_dir / "device_role_leave_one_summary.json")
    _write_report(aggregate, reports_dir / "device_role_leave_one_report.md")
    return aggregate
