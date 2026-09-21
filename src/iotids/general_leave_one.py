"""EXP-07 general leave-one-attack-out open-set evaluation."""

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

MAIN_METHODS = {
    "closed_dnn": "closed_dnn",
    "msp": "msp",
    "full_role_context": "msp_context__full_role_context__a0.999",
    "device_role_core": "msp_context__device_role_core__a0.999",
    "port_pair": "msp_context__port_pair__a0.999",
}


@dataclass
class GeneralLeaveOneConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/general_leave_one"
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


def _run_dir_name(held_out_attack: str) -> str:
    return held_out_attack.lower().replace(" ", "_").replace("/", "_")


def _clear_tensorflow() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _safe_get_zero_day(result: dict, held_out_attack: str) -> dict:
    return result.get("zero_day_breakdown", {}).get(held_out_attack, {})


def _method_metrics(result: dict, held_out_attack: str) -> dict[str, float | None | dict]:
    zero_day = _safe_get_zero_day(result, held_out_attack)
    benign_miss = zero_day.get("benign_miss_rate")
    return {
        "utdr": result.get("unknown_threat_detection_rate"),
        "far": result.get("benign_false_alarm_rate"),
        "macro_f1": result.get("open_set_macro_f1"),
        "known_accuracy": result.get("known_closed_label_accuracy"),
        "auroc": result.get("unknown_score_auc"),
        "reject_rate": zero_day.get("unknown_rate"),
        "benign_miss_rate": benign_miss,
        "attack_catch_rate": None if benign_miss is None else 1.0 - float(benign_miss),
        "prediction_distribution": zero_day.get("prediction_distribution", {}),
    }


def _summarize_run(metrics: dict, held_out_attack: str) -> dict:
    strategies = metrics["strategies"]
    methods = {}
    for short_name, strategy_name in MAIN_METHODS.items():
        if strategy_name in strategies:
            methods[short_name] = _method_metrics(strategies[strategy_name], held_out_attack)
    return {
        "held_out_attack": held_out_attack,
        "known_classes": metrics["config"]["known_classes"],
        "blind_distribution": metrics["dataset"]["blind_distribution"],
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
    fields = (
        "utdr",
        "far",
        "macro_f1",
        "known_accuracy",
        "auroc",
        "reject_rate",
        "benign_miss_rate",
        "attack_catch_rate",
    )
    summaries = {}
    for short_name in MAIN_METHODS:
        method_rows = [row["methods"][short_name] for row in rows if short_name in row["methods"]]
        if not method_rows:
            continue
        summaries[short_name] = {
            field: _mean_std([row[field] for row in method_rows])
            for field in fields
        }
    return summaries


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_mean_std(summary: dict[str, float | None]) -> str:
    if summary["mean"] is None:
        return "-"
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _dominant_absorption(prediction_distribution: dict) -> str:
    known_predictions = {
        label: count
        for label, count in prediction_distribution.items()
        if label not in {BENIGN_LABEL, "Unknown Attack"}
    }
    if not known_predictions:
        return "-"
    label, count = max(known_predictions.items(), key=lambda item: item[1])
    total = sum(prediction_distribution.values())
    ratio = count / total if total else 0.0
    return f"{label} ({ratio:.2%})"


def _write_report(aggregate: dict, output_path: Path) -> None:
    method_order = [method for method in MAIN_METHODS if method in aggregate["method_summaries"]]
    lines = [
        "# EXP-07 General Leave-One-Attack-Out Open-Set Report",
        "",
        "## Purpose",
        "",
        (
            "Test whether the IDS can handle many possible zero-day choices, rather than "
            "only the original Arp Spoofing / Port Scanning split. Each attack class is "
            "removed from training once and used as the blind unknown attack."
        ),
        "",
        "## What This Experiment Separates",
        "",
        "- Closed DNN attack catch rate: whether the paper-style DNN at least labels a zero-day flow as some attack, even if the attack name is wrong.",
        "- UTDR / reject rate: whether an open-set method explicitly outputs Unknown Attack.",
        "- Dominant absorption: which known class absorbs the held-out unknown attack when rejection fails.",
        "",
        "## Protocol",
        "",
        "- One attack class is held out at a time.",
        "- All remaining attacks plus Benign are used as known classes.",
        "- The held-out attack is blind-test-only and never enters training, validation, or calibration.",
        "- The same DNN, MSP threshold, and device-role context protocol are rerun for every held-out attack.",
        "- BotNet DDoS is excluded because Farm-Flow has too few rows for stable training and evaluation.",
        "",
        "## Per-Held-Out Results",
        "",
        "| Held-out attack | Method | Unknown reject / UTDR | Attack catch | Benign miss | FAR | Macro-F1 | Known acc. | Dominant absorption |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["runs"]:
        for method in method_order:
            if method not in row["methods"]:
                continue
            values = row["methods"][method]
            lines.append(
                f"| {row['held_out_attack']} | {method} | {_fmt(values['reject_rate'])} | "
                f"{_fmt(values['attack_catch_rate'])} | {_fmt(values['benign_miss_rate'])} | "
                f"{_fmt(values['far'])} | {_fmt(values['macro_f1'])} | "
                f"{_fmt(values['known_accuracy'])} | "
                f"{_dominant_absorption(values['prediction_distribution'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean +/- Std Across Held-Out Attacks",
            "",
            "| Method | UTDR | Attack catch | Benign miss | FAR | Macro-F1 | Known acc. | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        summary = aggregate["method_summaries"][method]
        lines.append(
            f"| {method} | {_fmt_mean_std(summary['reject_rate'])} | "
            f"{_fmt_mean_std(summary['attack_catch_rate'])} | "
            f"{_fmt_mean_std(summary['benign_miss_rate'])} | "
            f"{_fmt_mean_std(summary['far'])} | "
            f"{_fmt_mean_std(summary['macro_f1'])} | "
            f"{_fmt_mean_std(summary['known_accuracy'])} | "
            f"{_fmt_mean_std(summary['auroc'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If closed DNN attack catch is high but UTDR is low, the model detects attackness but cannot provide open-set rejection.",
            "- If a held-out attack is repeatedly absorbed by one known class, the next method should target known-class absorption instead of only tuning MSP.",
            "- A defensible open-set method should improve UTDR across several held-out attacks while keeping benign FAR and known-class accuracy acceptable.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_general_leave_one(config: GeneralLeaveOneConfig) -> dict:
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
        run_dir = output_dir / "runs" / _run_dir_name(held_out_attack)
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed EXP-07 run: {held_out_attack}", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting EXP-07 run: {held_out_attack}", flush=True)
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
            print(f"Finished EXP-07 run: {held_out_attack}", flush=True)
        rows.append(_summarize_run(metrics, held_out_attack))

    aggregate = {
        "experiment": "EXP-07",
        "config": asdict(config),
        "method_order": list(MAIN_METHODS),
        "runs": rows,
        "method_summaries": _summarize_methods(rows),
    }
    save_json(aggregate, output_dir / "general_leave_one_summary.json")
    _write_report(aggregate, reports_dir / "general_leave_one_report.md")
    return aggregate
