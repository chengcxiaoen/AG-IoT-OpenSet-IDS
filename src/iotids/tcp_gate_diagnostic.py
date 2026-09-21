"""EXP-04I diagnostics for TCP-gated rejection rules."""

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
class TcpGateDiagnosticConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/tcp_gate_diagnostic"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    base_seed: int = 42
    split_seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
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


def _q(diagnostics: dict, field: str, quantile: str = "p50") -> float | None:
    value = diagnostics.get(field, {}).get(quantile)
    return None if value is None else float(value)


def _tcp_block(metrics: dict, true_label: str) -> dict:
    source_key = (
        "zero_day_confidence_diagnostics"
        if true_label in metrics["config"]["zero_day_classes"]
        else "known_confidence_diagnostics"
    )
    label_diagnostics = metrics[source_key][true_label]
    predicted_tcp = label_diagnostics["by_predicted_class"].get("TCP Flood", {})
    support = int(label_diagnostics.get("support", 0))
    tcp_support = int(predicted_tcp.get("support", 0))
    return {
        "support": support,
        "predicted_tcp_support": tcp_support,
        "predicted_tcp_rate": float(tcp_support / support) if support else 0.0,
        "max_softmax_p10": _q(predicted_tcp, "max_softmax_quantiles", "p10"),
        "max_softmax_p50": _q(predicted_tcp, "max_softmax_quantiles", "p50"),
        "max_softmax_p90": _q(predicted_tcp, "max_softmax_quantiles", "p90"),
        "margin_p10": _q(predicted_tcp, "top1_margin_quantiles", "p10"),
        "margin_p50": _q(predicted_tcp, "top1_margin_quantiles", "p50"),
        "margin_p90": _q(predicted_tcp, "top1_margin_quantiles", "p90"),
        "entropy_p10": _q(predicted_tcp, "entropy_quantiles", "p10"),
        "entropy_p50": _q(predicted_tcp, "entropy_quantiles", "p50"),
        "entropy_p90": _q(predicted_tcp, "entropy_quantiles", "p90"),
    }


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    if len(valid) == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(valid)), "std": float(np.std(valid))}


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for label in ("true_tcp", "port_as_tcp"):
        summary[label] = {}
        for field in (
            "predicted_tcp_rate",
            "max_softmax_p50",
            "margin_p50",
            "entropy_p50",
        ):
            summary[label][field] = _mean_std([row[label][field] for row in rows])
    summary["gap"] = {
        "margin_p50_port_minus_tcp": _mean_std(
            [
                (
                    row["port_as_tcp"]["margin_p50"] - row["true_tcp"]["margin_p50"]
                    if row["port_as_tcp"]["margin_p50"] is not None
                    and row["true_tcp"]["margin_p50"] is not None
                    else None
                )
                for row in rows
            ]
        ),
        "entropy_p50_port_minus_tcp": _mean_std(
            [
                (
                    row["port_as_tcp"]["entropy_p50"] - row["true_tcp"]["entropy_p50"]
                    if row["port_as_tcp"]["entropy_p50"] is not None
                    and row["true_tcp"]["entropy_p50"] is not None
                    else None
                )
                for row in rows
            ]
        ),
        "confidence_p50_port_minus_tcp": _mean_std(
            [
                (
                    row["port_as_tcp"]["max_softmax_p50"]
                    - row["true_tcp"]["max_softmax_p50"]
                    if row["port_as_tcp"]["max_softmax_p50"] is not None
                    and row["true_tcp"]["max_softmax_p50"] is not None
                    else None
                )
                for row in rows
            ]
        ),
    }
    return summary


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _write_report(aggregate: dict, output_path: Path) -> None:
    lines = [
        "# EXP-04I TCP Gate Diagnostic Report",
        "",
        "## Purpose",
        "",
        "Compare true TCP Flood traffic and Port Scanning samples that are absorbed as TCP Flood, to decide whether a margin or entropy gate can protect known TCP Flood while preserving Port rejection.",
        "",
        "## Per-Seed TCP Geometry",
        "",
        "| Seed | Group | TCP pred rate | MSP p50 | Margin p50 | Entropy p50 | MSP p10-p90 | Margin p10-p90 | Entropy p10-p90 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        for group, label in (("true_tcp", "True TCP Flood"), ("port_as_tcp", "Port -> TCP")):
            values = row[group]
            lines.append(
                f"| {row['split_seed']} | {label} | "
                f"{_fmt(values['predicted_tcp_rate'])} | "
                f"{_fmt(values['max_softmax_p50'])} | "
                f"{_fmt(values['margin_p50'])} | "
                f"{_fmt(values['entropy_p50'])} | "
                f"{_fmt(values['max_softmax_p10'])}-{_fmt(values['max_softmax_p90'])} | "
                f"{_fmt(values['margin_p10'])}-{_fmt(values['margin_p90'])} | "
                f"{_fmt(values['entropy_p10'])}-{_fmt(values['entropy_p90'])} |"
            )

    summary = aggregate["summary"]
    lines.extend(
        [
            "",
            "## Mean +/- Std",
            "",
            "| Group | TCP pred rate | MSP p50 | Margin p50 | Entropy p50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group, label in (("true_tcp", "True TCP Flood"), ("port_as_tcp", "Port -> TCP")):
        values = summary[group]
        lines.append(
            f"| {label} | "
            f"{_fmt(values['predicted_tcp_rate']['mean'])} +/- {_fmt(values['predicted_tcp_rate']['std'])} | "
            f"{_fmt(values['max_softmax_p50']['mean'])} +/- {_fmt(values['max_softmax_p50']['std'])} | "
            f"{_fmt(values['margin_p50']['mean'])} +/- {_fmt(values['margin_p50']['std'])} | "
            f"{_fmt(values['entropy_p50']['mean'])} +/- {_fmt(values['entropy_p50']['std'])} |"
        )
    lines.extend(
        [
            "",
            "## Median Gaps: Port -> TCP minus True TCP",
            "",
            f"- MSP p50 gap: {_fmt(summary['gap']['confidence_p50_port_minus_tcp']['mean'])} +/- {_fmt(summary['gap']['confidence_p50_port_minus_tcp']['std'])}",
            f"- Margin p50 gap: {_fmt(summary['gap']['margin_p50_port_minus_tcp']['mean'])} +/- {_fmt(summary['gap']['margin_p50_port_minus_tcp']['std'])}",
            f"- Entropy p50 gap: {_fmt(summary['gap']['entropy_p50_port_minus_tcp']['mean'])} +/- {_fmt(summary['gap']['entropy_p50_port_minus_tcp']['std'])}",
            "",
            "## Interpretation Guide",
            "",
            "- If Port -> TCP has lower margin than true TCP, test a low-margin gate.",
            "- If Port -> TCP has higher entropy than true TCP, test a high-entropy gate.",
            "- If distributions overlap heavily, margin/entropy gates will probably not fix TCP true rejection, and the method should use calibration-tail or context evidence instead.",
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


def run_tcp_gate_diagnostic(config: TcpGateDiagnosticConfig) -> dict:
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    rows = []
    for split_seed in config.split_seeds:
        run_name = f"split{split_seed}_model{config.base_seed}_resample{config.base_seed}"
        run_dir = output_dir / "runs" / run_name
        metrics_path = run_dir / "metrics.json"
        metrics = None
        if config.reuse_completed and metrics_path.exists():
            candidate = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "known_confidence_diagnostics" in candidate:
                print(f"\nReusing completed TCP-gate run: {run_name}")
                metrics = candidate
        if metrics is None:
            print(f"\nStarting TCP-gate diagnostic run: {run_name}")
            metrics = run_class_conditional_experiment(
                ClassConditionalExperimentConfig(
                    experiment_id="EXP-04I",
                    report_filename="run_report.md",
                    farm_flow_path=config.farm_flow_path,
                    output_dir=str(run_dir),
                    known_classes=config.known_classes,
                    zero_day_classes=config.zero_day_classes,
                    seed=config.base_seed,
                    split_seed=split_seed,
                    model_seed=config.base_seed,
                    resampling_seed=config.base_seed,
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
            print(f"Finished TCP-gate diagnostic run: {run_name}")
        rows.append(
            {
                "split_seed": int(split_seed),
                "true_tcp": _tcp_block(metrics, "TCP Flood"),
                "port_as_tcp": _tcp_block(metrics, "Port Scanning"),
            }
        )

    aggregate = {
        "experiment": "EXP-04I",
        "config": asdict(config),
        "runs": rows,
        "summary": _summarize(rows),
    }
    save_json(aggregate, output_dir / "tcp_gate_diagnostic_summary.json")
    _write_report(aggregate, reports_dir / "tcp_gate_diagnostic_report.md")
    return aggregate
