"""EXP-04F calibration-stability evaluation for open-set rejection."""

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
class CalibrationStabilityConfig:
    experiment_id: str = "EXP-04F"
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/calibration_stability"
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
    stabilized_threshold_pool: str = "validation_calibration"
    stabilized_bootstrap_iterations: int = 200
    stabilized_bootstrap_quantile: float = 0.25
    attack_min_confidence_floors: tuple[float, ...] = (0.68, 0.69, 0.70)
    attack_floor_profiles: tuple[str, ...] = ()
    targeted_tail_acceptance_rates: tuple[float, ...] = ()
    targeted_tail_profiles: tuple[str, ...] = ()
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    reuse_completed: bool = True


def _rate_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _profile_names(profile_specs: tuple[str, ...]) -> tuple[str, ...]:
    names = []
    for raw_spec in profile_specs:
        if "=" not in raw_spec:
            raise ValueError(
                "attack_floor_profiles entries must look like "
                "'profile_name=Class A|Class B'."
            )
        name, _ = raw_spec.split("=", 1)
        names.append(name.strip())
    return tuple(names)


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


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summarize_methods(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    methods = sorted({method for row in rows for method in row["methods"]})
    output = {}
    for method in methods:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        output[method] = {
            field: _mean_std([row[field] for row in method_rows])
            for field in (
                "utdr",
                "far",
                "macro_f1",
                "known_accuracy",
                "arp_reject",
                "port_reject",
            )
        }
    return output


def _summarize_run(metrics: dict, split_seed: int, method_names: dict[str, str]) -> dict:
    methods = {}
    for short_name, strategy_name in method_names.items():
        if strategy_name in metrics["strategies"]:
            methods[short_name] = _method_metrics(metrics["strategies"][strategy_name])

    port_diagnostics = metrics["zero_day_confidence_diagnostics"]["Port Scanning"]
    tcp_diagnostics = port_diagnostics["by_predicted_class"].get("TCP Flood", {})
    tcp_index = list(metrics["config"]["known_classes"]).index("TCP Flood")
    single_thresholds = metrics["asymmetric_thresholds"]["0.95"]
    tcp_min_confidence = 1.0 - float(
        single_thresholds["confidence_anomaly_max"][tcp_index]
    )
    return {
        "split_seed": int(split_seed),
        "resolved_seeds": metrics["resolved_seeds"],
        "methods": methods,
        "port_predicted_tcp_rate": tcp_diagnostics.get("rate_within_attack", 0.0),
        "port_tcp_confidence_p50": tcp_diagnostics.get(
            "max_softmax_quantiles",
            {},
        ).get("p50"),
        "single_tcp_min_confidence": tcp_min_confidence,
    }


def _write_report(aggregate: dict, output_path: Path) -> None:
    config = aggregate["config"]
    experiment_id = aggregate["experiment"]
    method_order = aggregate["method_order"]
    lines = [
        f"# {experiment_id} Calibration Stabilization Report",
        "",
        "## Purpose",
        "",
        "Test whether known-only stabilized calibration can reduce Port Scanning rejection variance caused by split/calibration randomness.",
        "",
        "## Protocol",
        "",
        "- Zero-day attacks are blind-test-only in every run.",
        "- Model and resampling seeds are fixed to the base seed.",
        "- Only known-data split/calibration seeds are varied.",
        f"- Stabilized threshold pool: {config['stabilized_threshold_pool']}",
        f"- Bootstrap iterations: {config['stabilized_bootstrap_iterations']}",
        f"- Bootstrap aggregation quantile: {config['stabilized_bootstrap_quantile']:.4f}",
        f"- Attack-class confidence floors: {', '.join(str(x) for x in config['attack_min_confidence_floors'])}",
        f"- Attack floor profiles: {config['attack_floor_profiles'] or ['all_non_benign']}",
        f"- Targeted tail acceptance rates: {config['targeted_tail_acceptance_rates']}",
        f"- Targeted tail profiles: {config['targeted_tail_profiles']}",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Method | UTDR | FAR | Macro-F1 | Known acc. | Arp reject | Port reject |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        for method in method_order:
            if method not in row["methods"]:
                continue
            values = row["methods"][method]
            lines.append(
                f"| {row['split_seed']} | {method} | "
                f"{values['utdr']:.4f} | {values['far']:.4f} | "
                f"{values['macro_f1']:.4f} | {values['known_accuracy']:.4f} | "
                f"{values['arp_reject']:.4f} | {values['port_reject']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Mean +/- Std",
            "",
            "| Method | UTDR | FAR | Macro-F1 | Known acc. | Arp reject | Port reject |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        if method not in aggregate["method_summaries"]:
            continue
        summary = aggregate["method_summaries"][method]
        lines.append(
            f"| {method} | "
            f"{summary['utdr']['mean']:.4f} +/- {summary['utdr']['std']:.4f} | "
            f"{summary['far']['mean']:.4f} +/- {summary['far']['std']:.4f} | "
            f"{summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f} | "
            f"{summary['known_accuracy']['mean']:.4f} +/- {summary['known_accuracy']['std']:.4f} | "
            f"{summary['arp_reject']['mean']:.4f} +/- {summary['arp_reject']['std']:.4f} | "
            f"{summary['port_reject']['mean']:.4f} +/- {summary['port_reject']['std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Prefer methods that reduce Port reject standard deviation without increasing FAR above 1%.",
            "- Known accuracy loss should remain small because the method must still preserve closed-world IDS utility.",
            "- Floor variants are sensitivity analysis; the final floor must be justified without selecting on zero-day blind-test labels.",
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


def run_calibration_stability(config: CalibrationStabilityConfig) -> dict:
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    stable_rate_name = _rate_token(config.attack_acceptance_rate)
    method_names = {
        "baseline_global_msp_full_role": "global_msp_full_role",
        "single_class_msp_full_role": (
            f"asymmetric_class_msp_full_role__a{config.attack_acceptance_rate}"
        ),
        "stable_bootstrap": f"stable_asymmetric_full_role__a{stable_rate_name}",
    }
    floor_profiles = _profile_names(config.attack_floor_profiles)
    if floor_profiles:
        for floor in config.attack_min_confidence_floors:
            for profile_name in floor_profiles:
                method_names[f"{profile_name}_floor_{floor:.3f}"] = (
                    f"stable_asymmetric_full_role__a{stable_rate_name}"
                    f"__{profile_name}_floor{_rate_token(floor)}"
                )
    else:
        for floor in config.attack_min_confidence_floors:
            method_names[f"stable_floor_{floor:.2f}"] = (
                f"stable_asymmetric_full_role__a{stable_rate_name}__floor{_rate_token(floor)}"
            )
    for tail_rate in config.targeted_tail_acceptance_rates:
        for profile_name in _profile_names(config.targeted_tail_profiles):
            method_names[f"{profile_name}_tail_accept_{tail_rate:.3f}"] = (
                f"tail_gate_full_role__a{stable_rate_name}"
                f"__{profile_name}_accept{_rate_token(tail_rate)}"
            )

    rows = []
    for split_seed in config.split_seeds:
        run_name = f"split{split_seed}_model{config.base_seed}_resample{config.base_seed}"
        run_dir = output_dir / "runs" / run_name
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed calibration-stability run: {run_name}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting calibration-stability run: {run_name}")
            metrics = run_class_conditional_experiment(
                ClassConditionalExperimentConfig(
                    experiment_id=config.experiment_id,
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
                    stabilized_threshold_pool=config.stabilized_threshold_pool,
                    stabilized_bootstrap_iterations=config.stabilized_bootstrap_iterations,
                    stabilized_bootstrap_quantile=config.stabilized_bootstrap_quantile,
                    attack_min_confidence_floors=config.attack_min_confidence_floors,
                    attack_floor_profiles=config.attack_floor_profiles,
                    targeted_tail_acceptance_rates=config.targeted_tail_acceptance_rates,
                    targeted_tail_profiles=config.targeted_tail_profiles,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    dropout=config.dropout,
                    patience=config.patience,
                    save_model=False,
                )
            )
            _clear_tensorflow()
            print(f"Finished calibration-stability run: {run_name}")
        rows.append(_summarize_run(metrics, split_seed, method_names))

    method_order = [name for name in method_names if any(name in row["methods"] for row in rows)]
    aggregate = {
        "experiment": config.experiment_id,
        "config": asdict(config),
        "method_order": method_order,
        "runs": rows,
        "method_summaries": _summarize_methods(rows),
    }
    save_json(aggregate, output_dir / "calibration_stability_summary.json")
    _write_report(aggregate, reports_dir / "calibration_stability_report.md")
    return aggregate
