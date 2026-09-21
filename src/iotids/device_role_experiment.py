"""EXP-05B device-role context refinement for agriculture open-set IDS."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import BENIGN_LABEL, DEFAULT_ZERO_DAY_CLASSES, FARM_FLOW_KNOWN_CLASSES
from .context import (
    CONTEXT_ABLATIONS,
    RelationSpec,
    context_anomaly_score,
    context_surprisal_matrix,
    fit_context_profile,
    fit_surprisal_scales,
)
from .data import (
    cap_rows_per_class,
    encode_known_labels,
    fit_feature_preprocessor,
    read_farm_flow_csv,
    split_known_and_zero_day,
    split_training_validation_calibration,
    transform_features,
)
from .experiment import AgriOpenSetConfig, _distribution, _fit_classifier
from .metrics import add_score_auc, closed_world_decisions, evaluate_decisions, make_decisions
from .openset import calibrate_thresholds, compute_open_set_scores, fit_prototypes, predict_unknown_mask
from .utils import configure_tensorflow, ensure_dir, save_json, set_global_seed


DEVICE_ROLE_VARIANTS: dict[str, tuple[RelationSpec, ...]] = {
    "full_role_context": CONTEXT_ABLATIONS["full_role_context"],
    "device_role_core": (
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
        RelationSpec("source_role", ("id.orig_h", "proto", "service", "id.resp_p")),
    ),
    "device_role_no_source_role": (
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
    ),
    "port_pair": (
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
    ),
    "source_destination_port": (
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
    ),
    "service_destination_port": (
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
    ),
    "source_port_role": (
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
    ),
    "source_role": (
        RelationSpec("source_role", ("id.orig_h", "proto", "service", "id.resp_p")),
    ),
}


@dataclass
class DeviceRoleExperimentConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/device_role_seed42"
    label_column: str = "traffic"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.10
    calibration_size: float = 0.10
    max_rows_per_class: int | None = None
    balance_strategy: str = "median"
    max_train_per_class: int | None = 50000
    known_acceptance_rate: float = 0.95
    context_benign_acceptance_rates: tuple[float, ...] = (0.995, 0.999)
    context_scale_percentile: float = 99.0
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5


def _evaluate(
    true_labels: list[str],
    pred_indices: np.ndarray,
    known_classes: tuple[str, ...],
    zero_day_classes: tuple[str, ...],
    unknown_mask: np.ndarray,
    unknown_score: np.ndarray,
) -> dict:
    decisions = make_decisions(pred_indices, unknown_mask, known_classes)
    return add_score_auc(
        evaluate_decisions(true_labels, decisions, known_classes, zero_day_classes),
        true_labels,
        zero_day_classes,
        unknown_score,
    )


def _format(value: object) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _zero_day_metric(result: dict, zero_day_classes: tuple[str, ...], metric: str) -> dict:
    return {
        label: result.get("zero_day_breakdown", {}).get(label, {}).get(metric)
        for label in zero_day_classes
    }


def _summary_row(name: str, rate_name: str, relation_count: int, result: dict, zero_day_classes: tuple[str, ...]) -> dict:
    return {
        "method": name,
        "acceptance_rate": rate_name,
        "relation_count": relation_count,
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "known_accuracy": result["known_closed_label_accuracy"],
        "auroc": result.get("unknown_score_auc"),
        "zero_day_reject": _zero_day_metric(result, zero_day_classes, "unknown_rate"),
        "zero_day_benign_miss": _zero_day_metric(result, zero_day_classes, "benign_miss_rate"),
    }


def _hybrid_context_metrics(
    true_labels: list[str],
    pred_indices: np.ndarray,
    known_classes: tuple[str, ...],
    zero_day_classes: tuple[str, ...],
    predicted_benign: np.ndarray,
    msp_mask: np.ndarray,
    msp_score: np.ndarray,
    msp_min_confidence: float,
    context_score: np.ndarray,
    context_threshold: float,
) -> dict:
    context_mask = context_score > context_threshold
    hybrid_mask = msp_mask | (predicted_benign & context_mask)
    msp_ratio = msp_score / max(1.0 - msp_min_confidence, 1e-8)
    context_ratio = context_score / max(context_threshold, 1e-8)
    return _evaluate(
        true_labels,
        pred_indices,
        known_classes,
        zero_day_classes,
        hybrid_mask,
        np.maximum(msp_ratio, context_ratio),
    )


def _write_report(metrics: dict, output_path: Path) -> None:
    zero_day_classes = metrics["config"]["zero_day_classes"]
    lines = [
        "# EXP-05B Device-Role Context Refinement Report",
        "",
        "## Purpose",
        "",
        (
            "Refine the agriculture-device context branch after EXP-05A showed that "
            "device-role relationships carry most of the useful zero-day rejection "
            "signal. This experiment removes or isolates relations to find a lighter "
            "and more defensible context design."
        ),
        "",
        "## Protocol",
        "",
        "- Zero-day attacks are blind-test-only.",
        "- The DNN uses the same low-leakage feature space as previous open-set experiments.",
        "- Device-role profiles are fitted only on Benign training traffic.",
        "- Device-role thresholds are calibrated only on Benign calibration traffic.",
        "- Context rejection is applied only to samples predicted as Benign by the DNN.",
        "",
        "## Relation Variants",
        "",
        "| Variant | Relations |",
        "|---|---|",
    ]
    for name, relations in metrics["relation_variants"].items():
        lines.append(f"| {name} | {', '.join(relations)} |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Method | a | #Rel | UTDR | FAR | Macro-F1 | Known acc. | AUROC | "
            + " | ".join(f"{label} reject" for label in zero_day_classes)
            + " |",
            "|---|---:|---:|---:|---:|---:|---:|---:"
            + "".join("|---:" for _ in zero_day_classes)
            + "|",
        ]
    )
    for row in metrics["summary_rows"]:
        zero_values = " | ".join(
            _format(row["zero_day_reject"].get(label)) for label in zero_day_classes
        )
        lines.append(
            f"| {row['method']} | {row['acceptance_rate']} | {row['relation_count']} | "
            f"{_format(row['utdr'])} | {_format(row['far'])} | {_format(row['macro_f1'])} | "
            f"{_format(row['known_accuracy'])} | {_format(row['auroc'])} | {zero_values} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "Prefer the smallest relation variant that keeps Arp Spoofing rejection "
                "near the full-role context result, preserves Port Scanning rejection, "
                "keeps Benign FAR below 1%, and keeps known-class accuracy stable."
            ),
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_device_role_experiment(config: DeviceRoleExperimentConfig) -> dict:
    set_global_seed(config.seed)
    configure_tensorflow()
    start = time.perf_counter()
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")

    allowed_labels = list(config.known_classes) + list(config.zero_day_classes)
    df = read_farm_flow_csv(config.farm_flow_path, config.label_column, allowed_labels)
    df = cap_rows_per_class(df, config.label_column, config.max_rows_per_class, config.seed)
    known_pool, known_holdout, blind_df = split_known_and_zero_day(
        df,
        config.known_classes,
        config.zero_day_classes,
        config.label_column,
        config.test_size,
        config.seed,
    )
    train_df, validation_df, calibration_df = split_training_validation_calibration(
        known_pool,
        config.label_column,
        config.validation_size,
        config.calibration_size,
        config.seed,
    )

    preprocessor = fit_feature_preprocessor(
        train_df,
        config.label_column,
        columns_to_drop=(
            "id.orig_h",
            "id.orig_p",
            "id.resp_h",
            "id.resp_p",
            "history",
            "tunnel_parents",
            "local_orig",
            "local_resp",
        ),
        allowed_categorical_columns=("proto", "service", "conn_state"),
    )
    X_train = transform_features(train_df, preprocessor)
    X_validation = transform_features(validation_df, preprocessor)
    X_calibration = transform_features(calibration_df, preprocessor)
    X_blind = transform_features(blind_df, preprocessor)
    y_train = encode_known_labels(train_df[config.label_column], config.known_classes)
    y_validation = encode_known_labels(validation_df[config.label_column], config.known_classes)

    dnn_config = AgriOpenSetConfig(
        known_classes=config.known_classes,
        zero_day_classes=config.zero_day_classes,
        seed=config.seed,
        balance_strategy=config.balance_strategy,
        max_train_per_class=config.max_train_per_class,
        known_acceptance_rate=config.known_acceptance_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        dropout=config.dropout,
        patience=config.patience,
        enable_ae=False,
    )
    model, embedding_model, logits_model, history, resampling, training_seconds = _fit_classifier(
        dnn_config,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    train_embeddings = embedding_model.predict(X_train, batch_size=config.batch_size, verbose=0)
    prototypes = fit_prototypes(train_embeddings, y_train, len(config.known_classes))
    cal_probs = model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_logits = logits_model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_embeddings = embedding_model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_open_scores = compute_open_set_scores(cal_probs, cal_logits, cal_embeddings, prototypes)
    open_thresholds = calibrate_thresholds(cal_open_scores, config.known_acceptance_rate)

    blind_probs = model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_logits = logits_model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_embeddings = embedding_model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_open_scores = compute_open_set_scores(blind_probs, blind_logits, blind_embeddings, prototypes)
    pred_indices = np.argmax(blind_probs, axis=1)
    true_labels = blind_df[config.label_column].tolist()
    benign_index = list(config.known_classes).index(BENIGN_LABEL)
    predicted_benign = pred_indices == benign_index
    msp_mask = predict_unknown_mask(blind_open_scores, open_thresholds, strategy="msp")
    msp_score = 1.0 - blind_open_scores.max_softmax

    strategies: dict[str, dict] = {
        "closed_dnn": evaluate_decisions(
            true_labels,
            closed_world_decisions(pred_indices, config.known_classes),
            config.known_classes,
            config.zero_day_classes,
        ),
        "msp": _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            msp_mask,
            msp_score,
        ),
    }
    summary_rows = [
        _summary_row(
            "msp",
            "-",
            0,
            strategies["msp"],
            config.zero_day_classes,
        )
    ]

    benign_train = train_df.loc[train_df[config.label_column] == BENIGN_LABEL].reset_index(drop=True)
    benign_calibration = calibration_df.loc[
        calibration_df[config.label_column] == BENIGN_LABEL
    ].reset_index(drop=True)

    context_details: dict[str, dict] = {}
    for variant_name, relations in DEVICE_ROLE_VARIANTS.items():
        profile = fit_context_profile(benign_train, relations)
        calibration_matrix, relation_names = context_surprisal_matrix(profile, benign_calibration)
        scales = fit_surprisal_scales(calibration_matrix, config.context_scale_percentile)
        calibration_score = context_anomaly_score(calibration_matrix, scales)
        blind_matrix, _ = context_surprisal_matrix(profile, blind_df)
        blind_score = context_anomaly_score(blind_matrix, scales)

        thresholds = {}
        for acceptance_rate in config.context_benign_acceptance_rates:
            if not 0.5 <= acceptance_rate < 1.0:
                raise ValueError("Context benign acceptance rates must be in [0.5, 1.0).")
            rate_name = f"{acceptance_rate:.3f}".rstrip("0").rstrip(".")
            threshold = float(np.percentile(calibration_score, acceptance_rate * 100.0))
            method_name = f"msp_context__{variant_name}__a{rate_name}"
            strategies[method_name] = _hybrid_context_metrics(
                true_labels,
                pred_indices,
                config.known_classes,
                config.zero_day_classes,
                predicted_benign,
                msp_mask,
                msp_score,
                open_thresholds.msp_min_confidence,
                blind_score,
                threshold,
            )
            summary_rows.append(
                _summary_row(
                    method_name,
                    rate_name,
                    len(relations),
                    strategies[method_name],
                    config.zero_day_classes,
                )
            )
            thresholds[rate_name] = threshold

        context_details[variant_name] = {
            "relations": relation_names,
            "thresholds": thresholds,
            "scales": scales,
        }

    relation_variants = {
        name: [relation.name for relation in relations]
        for name, relations in DEVICE_ROLE_VARIANTS.items()
    }
    metrics = {
        "experiment": "EXP-05B",
        "config": asdict(config),
        "dataset": {
            "train_distribution": _distribution(train_df[config.label_column]),
            "validation_distribution": _distribution(validation_df[config.label_column]),
            "calibration_distribution": _distribution(calibration_df[config.label_column]),
            "known_holdout_distribution": _distribution(known_holdout[config.label_column]),
            "blind_distribution": _distribution(true_labels),
        },
        "feature_space": {
            "profile": "low_leakage",
            "input_dim": int(X_train.shape[1]),
            "numeric_columns": preprocessor.numeric_columns,
            "categorical_columns": preprocessor.categorical_columns,
            "dropped_columns": preprocessor.dropped_columns,
        },
        "model": {
            "parameter_count": int(model.count_params()),
            "training_seconds": training_seconds,
            "history": history,
        },
        "resampling": resampling,
        "open_thresholds": asdict(open_thresholds),
        "relation_variants": relation_variants,
        "context_details": context_details,
        "strategies": strategies,
        "summary_rows": summary_rows,
        "runtime_seconds": time.perf_counter() - start,
    }
    save_json(metrics, output_dir / "metrics.json")
    _write_report(metrics, reports_dir / "device_role_refinement_report.md")
    return metrics
