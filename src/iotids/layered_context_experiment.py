"""EXP-05A layered agriculture-context ablation for open-set IDS."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import BENIGN_LABEL, DEFAULT_ZERO_DAY_CLASSES, FARM_FLOW_KNOWN_CLASSES
from .context import (
    CONTEXT_ABLATIONS,
    LAYERED_CONTEXT_GROUPS,
    aggregate_layered_scores,
    context_anomaly_score,
    context_surprisal_matrix,
    fit_context_profile,
    fit_layered_context_model,
    fit_surprisal_scales,
    layered_context_score_matrix,
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


@dataclass
class LayeredContextExperimentConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/layered_context_seed42"
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
    context_benign_acceptance_rates: tuple[float, ...] = (0.99, 0.995, 0.999)
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


def _zero_day_value(result: dict, zero_day_classes: tuple[str, ...], metric: str) -> dict[str, float | None]:
    breakdown = result.get("zero_day_breakdown", {})
    return {label: breakdown.get(label, {}).get(metric) for label in zero_day_classes}


def _summary_row(name: str, family: str, result: dict, zero_day_classes: tuple[str, ...]) -> dict:
    row = {
        "method": name,
        "family": family,
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "known_accuracy": result["known_closed_label_accuracy"],
        "auroc": result.get("unknown_score_auc"),
        "zero_day_reject": _zero_day_value(result, zero_day_classes, "unknown_rate"),
        "zero_day_benign_miss": _zero_day_value(result, zero_day_classes, "benign_miss_rate"),
    }
    return row


def _context_score_ratio(score: np.ndarray, threshold: float) -> np.ndarray:
    return score / max(float(threshold), 1e-8)


def _add_context_strategy(
    strategies: dict[str, dict],
    name: str,
    true_labels: list[str],
    pred_indices: np.ndarray,
    known_classes: tuple[str, ...],
    zero_day_classes: tuple[str, ...],
    msp_mask: np.ndarray,
    msp_score: np.ndarray,
    msp_threshold: float,
    predicted_benign: np.ndarray,
    context_mask: np.ndarray,
    context_score: np.ndarray,
    context_threshold: float,
) -> None:
    context_ratio = _context_score_ratio(context_score, context_threshold)
    msp_ratio = msp_score / max(1.0 - msp_threshold, 1e-8)

    strategies[f"context_only__{name}"] = _evaluate(
        true_labels,
        pred_indices,
        known_classes,
        zero_day_classes,
        context_mask,
        context_ratio,
    )
    hybrid_mask = msp_mask | (predicted_benign & context_mask)
    strategies[f"msp_context__{name}"] = _evaluate(
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
        "# EXP-05A Layered Agriculture Context Ablation Report",
        "",
        "## Purpose",
        "",
        (
            "Verify that the proposed agriculture-device context branch is not a simple "
            "A+B attachment. The experiment decomposes normal agriculture communication "
            "relationships into protocol-service, device-role, and connection-behavior "
            "layers, then compares flat context, individual layers, and fused layered "
            "context under the same DNN and MSP calibration."
        ),
        "",
        "## Strict Protocol",
        "",
        "- Zero-day attacks are used only in the final blind test.",
        "- The DNN is trained only on the known classes.",
        "- MSP thresholds are calibrated only on known calibration traffic.",
        "- Context profiles are fitted only on Benign training traffic.",
        "- Context thresholds and scales are calibrated only on Benign calibration traffic.",
        "- Raw endpoint identifiers are relationship lookup keys, not DNN input features.",
        "",
        "## Context Layers",
        "",
        "| Layer | Weight | Relations |",
        "|---|---:|---|",
    ]
    for layer_name, layer in metrics["layer_definitions"].items():
        lines.append(
            f"| {layer_name} | {layer['weight']:.2f} | {', '.join(layer['relations'])} |"
        )

    lines.extend(
        [
            "",
            "## Main Results",
            "",
            "| Method | Family | UTDR | FAR | Macro-F1 | Known acc. | AUROC | "
            + " | ".join(f"{label} reject" for label in zero_day_classes)
            + " |",
            "|---|---|---:|---:|---:|---:|---:"
            + "".join("|---:" for _ in zero_day_classes)
            + "|",
        ]
    )
    for row in metrics["summary_rows"]:
        zero_values = " | ".join(
            _format(row["zero_day_reject"].get(label)) for label in zero_day_classes
        )
        lines.append(
            f"| {row['method']} | {row['family']} | {_format(row['utdr'])} | "
            f"{_format(row['far'])} | {_format(row['macro_f1'])} | "
            f"{_format(row['known_accuracy'])} | {_format(row['auroc'])} | "
            f"{zero_values} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This Experiment",
            "",
            "- `flat_full_role` is the previous flat context route: all relationships share one scoring space.",
            "- `layer_*` rows test which agriculture communication layer contributes useful rejection.",
            "- `layered_any` is the proposed layered threshold route: each layer keeps its own benign-calibrated boundary.",
            "- `layered_weighted` is a domain-weighted fusion route for sensitivity analysis.",
            "",
            "## Claim Rule",
            "",
            (
                "A layered context design is useful if it improves near-normal zero-day "
                "rejection over MSP and flat context, keeps benign FAR below 1%, and "
                "does not materially reduce known-class accuracy."
            ),
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_layered_context_experiment(config: LayeredContextExperimentConfig) -> dict:
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

    low_leakage_columns_to_drop = (
        "id.orig_h",
        "id.orig_p",
        "id.resp_h",
        "id.resp_p",
        "history",
        "tunnel_parents",
        "local_orig",
        "local_resp",
    )
    preprocessor = fit_feature_preprocessor(
        train_df,
        config.label_column,
        columns_to_drop=low_leakage_columns_to_drop,
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
        _summary_row("msp", "confidence", strategies["msp"], config.zero_day_classes)
    ]

    benign_train = train_df.loc[train_df[config.label_column] == BENIGN_LABEL].reset_index(drop=True)
    benign_calibration = calibration_df.loc[
        calibration_df[config.label_column] == BENIGN_LABEL
    ].reset_index(drop=True)

    context_details: dict[str, dict] = {}
    full_profile = fit_context_profile(benign_train, CONTEXT_ABLATIONS["full_role_context"])
    full_cal_matrix, full_relation_names = context_surprisal_matrix(full_profile, benign_calibration)
    full_scales = fit_surprisal_scales(full_cal_matrix, config.context_scale_percentile)
    full_cal_score = context_anomaly_score(full_cal_matrix, full_scales)
    full_blind_matrix, _ = context_surprisal_matrix(full_profile, blind_df)
    full_blind_score = context_anomaly_score(full_blind_matrix, full_scales)

    layered_model = fit_layered_context_model(
        benign_train,
        benign_calibration,
        scale_percentile=config.context_scale_percentile,
    )
    cal_layer_scores, layer_names = layered_context_score_matrix(layered_model, benign_calibration)
    blind_layer_scores, _ = layered_context_score_matrix(layered_model, blind_df)

    for acceptance_rate in config.context_benign_acceptance_rates:
        if not 0.5 <= acceptance_rate < 1.0:
            raise ValueError("Context benign acceptance rates must be in [0.5, 1.0).")
        rate_name = f"{acceptance_rate:.3f}".rstrip("0").rstrip(".")

        full_threshold = float(np.percentile(full_cal_score, acceptance_rate * 100.0))
        full_mask = full_blind_score > full_threshold
        full_name = f"flat_full_role__a{rate_name}"
        _add_context_strategy(
            strategies,
            full_name,
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            msp_mask,
            msp_score,
            open_thresholds.msp_min_confidence,
            predicted_benign,
            full_mask,
            full_blind_score,
            full_threshold,
        )
        summary_rows.append(
            _summary_row(
                f"msp_context__{full_name}",
                "flat_context",
                strategies[f"msp_context__{full_name}"],
                config.zero_day_classes,
            )
        )

        layer_thresholds = np.percentile(cal_layer_scores, acceptance_rate * 100.0, axis=0)
        layer_ratios = blind_layer_scores / np.maximum(layer_thresholds, 1e-8)
        any_layer_mask = np.any(blind_layer_scores > layer_thresholds, axis=1)
        any_layer_score = np.max(layer_ratios, axis=1)
        any_name = f"layered_any__a{rate_name}"
        _add_context_strategy(
            strategies,
            any_name,
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            msp_mask,
            msp_score,
            open_thresholds.msp_min_confidence,
            predicted_benign,
            any_layer_mask,
            any_layer_score,
            1.0,
        )
        summary_rows.append(
            _summary_row(
                f"msp_context__{any_name}",
                "layered_context",
                strategies[f"msp_context__{any_name}"],
                config.zero_day_classes,
            )
        )

        for aggregate_mode in ("max", "weighted_mean"):
            cal_aggregate = aggregate_layered_scores(
                cal_layer_scores,
                layer_names,
                layered_model.weights,
                mode=aggregate_mode,
            )
            blind_aggregate = aggregate_layered_scores(
                blind_layer_scores,
                layer_names,
                layered_model.weights,
                mode=aggregate_mode,
            )
            aggregate_threshold = float(np.percentile(cal_aggregate, acceptance_rate * 100.0))
            aggregate_mask = blind_aggregate > aggregate_threshold
            aggregate_name = f"layered_{aggregate_mode}__a{rate_name}"
            _add_context_strategy(
                strategies,
                aggregate_name,
                true_labels,
                pred_indices,
                config.known_classes,
                config.zero_day_classes,
                msp_mask,
                msp_score,
                open_thresholds.msp_min_confidence,
                predicted_benign,
                aggregate_mask,
                blind_aggregate,
                aggregate_threshold,
            )
            summary_rows.append(
                _summary_row(
                    f"msp_context__{aggregate_name}",
                    "layered_context",
                    strategies[f"msp_context__{aggregate_name}"],
                    config.zero_day_classes,
                )
            )

        for layer_index, layer_name in enumerate(layer_names):
            threshold = float(layer_thresholds[layer_index])
            layer_mask = blind_layer_scores[:, layer_index] > threshold
            method_name = f"layer_{layer_name}__a{rate_name}"
            _add_context_strategy(
                strategies,
                method_name,
                true_labels,
                pred_indices,
                config.known_classes,
                config.zero_day_classes,
                msp_mask,
                msp_score,
                open_thresholds.msp_min_confidence,
                predicted_benign,
                layer_mask,
                blind_layer_scores[:, layer_index],
                threshold,
            )
            summary_rows.append(
                _summary_row(
                    f"msp_context__{method_name}",
                    "single_layer",
                    strategies[f"msp_context__{method_name}"],
                    config.zero_day_classes,
                )
            )

        context_details[rate_name] = {
            "flat_full_role_threshold": full_threshold,
            "layer_thresholds": {
                layer_name: float(layer_thresholds[index])
                for index, layer_name in enumerate(layer_names)
            },
        }

    layer_definitions = {
        name: {
            "description": layer.description,
            "weight": layer.weight,
            "relations": [relation.name for relation in layer.relations],
            "columns": {relation.name: relation.columns for relation in layer.relations},
        }
        for name, layer in LAYERED_CONTEXT_GROUPS.items()
    }
    metrics = {
        "experiment": "EXP-05A",
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
        "layer_definitions": layer_definitions,
        "context_details": context_details,
        "flat_full_role_relations": full_relation_names,
        "layered_relation_names": layered_model.relation_names,
        "strategies": strategies,
        "summary_rows": summary_rows,
        "runtime_seconds": time.perf_counter() - start,
    }
    save_json(metrics, output_dir / "metrics.json")
    _write_report(metrics, reports_dir / "layered_context_report.md")
    return metrics
