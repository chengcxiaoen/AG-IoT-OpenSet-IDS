"""EXP-08 lightweight class-wise MSP leave-one-attack-out evaluation."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .class_conditional import (
    calibrate_class_confidence_thresholds,
    score_class_confidence,
)
from .constants import BENIGN_LABEL, UNKNOWN_LABEL
from .context import (
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
from .metrics import (
    add_score_auc,
    closed_world_decisions,
    evaluate_decisions,
    make_decisions,
)
from .openset import calibrate_thresholds, compute_open_set_scores, fit_prototypes, predict_unknown_mask
from .utils import configure_tensorflow, ensure_dir, save_json, set_global_seed


ATTACK_CLASSES = (
    "HTTP Flood",
    "ICMP Flood",
    "MQTT Flood",
    "TCP Flood",
    "UDP Flood",
    "Arp Spoofing",
    "Port Scanning",
)

LIGHTWEIGHT_PORT_CONTEXT = (
    RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
    RelationSpec("service_destination_port", ("service", "id.resp_p")),
)

METHOD_ORDER = (
    "closed_dnn",
    "global_msp",
    "classwise_msp",
    "global_msp_port_context",
    "classwise_msp_port_context",
)


@dataclass
class ClasswiseLeaveOneConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/classwise_leave_one"
    label_column: str = "traffic"
    held_out_attacks: tuple[str, ...] = ATTACK_CLASSES
    seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.10
    calibration_size: float = 0.10
    max_rows_per_class: int | None = None
    balance_strategy: str = "median"
    max_train_per_class: int | None = 50000
    known_acceptance_rate: float = 0.95
    context_benign_acceptance_rate: float = 0.999
    context_scale_percentile: float = 99.0
    minimum_class_calibration_samples: int = 30
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    reuse_completed: bool = True


def _clear_tensorflow() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _run_dir_name(held_out_attack: str) -> str:
    return held_out_attack.lower().replace(" ", "_").replace("/", "_")


def _evaluate(
    true_labels: list[str],
    pred_indices: np.ndarray,
    known_classes: tuple[str, ...],
    zero_day_classes: tuple[str, ...],
    unknown_mask: np.ndarray,
    unknown_score: np.ndarray,
) -> dict:
    predictions = make_decisions(pred_indices, unknown_mask, known_classes)
    return add_score_auc(
        evaluate_decisions(true_labels, predictions, known_classes, zero_day_classes),
        true_labels,
        zero_day_classes,
        unknown_score,
    )


def _extract_standard_metrics(result: dict) -> dict[str, float | None]:
    unknown_report = result.get("classification_report", {}).get(UNKNOWN_LABEL, {})
    return {
        "known_accuracy": result.get("known_closed_label_accuracy"),
        "macro_f1": result.get("open_set_macro_f1"),
        "unknown_precision": unknown_report.get("precision", 0.0),
        "unknown_recall": unknown_report.get("recall", 0.0),
        "unknown_f1": unknown_report.get("f1-score", 0.0),
        "fpr": result.get("benign_false_alarm_rate"),
        "auroc": result.get("unknown_score_auc"),
    }


def _method_summary(result: dict, held_out_attack: str) -> dict:
    standard = _extract_standard_metrics(result)
    zero_day = result.get("zero_day_breakdown", {}).get(held_out_attack, {})
    standard.update(
        {
            "held_out_support": zero_day.get("support"),
            "prediction_distribution": zero_day.get("prediction_distribution", {}),
        }
    )
    return standard


def _fit_port_context(
    train_df,
    calibration_df,
    blind_df,
    label_column: str,
    context_scale_percentile: float,
    context_benign_acceptance_rate: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    benign_train = train_df.loc[train_df[label_column] == BENIGN_LABEL].reset_index(drop=True)
    benign_calibration = calibration_df.loc[
        calibration_df[label_column] == BENIGN_LABEL
    ].reset_index(drop=True)

    profile = fit_context_profile(benign_train, LIGHTWEIGHT_PORT_CONTEXT)
    calibration_matrix, relation_names = context_surprisal_matrix(profile, benign_calibration)
    scales = fit_surprisal_scales(calibration_matrix, context_scale_percentile)
    calibration_score = context_anomaly_score(calibration_matrix, scales)
    threshold = float(np.percentile(calibration_score, context_benign_acceptance_rate * 100.0))
    blind_matrix, _ = context_surprisal_matrix(profile, blind_df)
    blind_score = context_anomaly_score(blind_matrix, scales)
    return blind_score, scales, threshold, relation_names


def _run_one(config: ClasswiseLeaveOneConfig, held_out_attack: str, output_dir: Path) -> dict:
    set_global_seed(config.seed)
    configure_tensorflow()
    start = time.perf_counter()

    known_classes = tuple(
        [BENIGN_LABEL] + [label for label in ATTACK_CLASSES if label != held_out_attack]
    )
    zero_day_classes = (held_out_attack,)
    allowed_labels = list(known_classes) + [held_out_attack]

    df = read_farm_flow_csv(config.farm_flow_path, config.label_column, allowed_labels)
    df = cap_rows_per_class(df, config.label_column, config.max_rows_per_class, config.seed)
    known_pool, known_holdout, blind_df = split_known_and_zero_day(
        df,
        known_classes,
        zero_day_classes,
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
    y_train = encode_known_labels(train_df[config.label_column], known_classes)
    y_validation = encode_known_labels(validation_df[config.label_column], known_classes)
    y_calibration = encode_known_labels(calibration_df[config.label_column], known_classes)

    dnn_config = AgriOpenSetConfig(
        known_classes=known_classes,
        zero_day_classes=zero_day_classes,
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
    calibration_probabilities = model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    calibration_logits = logits_model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    calibration_embeddings = embedding_model.predict(
        X_calibration,
        batch_size=config.batch_size,
        verbose=0,
    )
    blind_probabilities = model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_logits = logits_model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_embeddings = embedding_model.predict(X_blind, batch_size=config.batch_size, verbose=0)

    prototypes = fit_prototypes(train_embeddings, y_train, len(known_classes))
    calibration_scores = compute_open_set_scores(
        calibration_probabilities,
        calibration_logits,
        calibration_embeddings,
        prototypes,
    )
    global_thresholds = calibrate_thresholds(calibration_scores, config.known_acceptance_rate)
    blind_scores = compute_open_set_scores(
        blind_probabilities,
        blind_logits,
        blind_embeddings,
        prototypes,
    )

    pred_indices = np.argmax(blind_probabilities, axis=1)
    true_labels = blind_df[config.label_column].tolist()
    predicted_benign = pred_indices == known_classes.index(BENIGN_LABEL)

    global_msp_mask = predict_unknown_mask(blind_scores, global_thresholds, strategy="msp")
    global_msp_score = 1.0 - blind_scores.max_softmax

    class_thresholds = calibrate_class_confidence_thresholds(
        calibration_probabilities,
        y_calibration,
        known_acceptance_rate=config.known_acceptance_rate,
        minimum_correct_samples=config.minimum_class_calibration_samples,
    )
    classwise_score, classwise_mask = score_class_confidence(
        blind_probabilities,
        pred_indices,
        class_thresholds,
    )

    context_score, context_scales, context_threshold, context_relations = _fit_port_context(
        train_df,
        calibration_df,
        blind_df,
        config.label_column,
        config.context_scale_percentile,
        config.context_benign_acceptance_rate,
    )
    context_mask = predicted_benign & (context_score > context_threshold)
    context_relative_score = context_score / max(context_threshold, 1e-8)

    global_msp_relative_score = global_msp_score / max(
        1.0 - global_thresholds.msp_min_confidence,
        1e-8,
    )
    global_context_mask = global_msp_mask | context_mask
    global_context_score = np.maximum(
        global_msp_relative_score,
        np.where(predicted_benign, context_relative_score, 0.0),
    )

    classwise_context_mask = classwise_mask | context_mask
    classwise_context_score = np.maximum(
        classwise_score,
        np.where(predicted_benign, context_relative_score, 0.0),
    )

    strategies = {
        "closed_dnn": evaluate_decisions(
            true_labels,
            closed_world_decisions(pred_indices, known_classes),
            known_classes,
            zero_day_classes,
        ),
        "global_msp": _evaluate(
            true_labels,
            pred_indices,
            known_classes,
            zero_day_classes,
            global_msp_mask,
            global_msp_score,
        ),
        "classwise_msp": _evaluate(
            true_labels,
            pred_indices,
            known_classes,
            zero_day_classes,
            classwise_mask,
            classwise_score,
        ),
        "global_msp_port_context": _evaluate(
            true_labels,
            pred_indices,
            known_classes,
            zero_day_classes,
            global_context_mask,
            global_context_score,
        ),
        "classwise_msp_port_context": _evaluate(
            true_labels,
            pred_indices,
            known_classes,
            zero_day_classes,
            classwise_context_mask,
            classwise_context_score,
        ),
    }

    metrics = {
        "experiment": "EXP-08",
        "held_out_attack": held_out_attack,
        "known_classes": known_classes,
        "zero_day_classes": zero_day_classes,
        "config": asdict(config),
        "dataset": {
            "train_distribution": _distribution(train_df[config.label_column]),
            "validation_distribution": _distribution(validation_df[config.label_column]),
            "calibration_distribution": _distribution(calibration_df[config.label_column]),
            "known_holdout_distribution": _distribution(known_holdout[config.label_column]),
            "blind_distribution": _distribution(true_labels),
        },
        "feature_space": {
            "input_dim": int(X_train.shape[1]),
            "numeric_columns": preprocessor.numeric_columns,
            "categorical_columns": preprocessor.categorical_columns,
            "dropped_columns": preprocessor.dropped_columns,
        },
        "model": {
            "parameter_count": int(model.count_params()),
            "training_seconds": float(training_seconds),
            "history": history,
        },
        "resampling": resampling,
        "global_msp_threshold": float(global_thresholds.msp_min_confidence),
        "classwise_thresholds": {
            class_name: {
                "min_confidence": float(1.0 - class_thresholds.confidence_anomaly_max[class_id]),
                "calibration_count": class_thresholds.calibration_counts[class_id],
                "calibration_source": class_thresholds.calibration_sources[class_id],
            }
            for class_id, class_name in enumerate(known_classes)
        },
        "context": {
            "name": "port_pair",
            "relations": context_relations,
            "threshold": context_threshold,
            "scales": context_scales,
        },
        "strategies": strategies,
        "standard_summary": {
            method: _method_summary(result, held_out_attack)
            for method, result in strategies.items()
        },
        "runtime_seconds": time.perf_counter() - start,
    }
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return {"mean": None, "std": None}
    array = np.asarray(filtered, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _aggregate(rows: list[dict]) -> dict:
    fields = (
        "known_accuracy",
        "macro_f1",
        "unknown_precision",
        "unknown_recall",
        "unknown_f1",
        "fpr",
        "auroc",
    )
    method_summaries = {}
    for method in METHOD_ORDER:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        if not method_rows:
            continue
        method_summaries[method] = {
            field: _mean_std([row.get(field) for row in method_rows])
            for field in fields
        }
    return method_summaries


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_mean_std(summary: dict[str, float | None]) -> str:
    if summary["mean"] is None:
        return "-"
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def _write_report(aggregate: dict, output_path: Path) -> None:
    method_order = [method for method in METHOD_ORDER if method in aggregate["method_summaries"]]
    lines = [
        "# EXP-08 Lightweight Class-Wise MSP Open-Set Report",
        "",
        "## Purpose",
        "",
        (
            "Evaluate a lightweight class-wise MSP calibration method for open-set "
            "intrusion detection. The experiment avoids additional deep models and "
            "uses common metrics instead of custom primary indicators."
        ),
        "",
        "## Compared Methods",
        "",
        "- `closed_dnn`: paper-style DNN without Unknown output.",
        "- `global_msp`: one global MSP threshold calibrated on known validation traffic.",
        "- `classwise_msp`: one MSP threshold per predicted known class.",
        "- `global_msp_port_context`: global MSP plus lightweight agriculture port-pair context for predicted-Benign samples.",
        "- `classwise_msp_port_context`: class-wise MSP plus the same lightweight port-pair context.",
        "",
        "## Primary Metrics",
        "",
        "- `Known Acc`: accuracy on known-class blind-test samples.",
        "- `Macro-F1`: macro F1 over known classes and `Unknown Attack`.",
        "- `Unknown Precision / Recall / F1`: standard precision, recall, and F1 for the Unknown class.",
        "- `FPR`: false positive rate of Benign traffic.",
        "- `AUROC`: score-level known-vs-unknown separability.",
        "",
        "## Per-Held-Out Results",
        "",
        "| Held-out attack | Method | Known Acc | Macro-F1 | Unknown Precision | Unknown Recall | Unknown F1 | FPR | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["runs"]:
        for method in method_order:
            if method not in row["methods"]:
                continue
            values = row["methods"][method]
            lines.append(
                f"| {row['held_out_attack']} | {method} | "
                f"{_fmt(values['known_accuracy'])} | {_fmt(values['macro_f1'])} | "
                f"{_fmt(values['unknown_precision'])} | {_fmt(values['unknown_recall'])} | "
                f"{_fmt(values['unknown_f1'])} | {_fmt(values['fpr'])} | "
                f"{_fmt(values['auroc'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean +/- Std Across Held-Out Attacks",
            "",
            "| Method | Known Acc | Macro-F1 | Unknown Precision | Unknown Recall | Unknown F1 | FPR | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        summary = aggregate["method_summaries"][method]
        lines.append(
            f"| {method} | {_fmt_mean_std(summary['known_accuracy'])} | "
            f"{_fmt_mean_std(summary['macro_f1'])} | "
            f"{_fmt_mean_std(summary['unknown_precision'])} | "
            f"{_fmt_mean_std(summary['unknown_recall'])} | "
            f"{_fmt_mean_std(summary['unknown_f1'])} | "
            f"{_fmt_mean_std(summary['fpr'])} | "
            f"{_fmt_mean_std(summary['auroc'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "Prefer a lightweight method only if it improves Unknown Recall and "
                "Unknown F1 without unacceptable FPR or known-class accuracy loss. "
                "Prediction distributions are stored in JSON for error analysis, but "
                "they are not treated as primary metrics."
            ),
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_classwise_leave_one(config: ClasswiseLeaveOneConfig) -> dict:
    invalid = sorted(set(config.held_out_attacks) - set(ATTACK_CLASSES))
    if invalid:
        raise ValueError(f"Unknown held-out attacks: {invalid}")

    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    rows = []
    for held_out_attack in config.held_out_attacks:
        run_dir = ensure_dir(output_dir / "runs" / _run_dir_name(held_out_attack))
        metrics_path = run_dir / "metrics.json"
        if config.reuse_completed and metrics_path.exists():
            print(f"\nReusing completed EXP-08 run: {held_out_attack}", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\nStarting EXP-08 run: {held_out_attack}", flush=True)
            metrics = _run_one(config, held_out_attack, run_dir)
            _clear_tensorflow()
            print(f"Finished EXP-08 run: {held_out_attack}", flush=True)
        rows.append(
            {
                "held_out_attack": held_out_attack,
                "methods": metrics["standard_summary"],
            }
        )

    aggregate = {
        "experiment": "EXP-08",
        "config": asdict(config),
        "method_order": list(METHOD_ORDER),
        "runs": rows,
        "method_summaries": _aggregate(rows),
    }
    save_json(aggregate, output_dir / "classwise_leave_one_summary.json")
    _write_report(aggregate, reports_dir / "classwise_leave_one_report.md")
    return aggregate
