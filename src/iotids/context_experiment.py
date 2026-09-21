"""EXP-03B context-aware open-set evaluation on Farm-Flow."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from .constants import BENIGN_LABEL, DEFAULT_ZERO_DAY_CLASSES, FARM_FLOW_KNOWN_CLASSES, FEATURE_PROFILES
from .context import (
    CONTEXT_ABLATIONS,
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


@dataclass
class ContextExperimentConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/context_open_seed42"
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
    save_model: bool = False


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


def _summary_row(name: str, result: dict) -> dict:
    arp = result["zero_day_breakdown"].get("Arp Spoofing", {})
    port = result["zero_day_breakdown"].get("Port Scanning", {})
    return {
        "method": name,
        "utdr": result["unknown_threat_detection_rate"],
        "far": result["benign_false_alarm_rate"],
        "macro_f1": result["open_set_macro_f1"],
        "auroc": result["unknown_score_auc"],
        "arp_reject": arp.get("unknown_rate"),
        "arp_benign_miss": arp.get("benign_miss_rate"),
        "port_reject": port.get("unknown_rate"),
    }


def _format(value: object) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _write_report(metrics: dict, output_path: Path) -> None:
    lines = [
        "# EXP-03B Agriculture Device-Context Report",
        "",
        "## Purpose",
        "",
        "Test whether normal-training-only device relationship rarity can detect near-normal zero-day attacks without directly feeding raw endpoint identities into the DNN.",
        "",
        "## Protocol",
        "",
        "- The DNN uses the low-leakage feature profile.",
        "- Context profiles are fitted only on Benign training traffic.",
        "- Context thresholds and normalization scales are fitted only on Benign calibration traffic.",
        "- Raw IP addresses are lookup keys for relationships, not DNN input features.",
        "- Context rejection is applied only when the DNN predicts Benign in the hybrid method.",
        "",
        "## Results",
        "",
        "| Method | UTDR | FAR | Macro-F1 | AUROC | Arp reject | Arp benign miss | Port reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["summary_rows"]:
        lines.append(
            f"| {row['method']} | {_format(row['utdr'])} | {_format(row['far'])} | "
            f"{_format(row['macro_f1'])} | {_format(row['auroc'])} | "
            f"{_format(row['arp_reject'])} | {_format(row['arp_benign_miss'])} | "
            f"{_format(row['port_reject'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Rule",
            "",
            "The context route is useful only if it materially improves Arp Spoofing rejection while keeping benign FAR below 1% and preserving Port Scanning rejection.",
            "",
            "## Limitation",
            "",
            "Farm-Flow contains only a small number of source devices and one destination address. Even relationship-based improvements require held-out-device or cross-dataset validation before a generalization claim.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_context_experiment(config: ContextExperimentConfig) -> dict:
    set_global_seed(config.seed)
    configure_tensorflow()
    start = time.perf_counter()
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    models_dir = ensure_dir(output_dir / "models")

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

    low_leakage = FEATURE_PROFILES["low_leakage"]
    preprocessor = fit_feature_preprocessor(
        train_df,
        config.label_column,
        columns_to_drop=low_leakage["columns_to_drop"],
        allowed_categorical_columns=low_leakage["categorical_columns"],
    )
    X_train = transform_features(train_df, preprocessor)
    X_validation = transform_features(validation_df, preprocessor)
    X_calibration = transform_features(calibration_df, preprocessor)
    X_blind = transform_features(blind_df, preprocessor)
    y_train = encode_known_labels(train_df[config.label_column], config.known_classes)
    y_validation = encode_known_labels(validation_df[config.label_column], config.known_classes)
    benign_index = list(config.known_classes).index(BENIGN_LABEL)

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
    predicted_benign = pred_indices == benign_index
    msp_mask = predict_unknown_mask(blind_open_scores, open_thresholds, strategy="msp")
    msp_score = 1.0 - blind_open_scores.max_softmax

    strategies: dict[str, dict] = {}
    strategies["closed_dnn"] = evaluate_decisions(
        true_labels,
        closed_world_decisions(pred_indices, config.known_classes),
        config.known_classes,
        config.zero_day_classes,
    )
    strategies["msp"] = _evaluate(
        true_labels,
        pred_indices,
        config.known_classes,
        config.zero_day_classes,
        msp_mask,
        msp_score,
    )

    benign_train = train_df.loc[train_df[config.label_column] == BENIGN_LABEL].reset_index(drop=True)
    benign_calibration = calibration_df.loc[
        calibration_df[config.label_column] == BENIGN_LABEL
    ].reset_index(drop=True)
    context_metadata = {}
    saved_profiles = {}
    for ablation_name, relation_specs in CONTEXT_ABLATIONS.items():
        profile = fit_context_profile(benign_train, relation_specs)
        calibration_matrix, relation_names = context_surprisal_matrix(profile, benign_calibration)
        scales = fit_surprisal_scales(calibration_matrix, config.context_scale_percentile)
        calibration_score = context_anomaly_score(calibration_matrix, scales)
        blind_matrix, _ = context_surprisal_matrix(profile, blind_df)
        blind_context_score = context_anomaly_score(blind_matrix, scales)
        thresholds_by_rate = {}
        for acceptance_rate in config.context_benign_acceptance_rates:
            if not 0.5 <= acceptance_rate < 1.0:
                raise ValueError("Context benign acceptance rates must be in [0.5, 1.0).")
            rate_name = f"{acceptance_rate:.3f}".rstrip("0").rstrip(".")
            context_threshold = float(np.percentile(calibration_score, acceptance_rate * 100.0))
            thresholds_by_rate[rate_name] = context_threshold
            context_mask = blind_context_score > context_threshold
            hybrid_mask = msp_mask | (predicted_benign & context_mask)
            strategies[f"context_only__{ablation_name}__a{rate_name}"] = _evaluate(
                true_labels,
                pred_indices,
                config.known_classes,
                config.zero_day_classes,
                context_mask,
                blind_context_score,
            )
            strategies[f"msp_context__{ablation_name}__a{rate_name}"] = _evaluate(
                true_labels,
                pred_indices,
                config.known_classes,
                config.zero_day_classes,
                hybrid_mask,
                np.maximum(
                    msp_score / max(1.0 - open_thresholds.msp_min_confidence, 1e-8),
                    blind_context_score / max(context_threshold, 1e-8),
                ),
            )
        context_metadata[ablation_name] = {
            "relations": relation_names,
            "thresholds_by_acceptance_rate": thresholds_by_rate,
            "scales": scales,
            "benign_train_rows": len(benign_train),
            "benign_calibration_rows": len(benign_calibration),
        }
        saved_profiles[ablation_name] = {
            "profile": profile,
            "scales": scales,
            "thresholds_by_acceptance_rate": thresholds_by_rate,
        }

    preferred_order = ["msp"]
    for name in CONTEXT_ABLATIONS:
        for acceptance_rate in config.context_benign_acceptance_rates:
            rate_name = f"{acceptance_rate:.3f}".rstrip("0").rstrip(".")
            preferred_order.extend(
                [
                    f"context_only__{name}__a{rate_name}",
                    f"msp_context__{name}__a{rate_name}",
                ]
            )
    summary_rows = [_summary_row(name, strategies[name]) for name in preferred_order]
    metrics = {
        "experiment": "EXP-03B",
        "config": asdict(config),
        "dataset": {
            "train_distribution": _distribution(train_df[config.label_column]),
            "validation_distribution": _distribution(validation_df[config.label_column]),
            "calibration_distribution": _distribution(calibration_df[config.label_column]),
            "known_holdout_distribution": _distribution(known_holdout[config.label_column]),
            "blind_distribution": _distribution(true_labels),
        },
        "feature_space": {"profile": "low_leakage", "input_dim": int(X_train.shape[1])},
        "model": {
            "parameter_count": int(model.count_params()),
            "training_seconds": training_seconds,
            "history": history,
        },
        "resampling": resampling,
        "open_thresholds": asdict(open_thresholds),
        "context": context_metadata,
        "strategies": strategies,
        "summary_rows": summary_rows,
        "runtime_seconds": time.perf_counter() - start,
    }
    save_json(metrics, output_dir / "metrics.json")
    _write_report(metrics, reports_dir / "context_report.md")
    if config.save_model:
        model.save(models_dir / "lightweight_dnn.h5", include_optimizer=False)
        joblib.dump(preprocessor, models_dir / "feature_preprocessor.joblib")
        joblib.dump(saved_profiles, models_dir / "context_profiles.joblib")
    return metrics
