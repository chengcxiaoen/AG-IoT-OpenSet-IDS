"""Strict open-set experiment pipeline for smart-agriculture network IDS."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .constants import (
    BENIGN_LABEL,
    DEFAULT_ZERO_DAY_CLASSES,
    FARM_FLOW_KNOWN_CLASSES,
    FEATURE_PROFILES,
)
from .data import (
    balanced_sample_indices,
    cap_rows_per_class,
    encode_known_labels,
    fit_feature_preprocessor,
    read_farm_flow_csv,
    split_known_and_zero_day,
    split_training_validation_calibration,
    transform_features,
)
from .metrics import add_score_auc, closed_world_decisions, evaluate_decisions, make_decisions
from .models import build_benign_autoencoder, build_lightweight_dnn, reconstruction_error
from .openset import (
    calibrate_benign_boundary,
    calibrate_thresholds,
    compute_open_set_scores,
    fit_prototypes,
    predict_msp_benign_ae_unknown_mask,
    predict_unknown_mask,
)
from .sampling import paper_resample
from .utils import configure_tensorflow, ensure_dir, save_json, set_global_seed


@dataclass
class AgriOpenSetConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/agri_openset"
    label_column: str = "traffic"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    seed: int = 42
    resampling_seed: int | None = None
    test_size: float = 0.20
    validation_size: float = 0.10
    calibration_size: float = 0.10
    max_rows_per_class: int | None = None
    resampling_protocol: str = "paper"
    balance_strategy: str = "median"
    max_train_per_class: int | None = None
    known_acceptance_rate: float = 0.95
    benign_acceptance_rate: float = 0.99
    epochs: int = 20
    ae_epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    save_model: bool = True
    feature_profile: str = "low_leakage"
    enable_ae: bool = True


def _distribution(labels: Sequence[str]) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _fit_classifier(
    config: AgriOpenSetConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, dict, dict, float]:
    resampling_seed = (
        config.seed if config.resampling_seed is None else config.resampling_seed
    )
    if config.resampling_protocol == "paper":
        resampled = paper_resample(
            X_train,
            y_train,
            target_strategy=config.balance_strategy,
            max_per_class=config.max_train_per_class,
            seed=resampling_seed,
        )
        X_fit, y_fit = resampled.X, resampled.y
        resampling_metrics = {
            "protocol": "paper",
            "target_size": resampled.target_size,
            "before": resampled.before,
            "after_under_sampling": resampled.after_under_sampling,
            "after_smote": resampled.after_smote,
            "after_tomek": resampled.after_tomek,
            "fit_rows": int(len(y_fit)),
        }
    elif config.resampling_protocol == "simple":
        indices = balanced_sample_indices(
            y_train,
            strategy=config.balance_strategy,
            seed=resampling_seed,
            max_per_class=config.max_train_per_class,
        )
        X_fit, y_fit = X_train[indices], y_train[indices]
        resampling_metrics = {
            "protocol": "simple",
            "fit_rows": int(len(y_fit)),
        }
    else:
        raise ValueError(f"Unknown resampling protocol: {config.resampling_protocol}")

    resampling_metrics["seed"] = int(resampling_seed)

    model, embedding_model, logits_model = build_lightweight_dnn(
        input_dim=X_fit.shape[1],
        num_classes=len(config.known_classes),
        learning_rate=config.learning_rate,
        dropout=config.dropout,
    )
    start = time.perf_counter()
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_validation, y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        ],
    )
    training_seconds = time.perf_counter() - start
    return (
        model,
        embedding_model,
        logits_model,
        history.history,
        resampling_metrics,
        training_seconds,
    )


def _fit_benign_boundary(
    config: AgriOpenSetConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    benign_index: int,
) -> tuple[tf.keras.Model, object, dict]:
    benign_train = X_train[y_train == benign_index]
    benign_validation = X_validation[y_validation == benign_index]
    benign_calibration = X_calibration[y_calibration == benign_index]

    model = build_benign_autoencoder(input_dim=X_train.shape[1])
    history = model.fit(
        benign_train,
        benign_train,
        validation_data=(benign_validation, benign_validation),
        epochs=config.ae_epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        ],
    )
    calibration_errors = reconstruction_error(model, benign_calibration, config.batch_size)
    boundary = calibrate_benign_boundary(
        calibration_errors,
        benign_acceptance_rate=config.benign_acceptance_rate,
    )
    return model, boundary, history.history


def _evaluate_open_set_strategies(
    true_labels: list[str],
    pred_indices: np.ndarray,
    class_order: Sequence[str],
    zero_day_classes: Sequence[str],
    scores: object,
    thresholds: object,
    reconstruction_errors: np.ndarray | None,
    benign_boundary: object | None,
    benign_index: int,
) -> dict[str, dict]:
    strategies: dict[str, dict] = {}

    score_by_strategy = {
        "msp": 1.0 - scores.max_softmax,
        "energy": scores.energy,
        "prototype": scores.nearest_prototype_distance,
        "energy_proto": np.maximum(
            (scores.energy - thresholds.energy_mean) / thresholds.energy_std,
            (
                scores.nearest_prototype_distance - thresholds.prototype_mean
            )
            / thresholds.prototype_std,
        ),
    }
    for strategy, unknown_scores in score_by_strategy.items():
        unknown_mask = predict_unknown_mask(scores, thresholds, strategy=strategy)
        predictions = make_decisions(pred_indices, unknown_mask, class_order)
        strategies[strategy] = add_score_auc(
            evaluate_decisions(true_labels, predictions, class_order, zero_day_classes),
            true_labels,
            zero_day_classes,
            unknown_scores,
        )

    if reconstruction_errors is None or benign_boundary is None:
        return strategies

    ae_unknown_mask = reconstruction_errors > benign_boundary.reconstruction_error_max
    ae_predictions = make_decisions(pred_indices, ae_unknown_mask, class_order)
    strategies["benign_ae"] = add_score_auc(
        evaluate_decisions(true_labels, ae_predictions, class_order, zero_day_classes),
        true_labels,
        zero_day_classes,
        reconstruction_errors,
    )

    hybrid_unknown_mask = predict_msp_benign_ae_unknown_mask(
        scores=scores,
        thresholds=thresholds,
        pred_indices=pred_indices,
        benign_class_index=benign_index,
        reconstruction_errors=reconstruction_errors,
        benign_boundary=benign_boundary,
    )
    hybrid_predictions = make_decisions(pred_indices, hybrid_unknown_mask, class_order)
    msp_scores = (1.0 - scores.max_softmax) / max(
        1.0 - thresholds.msp_min_confidence,
        1e-8,
    )
    benign_scores = np.zeros_like(reconstruction_errors)
    predicted_benign = pred_indices == benign_index
    benign_scores[predicted_benign] = (
        reconstruction_errors[predicted_benign]
        / max(benign_boundary.reconstruction_error_max, 1e-8)
    )
    hybrid_scores = np.maximum(msp_scores, benign_scores)
    strategies["msp_benign_ae"] = add_score_auc(
        evaluate_decisions(true_labels, hybrid_predictions, class_order, zero_day_classes),
        true_labels,
        zero_day_classes,
        hybrid_scores,
    )
    return strategies


def _plot_strategy_bars(strategy_metrics: dict[str, dict], output_path: Path) -> None:
    strategies = list(strategy_metrics)
    utdr = [strategy_metrics[name]["unknown_threat_detection_rate"] for name in strategies]
    far = [strategy_metrics[name]["benign_false_alarm_rate"] for name in strategies]
    x = np.arange(len(strategies))
    width = 0.36
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, utdr, width=width, label="Unknown detection rate")
    plt.bar(x + width / 2, far, width=width, label="Benign false alarm rate")
    plt.xticks(x, strategies, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Rate")
    plt.title("Open-set strategies on the Farm-Flow blind test")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _write_summary(metrics: dict, output_path: Path) -> None:
    msp = metrics["strategies"]["msp"]
    closed = metrics["closed_world_dnn"]
    if "msp_benign_ae" not in metrics["strategies"]:
        lines = [
            "# Agri-OpenSet-IDS Experiment Summary",
            "",
            "## Protocol",
            "",
            "- Validation, open-set calibration, and blind-test subsets are disjoint.",
            f"- Feature profile: {metrics['feature_space']['profile']}",
            f"- Zero-day classes: {', '.join(metrics['config']['zero_day_classes'])}",
            f"- Resampling: {metrics['config']['resampling_protocol']} / {metrics['config']['balance_strategy']}",
            "- Benign-AE is disabled for this experiment.",
            "",
            "## Results",
            "",
            f"- Closed DNN zero-day miss-as-benign: {closed['zero_day_miss_as_benign_rate']:.4f}",
            f"- MSP UTDR / FAR: {msp['unknown_threat_detection_rate']:.4f} / {msp['benign_false_alarm_rate']:.4f}",
            f"- MSP open-set macro F1: {msp['open_set_macro_f1']:.4f}",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    main = metrics["strategies"]["msp_benign_ae"]
    lines = [
        "# Revised Agri-OpenSet-IDS Experiment Summary",
        "",
        "## Protocol",
        "",
        "- Paper-aligned DNN training preprocessing is isolated to the training subset.",
        "- Validation, open-set calibration, and blind-test subsets are disjoint.",
        f"- Zero-day classes: {', '.join(metrics['config']['zero_day_classes'])}",
        f"- Resampling: {metrics['config']['resampling_protocol']} / {metrics['config']['balance_strategy']}",
        "",
        "## Results",
        "",
        f"- Closed DNN zero-day miss-as-benign: {closed['zero_day_miss_as_benign_rate']:.4f}",
        f"- MSP UTDR / FAR: {msp['unknown_threat_detection_rate']:.4f} / {msp['benign_false_alarm_rate']:.4f}",
        f"- MSP + Benign-AE UTDR / FAR: {main['unknown_threat_detection_rate']:.4f} / {main['benign_false_alarm_rate']:.4f}",
        f"- MSP + Benign-AE open-set macro F1: {main['open_set_macro_f1']:.4f}",
        "",
        "## Revised Method",
        "",
        (
            "MSP rejects generic low-confidence unknowns. The benign-only autoencoder is "
            "consulted only when the DNN predicts Benign, targeting near-normal attacks "
            "such as Arp Spoofing while limiting unnecessary rejection of known attacks."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_agri_openset_experiment(config: AgriOpenSetConfig) -> dict:
    set_global_seed(config.seed)
    configure_tensorflow()
    output_dir = ensure_dir(config.output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    models_dir = ensure_dir(output_dir / "models")
    reports_dir = ensure_dir(output_dir / "reports")
    start = time.perf_counter()

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

    if config.feature_profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile '{config.feature_profile}'. "
            f"Choose from: {', '.join(FEATURE_PROFILES)}"
        )
    feature_profile = FEATURE_PROFILES[config.feature_profile]
    preprocessor = fit_feature_preprocessor(
        train_df,
        config.label_column,
        columns_to_drop=feature_profile["columns_to_drop"],
        allowed_categorical_columns=feature_profile["categorical_columns"],
    )
    X_train = transform_features(train_df, preprocessor)
    X_validation = transform_features(validation_df, preprocessor)
    X_calibration = transform_features(calibration_df, preprocessor)
    X_blind = transform_features(blind_df, preprocessor)
    y_train = encode_known_labels(train_df[config.label_column], config.known_classes)
    y_validation = encode_known_labels(validation_df[config.label_column], config.known_classes)
    y_calibration = encode_known_labels(calibration_df[config.label_column], config.known_classes)

    (
        model,
        embedding_model,
        logits_model,
        dnn_history,
        resampling_metrics,
        training_seconds,
    ) = _fit_classifier(config, X_train, y_train, X_validation, y_validation)

    benign_index = list(config.known_classes).index(BENIGN_LABEL)
    benign_autoencoder = None
    benign_boundary = None
    ae_history = None
    if config.enable_ae:
        benign_autoencoder, benign_boundary, ae_history = _fit_benign_boundary(
            config,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_calibration,
            y_calibration,
            benign_index,
        )

    train_embeddings = embedding_model.predict(X_train, batch_size=config.batch_size, verbose=0)
    prototypes = fit_prototypes(train_embeddings, y_train, len(config.known_classes))
    cal_probs = model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_logits = logits_model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_embeddings = embedding_model.predict(X_calibration, batch_size=config.batch_size, verbose=0)
    cal_scores = compute_open_set_scores(cal_probs, cal_logits, cal_embeddings, prototypes)
    thresholds = calibrate_thresholds(cal_scores, config.known_acceptance_rate)

    infer_start = time.perf_counter()
    blind_probs = model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_logits = logits_model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    blind_embeddings = embedding_model.predict(X_blind, batch_size=config.batch_size, verbose=0)
    reconstruction_errors = None
    if benign_autoencoder is not None:
        reconstruction_errors = reconstruction_error(benign_autoencoder, X_blind, config.batch_size)
    inference_seconds = time.perf_counter() - infer_start
    blind_scores = compute_open_set_scores(blind_probs, blind_logits, blind_embeddings, prototypes)
    pred_indices = np.argmax(blind_probs, axis=1)
    true_labels = blind_df[config.label_column].tolist()

    closed_metrics = evaluate_decisions(
        true_labels,
        closed_world_decisions(pred_indices, config.known_classes),
        config.known_classes,
        config.zero_day_classes,
    )
    strategy_metrics = _evaluate_open_set_strategies(
        true_labels,
        pred_indices,
        config.known_classes,
        config.zero_day_classes,
        blind_scores,
        thresholds,
        reconstruction_errors,
        benign_boundary,
        benign_index,
    )

    metrics = {
        "config": asdict(config),
        "dataset": {
            "used_distribution": _distribution(df[config.label_column]),
            "train_distribution": _distribution(train_df[config.label_column]),
            "validation_distribution": _distribution(validation_df[config.label_column]),
            "calibration_distribution": _distribution(calibration_df[config.label_column]),
            "known_holdout_distribution": _distribution(known_holdout[config.label_column]),
            "blind_distribution": _distribution(true_labels),
        },
        "feature_space": {
            "profile": config.feature_profile,
            "description": feature_profile["description"],
            "leakage_risk": feature_profile["leakage_risk"],
            "input_dim": int(X_train.shape[1]),
            "numeric_columns": preprocessor.numeric_columns,
            "categorical_columns": preprocessor.categorical_columns,
            "dropped_columns": preprocessor.dropped_columns,
            "feature_names": preprocessor.feature_names,
        },
        "resampling": resampling_metrics,
        "model": {
            "name": "paper_architecture_dnn",
            "parameter_count": int(model.count_params()),
            "training_seconds": float(training_seconds),
            "blind_inference_seconds": float(inference_seconds),
            "seconds_per_sample": float(inference_seconds / max(len(X_blind), 1)),
            "history": {key: [float(v) for v in values] for key, values in dnn_history.items()},
        },
        "benign_autoencoder": (
            {
                "parameter_count": int(benign_autoencoder.count_params()),
                "history": {key: [float(v) for v in values] for key, values in ae_history.items()},
                "boundary": asdict(benign_boundary),
            }
            if benign_autoencoder is not None
            else None
        ),
        "thresholds": asdict(thresholds),
        "closed_world_dnn": closed_metrics,
        "strategies": strategy_metrics,
        "main_strategy": "msp_benign_ae" if config.enable_ae else "msp",
        "runtime_seconds": float(time.perf_counter() - start),
    }
    save_json(metrics, output_dir / "metrics.json")
    _plot_strategy_bars(strategy_metrics, plots_dir / "open_set_strategy_rates.png")
    _write_summary(metrics, reports_dir / "summary.md")

    if config.save_model:
        model.save(models_dir / "lightweight_dnn.h5", include_optimizer=False)
        if benign_autoencoder is not None:
            benign_autoencoder.save(models_dir / "benign_autoencoder.h5", include_optimizer=False)
        joblib.dump(preprocessor, models_dir / "feature_preprocessor.joblib")
        joblib.dump(
            {
                "thresholds": thresholds,
                "prototypes": prototypes,
                "benign_boundary": benign_boundary,
            },
            models_dir / "open_set_calibrator.joblib",
        )
    return metrics
