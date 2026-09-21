"""EXP-04B class-conditional dual-evidence open-set evaluation."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from .class_conditional import (
    calibrate_bootstrap_class_confidence_thresholds,
    calibrate_class_confidence_thresholds,
    calibrate_class_conditional_thresholds,
    class_conditional_unknown_masks,
    compute_class_conditional_scores,
    copy_confidence_thresholds_with_floors,
    fit_class_feature_distributions,
    normalized_dual_unknown_score,
    score_class_confidence,
)
from .constants import (
    BENIGN_LABEL,
    DEFAULT_ZERO_DAY_CLASSES,
    FARM_FLOW_KNOWN_CLASSES,
    FEATURE_PROFILES,
)
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
from .unknown_inspection import add_cluster_summaries, cluster_rejected_embeddings
from .utils import configure_tensorflow, ensure_dir, save_json, set_global_seed


@dataclass
class ClassConditionalExperimentConfig:
    experiment_id: str = "EXP-04B"
    report_filename: str = "class_conditional_report.md"
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/class_conditional_seed42"
    label_column: str = "traffic"
    known_classes: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    zero_day_classes: tuple[str, ...] = tuple(DEFAULT_ZERO_DAY_CLASSES)
    seed: int = 42
    split_seed: int | None = None
    model_seed: int | None = None
    resampling_seed: int | None = None
    test_size: float = 0.20
    validation_size: float = 0.10
    calibration_size: float = 0.10
    max_rows_per_class: int | None = None
    balance_strategy: str = "median"
    max_train_per_class: int | None = 50000
    known_acceptance_rate: float = 0.95
    attack_acceptance_rates: tuple[float, ...] = ()
    context_benign_acceptance_rate: float = 0.999
    context_scale_percentile: float = 99.0
    confidence_weight: float = 0.50
    variance_shrinkage: float = 0.10
    minimum_class_calibration_samples: int = 30
    stabilized_threshold_pool: str = "calibration"
    stabilized_bootstrap_iterations: int = 0
    stabilized_bootstrap_quantile: float = 0.25
    attack_min_confidence_floors: tuple[float, ...] = ()
    attack_floor_profiles: tuple[str, ...] = ()
    targeted_tail_acceptance_rates: tuple[float, ...] = ()
    targeted_tail_profiles: tuple[str, ...] = ()
    epochs: int = 30
    batch_size: int = 1024
    inference_batch_size: int = 256
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
        "auroc": result.get("unknown_score_auc"),
        "arp_reject": arp.get("unknown_rate"),
        "arp_benign_miss": arp.get("benign_miss_rate"),
        "port_reject": port.get("unknown_rate"),
        "known_accuracy": result["known_closed_label_accuracy"],
    }


def _format(value: object) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _rejection_by_predicted_class(
    pred_indices: np.ndarray,
    unknown_mask: np.ndarray,
    class_order: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    output = {}
    for class_id, class_name in enumerate(class_order):
        class_mask = pred_indices == class_id
        support = int(np.sum(class_mask))
        rejected = int(np.sum(class_mask & unknown_mask))
        output[class_name] = {
            "support": support,
            "rejected": rejected,
            "rejection_rate": float(rejected / support) if support else 0.0,
        }
    return output


def _evidence_diagnostics(
    true_labels: list[str],
    zero_day_classes: tuple[str, ...],
    confidence_mask: np.ndarray,
    distance_mask: np.ndarray,
    dual_mask: np.ndarray,
) -> dict[str, dict[str, int]]:
    true_array = np.asarray(true_labels, dtype=object)
    groups = {"all_blind": np.ones(len(true_array), dtype=bool)}
    groups.update({label: true_array == label for label in zero_day_classes})
    output = {}
    for name, group_mask in groups.items():
        output[name] = {
            "support": int(np.sum(group_mask)),
            "confidence_only": int(np.sum(group_mask & confidence_mask & ~distance_mask)),
            "distance_only": int(np.sum(group_mask & ~confidence_mask & distance_mask)),
            "both_individual_rules": int(np.sum(group_mask & confidence_mask & distance_mask)),
            "dual_rejected": int(np.sum(group_mask & dual_mask)),
        }
    return output


def _named_class_thresholds(
    class_order: tuple[str, ...],
    thresholds: object,
) -> dict[str, dict[str, object]]:
    output = {}
    for class_id, class_name in enumerate(class_order):
        output[class_name] = {
            "confidence_anomaly_max": float(thresholds.confidence_anomaly_max[class_id]),
            "distance_max": float(thresholds.distance_max[class_id]),
            "dual_score_max": float(thresholds.dual_score_max[class_id]),
            "calibration_count": thresholds.calibration_counts[class_id],
            "calibration_source": thresholds.calibration_sources[class_id],
        }
    return output


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    levels = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    names = ("min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    return {
        name: float(value)
        for name, value in zip(names, np.quantile(values, levels))
    }


def _confidence_diagnostics_for_labels(
    true_labels: list[str],
    probabilities: np.ndarray,
    pred_indices: np.ndarray,
    class_order: tuple[str, ...],
    target_labels: tuple[str, ...],
) -> dict[str, dict]:
    """Describe confidence geometry without using target labels for decisions."""

    true_array = np.asarray(true_labels, dtype=object)
    max_softmax = np.max(probabilities, axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    top1_margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)),
        axis=1,
    )

    output = {}
    for target_label in target_labels:
        attack_mask = true_array == target_label
        by_predicted_class = {}
        for class_id, class_name in enumerate(class_order):
            group_mask = attack_mask & (pred_indices == class_id)
            support = int(np.sum(group_mask))
            if support == 0:
                continue
            by_predicted_class[class_name] = {
                "support": support,
                "rate_within_attack": float(support / max(int(np.sum(attack_mask)), 1)),
                "max_softmax_quantiles": _quantiles(max_softmax[group_mask]),
                "top1_margin_quantiles": _quantiles(top1_margin[group_mask]),
                "entropy_quantiles": _quantiles(entropy[group_mask]),
            }
        output[target_label] = {
            "support": int(np.sum(attack_mask)),
            "max_softmax_quantiles": _quantiles(max_softmax[attack_mask]),
            "top1_margin_quantiles": _quantiles(top1_margin[attack_mask]),
            "entropy_quantiles": _quantiles(entropy[attack_mask]),
            "by_predicted_class": by_predicted_class,
        }
    return output


def _zero_day_confidence_diagnostics(
    true_labels: list[str],
    probabilities: np.ndarray,
    pred_indices: np.ndarray,
    class_order: tuple[str, ...],
    zero_day_classes: tuple[str, ...],
) -> dict[str, dict]:
    """Describe zero-day confidence geometry without using labels for decisions."""

    return _confidence_diagnostics_for_labels(
        true_labels,
        probabilities,
        pred_indices,
        class_order,
        zero_day_classes,
    )


def _rate_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_floor_profiles(
    profile_specs: tuple[str, ...],
    known_classes: tuple[str, ...],
) -> list[tuple[str | None, list[int]]]:
    """Parse targeted floor profiles such as 'tcp=TCP Flood|HTTP Flood'."""

    if not profile_specs:
        return [
            (
                None,
                [
                    class_id
                    for class_id, class_name in enumerate(known_classes)
                    if class_name != BENIGN_LABEL
                ],
            )
        ]

    class_to_id = {class_name: class_id for class_id, class_name in enumerate(known_classes)}
    profiles: list[tuple[str | None, list[int]]] = []
    for raw_spec in profile_specs:
        if "=" not in raw_spec:
            raise ValueError(
                "attack_floor_profiles entries must look like "
                "'profile_name=Class A|Class B'."
            )
        name, class_text = raw_spec.split("=", 1)
        profile_name = name.strip()
        target_names = [item.strip() for item in class_text.split("|") if item.strip()]
        if not profile_name or not target_names:
            raise ValueError(f"Invalid attack floor profile: {raw_spec!r}")
        missing = [class_name for class_name in target_names if class_name not in class_to_id]
        if missing:
            raise ValueError(f"Unknown class names in floor profile {profile_name}: {missing}")
        profiles.append((profile_name, [class_to_id[class_name] for class_name in target_names]))
    return profiles

def _write_report(metrics: dict, output_path: Path) -> None:
    config = metrics["config"]
    is_asymmetric = metrics["experiment"] == "EXP-04C"
    title = (
        "# EXP-04C Asymmetric Class-Conditional Rejection Report"
        if is_asymmetric
        else "# EXP-04B Class-Conditional Dual-Evidence Rejection Report"
    )
    purpose = (
        "Test whether attack-class conditional MSP can improve high-confidence "
        "unknown rejection while the Benign route retains global MSP and "
        "Full-Role Context to control normal-traffic FAR."
        if is_asymmetric
        else "Test whether class-conditional confidence and embedding-distance evidence can reject high-confidence unknown attacks that global MSP accepts, while Full-Role Context preserves near-normal Arp Spoofing detection."
    )
    lines = [
        title,
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Strict Protocol",
        "",
        "- Zero-day attacks are used only in the final blind test.",
        "- The paper-style DNN is trained only on the six known classes.",
        "- Per-class feature distributions are fitted only on known-class training embeddings.",
        "- Per-class rejection thresholds are fitted only on disjoint known calibration data.",
        "- Full-Role Context is fitted only on Benign training traffic and calibrated only on Benign calibration traffic.",
        f"- Known acceptance rate: {config['known_acceptance_rate']:.4f}",
        f"- Confidence weight: {config['confidence_weight']:.4f}",
        f"- Variance shrinkage: {config['variance_shrinkage']:.4f}",
        "",
        "## Results",
        "",
        "| Method | UTDR | FAR | Macro-F1 | AUROC | Arp reject | Arp benign miss | Port reject | Known acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["summary_rows"]:
        lines.append(
            f"| {row['method']} | {_format(row['utdr'])} | {_format(row['far'])} | "
            f"{_format(row['macro_f1'])} | {_format(row['auroc'])} | "
            f"{_format(row['arp_reject'])} | {_format(row['arp_benign_miss'])} | "
            f"{_format(row['port_reject'])} | {_format(row['known_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Class Calibration",
            "",
            "| Predicted class | Confidence anomaly max | Distance max | Dual max | Rows | Source |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for class_name, values in metrics["class_thresholds_by_name"].items():
        lines.append(
            f"| {class_name} | {_format(values['confidence_anomaly_max'])} | "
            f"{_format(values['distance_max'])} | {_format(values['dual_score_max'])} | "
            f"{values['calibration_count']} | {values['calibration_source']} |"
        )
    if metrics.get("asymmetric_thresholds"):
        lines.extend(
            [
                "",
                "## Asymmetric Decision Rule",
                "",
                "- Predicted Benign: global MSP OR conditional Full-Role Context.",
                "- Predicted known attack: class-conditional MSP.",
                "- Class distance is excluded because EXP-04B showed that it weakens Port Scanning rejection.",
                "- This run is threshold sensitivity analysis only. No attack acceptance rate is selected from blind-test zero-day performance.",
            ]
        )
    if metrics.get("stabilized_thresholds"):
        lines.extend(
            [
                "",
                "## Calibration Stabilization",
                "",
                f"- Threshold pool: {metrics['config']['stabilized_threshold_pool']}",
                f"- Bootstrap iterations: {metrics['config']['stabilized_bootstrap_iterations']}",
                f"- Bootstrap aggregation quantile: {metrics['config']['stabilized_bootstrap_quantile']:.4f}",
                f"- Attack floor profiles: {metrics['config']['attack_floor_profiles'] or ['all_non_benign']}",
        "- Optional confidence floors or targeted tail thresholds are applied only to the configured predicted attack classes.",
        "- Zero-day attacks are never used to fit or select these thresholds.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The proposed route is useful only if it improves multi-seed Port Scanning rejection stability over global MSP, preserves the Full-Role Context gain on Arp Spoofing, keeps Benign FAR below 1%, and does not materially damage known-class accuracy.",
            "",
            "## Important Limitation",
            "",
            "This experiment validates rejection on Farm-Flow only. Cross-device and cross-dataset evaluation is still required before claiming general agricultural deployment robustness.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_class_conditional_experiment(config: ClassConditionalExperimentConfig) -> dict:
    if not 0.5 <= config.context_benign_acceptance_rate < 1.0:
        raise ValueError("context_benign_acceptance_rate should be in [0.5, 1.0).")
    if not 0.0 < config.context_scale_percentile <= 100.0:
        raise ValueError("context_scale_percentile should be in (0.0, 100.0].")
    if config.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")

    split_seed = config.seed if config.split_seed is None else config.split_seed
    model_seed = config.seed if config.model_seed is None else config.model_seed
    resampling_seed = (
        config.seed if config.resampling_seed is None else config.resampling_seed
    )
    set_global_seed(model_seed)
    configure_tensorflow()
    start = time.perf_counter()
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    models_dir = ensure_dir(output_dir / "models")

    allowed_labels = list(config.known_classes) + list(config.zero_day_classes)
    df = read_farm_flow_csv(config.farm_flow_path, config.label_column, allowed_labels)
    df = cap_rows_per_class(df, config.label_column, config.max_rows_per_class, split_seed)
    known_pool, known_holdout, blind_df = split_known_and_zero_day(
        df,
        config.known_classes,
        config.zero_day_classes,
        config.label_column,
        config.test_size,
        split_seed,
    )
    train_df, validation_df, calibration_df = split_training_validation_calibration(
        known_pool,
        config.label_column,
        config.validation_size,
        config.calibration_size,
        split_seed,
    )

    feature_profile = FEATURE_PROFILES["low_leakage"]
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
    # Rare Farm-Flow outliers can produce extreme standardized values outside
    # the training range. This label-free clip preserves every sample while
    # preventing overflow in validation/blind softmax calculations.
    for name, matrix in {
        "train": X_train,
        "validation": X_validation,
        "calibration": X_calibration,
        "blind": X_blind,
    }.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values found in {name} feature matrix.")
        np.clip(matrix, -20.0, 20.0, out=matrix)
    y_train = encode_known_labels(train_df[config.label_column], config.known_classes)
    y_validation = encode_known_labels(validation_df[config.label_column], config.known_classes)
    y_calibration = encode_known_labels(calibration_df[config.label_column], config.known_classes)
    benign_index = list(config.known_classes).index(BENIGN_LABEL)

    dnn_config = AgriOpenSetConfig(
        known_classes=config.known_classes,
        zero_day_classes=config.zero_day_classes,
        seed=model_seed,
        resampling_seed=resampling_seed,
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
    set_global_seed(model_seed)
    model, embedding_model, logits_model, history, resampling, training_seconds = _fit_classifier(
        dnn_config,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    inference_batch_size = min(config.inference_batch_size, config.batch_size)
    train_embeddings = embedding_model.predict(
        X_train, batch_size=inference_batch_size, verbose=0
    )
    calibration_probabilities = model.predict(
        X_calibration,
        batch_size=inference_batch_size,
        verbose=0,
    )
    calibration_logits = logits_model.predict(
        X_calibration,
        batch_size=inference_batch_size,
        verbose=0,
    )
    calibration_embeddings = embedding_model.predict(
        X_calibration,
        batch_size=inference_batch_size,
        verbose=0,
    )
    validation_probabilities = model.predict(
        X_validation,
        batch_size=inference_batch_size,
        verbose=0,
    )
    blind_probabilities = model.predict(X_blind, batch_size=inference_batch_size, verbose=0)
    blind_logits = logits_model.predict(X_blind, batch_size=inference_batch_size, verbose=0)
    blind_embeddings = embedding_model.predict(X_blind, batch_size=inference_batch_size, verbose=0)
    pred_indices = np.argmax(blind_probabilities, axis=1)
    true_labels = blind_df[config.label_column].tolist()

    prototypes = fit_prototypes(train_embeddings, y_train, len(config.known_classes))
    calibration_open_scores = compute_open_set_scores(
        calibration_probabilities,
        calibration_logits,
        calibration_embeddings,
        prototypes,
    )
    global_thresholds = calibrate_thresholds(
        calibration_open_scores,
        config.known_acceptance_rate,
    )
    blind_open_scores = compute_open_set_scores(
        blind_probabilities,
        blind_logits,
        blind_embeddings,
        prototypes,
    )
    global_msp_mask = predict_unknown_mask(blind_open_scores, global_thresholds, strategy="msp")
    global_msp_score = 1.0 - blind_open_scores.max_softmax

    distributions = fit_class_feature_distributions(
        train_embeddings,
        y_train,
        len(config.known_classes),
        variance_shrinkage=config.variance_shrinkage,
    )
    class_thresholds = calibrate_class_conditional_thresholds(
        calibration_probabilities,
        calibration_embeddings,
        y_calibration,
        distributions,
        known_acceptance_rate=config.known_acceptance_rate,
        confidence_weight=config.confidence_weight,
        minimum_correct_samples=config.minimum_class_calibration_samples,
    )
    class_scores = compute_class_conditional_scores(
        blind_probabilities,
        blind_embeddings,
        pred_indices,
        distributions,
        class_thresholds,
    )
    class_masks = class_conditional_unknown_masks(class_scores, pred_indices, class_thresholds)
    class_dual_score = normalized_dual_unknown_score(class_scores, pred_indices, class_thresholds)

    strategies: dict[str, dict] = {
        "closed_dnn": evaluate_decisions(
            true_labels,
            closed_world_decisions(pred_indices, config.known_classes),
            config.known_classes,
            config.zero_day_classes,
        ),
        "global_msp": _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            global_msp_mask,
            global_msp_score,
        ),
        "class_msp": _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            class_masks["class_msp"],
            class_scores.normalized_confidence,
        ),
        "class_distance": _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            class_masks["class_distance"],
            class_scores.normalized_distance,
        ),
        "class_dual": _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            class_masks["class_dual"],
            class_dual_score,
        ),
    }

    benign_train = train_df.loc[train_df[config.label_column] == BENIGN_LABEL].reset_index(drop=True)
    benign_calibration = calibration_df.loc[
        calibration_df[config.label_column] == BENIGN_LABEL
    ].reset_index(drop=True)
    full_role_profile = fit_context_profile(
        benign_train,
        CONTEXT_ABLATIONS["full_role_context"],
    )
    calibration_context_matrix, relation_names = context_surprisal_matrix(
        full_role_profile,
        benign_calibration,
    )
    context_scales = fit_surprisal_scales(
        calibration_context_matrix,
        config.context_scale_percentile,
    )
    calibration_context_score = context_anomaly_score(
        calibration_context_matrix,
        context_scales,
    )
    context_threshold = float(
        np.percentile(
            calibration_context_score,
            config.context_benign_acceptance_rate * 100.0,
        )
    )
    blind_context_matrix, _ = context_surprisal_matrix(full_role_profile, blind_df)
    blind_context_score = context_anomaly_score(blind_context_matrix, context_scales)
    predicted_benign = pred_indices == benign_index
    context_mask = blind_context_score > context_threshold
    final_mask = class_masks["class_dual"] | (predicted_benign & context_mask)
    final_score = np.maximum(
        class_dual_score,
        np.where(
            predicted_benign,
            blind_context_score / max(context_threshold, 1e-8),
            0.0,
        ),
    )
    strategies["class_dual_full_role"] = _evaluate(
        true_labels,
        pred_indices,
        config.known_classes,
        config.zero_day_classes,
        final_mask,
        final_score,
    )

    method_order = [
        "global_msp",
        "class_msp",
        "class_distance",
        "class_dual",
        "class_dual_full_role",
    ]
    asymmetric_thresholds = {}
    stabilized_thresholds = {}
    global_msp_relative_score = global_msp_score / max(
        1.0 - global_thresholds.msp_min_confidence,
        1e-8,
    )
    benign_route_mask = predicted_benign & (global_msp_mask | context_mask)
    benign_route_score = np.maximum(
        global_msp_relative_score,
        blind_context_score / max(context_threshold, 1e-8),
    )
    global_msp_full_role_mask = global_msp_mask | (predicted_benign & context_mask)
    global_msp_full_role_score = np.maximum(
        global_msp_relative_score,
        np.where(predicted_benign, benign_route_score, 0.0),
    )
    strategies["global_msp_full_role"] = _evaluate(
        true_labels,
        pred_indices,
        config.known_classes,
        config.zero_day_classes,
        global_msp_full_role_mask,
        global_msp_full_role_score,
    )
    method_order.append("global_msp_full_role")

    # EXP-05A: preserve the low-FAR Benign route and apply class-conditional
    # evidence only when the closed DNN already predicts a known attack. This
    # is a fixed architectural ablation, not a threshold selected on zero-day
    # labels.
    asymmetric_dual_attack_mask = (~predicted_benign) & class_masks["class_dual"]
    asymmetric_dual_full_role_mask = benign_route_mask | asymmetric_dual_attack_mask
    asymmetric_dual_full_role_score = np.where(
        predicted_benign,
        benign_route_score,
        class_dual_score,
    )
    strategies["asymmetric_class_dual_full_role"] = _evaluate(
        true_labels,
        pred_indices,
        config.known_classes,
        config.zero_day_classes,
        asymmetric_dual_full_role_mask,
        asymmetric_dual_full_role_score,
    )
    method_order.append("asymmetric_class_dual_full_role")

    # ORI-style inspection is deliberately downstream of rejection.  It groups
    # only rejected flows and cannot alter online decisions or threshold tuning.
    unknown_cluster_inspection = add_cluster_summaries(
        cluster_rejected_embeddings(blind_embeddings, final_mask),
        true_labels,
        pred_indices,
        config.known_classes,
    )

    asymmetric_masks = {}
    for acceptance_rate in config.attack_acceptance_rates:
        rate_name = f"{acceptance_rate:.4f}".rstrip("0").rstrip(".")
        confidence_thresholds = calibrate_class_confidence_thresholds(
            calibration_probabilities,
            y_calibration,
            known_acceptance_rate=acceptance_rate,
            minimum_correct_samples=config.minimum_class_calibration_samples,
        )
        class_confidence_score, class_confidence_mask = score_class_confidence(
            blind_probabilities,
            pred_indices,
            confidence_thresholds,
        )
        attack_route_mask = (~predicted_benign) & class_confidence_mask
        asymmetric_mask = benign_route_mask | attack_route_mask
        asymmetric_score = np.where(
            predicted_benign,
            benign_route_score,
            class_confidence_score,
        )
        strategy_name = f"asymmetric_class_msp_full_role__a{rate_name}"
        asymmetric_masks[strategy_name] = asymmetric_mask
        strategies[strategy_name] = _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            asymmetric_mask,
            asymmetric_score,
        )
        method_order.append(strategy_name)
        asymmetric_thresholds[rate_name] = {
            "known_acceptance_rate": acceptance_rate,
            "confidence_anomaly_max": confidence_thresholds.confidence_anomaly_max,
            "calibration_counts": confidence_thresholds.calibration_counts,
            "calibration_sources": confidence_thresholds.calibration_sources,
        }

    if config.stabilized_bootstrap_iterations > 0:
        if not config.attack_acceptance_rates:
            raise ValueError(
                "stabilized_bootstrap_iterations requires one attack_acceptance_rate."
            )
        if config.stabilized_threshold_pool == "calibration":
            threshold_pool_probabilities = calibration_probabilities
            threshold_pool_labels = y_calibration
        elif config.stabilized_threshold_pool == "validation_calibration":
            threshold_pool_probabilities = np.vstack(
                [validation_probabilities, calibration_probabilities]
            )
            threshold_pool_labels = np.concatenate([y_validation, y_calibration])
        else:
            raise ValueError(
                "stabilized_threshold_pool must be 'calibration' or "
                "'validation_calibration'."
            )

        stable_rate = config.attack_acceptance_rates[0]
        stable_rate_name = _rate_token(stable_rate)
        stable_thresholds = calibrate_bootstrap_class_confidence_thresholds(
            threshold_pool_probabilities,
            threshold_pool_labels,
            known_acceptance_rate=stable_rate,
            minimum_correct_samples=config.minimum_class_calibration_samples,
            bootstrap_iterations=config.stabilized_bootstrap_iterations,
            aggregation_quantile=config.stabilized_bootstrap_quantile,
            seed=split_seed,
        )
        stable_score, stable_mask_all_classes = score_class_confidence(
            blind_probabilities,
            pred_indices,
            stable_thresholds,
        )
        stable_attack_mask = (~predicted_benign) & stable_mask_all_classes
        stable_asymmetric_mask = benign_route_mask | stable_attack_mask
        stable_asymmetric_score = np.where(
            predicted_benign,
            benign_route_score,
            stable_score,
        )
        stable_strategy_name = (
            f"stable_asymmetric_full_role__a{stable_rate_name}"
        )
        asymmetric_masks[stable_strategy_name] = stable_asymmetric_mask
        strategies[stable_strategy_name] = _evaluate(
            true_labels,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
            stable_asymmetric_mask,
            stable_asymmetric_score,
        )
        method_order.append(stable_strategy_name)
        stabilized_thresholds[stable_strategy_name] = {
            "known_acceptance_rate": stable_rate,
            "pool": config.stabilized_threshold_pool,
            "bootstrap_iterations": config.stabilized_bootstrap_iterations,
            "bootstrap_quantile": config.stabilized_bootstrap_quantile,
            "confidence_anomaly_max": stable_thresholds.confidence_anomaly_max,
            "calibration_counts": stable_thresholds.calibration_counts,
            "calibration_sources": stable_thresholds.calibration_sources,
        }

        floor_variants = {}
        floor_profiles = _parse_floor_profiles(
            config.attack_floor_profiles,
            config.known_classes,
        )
        for floor in config.attack_min_confidence_floors:
            for profile_name, target_class_ids in floor_profiles:
                min_confidence_by_class = {
                    class_id: float(floor) for class_id in target_class_ids
                }
                floored_thresholds = copy_confidence_thresholds_with_floors(
                    stable_thresholds,
                    min_confidence_by_class,
                )
                floored_score, floored_mask_all_classes = score_class_confidence(
                    blind_probabilities,
                    pred_indices,
                    floored_thresholds,
                )
                floored_attack_mask = (~predicted_benign) & floored_mask_all_classes
                floored_asymmetric_mask = benign_route_mask | floored_attack_mask
                floored_asymmetric_score = np.where(
                    predicted_benign,
                    benign_route_score,
                    floored_score,
                )
                floor_name = _rate_token(float(floor))
                if profile_name is None:
                    floored_strategy_name = (
                        f"stable_asymmetric_full_role__a{stable_rate_name}__floor{floor_name}"
                    )
                    target_profile_name = "all_non_benign"
                else:
                    floored_strategy_name = (
                        f"stable_asymmetric_full_role__a{stable_rate_name}"
                        f"__{profile_name}_floor{floor_name}"
                    )
                    target_profile_name = profile_name
                asymmetric_masks[floored_strategy_name] = floored_asymmetric_mask
                strategies[floored_strategy_name] = _evaluate(
                    true_labels,
                    pred_indices,
                    config.known_classes,
                    config.zero_day_classes,
                    floored_asymmetric_mask,
                    floored_asymmetric_score,
                )
                method_order.append(floored_strategy_name)
                floor_variants[floored_strategy_name] = {
                    "floor": float(floor),
                    "profile": target_profile_name,
                    "target_classes": [
                        config.known_classes[class_id] for class_id in target_class_ids
                    ],
                    "confidence_anomaly_max": floored_thresholds.confidence_anomaly_max,
                    "calibration_counts": floored_thresholds.calibration_counts,
                    "calibration_sources": floored_thresholds.calibration_sources,
                }
        stabilized_thresholds["floored_variants"] = floor_variants

        tail_variants = {}
        tail_profiles = _parse_floor_profiles(
            config.targeted_tail_profiles,
            config.known_classes,
        )
        for tail_rate in config.targeted_tail_acceptance_rates:
            if config.stabilized_bootstrap_iterations > 0:
                tail_thresholds_all = calibrate_bootstrap_class_confidence_thresholds(
                    threshold_pool_probabilities,
                    threshold_pool_labels,
                    known_acceptance_rate=tail_rate,
                    minimum_correct_samples=config.minimum_class_calibration_samples,
                    bootstrap_iterations=config.stabilized_bootstrap_iterations,
                    aggregation_quantile=config.stabilized_bootstrap_quantile,
                    seed=split_seed,
                )
            else:
                tail_thresholds_all = calibrate_class_confidence_thresholds(
                    threshold_pool_probabilities,
                    threshold_pool_labels,
                    known_acceptance_rate=tail_rate,
                    minimum_correct_samples=config.minimum_class_calibration_samples,
                )
            tail_rate_name = _rate_token(tail_rate)
            for profile_name, target_class_ids in tail_profiles:
                targeted_anomaly = np.asarray(
                    stable_thresholds.confidence_anomaly_max,
                    dtype=np.float32,
                ).copy()
                for class_id in target_class_ids:
                    targeted_anomaly[class_id] = tail_thresholds_all.confidence_anomaly_max[
                        class_id
                    ]
                targeted_thresholds = copy_confidence_thresholds_with_floors(
                    stable_thresholds,
                    {},
                )
                targeted_thresholds.confidence_anomaly_max = targeted_anomaly
                targeted_score, targeted_mask_all_classes = score_class_confidence(
                    blind_probabilities,
                    pred_indices,
                    targeted_thresholds,
                )
                targeted_attack_mask = (~predicted_benign) & targeted_mask_all_classes
                targeted_asymmetric_mask = benign_route_mask | targeted_attack_mask
                targeted_asymmetric_score = np.where(
                    predicted_benign,
                    benign_route_score,
                    targeted_score,
                )
                tail_strategy_name = (
                    f"tail_gate_full_role__a{stable_rate_name}"
                    f"__{profile_name}_accept{tail_rate_name}"
                )
                asymmetric_masks[tail_strategy_name] = targeted_asymmetric_mask
                strategies[tail_strategy_name] = _evaluate(
                    true_labels,
                    pred_indices,
                    config.known_classes,
                    config.zero_day_classes,
                    targeted_asymmetric_mask,
                    targeted_asymmetric_score,
                )
                method_order.append(tail_strategy_name)
                tail_variants[tail_strategy_name] = {
                    "target_acceptance_rate": float(tail_rate),
                    "profile": profile_name,
                    "target_classes": [
                        config.known_classes[class_id] for class_id in target_class_ids
                    ],
                    "confidence_anomaly_max": targeted_anomaly,
                    "tail_confidence_anomaly_max": tail_thresholds_all.confidence_anomaly_max,
                    "calibration_counts": tail_thresholds_all.calibration_counts,
                    "calibration_sources": tail_thresholds_all.calibration_sources,
                }
        stabilized_thresholds["targeted_tail_variants"] = tail_variants

    rejection_diagnostics = {
        name: _rejection_by_predicted_class(pred_indices, mask, config.known_classes)
        for name, mask in {
            "global_msp": global_msp_mask,
            "class_msp": class_masks["class_msp"],
            "class_distance": class_masks["class_distance"],
            "class_dual": class_masks["class_dual"],
            "class_dual_full_role": final_mask,
            "global_msp_full_role": global_msp_full_role_mask,
            "asymmetric_class_dual_full_role": asymmetric_dual_full_role_mask,
            **asymmetric_masks,
        }.items()
    }
    metrics = {
        "experiment": config.experiment_id,
        "main_strategy": (
            "threshold_sensitivity_no_blind_selection"
            if config.attack_acceptance_rates
            else "class_dual_full_role"
        ),
        "config": asdict(config),
        "resolved_seeds": {
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "resampling_seed": int(resampling_seed),
        },
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
            "standardized_feature_clip": 20.0,
        },
        "model": {
            "parameter_count": int(model.count_params()),
            "training_seconds": training_seconds,
            "history": history,
        },
        "resampling": resampling,
        "global_msp_thresholds": asdict(global_thresholds),
        "class_feature_distributions": {
            "class_counts": distributions.class_counts,
            "variance_shrinkage": distributions.variance_shrinkage,
        },
        "class_conditional_thresholds": asdict(class_thresholds),
        "class_thresholds_by_name": _named_class_thresholds(
            config.known_classes,
            class_thresholds,
        ),
        "context": {
            "profile": "full_role_context",
            "relations": relation_names,
            "threshold": context_threshold,
            "scales": context_scales,
            "benign_acceptance_rate": config.context_benign_acceptance_rate,
        },
        "unknown_cluster_inspection": unknown_cluster_inspection,
        "asymmetric_thresholds": asymmetric_thresholds,
        "stabilized_thresholds": stabilized_thresholds,
        "strategies": strategies,
        "rejection_by_predicted_class": rejection_diagnostics,
        "evidence_diagnostics": _evidence_diagnostics(
            true_labels,
            config.zero_day_classes,
            class_masks["class_msp"],
            class_masks["class_distance"],
            class_masks["class_dual"],
        ),
        "zero_day_confidence_diagnostics": _zero_day_confidence_diagnostics(
            true_labels,
            blind_probabilities,
            pred_indices,
            config.known_classes,
            config.zero_day_classes,
        ),
        "known_confidence_diagnostics": _confidence_diagnostics_for_labels(
            true_labels,
            blind_probabilities,
            pred_indices,
            config.known_classes,
            config.known_classes,
        ),
        "summary_rows": [_summary_row(name, strategies[name]) for name in method_order],
        "runtime_seconds": time.perf_counter() - start,
    }
    save_json(metrics, output_dir / "metrics.json")
    _write_report(metrics, reports_dir / config.report_filename)
    if config.save_model:
        model.save(models_dir / "lightweight_dnn.h5", include_optimizer=False)
        joblib.dump(preprocessor, models_dir / "feature_preprocessor.joblib")
        joblib.dump(distributions, models_dir / "class_feature_distributions.joblib")
        joblib.dump(class_thresholds, models_dir / "class_conditional_thresholds.joblib")
        joblib.dump(
            {
                "profile": full_role_profile,
                "scales": context_scales,
                "threshold": context_threshold,
            },
            models_dir / "full_role_context.joblib",
        )
    # The command-line runner isolates seeds in separate Python processes. This
    # explicit cleanup additionally protects future interactive multi-seed use.
    del blind_embeddings, blind_logits, blind_probabilities
    del calibration_embeddings, calibration_logits, calibration_probabilities
    del train_embeddings, X_blind, X_calibration, X_validation, X_train
    gc.collect()
    return metrics
