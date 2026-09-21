"""Class-conditional open-set rejection for high-confidence unknown attacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassFeatureDistributions:
    """Diagonal-Gaussian feature distributions fitted for each known class."""

    means: np.ndarray
    variances: np.ndarray
    class_counts: dict[int, int]
    variance_shrinkage: float


@dataclass
class ClassConditionalThresholds:
    """Per-predicted-class thresholds fitted only on known calibration data."""

    confidence_anomaly_max: np.ndarray
    distance_max: np.ndarray
    dual_score_max: np.ndarray
    calibration_counts: dict[int, int]
    calibration_sources: dict[int, str]
    known_acceptance_rate: float
    confidence_weight: float


@dataclass
class ClassConditionalScores:
    """Open-set scores associated with the DNN-predicted known class."""

    confidence_anomaly: np.ndarray
    feature_distance: np.ndarray
    normalized_confidence: np.ndarray
    normalized_distance: np.ndarray
    dual_score: np.ndarray


@dataclass
class ClassConfidenceThresholds:
    """Per-class confidence thresholds for asymmetric MSP rejection."""

    confidence_anomaly_max: np.ndarray
    calibration_counts: dict[int, int]
    calibration_sources: dict[int, str]
    known_acceptance_rate: float


def copy_confidence_thresholds_with_floors(
    thresholds: ClassConfidenceThresholds,
    min_confidence_by_class: dict[int, float],
) -> ClassConfidenceThresholds:
    """Return stricter per-class MSP thresholds using minimum confidence floors.

    The stored value is the maximum allowed confidence anomaly: 1 - confidence.
    A larger minimum confidence therefore means a smaller anomaly threshold.
    """

    adjusted = np.asarray(thresholds.confidence_anomaly_max, dtype=np.float32).copy()
    for class_id, min_confidence in min_confidence_by_class.items():
        if not 0.0 < min_confidence < 1.0:
            raise ValueError("Minimum confidence floors should be in (0, 1).")
        adjusted[int(class_id)] = min(float(adjusted[int(class_id)]), 1.0 - min_confidence)

    sources = dict(thresholds.calibration_sources)
    for class_id, min_confidence in min_confidence_by_class.items():
        original = sources.get(int(class_id), "unknown")
        sources[int(class_id)] = f"{original}+floor_{min_confidence:.3f}"

    return ClassConfidenceThresholds(
        confidence_anomaly_max=adjusted,
        calibration_counts=dict(thresholds.calibration_counts),
        calibration_sources=sources,
        known_acceptance_rate=thresholds.known_acceptance_rate,
    )


def _validate_rate(rate: float) -> None:
    if not 0.5 <= rate < 1.0:
        raise ValueError("known_acceptance_rate should be in [0.5, 1.0).")


def fit_class_feature_distributions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    variance_shrinkage: float = 0.10,
    variance_floor: float = 1e-4,
) -> ClassFeatureDistributions:
    """Fit regularized class means and diagonal variances in embedding space."""

    if not 0.0 <= variance_shrinkage <= 1.0:
        raise ValueError("variance_shrinkage should be in [0.0, 1.0].")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must contain the same number of rows.")

    global_variance = np.var(embeddings, axis=0)
    global_variance = np.maximum(global_variance, variance_floor)
    means = []
    variances = []
    class_counts: dict[int, int] = {}

    for class_id in range(num_classes):
        class_embeddings = embeddings[labels == class_id]
        if len(class_embeddings) == 0:
            raise ValueError(f"No training embeddings available for class {class_id}.")
        class_variance = np.var(class_embeddings, axis=0)
        regularized_variance = (
            (1.0 - variance_shrinkage) * class_variance
            + variance_shrinkage * global_variance
        )
        means.append(np.mean(class_embeddings, axis=0))
        variances.append(np.maximum(regularized_variance, variance_floor))
        class_counts[class_id] = int(len(class_embeddings))

    return ClassFeatureDistributions(
        means=np.asarray(means, dtype=np.float32),
        variances=np.asarray(variances, dtype=np.float32),
        class_counts=class_counts,
        variance_shrinkage=float(variance_shrinkage),
    )


def distance_to_each_class(
    embeddings: np.ndarray,
    distributions: ClassFeatureDistributions,
) -> np.ndarray:
    """Return dimension-normalized diagonal Mahalanobis distances."""

    differences = embeddings[:, None, :] - distributions.means[None, :, :]
    squared_standardized = differences**2 / distributions.variances[None, :, :]
    return np.sqrt(np.mean(squared_standardized, axis=2)).astype(np.float32)


def _select_calibration_rows(
    true_labels: np.ndarray,
    pred_indices: np.ndarray,
    class_id: int,
    minimum_correct_samples: int,
) -> tuple[np.ndarray, str]:
    correct = (true_labels == class_id) & (pred_indices == class_id)
    if int(np.sum(correct)) >= minimum_correct_samples:
        return correct, "correctly_classified"

    true_class = true_labels == class_id
    if int(np.sum(true_class)) > 0:
        return true_class, "true_class_fallback"

    predicted_class = pred_indices == class_id
    if int(np.sum(predicted_class)) > 0:
        return predicted_class, "predicted_class_fallback"

    return np.ones(len(true_labels), dtype=bool), "global_fallback"


def calibrate_class_conditional_thresholds(
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    distributions: ClassFeatureDistributions,
    known_acceptance_rate: float,
    confidence_weight: float = 0.50,
    minimum_correct_samples: int = 30,
) -> ClassConditionalThresholds:
    """Fit per-class rejection thresholds from disjoint known calibration data."""

    _validate_rate(known_acceptance_rate)
    if not 0.0 <= confidence_weight <= 1.0:
        raise ValueError("confidence_weight should be in [0.0, 1.0].")

    num_classes = probabilities.shape[1]
    pred_indices = np.argmax(probabilities, axis=1)
    all_distances = distance_to_each_class(embeddings, distributions)
    percentile = known_acceptance_rate * 100.0

    confidence_thresholds = np.zeros(num_classes, dtype=np.float32)
    distance_thresholds = np.zeros(num_classes, dtype=np.float32)
    calibration_counts: dict[int, int] = {}
    calibration_sources: dict[int, str] = {}
    selected_masks: dict[int, np.ndarray] = {}

    for class_id in range(num_classes):
        mask, source = _select_calibration_rows(
            true_labels,
            pred_indices,
            class_id,
            minimum_correct_samples,
        )
        selected_masks[class_id] = mask
        calibration_counts[class_id] = int(np.sum(mask))
        calibration_sources[class_id] = source
        confidence_anomaly = 1.0 - probabilities[mask, class_id]
        class_distance = all_distances[mask, class_id]
        confidence_thresholds[class_id] = np.percentile(confidence_anomaly, percentile)
        distance_thresholds[class_id] = np.percentile(class_distance, percentile)

    dual_thresholds = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        mask = selected_masks[class_id]
        confidence_score = (1.0 - probabilities[mask, class_id]) / max(
            float(confidence_thresholds[class_id]),
            1e-8,
        )
        distance_score = all_distances[mask, class_id] / max(
            float(distance_thresholds[class_id]),
            1e-8,
        )
        dual_score = (
            confidence_weight * confidence_score
            + (1.0 - confidence_weight) * distance_score
        )
        dual_thresholds[class_id] = np.percentile(dual_score, percentile)

    return ClassConditionalThresholds(
        confidence_anomaly_max=confidence_thresholds,
        distance_max=distance_thresholds,
        dual_score_max=dual_thresholds,
        calibration_counts=calibration_counts,
        calibration_sources=calibration_sources,
        known_acceptance_rate=float(known_acceptance_rate),
        confidence_weight=float(confidence_weight),
    )


def calibrate_class_confidence_thresholds(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    known_acceptance_rate: float,
    minimum_correct_samples: int = 30,
) -> ClassConfidenceThresholds:
    """Fit confidence-only thresholds without introducing distance evidence."""

    _validate_rate(known_acceptance_rate)
    num_classes = probabilities.shape[1]
    pred_indices = np.argmax(probabilities, axis=1)
    percentile = known_acceptance_rate * 100.0
    thresholds = np.zeros(num_classes, dtype=np.float32)
    calibration_counts: dict[int, int] = {}
    calibration_sources: dict[int, str] = {}

    for class_id in range(num_classes):
        mask, source = _select_calibration_rows(
            true_labels,
            pred_indices,
            class_id,
            minimum_correct_samples,
        )
        thresholds[class_id] = np.percentile(
            1.0 - probabilities[mask, class_id],
            percentile,
        )
        calibration_counts[class_id] = int(np.sum(mask))
        calibration_sources[class_id] = source

    return ClassConfidenceThresholds(
        confidence_anomaly_max=thresholds,
        calibration_counts=calibration_counts,
        calibration_sources=calibration_sources,
        known_acceptance_rate=float(known_acceptance_rate),
    )


def calibrate_bootstrap_class_confidence_thresholds(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    known_acceptance_rate: float,
    minimum_correct_samples: int = 30,
    bootstrap_iterations: int = 200,
    aggregation_quantile: float = 0.25,
    seed: int = 42,
) -> ClassConfidenceThresholds:
    """Fit class thresholds by bootstrapping known-only calibration rows.

    A lower aggregation quantile gives a stricter and more conservative threshold.
    This is useful when the unknown confidence distribution lies close to a class
    threshold, as observed for Port Scanning being absorbed by TCP Flood.
    """

    _validate_rate(known_acceptance_rate)
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    if not 0.0 <= aggregation_quantile <= 1.0:
        raise ValueError("aggregation_quantile should be in [0, 1].")

    num_classes = probabilities.shape[1]
    pred_indices = np.argmax(probabilities, axis=1)
    percentile = known_acceptance_rate * 100.0
    rng = np.random.default_rng(seed)
    thresholds = np.zeros(num_classes, dtype=np.float32)
    calibration_counts: dict[int, int] = {}
    calibration_sources: dict[int, str] = {}

    for class_id in range(num_classes):
        mask, source = _select_calibration_rows(
            true_labels,
            pred_indices,
            class_id,
            minimum_correct_samples,
        )
        anomalies = 1.0 - probabilities[mask, class_id]
        calibration_counts[class_id] = int(len(anomalies))
        calibration_sources[class_id] = (
            f"{source}+bootstrap_q{aggregation_quantile:.2f}"
        )
        if len(anomalies) <= 1:
            thresholds[class_id] = float(np.percentile(anomalies, percentile))
            continue

        bootstrapped = np.empty(bootstrap_iterations, dtype=np.float32)
        for i in range(bootstrap_iterations):
            sample = rng.choice(anomalies, size=len(anomalies), replace=True)
            bootstrapped[i] = np.percentile(sample, percentile)
        thresholds[class_id] = float(np.quantile(bootstrapped, aggregation_quantile))

    return ClassConfidenceThresholds(
        confidence_anomaly_max=thresholds,
        calibration_counts=calibration_counts,
        calibration_sources=calibration_sources,
        known_acceptance_rate=float(known_acceptance_rate),
    )


def score_class_confidence(
    probabilities: np.ndarray,
    pred_indices: np.ndarray,
    thresholds: ClassConfidenceThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """Return threshold-relative confidence scores and rejection decisions."""

    row_indices = np.arange(len(pred_indices))
    confidence_anomaly = 1.0 - probabilities[row_indices, pred_indices]
    normalized_score = confidence_anomaly / np.maximum(
        thresholds.confidence_anomaly_max[pred_indices],
        1e-8,
    )
    return normalized_score.astype(np.float32), normalized_score > 1.0


def compute_class_conditional_scores(
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    pred_indices: np.ndarray,
    distributions: ClassFeatureDistributions,
    thresholds: ClassConditionalThresholds,
) -> ClassConditionalScores:
    """Score each sample against the distribution of its predicted class."""

    row_indices = np.arange(len(pred_indices))
    confidence_anomaly = 1.0 - probabilities[row_indices, pred_indices]
    all_distances = distance_to_each_class(embeddings, distributions)
    feature_distance = all_distances[row_indices, pred_indices]
    normalized_confidence = confidence_anomaly / np.maximum(
        thresholds.confidence_anomaly_max[pred_indices],
        1e-8,
    )
    normalized_distance = feature_distance / np.maximum(
        thresholds.distance_max[pred_indices],
        1e-8,
    )
    dual_score = (
        thresholds.confidence_weight * normalized_confidence
        + (1.0 - thresholds.confidence_weight) * normalized_distance
    )
    return ClassConditionalScores(
        confidence_anomaly=confidence_anomaly.astype(np.float32),
        feature_distance=feature_distance.astype(np.float32),
        normalized_confidence=normalized_confidence.astype(np.float32),
        normalized_distance=normalized_distance.astype(np.float32),
        dual_score=dual_score.astype(np.float32),
    )


def class_conditional_unknown_masks(
    scores: ClassConditionalScores,
    pred_indices: np.ndarray,
    thresholds: ClassConditionalThresholds,
) -> dict[str, np.ndarray]:
    """Return confidence-only, distance-only, and dual-evidence decisions."""

    return {
        "class_msp": (
            scores.confidence_anomaly
            > thresholds.confidence_anomaly_max[pred_indices]
        ),
        "class_distance": (
            scores.feature_distance
            > thresholds.distance_max[pred_indices]
        ),
        "class_dual": scores.dual_score > thresholds.dual_score_max[pred_indices],
    }


def normalized_dual_unknown_score(
    scores: ClassConditionalScores,
    pred_indices: np.ndarray,
    thresholds: ClassConditionalThresholds,
) -> np.ndarray:
    """Return a threshold-relative score; values above one are rejected."""

    return scores.dual_score / np.maximum(
        thresholds.dual_score_max[pred_indices],
        1e-8,
    )
