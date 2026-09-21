"""Open-set rejection methods for unknown attack detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OpenSetStrategy = Literal["msp", "energy", "prototype", "energy_proto"]


@dataclass
class PrototypeStore:
    prototypes: np.ndarray
    class_counts: dict[int, int]


@dataclass
class OpenSetThresholds:
    msp_min_confidence: float
    energy_max: float
    prototype_max_distance: float
    fusion_max: float
    energy_mean: float
    energy_std: float
    prototype_mean: float
    prototype_std: float
    known_acceptance_rate: float


@dataclass
class OpenSetScores:
    max_softmax: np.ndarray
    energy: np.ndarray
    nearest_prototype_distance: np.ndarray


@dataclass
class BenignBoundary:
    reconstruction_error_max: float
    benign_acceptance_rate: float


def compute_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_value = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_value + np.log(np.sum(np.exp(scaled - max_value), axis=1, keepdims=True))
    return (-temperature * log_sum_exp).reshape(-1)


def fit_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> PrototypeStore:
    prototypes = []
    class_counts: dict[int, int] = {}
    for class_id in range(num_classes):
        class_embeddings = embeddings[labels == class_id]
        if len(class_embeddings) == 0:
            raise ValueError(f"No embeddings available for class {class_id}.")
        prototypes.append(class_embeddings.mean(axis=0))
        class_counts[class_id] = int(len(class_embeddings))
    return PrototypeStore(prototypes=np.vstack(prototypes), class_counts=class_counts)


def nearest_prototype_distance(
    embeddings: np.ndarray,
    prototype_store: PrototypeStore,
) -> tuple[np.ndarray, np.ndarray]:
    diff = embeddings[:, None, :] - prototype_store.prototypes[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    nearest_class = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(embeddings)), nearest_class]
    return nearest_distance, nearest_class


def compute_open_set_scores(
    probabilities: np.ndarray,
    logits: np.ndarray,
    embeddings: np.ndarray,
    prototype_store: PrototypeStore,
    temperature: float = 1.0,
) -> OpenSetScores:
    nearest_distance, _ = nearest_prototype_distance(embeddings, prototype_store)
    return OpenSetScores(
        max_softmax=np.max(probabilities, axis=1),
        energy=compute_energy(logits, temperature=temperature),
        nearest_prototype_distance=nearest_distance,
    )


def calibrate_thresholds(
    scores: OpenSetScores,
    known_acceptance_rate: float,
) -> OpenSetThresholds:
    if not 0.5 <= known_acceptance_rate < 1.0:
        raise ValueError("known_acceptance_rate should be in [0.5, 1.0).")
    rejection_quantile = known_acceptance_rate * 100.0
    energy_mean = float(np.mean(scores.energy))
    energy_std = float(np.std(scores.energy) + 1e-8)
    prototype_mean = float(np.mean(scores.nearest_prototype_distance))
    prototype_std = float(np.std(scores.nearest_prototype_distance) + 1e-8)
    fusion_scores = np.maximum(
        (scores.energy - energy_mean) / energy_std,
        (scores.nearest_prototype_distance - prototype_mean) / prototype_std,
    )
    return OpenSetThresholds(
        msp_min_confidence=float(np.percentile(scores.max_softmax, (1.0 - known_acceptance_rate) * 100.0)),
        energy_max=float(np.percentile(scores.energy, rejection_quantile)),
        prototype_max_distance=float(np.percentile(scores.nearest_prototype_distance, rejection_quantile)),
        fusion_max=float(np.percentile(fusion_scores, rejection_quantile)),
        energy_mean=energy_mean,
        energy_std=energy_std,
        prototype_mean=prototype_mean,
        prototype_std=prototype_std,
        known_acceptance_rate=float(known_acceptance_rate),
    )


def predict_unknown_mask(
    scores: OpenSetScores,
    thresholds: OpenSetThresholds,
    strategy: OpenSetStrategy,
) -> np.ndarray:
    if strategy == "msp":
        return scores.max_softmax < thresholds.msp_min_confidence
    if strategy == "energy":
        return scores.energy > thresholds.energy_max
    if strategy == "prototype":
        return scores.nearest_prototype_distance > thresholds.prototype_max_distance
    if strategy == "energy_proto":
        fusion_scores = np.maximum(
            (scores.energy - thresholds.energy_mean) / thresholds.energy_std,
            (
                scores.nearest_prototype_distance - thresholds.prototype_mean
            ) / thresholds.prototype_std,
        )
        return fusion_scores > thresholds.fusion_max
    raise ValueError(f"Unknown open-set strategy: {strategy}")


def calibrate_benign_boundary(
    benign_reconstruction_errors: np.ndarray,
    benign_acceptance_rate: float,
) -> BenignBoundary:
    if not 0.5 <= benign_acceptance_rate < 1.0:
        raise ValueError("benign_acceptance_rate should be in [0.5, 1.0).")
    if len(benign_reconstruction_errors) == 0:
        raise ValueError("No benign calibration errors were provided.")
    threshold = float(
        np.percentile(benign_reconstruction_errors, benign_acceptance_rate * 100.0)
    )
    return BenignBoundary(
        reconstruction_error_max=threshold,
        benign_acceptance_rate=float(benign_acceptance_rate),
    )


def predict_msp_benign_ae_unknown_mask(
    scores: OpenSetScores,
    thresholds: OpenSetThresholds,
    pred_indices: np.ndarray,
    benign_class_index: int,
    reconstruction_errors: np.ndarray,
    benign_boundary: BenignBoundary,
) -> np.ndarray:
    """Reject low-confidence samples and suspicious high-confidence Benign samples."""

    msp_unknown = scores.max_softmax < thresholds.msp_min_confidence
    predicted_benign = pred_indices == benign_class_index
    outside_benign_boundary = (
        reconstruction_errors > benign_boundary.reconstruction_error_max
    )
    return msp_unknown | (predicted_benign & outside_benign_boundary)
