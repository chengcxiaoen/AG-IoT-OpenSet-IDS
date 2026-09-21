"""Training-only resampling protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks


@dataclass
class ResamplingResult:
    X: np.ndarray
    y: np.ndarray
    before: dict[int, int]
    after_under_sampling: dict[int, int]
    after_smote: dict[int, int]
    after_tomek: dict[int, int]
    target_size: int


def class_distribution(y: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(class_id): int(count) for class_id, count in zip(classes, counts)}


def determine_target_size(
    y: np.ndarray,
    strategy: str,
    max_per_class: int | None,
) -> int:
    counts = np.asarray(list(class_distribution(y).values()), dtype=int)
    if strategy == "min":
        target = int(np.min(counts))
    elif strategy == "median":
        target = int(np.median(counts))
    elif strategy == "mean":
        target = int(round(float(np.mean(counts))))
    elif strategy == "max":
        target = int(np.max(counts))
    else:
        raise ValueError(f"Unknown target strategy: {strategy}")
    if max_per_class is not None and max_per_class > 0:
        target = min(target, int(max_per_class))
    return max(target, 2)


def paper_resample(
    X: np.ndarray,
    y: np.ndarray,
    target_strategy: str = "median",
    max_per_class: int | None = None,
    seed: int = 42,
    apply_tomek: bool = True,
) -> ResamplingResult:
    """Apply the paper-aligned RUS -> SMOTE -> Tomek Links training chain.

    Resampling must only receive a training fold. Validation, calibration, and
    blind-test samples must never pass through this function.
    """

    before = class_distribution(y)
    target = determine_target_size(y, target_strategy, max_per_class)

    under_strategy = {
        class_id: target
        for class_id, count in before.items()
        if count > target
    }
    if under_strategy:
        X_under, y_under = RandomUnderSampler(
            sampling_strategy=under_strategy,
            random_state=seed,
        ).fit_resample(X, y)
    else:
        X_under, y_under = X, y
    after_under = class_distribution(y_under)

    over_strategy = {
        class_id: target
        for class_id, count in after_under.items()
        if count < target
    }
    if over_strategy:
        minimum_count = min(after_under[class_id] for class_id in over_strategy)
        neighbors = max(1, min(5, minimum_count - 1))
        X_smote, y_smote = SMOTE(
            sampling_strategy=over_strategy,
            random_state=seed,
            k_neighbors=neighbors,
        ).fit_resample(X_under, y_under)
    else:
        X_smote, y_smote = X_under, y_under
    after_smote = class_distribution(y_smote)

    if apply_tomek:
        X_final, y_final = TomekLinks(sampling_strategy="all").fit_resample(X_smote, y_smote)
    else:
        X_final, y_final = X_smote, y_smote

    return ResamplingResult(
        X=np.asarray(X_final, dtype=np.float32),
        y=np.asarray(y_final, dtype=np.int64),
        before=before,
        after_under_sampling=after_under,
        after_smote=after_smote,
        after_tomek=class_distribution(y_final),
        target_size=target,
    )
