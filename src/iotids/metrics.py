"""Evaluation helpers for closed-set and open-set IDS decisions."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .constants import BENIGN_LABEL, UNKNOWN_LABEL


def make_open_truth(
    true_labels: Sequence[str],
    zero_day_classes: Sequence[str],
) -> list[str]:
    zero_day = set(zero_day_classes)
    return [UNKNOWN_LABEL if label in zero_day else str(label) for label in true_labels]


def make_decisions(
    pred_indices: np.ndarray,
    unknown_mask: np.ndarray,
    class_order: Sequence[str],
) -> list[str]:
    predictions = []
    for index, is_unknown in zip(pred_indices, unknown_mask):
        predictions.append(UNKNOWN_LABEL if is_unknown else class_order[int(index)])
    return predictions


def closed_world_decisions(pred_indices: np.ndarray, class_order: Sequence[str]) -> list[str]:
    return [class_order[int(index)] for index in pred_indices]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_decisions(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    known_classes: Sequence[str],
    zero_day_classes: Sequence[str],
) -> dict:
    truth_open = make_open_truth(true_labels, zero_day_classes)
    label_order = list(known_classes) + [UNKNOWN_LABEL]

    true_attack = np.asarray([label != BENIGN_LABEL for label in true_labels], dtype=int)
    pred_attack = np.asarray([label != BENIGN_LABEL for label in predicted_labels], dtype=int)
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        true_attack,
        pred_attack,
        average="binary",
        zero_division=0,
    )

    zero_day_set = set(zero_day_classes)
    zero_day_mask = np.asarray([label in zero_day_set for label in true_labels], dtype=bool)
    benign_mask = np.asarray([label == BENIGN_LABEL for label in true_labels], dtype=bool)
    known_mask = ~zero_day_mask
    known_attack_mask = known_mask & ~benign_mask
    unknown_pred_mask = np.asarray([label == UNKNOWN_LABEL for label in predicted_labels], dtype=bool)

    result = {
        "open_set_accuracy": accuracy_score(truth_open, predicted_labels),
        "open_set_balanced_accuracy": balanced_accuracy_score(truth_open, predicted_labels),
        "open_set_macro_f1": f1_score(truth_open, predicted_labels, average="macro", zero_division=0),
        "open_set_weighted_f1": f1_score(truth_open, predicted_labels, average="weighted", zero_division=0),
        "binary_attack_precision": float(binary_precision),
        "binary_attack_recall": float(binary_recall),
        "binary_attack_f1": float(binary_f1),
        "binary_attack_mcc": float(matthews_corrcoef(true_attack, pred_attack)),
        "unknown_threat_detection_rate": _safe_divide(
            int(np.sum(zero_day_mask & unknown_pred_mask)),
            int(np.sum(zero_day_mask)),
        ),
        "zero_day_miss_as_benign_rate": _safe_divide(
            int(np.sum(zero_day_mask & (np.asarray(predicted_labels) == BENIGN_LABEL))),
            int(np.sum(zero_day_mask)),
        ),
        "benign_false_alarm_rate": _safe_divide(
            int(np.sum(benign_mask & pred_attack.astype(bool))),
            int(np.sum(benign_mask)),
        ),
        "known_closed_label_accuracy": _safe_divide(
            sum(t == p for t, p, keep in zip(true_labels, predicted_labels, known_mask) if keep),
            int(np.sum(known_mask)),
        ),
        "known_attack_label_accuracy": _safe_divide(
            sum(t == p for t, p, keep in zip(true_labels, predicted_labels, known_attack_mask) if keep),
            int(np.sum(known_attack_mask)),
        ),
        "support": {
            "total": int(len(true_labels)),
            "known": int(np.sum(known_mask)),
            "known_attack": int(np.sum(known_attack_mask)),
            "benign": int(np.sum(benign_mask)),
            "zero_day": int(np.sum(zero_day_mask)),
        },
        "prediction_distribution": dict(Counter(predicted_labels)),
        "classification_report": classification_report(
            truth_open,
            predicted_labels,
            labels=label_order,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth_open,
            predicted_labels,
            labels=label_order,
        ).tolist(),
        "confusion_matrix_labels": label_order,
        "zero_day_breakdown": {},
    }

    predicted_array = np.asarray(predicted_labels)
    true_array = np.asarray(true_labels)
    for zero_label in zero_day_classes:
        mask = true_array == zero_label
        result["zero_day_breakdown"][zero_label] = {
            "support": int(np.sum(mask)),
            "unknown_rate": _safe_divide(int(np.sum(mask & unknown_pred_mask)), int(np.sum(mask))),
            "benign_miss_rate": _safe_divide(
                int(np.sum(mask & (predicted_array == BENIGN_LABEL))),
                int(np.sum(mask)),
            ),
            "prediction_distribution": dict(Counter(predicted_array[mask].tolist())),
        }
    return result


def add_score_auc(
    metrics: dict,
    true_labels: Sequence[str],
    zero_day_classes: Sequence[str],
    unknown_scores: np.ndarray,
) -> dict:
    zero_day_set = set(zero_day_classes)
    y_true = np.asarray([label in zero_day_set for label in true_labels], dtype=int)
    if len(np.unique(y_true)) == 2:
        metrics["unknown_score_auc"] = float(roc_auc_score(y_true, unknown_scores))
    else:
        metrics["unknown_score_auc"] = None
    return metrics
