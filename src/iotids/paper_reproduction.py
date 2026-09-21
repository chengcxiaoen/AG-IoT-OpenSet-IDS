"""Paper-aligned closed-world reproduction on Farm-Flow."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from .constants import FARM_FLOW_KNOWN_CLASSES
from .data import (
    cap_rows_per_class,
    encode_known_labels,
    fit_feature_preprocessor,
    read_farm_flow_csv,
    transform_features,
)
from .models import build_lightweight_dnn
from .sampling import paper_resample
from .utils import configure_tensorflow, ensure_dir, save_json, set_global_seed


@dataclass
class PaperReproductionConfig:
    farm_flow_path: str = "datasets/Farm-Flow/Farm-Flows.csv"
    output_dir: str = "outputs/paper_reproduction"
    label_column: str = "traffic"
    class_order: tuple[str, ...] = tuple(FARM_FLOW_KNOWN_CLASSES)
    folds: int = 10
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dropout: float = 0.30
    balance_strategy: str = "median"
    max_train_per_class: int | None = None
    max_rows_per_class: int | None = None
    seed: int = 42


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_summary(metrics: dict, path: Path) -> None:
    aggregate = metrics["aggregate"]
    lines = [
        "# Paper-Aligned Closed-World Reproduction",
        "",
        "## Purpose",
        "",
        (
            "This experiment evaluates the original lightweight DNN architecture under a "
            "closed-world stratified cross-validation protocol. It is kept separate from "
            "the proposed open-set experiment."
        ),
        "",
        "## Configuration",
        "",
        f"- Folds: {metrics['config']['folds']}",
        f"- Classes: {', '.join(metrics['config']['class_order'])}",
        "- Training-only preprocessing: StandardScaler / RUS / SMOTE / Tomek Links",
        f"- Balance target: {metrics['config']['balance_strategy']}",
        "",
        "## Aggregate Results",
        "",
        f"- Accuracy: {aggregate['accuracy']['mean']:.4f} +/- {aggregate['accuracy']['std']:.4f}",
        f"- Macro F1: {aggregate['macro_f1']['mean']:.4f} +/- {aggregate['macro_f1']['std']:.4f}",
        f"- Macro precision: {aggregate['macro_precision']['mean']:.4f} +/- {aggregate['macro_precision']['std']:.4f}",
        f"- Macro recall: {aggregate['macro_recall']['mean']:.4f} +/- {aggregate['macro_recall']['std']:.4f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_paper_reproduction(config: PaperReproductionConfig) -> dict:
    set_global_seed(config.seed)
    configure_tensorflow()
    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    start = time.perf_counter()

    df = read_farm_flow_csv(
        config.farm_flow_path,
        label_column=config.label_column,
        allowed_labels=config.class_order,
    )
    df = cap_rows_per_class(df, config.label_column, config.max_rows_per_class, config.seed)
    labels = encode_known_labels(df[config.label_column], config.class_order)
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(df, labels), start=1):
        print(f"\n=== Paper reproduction fold {fold}/{config.folds} ===")
        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        preprocessor = fit_feature_preprocessor(train_df, config.label_column)
        X_train = transform_features(train_df, preprocessor)
        X_test = transform_features(test_df, preprocessor)
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        resampled = paper_resample(
            X_train,
            y_train,
            target_strategy=config.balance_strategy,
            max_per_class=config.max_train_per_class,
            seed=config.seed + fold,
        )
        model, _, _ = build_lightweight_dnn(
            input_dim=X_train.shape[1],
            num_classes=len(config.class_order),
            learning_rate=config.learning_rate,
            dropout=config.dropout,
        )
        fold_start = time.perf_counter()
        model.fit(
            resampled.X,
            resampled.y,
            validation_split=0.10,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=2,
        )
        probabilities = model.predict(X_test, batch_size=config.batch_size, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        fold_results.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_precision": float(
                    precision_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "macro_recall": float(
                    recall_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                "seconds": float(time.perf_counter() - fold_start),
                "input_dim": int(X_train.shape[1]),
                "parameter_count": int(model.count_params()),
                "resampling": {
                    "target_size": resampled.target_size,
                    "before": resampled.before,
                    "after_under_sampling": resampled.after_under_sampling,
                    "after_smote": resampled.after_smote,
                    "after_tomek": resampled.after_tomek,
                },
            }
        )
        tf.keras.backend.clear_session()

    aggregate = {
        metric: _summarize([fold[metric] for fold in fold_results])
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    metrics = {
        "config": asdict(config),
        "folds": fold_results,
        "aggregate": aggregate,
        "runtime_seconds": float(time.perf_counter() - start),
        "scope_note": (
            "Closed-world paper-aligned reproduction. Do not use these folds as zero-day evidence."
        ),
    }
    save_json(metrics, output_dir / "metrics.json")
    _write_summary(metrics, reports_dir / "summary.md")
    return metrics
