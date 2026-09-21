"""Data loading and feature engineering for agriculture IDS datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    ALLOWED_CATEGORICAL_COLUMNS,
    DEFAULT_COLUMNS_TO_DROP,
    LABEL_ALIASES,
)


@dataclass
class FeaturePreprocessor:
    preprocessor: ColumnTransformer
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str]
    feature_names: list[str]


def normalize_label_name(label: object) -> str:
    raw = str(label).strip()
    key = re.sub(r"[\s\-]+", "_", raw.lower())
    key = re.sub(r"_+", "_", key)
    return LABEL_ALIASES.get(key, raw.replace("_", " "))


def read_farm_flow_csv(
    csv_path: str | Path,
    label_column: str = "traffic",
    allowed_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Farm-Flow CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}")

    df = df.copy()
    df[label_column] = df[label_column].map(normalize_label_name)
    if allowed_labels is not None:
        allowed = set(allowed_labels)
        df = df.loc[df[label_column].isin(allowed)].copy()
    if df.empty:
        raise ValueError("No rows remain after Farm-Flow label filtering.")
    return df.reset_index(drop=True)


def read_smart_farm_csv(
    csv_path: str | Path,
    label_column: str = "Class",
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Smart-Farm-IDS CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}")
    return df.reset_index(drop=True)


def cap_rows_per_class(
    df: pd.DataFrame,
    label_column: str,
    max_rows_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows_per_class is None or max_rows_per_class <= 0:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby(label_column, sort=False):
        if len(group) > max_rows_per_class:
            parts.append(group.sample(n=max_rows_per_class, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_known_and_zero_day(
    df: pd.DataFrame,
    known_classes: Sequence[str],
    zero_day_classes: Sequence[str],
    label_column: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_df = df.loc[df[label_column].isin(list(known_classes))].copy()
    zero_day_df = df.loc[df[label_column].isin(list(zero_day_classes))].copy()
    if known_df.empty:
        raise ValueError("No known-class rows are available for training.")
    if zero_day_df.empty:
        raise ValueError("No zero-day rows are available for blind testing.")

    stratify = known_df[label_column] if known_df[label_column].value_counts().min() >= 2 else None
    known_train_df, known_holdout_df = train_test_split(
        known_df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    blind_df = pd.concat([known_holdout_df, zero_day_df], axis=0, ignore_index=True)
    blind_df = blind_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return known_train_df.reset_index(drop=True), known_holdout_df.reset_index(drop=True), blind_df


def split_training_validation_calibration(
    known_train_df: pd.DataFrame,
    label_column: str,
    validation_size: float,
    calibration_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint training, validation, and open-set calibration subsets."""

    if validation_size <= 0 or calibration_size <= 0:
        raise ValueError("validation_size and calibration_size must be positive.")
    if validation_size + calibration_size >= 1:
        raise ValueError("validation_size + calibration_size must be less than 1.")

    stratify = (
        known_train_df[label_column]
        if known_train_df[label_column].value_counts().min() >= 2
        else None
    )
    train_df, tuning_df = train_test_split(
        known_train_df,
        test_size=validation_size + calibration_size,
        random_state=seed,
        stratify=stratify,
    )
    calibration_fraction = calibration_size / (validation_size + calibration_size)
    tuning_stratify = (
        tuning_df[label_column]
        if tuning_df[label_column].value_counts().min() >= 2
        else None
    )
    validation_df, calibration_df = train_test_split(
        tuning_df,
        test_size=calibration_fraction,
        random_state=seed + 1,
        stratify=tuning_stratify,
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        calibration_df.reset_index(drop=True),
    )


def infer_feature_columns(
    train_df: pd.DataFrame,
    label_column: str,
    columns_to_drop: Sequence[str] = DEFAULT_COLUMNS_TO_DROP,
    allowed_categorical_columns: Sequence[str] = ALLOWED_CATEGORICAL_COLUMNS,
) -> tuple[list[str], list[str], list[str]]:
    drop_set = set(columns_to_drop) | {label_column, "is_attack"}
    allowed_cats = set(allowed_categorical_columns)
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    dropped_columns: list[str] = []

    for column in train_df.columns:
        if column in drop_set:
            dropped_columns.append(column)
            continue
        if column in allowed_cats:
            categorical_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(train_df[column]):
            numeric_columns.append(column)
            continue
        coerced = pd.to_numeric(train_df[column].replace("-", np.nan), errors="coerce")
        if coerced.notna().mean() >= 0.8:
            numeric_columns.append(column)
        else:
            dropped_columns.append(column)

    return numeric_columns, categorical_columns, sorted(set(dropped_columns))


def _prepare_feature_frame(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        output[column] = pd.to_numeric(df[column].replace("-", np.nan), errors="coerce")
    for column in categorical_columns:
        output[column] = df[column].astype("object").where(df[column].notna(), "missing")
    return output


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    label_column: str,
    columns_to_drop: Sequence[str] = DEFAULT_COLUMNS_TO_DROP,
    allowed_categorical_columns: Sequence[str] = ALLOWED_CATEGORICAL_COLUMNS,
) -> FeaturePreprocessor:
    numeric_columns, categorical_columns, dropped_columns = infer_feature_columns(
        train_df=train_df,
        label_column=label_column,
        columns_to_drop=columns_to_drop,
        allowed_categorical_columns=allowed_categorical_columns,
    )
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were detected.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    prepared_train = _prepare_feature_frame(train_df, numeric_columns, categorical_columns)
    preprocessor.fit(prepared_train)

    try:
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        feature_names = list(numeric_columns) + list(categorical_columns)

    return FeaturePreprocessor(
        preprocessor=preprocessor,
        numeric_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
        dropped_columns=dropped_columns,
        feature_names=feature_names,
    )


def transform_features(df: pd.DataFrame, feature_preprocessor: FeaturePreprocessor) -> np.ndarray:
    prepared = _prepare_feature_frame(
        df,
        feature_preprocessor.numeric_columns,
        feature_preprocessor.categorical_columns,
    )
    matrix = feature_preprocessor.preprocessor.transform(prepared)
    return np.asarray(matrix, dtype=np.float32)


def encode_known_labels(labels: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(class_order)}
    missing = sorted(set(labels) - set(mapping))
    if missing:
        raise ValueError(f"Labels not present in class order: {missing}")
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def balanced_sample_indices(
    y: np.ndarray,
    strategy: str,
    seed: int,
    max_per_class: int | None = None,
) -> np.ndarray:
    if strategy == "none":
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if strategy == "min":
        target = int(counts.min())
    elif strategy == "median":
        target = int(np.median(counts))
    elif strategy == "mean":
        target = int(round(float(np.mean(counts))))
    else:
        raise ValueError(f"Unknown balance strategy: {strategy}")
    if max_per_class is not None and max_per_class > 0:
        target = min(target, int(max_per_class))

    selected: list[np.ndarray] = []
    for class_id in classes:
        class_indices = np.flatnonzero(y == class_id)
        selected.append(
            rng.choice(
                class_indices,
                size=target,
                replace=len(class_indices) < target,
            )
        )
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
