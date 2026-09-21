"""Normal-only device relationship profiles for agriculture-network context."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RelationSpec:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ContextLayerSpec:
    name: str
    description: str
    relations: tuple[RelationSpec, ...]
    weight: float


CONTEXT_ABLATIONS: dict[str, tuple[RelationSpec, ...]] = {
    "service_context": (
        RelationSpec("source_service", ("id.orig_h", "service")),
        RelationSpec("source_protocol", ("id.orig_h", "proto")),
        RelationSpec("source_state", ("id.orig_h", "conn_state")),
    ),
    "service_port_context": (
        RelationSpec("source_service", ("id.orig_h", "service")),
        RelationSpec("source_protocol", ("id.orig_h", "proto")),
        RelationSpec("source_state", ("id.orig_h", "conn_state")),
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
    ),
    "service_port_history_context": (
        RelationSpec("source_service", ("id.orig_h", "service")),
        RelationSpec("source_protocol", ("id.orig_h", "proto")),
        RelationSpec("source_state", ("id.orig_h", "conn_state")),
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
        RelationSpec("global_history", ("history",)),
        RelationSpec("source_history", ("id.orig_h", "history")),
        RelationSpec("service_history", ("service", "history")),
    ),
    "full_role_context": (
        RelationSpec("source_service", ("id.orig_h", "service")),
        RelationSpec("source_protocol", ("id.orig_h", "proto")),
        RelationSpec("source_state", ("id.orig_h", "conn_state")),
        RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
        RelationSpec("service_destination_port", ("service", "id.resp_p")),
        RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
        RelationSpec("global_history", ("history",)),
        RelationSpec("source_history", ("id.orig_h", "history")),
        RelationSpec("service_history", ("service", "history")),
        RelationSpec(
            "source_role",
            ("id.orig_h", "proto", "service", "id.resp_p"),
        ),
        RelationSpec(
            "source_service_state",
            ("id.orig_h", "service", "conn_state"),
        ),
    ),
}


LAYERED_CONTEXT_GROUPS: dict[str, ContextLayerSpec] = {
    "protocol_service": ContextLayerSpec(
        name="protocol_service",
        description=(
            "Protocol and service habits of each agriculture device, "
            "including service, protocol, and connection-state usage."
        ),
        relations=(
            RelationSpec("source_service", ("id.orig_h", "service")),
            RelationSpec("source_protocol", ("id.orig_h", "proto")),
            RelationSpec("source_state", ("id.orig_h", "conn_state")),
        ),
        weight=0.30,
    ),
    "device_role": ContextLayerSpec(
        name="device_role",
        description=(
            "Device communication role, including destination port, service-port "
            "mapping, source port role, and source-service-port role."
        ),
        relations=(
            RelationSpec("source_destination_port", ("id.orig_h", "id.resp_p")),
            RelationSpec("service_destination_port", ("service", "id.resp_p")),
            RelationSpec("source_port_role", ("id.orig_h", "_orig_port_bucket")),
            RelationSpec("source_role", ("id.orig_h", "proto", "service", "id.resp_p")),
        ),
        weight=0.45,
    ),
    "connection_behavior": ContextLayerSpec(
        name="connection_behavior",
        description=(
            "Connection behavior patterns from Zeek history and service-state "
            "combinations."
        ),
        relations=(
            RelationSpec("global_history", ("history",)),
            RelationSpec("source_history", ("id.orig_h", "history")),
            RelationSpec("service_history", ("service", "history")),
            RelationSpec("source_service_state", ("id.orig_h", "service", "conn_state")),
        ),
        weight=0.25,
    ),
}


@dataclass
class RelationCounter:
    spec: RelationSpec
    joint_counts: Counter
    parent_counts: Counter
    vocabulary_size: int


@dataclass
class ContextProfile:
    relations: dict[str, RelationCounter]
    alpha: float


@dataclass
class LayeredContextModel:
    layers: dict[str, ContextProfile]
    scales: dict[str, np.ndarray]
    relation_names: dict[str, list[str]]
    weights: dict[str, float]
    scale_percentile: float


def _value(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def prepare_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        spec_column
        for specs in CONTEXT_ABLATIONS.values()
        for spec in specs
        for spec_column in spec.columns
        if not spec_column.startswith("_")
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing context columns: {', '.join(missing)}")

    output = df.copy()
    orig_port = pd.to_numeric(output["id.orig_p"], errors="coerce").fillna(-1)
    output["_orig_port_bucket"] = (orig_port // 1024).astype(int).astype(str)
    for column in required:
        output[column] = output[column].map(_value)
    return output


def fit_context_profile(
    benign_train_df: pd.DataFrame,
    relation_specs: Sequence[RelationSpec],
    alpha: float = 1.0,
) -> ContextProfile:
    frame = prepare_context_frame(benign_train_df)
    relations: dict[str, RelationCounter] = {}
    for spec in relation_specs:
        joint_counts: Counter = Counter()
        parent_counts: Counter = Counter()
        vocabulary = set()
        for values in frame.loc[:, list(spec.columns)].itertuples(index=False, name=None):
            key = tuple(_value(value) for value in values)
            parent = key[:-1]
            joint_counts[key] += 1
            parent_counts[parent] += 1
            vocabulary.add(key[-1])
        relations[spec.name] = RelationCounter(
            spec=spec,
            joint_counts=joint_counts,
            parent_counts=parent_counts,
            vocabulary_size=max(len(vocabulary), 2),
        )
    return ContextProfile(relations=relations, alpha=float(alpha))


def context_surprisal_matrix(profile: ContextProfile, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frame = prepare_context_frame(df)
    names = list(profile.relations)
    matrix = np.zeros((len(frame), len(names)), dtype=np.float32)
    for column_index, name in enumerate(names):
        relation = profile.relations[name]
        values_iter = frame.loc[:, list(relation.spec.columns)].itertuples(index=False, name=None)
        for row_index, values in enumerate(values_iter):
            key = tuple(_value(value) for value in values)
            parent = key[:-1]
            numerator = relation.joint_counts.get(key, 0) + profile.alpha
            denominator = (
                relation.parent_counts.get(parent, 0)
                + profile.alpha * relation.vocabulary_size
            )
            probability = numerator / max(denominator, profile.alpha)
            matrix[row_index, column_index] = -math.log(max(probability, 1e-12))
    return matrix, names


def fit_surprisal_scales(
    calibration_matrix: np.ndarray,
    scale_percentile: float = 99.0,
) -> np.ndarray:
    scales = np.percentile(calibration_matrix, scale_percentile, axis=0)
    return np.maximum(scales.astype(np.float32), 1e-6)


def context_anomaly_score(surprisal_matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
    normalized = surprisal_matrix / scales
    return np.max(normalized, axis=1).astype(np.float32)


def fit_layered_context_model(
    benign_train_df: pd.DataFrame,
    benign_calibration_df: pd.DataFrame,
    layer_specs: Mapping[str, ContextLayerSpec] = LAYERED_CONTEXT_GROUPS,
    scale_percentile: float = 99.0,
    alpha: float = 1.0,
) -> LayeredContextModel:
    layers: dict[str, ContextProfile] = {}
    scales: dict[str, np.ndarray] = {}
    relation_names: dict[str, list[str]] = {}
    weights: dict[str, float] = {}

    for layer_name, layer in layer_specs.items():
        profile = fit_context_profile(
            benign_train_df,
            relation_specs=layer.relations,
            alpha=alpha,
        )
        calibration_matrix, names = context_surprisal_matrix(profile, benign_calibration_df)
        layers[layer_name] = profile
        scales[layer_name] = fit_surprisal_scales(calibration_matrix, scale_percentile)
        relation_names[layer_name] = names
        weights[layer_name] = float(layer.weight)

    return LayeredContextModel(
        layers=layers,
        scales=scales,
        relation_names=relation_names,
        weights=weights,
        scale_percentile=float(scale_percentile),
    )


def layered_context_score_matrix(
    model: LayeredContextModel,
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    layer_names = list(model.layers)
    scores = np.zeros((len(df), len(layer_names)), dtype=np.float32)
    for column_index, layer_name in enumerate(layer_names):
        matrix, _ = context_surprisal_matrix(model.layers[layer_name], df)
        scores[:, column_index] = context_anomaly_score(
            matrix,
            model.scales[layer_name],
        )
    return scores, layer_names


def aggregate_layered_scores(
    layer_scores: np.ndarray,
    layer_names: Sequence[str],
    weights: Mapping[str, float],
    mode: str,
) -> np.ndarray:
    if mode == "max":
        return np.max(layer_scores, axis=1).astype(np.float32)
    if mode == "weighted_mean":
        weight_array = np.asarray([weights[name] for name in layer_names], dtype=np.float32)
        weight_array = weight_array / max(float(np.sum(weight_array)), 1e-8)
        return np.matmul(layer_scores, weight_array).astype(np.float32)
    raise ValueError(f"Unknown layered context aggregation mode: {mode}")
