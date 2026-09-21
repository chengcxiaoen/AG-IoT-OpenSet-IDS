"""Label-free inspection of flows rejected by the online open-set gate."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def cluster_rejected_embeddings(
    embeddings: np.ndarray,
    rejected_mask: np.ndarray,
    min_cluster_size: int = 20,
) -> dict:
    """Cluster rejected flows without using their true labels or attack names.

    The DBSCAN radius is derived from the rejected embeddings' k-neighbour
    distance distribution, so no zero-day labels participate in tuning.
    """

    rejected_indices = np.flatnonzero(rejected_mask)
    if len(rejected_indices) < min_cluster_size:
        return {
            "rejected_count": int(len(rejected_indices)),
            "min_cluster_size": int(min_cluster_size),
            "algorithm": "not_run",
            "eps": None,
            "clusters": [],
            "noise_count": int(len(rejected_indices)),
        }

    features = StandardScaler(copy=False).fit_transform(embeddings[rejected_indices])

    # DBSCAN requires a potentially quadratic neighbourhood structure. Keep it
    # for small rejected sets, but use a streaming-capable clusterer on a full
    # Farm-Flow blind set. Neither path accesses true labels.
    if len(features) > 50_000:
        cluster_count = min(64, max(2, int(np.ceil(len(features) / 20_000))))
        labels = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=4096,
            n_init=3,
            random_state=42,
        ).fit_predict(features)
        clusters = []
        for cluster_id in range(cluster_count):
            member_indices = rejected_indices[labels == cluster_id]
            clusters.append({
                "cluster_id": int(cluster_id),
                "size": int(len(member_indices)),
                "blind_indices": member_indices.astype(int).tolist(),
            })
        return {
            "rejected_count": int(len(rejected_indices)),
            "min_cluster_size": int(min_cluster_size),
            "algorithm": "minibatch_kmeans",
            "cluster_count": int(cluster_count),
            "clusters": clusters,
            "noise_count": 0,
        }
    neighbours = min(min_cluster_size, len(features))
    distances = NearestNeighbors(n_neighbors=neighbours).fit(features).kneighbors(
        features,
        return_distance=True,
    )[0][:, -1]
    eps = float(np.quantile(distances, 0.90))
    labels = DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(features)
    clusters = []
    for cluster_id in sorted(set(labels) - {-1}):
        member_indices = rejected_indices[labels == cluster_id]
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(member_indices)),
            "blind_indices": member_indices.astype(int).tolist(),
        })
    return {
        "rejected_count": int(len(rejected_indices)),
        "min_cluster_size": int(min_cluster_size),
        "algorithm": "dbscan",
        "eps": eps,
        "clusters": clusters,
        "noise_count": int(np.sum(labels == -1)),
    }


def add_cluster_summaries(
    inspection: dict,
    true_labels: list[str],
    predicted_indices: np.ndarray,
    known_classes: tuple[str, ...],
) -> dict:
    """Attach post-hoc evaluation summaries; never used by the clustering gate."""

    label_array = np.asarray(true_labels, dtype=object)
    for cluster in inspection["clusters"]:
        members = np.asarray(cluster.pop("blind_indices"), dtype=int)
        labels, label_counts = np.unique(label_array[members], return_counts=True)
        predicted, predicted_counts = np.unique(predicted_indices[members], return_counts=True)
        dominant = int(np.argmax(label_counts))
        cluster["true_label_distribution_eval_only"] = {
            str(label): int(count) for label, count in zip(labels, label_counts)
        }
        cluster["purity_eval_only"] = float(label_counts[dominant] / len(members))
        cluster["dominant_true_label_eval_only"] = str(labels[dominant])
        cluster["predicted_known_distribution"] = {
            known_classes[int(label)]: int(count)
            for label, count in zip(predicted, predicted_counts)
        }
    return inspection
