"""Supplemental observable-group splits, independent of TensorFlow.

API::

    frame = iotids.paper_protocol.load_frame(path, config["label_column"])
    splits, audit = grouped_split_indices(frame, unknown_classes, seed, config)

Pass the *full, already repaired* frame from the unchanged ``load_frame``.
This module neither repeats its repairs nor fits preprocessing, samples rows,
or mutates the frame. All columns except ``config['label_column']`` and
``is_attack`` define observable equality, including columns later dropped by
the model. The DataFrame index is not a feature. Missing values compare equal
within a column. Grouping uses exact pandas multi-column factorization, not
concatenated strings or an unchecked fixed-width record hash.

``splits`` has the legacy keys train/validation/calibration/test/excluded;
values are sorted int64 arrays of original *positional* row IDs for ``iloc``.
Use ``splits`` (not the returned pair) in a scoped legacy trainer adapter.
Such an adapter must capture the repaired frame: labels alone cannot recover
groups. The legacy trainer imports its own binding, so scope a patch to
``iotids.paper_train.split_indices``, not only ``paper_protocol.split_indices``.
Persist ``audit`` as the supplemental audit, not the legacy trainer's random-
row audit. Use a new supplemental run identity/output directory; the unchanged
legacy source digest does not identify this splitter. This module does not
patch or import that trainer.

The three legacy fraction keys retain their nested meaning, but apply to
eligible known GROUPS. Defaults in paper_final.json imply 64/8/8/20 percent
of those groups, subject to sklearn rounding; these are NOT row guarantees.
Group strata are majority known labels by row count, with lexical tie-breaking.
Unknown-touching groups go wholly to test; excluded-touching groups go wholly
to excluded, including their known rows. A group touching both is infeasible
and raises ValueError rather than leaking, dropping, or relocating those rows.

Audit values are JSON-serializable. ``group_counts``/``row_counts`` and
``distribution`` describe all five final partitions (class counts include
zeros). ``eligible_known_group_distribution`` counts majority-label strata,
not every label touching a group. ``row_fractions`` uses all source rows as its denominator; the two
``eligible_known_*_fractions`` fields use eligible groups/rows only, excluding
forced-test and excluded groups. Duplicate rows beyond the first, all rows
in duplicate groups, mixed-label group details, and forced destinations are
reported separately. Group IDs in mixed-group details are local to this frame.

Only the supplied seed is attempted. Invalid inputs, incompatible destinations,
insufficient stratification groups, or any observed known class missing from
train/validation/calibration raise ValueError. No fallback or best-seed search
is performed. Random observable groups do not establish temporal or unseen-
device generalization. Compatible with Python 3.10 / NumPy 1.23 / pandas 1.5 /
scikit-learn 1.2.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


_PARTITIONS = ("train", "validation", "calibration", "test", "excluded")
_FRACTION_KEYS = (
    "test_fraction_of_known",
    "validation_fraction_of_known_pool",
    "calibration_fraction_of_known_pool",
)


def _class_list(values: Iterable[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of class names, not a string")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of class names") from exc
    if any(not isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain string class names")
    return sorted(set(result))


def _observable_groups(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Exact equality groups; pandas resolves hash-table collisions by equality.

    Object-cast categoricals specifically: pandas 1.5 groupby can discard NA
    categorical keys even with dropna=False. The copy also prevents mutations
    to caller data. No target, index, rounding, or string serialization is used.
    """
    observable = frame.loc[:, columns].copy(deep=False)
    for column in columns:
        if pd.api.types.is_categorical_dtype(observable[column].dtype):
            observable[column] = observable[column].astype(object)
    try:
        groups = observable.groupby(
            columns, sort=False, dropna=False, observed=True
        ).ngroup()
    except (TypeError, ValueError) as exc:
        raise ValueError("Observable columns must contain groupable scalar values") from exc
    if groups.isna().any():
        raise ValueError("Exact observable grouping failed to account for missing values")
    return groups.to_numpy(dtype=np.int64)


def grouped_split_indices(
    frame: pd.DataFrame,
    unknown_classes: Iterable[str],
    seed: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return group-disjoint legacy positional indices and a supplemental audit.

    See the module API documentation for input repairs, fractions, exclusions,
    audit denominators, and the scoped legacy-adapter contract.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a nonempty, repaired pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("frame must have unique column names")
    label_column = config.get("label_column")
    if label_column not in frame.columns:
        raise ValueError("config['label_column'] must name a column in frame")
    if frame[label_column].isna().any():
        raise ValueError("Missing labels; refusing silent filtering")
    grouping_columns = [c for c in frame.columns if c not in (label_column, "is_attack")]
    if not grouping_columns:
        raise ValueError("At least one observable grouping column is required")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 2] (also used as seed + 1)")
    seed = int(seed)
    try:
        test_fraction, val_fraction, cal_fraction = (float(config[k]) for k in _FRACTION_KEYS)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config must supply numeric fractions: {_FRACTION_KEYS}") from exc
    fractions = (test_fraction, val_fraction, cal_fraction)
    if (not all(np.isfinite(f) and 0 < f < 1 for f in fractions)
            or val_fraction + cal_fraction >= 1):
        raise ValueError("Fractions must be finite and positive, test < 1, and validation + calibration < 1")

    unknown = _class_list(unknown_classes, "unknown_classes")
    excluded = _class_list(config.get("excluded_classes", []), "excluded_classes")
    if not unknown or "Benign" in unknown:
        raise ValueError("Benign must stay known and unknown classes cannot be empty")
    if "Benign" in excluded or set(unknown) & set(excluded):
        raise ValueError("Exclusions must not contain Benign or held-out unknown classes")
    labels = frame[label_column].to_numpy(dtype=str)
    label_codes, classes = pd.factorize(labels, sort=True)
    class_names = classes.tolist()
    missing_unknown = set(unknown) - set(class_names)
    if missing_unknown:
        raise ValueError(f"Unknown class absent from dataset: {sorted(missing_unknown)}")
    if "Benign" not in class_names:
        raise ValueError("Benign class absent from dataset")
    known_classes = sorted(set(class_names) - set(unknown) - set(excluded))
    known_class_mask = np.isin(classes, known_classes)
    unknown_rows = np.isin(labels, unknown)
    excluded_rows = np.isin(labels, excluded)

    group_ids = _observable_groups(frame, grouping_columns)
    group_sizes = np.bincount(group_ids)
    n_groups = len(group_sizes)
    if np.any(group_sizes == 0) or int(group_sizes.sum()) != len(frame):
        raise AssertionError("Exact groups must be contiguous and account for every row")
    first_rows = np.full(n_groups, len(frame), dtype=np.int64)
    np.minimum.at(first_rows, group_ids, np.arange(len(frame), dtype=np.int64))
    forced_test = np.zeros(n_groups, dtype=bool)
    forced_excluded = np.zeros(n_groups, dtype=bool)
    forced_test[group_ids[unknown_rows]] = True
    forced_excluded[group_ids[excluded_rows]] = True
    conflict = np.flatnonzero(forced_test & forced_excluded)
    if len(conflict):
        raise ValueError(
            f"Incompatible destinations: {len(conflict)} observable group(s) touch both "
            "unknown and excluded classes; whole-group test, retained exclusions, and "
            "group-disjoint partitions cannot all hold. "
            f"Example original positional row IDs: {first_rows[conflict[:5]].tolist()}"
        )
    eligible = np.flatnonzero(~(forced_test | forced_excluded))

    # Sparse group/class counts avoid an n_groups * n_classes dense matrix.
    counts = pd.DataFrame({"group": group_ids, "label": label_codes}).groupby(
        ["group", "label"], sort=True
    ).size()
    pair_groups = counts.index.get_level_values("group").to_numpy(dtype=np.int64)
    pair_labels = counts.index.get_level_values("label").to_numpy(dtype=np.int64)
    pair_counts = counts.to_numpy(dtype=np.int64)
    labels_per_group = np.bincount(pair_groups, minlength=n_groups)
    known_pairs = known_class_mask[pair_labels]
    # Largest row count wins; the sorted class code resolves ties lexically.
    order = np.lexsort((pair_labels[known_pairs], -pair_counts[known_pairs], pair_groups[known_pairs]))
    ranked_groups = pair_groups[known_pairs][order]
    ranked_labels = pair_labels[known_pairs][order]
    first = np.r_[True, ranked_groups[1:] != ranked_groups[:-1]]
    majority = np.full(n_groups, -1, dtype=np.int64)
    majority[ranked_groups[first]] = ranked_labels[first]

    eligible_mask = np.zeros(n_groups, dtype=bool)
    eligible_mask[eligible] = True
    eligible_presence = np.bincount(
        pair_labels[eligible_mask[pair_groups]], minlength=len(classes)
    )
    insufficient = {c: int(eligible_presence[i]) for i, c in enumerate(class_names)
                    if known_class_mask[i] and eligible_presence[i] < 3}
    if insufficient:
        raise ValueError(
            "Insufficient eligible known groups: each known class needs at least three "
            "distinct groups to appear in train/validation/calibration after forced "
            f"destinations; group counts={insufficient}. Seed {seed}; no seed retry."
        )

    def stratified(groups: np.ndarray, size: float, state: int, stage: str):
        strata = majority[groups]
        try:
            return train_test_split(groups, test_size=size, random_state=state, stratify=strata)
        except ValueError as exc:
            values, sizes = np.unique(strata, return_counts=True)
            summary = {class_names[int(v)]: int(s) for v, s in zip(values, sizes)}
            raise ValueError(
                f"Insufficient groups for stratified {stage}: majority-label group "
                f"counts={summary}; seed={state}. {exc} No seed retry."
            ) from exc

    pool, known_test = stratified(eligible, test_fraction, seed, "known pool/test")
    train, tuning = stratified(pool, val_fraction + cal_fraction, seed, "train/tuning")
    validation, calibration = stratified(
        tuning, cal_fraction / (val_fraction + cal_fraction), seed + 1, "validation/calibration"
    )
    partition_groups = dict(
        train=train, validation=validation, calibration=calibration,
        test=np.concatenate((known_test, np.flatnonzero(forced_test))),
        excluded=np.flatnonzero(forced_excluded),
    )
    destination = np.full(n_groups, -1, dtype=np.int8)
    for code, name in enumerate(_PARTITIONS):
        selected = partition_groups[name]
        if len(np.unique(selected)) != len(selected) or np.any(destination[selected] != -1):
            raise AssertionError("Observable groups are shared between partitions")
        destination[selected] = code
    if np.any(destination == -1):
        raise AssertionError("Observable groups are missing from partitions")
    row_destination = destination[group_ids]
    splits = {name: np.flatnonzero(row_destination == code).astype(np.int64, copy=False)
              for code, name in enumerate(_PARTITIONS)}
    if sum(map(len, splits.values())) != len(frame):
        raise AssertionError("Rows are missing from partitions")
    if np.any(row_destination[unknown_rows] != 3) or np.any(row_destination[excluded_rows] != 4):
        raise AssertionError("Unknown or excluded rows are in the wrong partition")

    distribution = {}
    missing_by_partition = {}
    for name, rows in splits.items():
        sizes = np.bincount(label_codes[rows], minlength=len(classes))
        distribution[name] = {c: int(sizes[i]) for i, c in enumerate(class_names)}
        if name in ("train", "validation", "calibration"):
            missing = [c for i, c in enumerate(class_names) if known_class_mask[i] and sizes[i] == 0]
            if missing:
                missing_by_partition[name] = missing
    if missing_by_partition:
        raise ValueError(
            f"Known classes absent from required partitions: {missing_by_partition}; "
            f"seed={seed}. Mixed-label majority stratification cannot guarantee row-class "
            "coverage; no seed retry."
        )

    mixed_ids = np.flatnonzero(labels_per_group > 1)
    offsets = np.r_[0, np.cumsum(labels_per_group)]
    mixed_groups = []
    for group in mixed_ids:
        start, stop = offsets[group:group + 2]
        mixed_groups.append({
            "group_id": int(group), "first_row_id": int(first_rows[group]),
            "rows": int(group_sizes[group]), "partition": _PARTITIONS[destination[group]],
            "class_counts": {class_names[int(c)]: int(n)
                             for c, n in zip(pair_labels[start:stop], pair_counts[start:stop])},
            "majority_known_label": class_names[majority[group]] if majority[group] >= 0 else None,
        })
    eligible_rows = int(group_sizes[eligible].sum())
    eligible_group_counts = {name: int(eligible_mask[groups].sum())
                             for name, groups in partition_groups.items()}
    eligible_row_counts = {name: int(group_sizes[groups[eligible_mask[groups]]].sum())
                           for name, groups in partition_groups.items()}
    eligible_group_distribution = {}
    for name, groups in partition_groups.items():
        sizes = np.bincount(majority[groups[eligible_mask[groups]]], minlength=len(classes))
        eligible_group_distribution[name] = {
            c: int(sizes[i]) for i, c in enumerate(class_names) if known_class_mask[i]
        }
    audit = {
        "split_type": "stratified_random_observable_groups; NOT device/time-independent",
        "seed": seed, "seed_attempts": 1,
        "grouping_method": "exact_pandas_multi_column_factorization",
        "grouping_columns": grouping_columns,
        "grouping_input": "full frame after fixed paper_protocol.load_frame repairs",
        "stratification": "majority known row label per eligible group; lexical ties",
        "source_rows": len(frame), "source_groups": n_groups,
        "row_cap": None, "resampling": "none",
        "all_rows_accounted_for": True,
        "partitions_disjoint_by_row_id": True, "partitions_disjoint_by_group": True,
        "unknown_excluded_before_test": True, "unknown_test_fraction": 1.0,
        "unknown_classes": unknown, "known_classes": known_classes,
        "explicitly_excluded_classes": excluded,
        "explicitly_excluded_rows": int(excluded_rows.sum()),
        "additional_known_rows_excluded_with_group": len(splits["excluded"]) - int(excluded_rows.sum()),
        "protocol_rows": len(frame) - len(splits["excluded"]),
        "forced_test_groups": int(forced_test.sum()),
        "forced_test_rows": int(group_sizes[forced_test].sum()),
        "known_rows_forced_test": int(group_sizes[forced_test].sum() - unknown_rows.sum()),
        "excluded_groups": int(forced_excluded.sum()),
        "eligible_known_groups": len(eligible), "eligible_known_rows": eligible_rows,
        "duplicate_groups": int((group_sizes > 1).sum()),
        "rows_in_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
        "duplicate_rows_beyond_first": len(frame) - n_groups,
        "mixed_label_group_count": len(mixed_ids),
        "rows_in_mixed_label_groups": int(group_sizes[mixed_ids].sum()),
        "mixed_label_groups": mixed_groups,
        "group_counts": {name: len(groups) for name, groups in partition_groups.items()},
        "row_counts": {name: len(rows) for name, rows in splits.items()},
        "row_fractions": {name: len(rows) / len(frame) for name, rows in splits.items()},
        "distribution": distribution,
        "requested_eligible_known_group_fractions": {
            "train": (1 - test_fraction) * (1 - val_fraction - cal_fraction),
            "validation": (1 - test_fraction) * val_fraction,
            "calibration": (1 - test_fraction) * cal_fraction, "test": test_fraction,
        },
        "eligible_known_group_counts": eligible_group_counts,
        "eligible_known_group_distribution": eligible_group_distribution,
        "eligible_known_row_counts": eligible_row_counts,
        "eligible_known_group_fractions": {k: v / len(eligible) for k, v in eligible_group_counts.items()},
        "eligible_known_row_fractions": {k: v / eligible_rows for k, v in eligible_row_counts.items()},
        "warning": "Configured fractions target eligible GROUPS, not rows; rounding and forced destinations change totals.",
    }
    return splits, audit
