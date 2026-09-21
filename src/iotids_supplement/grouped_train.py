"""Reuse the frozen training pipeline, changing only the partition protocol.

Hooks are process-local and restored in finally. This avoids copying/changing
the original classifier, preprocessing, losses, or rejection implementation.
"""
from __future__ import annotations

from pathlib import Path

from .splits import grouped_split_indices


def run_grouped(config, root, output, scenario, seed, allow_cpu=False):
    import iotids.paper_train as trainer
    original_load, original_split, original_json = trainer.load_frame, trainer.split_indices, trainer.atomic_json
    state = {}

    def load(path, label):
        frame = original_load(path, label)
        state["frame"] = frame
        return frame

    def split(labels, unknown, random_seed, cfg):
        partitions, audit = grouped_split_indices(state["frame"], unknown, random_seed, cfg)
        state["audit"] = audit
        original_json(Path(output) / "group_audit.json", audit)
        return partitions

    def write(path, value):
        if Path(path).name == "data_audit.json":
            value["split_type"] = "observable-duplicate-group-disjoint; NOT device/time-independent"
            value["known_partition_fractions"] = "Nominal 64/8/8/20 percent of known-only groups; actual row counts in distribution"
            value["group_audit"] = state["audit"]
            value["warning"] = "Duplicate isolation does not establish new-device or chronological generalization. Group sizes change realized row/class proportions."
            if value["test_rows_with_observable_duplicate_in_train"]:
                raise AssertionError("Observable duplicates crossed grouped train/test partitions")
        original_json(path, value)

    try:
        trainer.load_frame, trainer.split_indices, trainer.atomic_json = load, split, write
        trainer.run(config, root, output, scenario, seed, allow_cpu=allow_cpu)
    finally:
        trainer.load_frame, trainer.split_indices, trainer.atomic_json = original_load, original_split, original_json
