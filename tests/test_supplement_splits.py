"""CPU-only grouped protocol tests; synthetic data, not experiment results."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iotids_supplement.splits import grouped_split_indices


CONFIG = {
    "label_column": "traffic",
    "excluded_classes": ["BotNet DDoS"],
    "test_fraction_of_known": 0.20,
    "validation_fraction_of_known_pool": 0.10,
    "calibration_fraction_of_known_pool": 0.10,
}
PARTITIONS = ("train", "validation", "calibration", "test", "excluded")


def fixture(groups_per_class=100):
    rows = []
    for class_id, label in enumerate(("Benign", "TCP Flood")):
        for group in range(groups_per_class):
            for duplicate in range(1 + group % 5):
                rows.append((class_id * 1000 + group, "tcp", label, duplicate % 2))
    rows.extend((10000 + i, "udp", "Arp Spoofing", 1) for i in range(12))
    rows.extend((20000, "tcp", "BotNet DDoS", 1) for _ in range(64))
    return pd.DataFrame(rows, columns=["duration", "proto", "traffic", "is_attack"])


class GroupedSplitTests(unittest.TestCase):
    def split(self, frame, seed=42, config=None, unknown=None):
        return grouped_split_indices(frame, ["Arp Spoofing"] if unknown is None else unknown,
                                     seed, CONFIG if config is None else config)

    def assert_partition(self, frame, splits, label="traffic"):
        self.assertEqual(tuple(splits), PARTITIONS)
        combined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(frame)))
        destinations = np.full(len(frame), -1, dtype=int)
        for code, rows in enumerate(splits.values()):
            self.assertEqual(rows.dtype, np.dtype("int64"))
            self.assertTrue(np.all(rows[:-1] < rows[1:]))
            destinations[rows] = code
        # Independent exact tuple oracle, including null and signed-zero equality.
        observed = {}
        columns = [c for c in frame if c not in (label, "is_attack")]
        for row, values in enumerate(frame[columns].itertuples(index=False, name=None)):
            key = tuple(None if pd.isna(v) else v for v in values)
            observed.setdefault(key, set()).add(int(destinations[row]))
        self.assertTrue(all(len(parts) == 1 for parts in observed.values()))
        return destinations, len(observed)

    def test_all_rows_original_positions_and_audit(self):
        frame = fixture()
        frame.index = np.repeat("not-a-row-id", len(frame))
        original = frame.copy(deep=True)
        splits, audit = self.split(frame)
        _, groups = self.assert_partition(frame, splits)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(audit["source_groups"], groups)
        self.assertEqual(groups, 213)
        self.assertEqual(audit["duplicate_rows_beyond_first"], len(frame) - groups)
        self.assertEqual(audit["duplicate_groups"], 161)
        self.assertEqual(audit["rows_in_duplicate_groups"], 624)
        self.assertEqual(audit["grouping_columns"], ["duration", "proto"])
        self.assertEqual(audit["eligible_known_group_counts"],
                         dict(train=128, validation=16, calibration=16, test=40, excluded=0))
        for name, per_class in dict(train=64, validation=8, calibration=8, test=20, excluded=0).items():
            self.assertEqual(audit["eligible_known_group_distribution"][name],
                             {"Benign": per_class, "TCP Flood": per_class})
        self.assertEqual(audit["group_counts"]["test"], 52)
        self.assertAlmostEqual(sum(audit["row_fractions"].values()), 1.0)
        for name, rows in splits.items():
            expected = frame.iloc[rows]["traffic"].value_counts().to_dict()
            self.assertEqual({k: v for k, v in audit["distribution"][name].items() if v}, expected)
            self.assertEqual(audit["row_counts"][name], len(rows))
            self.assertAlmostEqual(audit["row_fractions"][name], len(rows) / len(frame))
        self.assertTrue(audit["all_rows_accounted_for"])
        self.assertTrue(audit["partitions_disjoint_by_group"])
        json.dumps(audit, allow_nan=False)

    def test_unknown_known_shared_group_wholly_test_and_multiple_unknown(self):
        frame = fixture()
        shared = frame.iloc[[0]].copy()
        shared["traffic"] = "Arp Spoofing"
        shared["is_attack"] = 1
        other_unknown = frame.iloc[[10]].copy()
        other_unknown["traffic"] = "Port Scanning"
        frame = pd.concat([frame, shared, other_unknown], ignore_index=True)
        splits, audit = self.split(frame, unknown=["Arp Spoofing", "Port Scanning"])
        destinations, _ = self.assert_partition(frame, splits)
        touched = frame["duration"].isin([shared.iloc[0]["duration"], other_unknown.iloc[0]["duration"]])
        self.assertTrue(np.all(destinations[touched] == 3))
        unknown = frame["traffic"].isin(["Arp Spoofing", "Port Scanning"])
        self.assertTrue(np.all(destinations[unknown] == 3))
        self.assertEqual(audit["unknown_test_fraction"], 1.0)
        self.assertEqual(audit["forced_test_groups"], 14)
        self.assertEqual(audit["known_rows_forced_test"], int((touched & ~unknown).sum()))
        self.assertEqual(audit["mixed_label_group_count"], 2)
        self.assertTrue(all(g["partition"] == "test" for g in audit["mixed_label_groups"]))

    def test_excluded_64_rows_retained_in_mixed_group(self):
        frame = fixture()
        extra = frame.iloc[[-1]].copy()
        extra["traffic"] = "Benign"
        extra["is_attack"] = 0
        frame = pd.concat([frame, extra], ignore_index=True)
        splits, audit = self.split(frame)
        self.assert_partition(frame, splits)
        self.assertEqual(len(splits["excluded"]), 65)
        self.assertEqual((frame.iloc[splits["excluded"]]["traffic"] == "BotNet DDoS").sum(), 64)
        self.assertEqual(audit["explicitly_excluded_rows"], 64)
        self.assertEqual(audit["additional_known_rows_excluded_with_group"], 1)
        self.assertEqual(audit["mixed_label_groups"][0]["partition"], "excluded")
        for name in PARTITIONS[:-1]:
            self.assertNotIn("BotNet DDoS", frame.iloc[splits[name]]["traffic"].to_list())

    def test_conflicting_unknown_and_excluded_group_fails(self):
        frame = fixture()
        extra = frame.iloc[[-1]].copy()
        extra["traffic"] = "Arp Spoofing"
        with self.assertRaisesRegex(ValueError, "Incompatible destinations.*both unknown and excluded"):
            self.split(pd.concat([frame, extra], ignore_index=True))

    def test_grouping_never_uses_target_is_attack_or_dataframe_index(self):
        frame = fixture().rename(columns={"traffic": "target"})
        mixed = frame.iloc[[0]].copy()
        mixed["target"] = "TCP Flood"
        mixed["is_attack"] = 12345
        frame = pd.concat([frame, mixed], ignore_index=True)
        config = {**CONFIG, "label_column": "target"}
        splits, audit = self.split(frame, config=config)
        destinations, groups = self.assert_partition(frame, splits, label="target")
        self.assertEqual(destinations[0], destinations[-1])
        self.assertEqual(audit["source_groups"], groups)
        self.assertEqual(audit["grouping_columns"], ["duration", "proto"])
        self.assertEqual(audit["mixed_label_groups"][0]["majority_known_label"], "Benign")
        changed = frame.copy()
        changed.index = np.arange(len(frame)) * -13
        changed["is_attack"] = np.arange(len(frame))
        again, other = self.split(changed, config=config)
        for name in splits:
            np.testing.assert_array_equal(splits[name], again[name])
        self.assertEqual(audit, other)

    def test_majority_strata_follow_exact_nested_legacy_pattern(self):
        frame = fixture()
        # Each originally Benign group gets one TCP row: ties or Benign majority.
        mixed = frame[frame["traffic"] == "Benign"].drop_duplicates("duration").copy()
        mixed["traffic"] = "TCP Flood"
        frame = pd.concat([frame, mixed], ignore_index=True)
        with patch("iotids_supplement.splits.train_test_split", wraps=train_test_split) as splitter:
            splits, audit = self.split(frame)
        self.assertEqual(splitter.call_count, 3)
        calls = splitter.call_args_list
        self.assertEqual([c.kwargs["random_state"] for c in calls], [42, 42, 43])
        self.assertEqual([c.kwargs["test_size"] for c in calls], [.2, .2, .5])
        np.testing.assert_array_equal(calls[0].args[0], np.arange(200))
        # Factorized lexical codes: Arp=0, Benign=1, BotNet=2, TCP=3.
        np.testing.assert_array_equal(calls[0].kwargs["stratify"], np.repeat([1, 3], 100))
        strata = np.repeat([1, 3], 100)
        pool, test = train_test_split(np.arange(200), test_size=.2, random_state=42, stratify=strata)
        train, tuning = train_test_split(pool, test_size=.2, random_state=42, stratify=strata[pool])
        val, cal = train_test_split(tuning, test_size=.5, random_state=43, stratify=strata[tuning])
        representative_features = np.r_[np.arange(100), 1000 + np.arange(100)]
        for name, expected in dict(train=train, validation=val, calibration=cal, test=test).items():
            actual = frame.iloc[splits[name]]["duration"].unique()
            actual = actual[actual < 10000]
            np.testing.assert_array_equal(np.sort(actual), np.sort(representative_features[expected]))
        self.assert_partition(frame, splits)
        self.assertEqual(audit["mixed_label_group_count"], 100)
        self.assertTrue(all(g["majority_known_label"] == "Benign" for g in audit["mixed_label_groups"]))

    def test_deterministic_seed_without_global_rng_side_effect(self):
        frame = fixture()
        np.random.seed(989)
        expected = np.random.random(4)
        np.random.seed(989)
        first, audit = self.split(frame, seed=42)
        np.testing.assert_array_equal(np.random.random(4), expected)
        second, other = self.split(frame, seed=np.int64(42))
        different, _ = self.split(frame, seed=52)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(audit, other)
        self.assertFalse(np.array_equal(first["train"], different["train"]))

    def test_group_proportions_are_not_row_proportions_no_downsample(self):
        frame = fixture()
        first, _ = self.split(frame)
        heavy = frame.iloc[first["train"][:1]]
        frame = pd.concat([frame, pd.concat([heavy] * 10000, ignore_index=True)], ignore_index=True)
        splits, audit = self.split(frame)
        self.assert_partition(frame, splits)
        for name, fraction in dict(train=.64, validation=.08, calibration=.08, test=.20).items():
            self.assertAlmostEqual(audit["requested_eligible_known_group_fractions"][name], fraction)
            self.assertAlmostEqual(audit["eligible_known_group_fractions"][name], fraction)
        self.assertGreater(audit["eligible_known_row_fractions"]["train"], .95)
        self.assertIsNone(audit["row_cap"])
        self.assertEqual(audit["resampling"], "none")

    def test_exact_grouping_nulls_categories_signed_zero_and_no_concat(self):
        frame = fixture()
        special = pd.DataFrame({
            "duration": [np.nan, np.nan, 0.0, -0.0, 777., 777.],
            "proto": [None, np.nan, "zero", "zero", "a|b", "a"],
            "traffic": ["Benign", "Arp Spoofing"] * 2 + ["Benign", "TCP Flood"],
            "is_attack": [0, 1, 0, 1, 0, 1],
        })
        frame = pd.concat([frame, special], ignore_index=True)
        frame["service"] = "base"
        frame.loc[len(frame) - 2:, "service"] = ["c", "b|c"]
        frame["proto"] = pd.Categorical(frame["proto"], categories=["tcp", "udp", "zero", "a|b", "a", "unused"])
        splits, audit = self.split(frame)
        destinations, groups = self.assert_partition(frame, splits)
        self.assertEqual(groups, 217)
        self.assertEqual(audit["source_groups"], groups)
        self.assertTrue(np.all(destinations[-6:-2] == 3))
        self.assertEqual(audit["grouping_columns"], ["duration", "proto", "service"])

    def test_group_after_fixed_load_frame_repairs(self):
        # Feed load_frame a CSV-shaped in-memory chunk, without writing fixtures.
        from iotids.paper_protocol import load_frame

        raw = fixture()
        special = pd.DataFrame({"duration": ["-", "not-a-number", np.inf, -np.inf],
                                "proto": ["tcp"] * 4,
                                "traffic": ["Benign", "arp_spoofing", "Benign", "Benign"],
                                "is_attack": [0, 1, 0, 0]})
        raw = pd.concat([raw, special], ignore_index=True)
        with patch("iotids.paper_protocol.pd.read_csv", return_value=iter([raw])):
            frame = load_frame("unused.csv", "traffic")
        self.assertTrue(frame["duration"].iloc[-4:].isna().all())
        self.assertEqual(frame["traffic"].iloc[-3], "Arp Spoofing")
        splits, audit = self.split(frame)
        destinations, _ = self.assert_partition(frame, splits)
        self.assertTrue(np.all(destinations[-4:] == 3))
        self.assertEqual(audit["known_rows_forced_test"], 3)

    def test_observable_columns_later_dropped_by_model_still_define_groups(self):
        frame = fixture()
        frame["uid"] = "original"
        extra = frame.iloc[[0]].copy()
        extra["traffic"] = "Arp Spoofing"
        extra["uid"] = "different-observable-record"
        frame = pd.concat([frame, extra], ignore_index=True)
        splits, audit = self.split(frame, config={**CONFIG, "columns_to_drop": ["uid"]})
        self.assert_partition(frame, splits)
        self.assertIn("uid", audit["grouping_columns"])
        self.assertEqual(audit["source_groups"], 214)
        self.assertEqual(audit["mixed_label_group_count"], 0)
        self.assertEqual(audit["known_rows_forced_test"], 0)

    def test_configurable_nested_group_fractions(self):
        splits, audit = self.split(fixture(), config={
            **CONFIG, "test_fraction_of_known": .4,
            "validation_fraction_of_known_pool": .2,
            "calibration_fraction_of_known_pool": .3,
        })
        self.assert_partition(fixture(), splits)
        self.assertEqual(audit["eligible_known_group_counts"],
                         dict(train=60, validation=24, calibration=36, test=80, excluded=0))
        for name, fraction in dict(train=.30, validation=.12, calibration=.18, test=.40).items():
            self.assertAlmostEqual(audit["requested_eligible_known_group_fractions"][name], fraction)

    def test_insufficient_distinct_groups_not_rescued_by_duplicate_rows(self):
        frame = fixture()
        frame.loc[frame["traffic"] == "TCP Flood", "duration"] = 123456
        with patch("iotids_supplement.splits.train_test_split", wraps=train_test_split) as splitter:
            with self.assertRaisesRegex(ValueError, "Insufficient eligible known groups.*TCP Flood.*1"):
                self.split(frame)
        self.assertEqual(splitter.call_count, 0)

    def test_class_with_all_groups_forced_test_fails(self):
        frame = fixture()
        unknown = frame[frame["traffic"] == "TCP Flood"].drop_duplicates("duration").copy()
        unknown["traffic"] = "Arp Spoofing"
        frame = pd.concat([frame, unknown], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Insufficient eligible known groups.*TCP Flood.*0"):
            self.split(frame)

    def test_nested_stratification_failure_has_stage_and_never_retries(self):
        with patch("iotids_supplement.splits.train_test_split", wraps=train_test_split) as splitter:
            with self.assertRaisesRegex(ValueError, "Insufficient groups.*validation/calibration.*No seed retry"):
                self.split(fixture(groups_per_class=5))
        self.assertEqual(splitter.call_count, 3)

    def test_minority_class_missing_required_partition_fails_without_retry(self):
        frame = fixture()
        base, _ = self.split(frame)
        # A rare class occurs in three different train groups but is never majority.
        representatives = frame.iloc[base["train"]].drop_duplicates(["duration", "proto"])
        representatives = representatives[representatives["traffic"] == "Benign"].head(3).copy()
        representatives["traffic"] = "Z minority"
        frame = pd.concat([frame, representatives], ignore_index=True)
        with patch("iotids_supplement.splits.train_test_split", wraps=train_test_split) as splitter:
            with self.assertRaisesRegex(ValueError, "Known classes absent.*validation.*calibration.*no seed retry"):
                self.split(frame)
        self.assertEqual(splitter.call_count, 3)

    def test_minority_only_class_can_succeed_with_actual_row_coverage(self):
        frame = fixture()
        minority = frame[frame["traffic"] == "Benign"].drop_duplicates("duration").copy()
        minority["traffic"] = "Z minority"
        frame = pd.concat([frame, minority], ignore_index=True)
        splits, audit = self.split(frame)
        self.assert_partition(frame, splits)
        for name in ("train", "validation", "calibration"):
            self.assertGreater(audit["distribution"][name]["Z minority"], 0)

    def test_rejects_invalid_input_and_protocol(self):
        frame = fixture()
        cases = [
            (frame.iloc[:0], {}, ["Arp Spoofing"], 42, "nonempty"),
            (frame, {}, [], 42, "cannot be empty"),
            (frame, {}, ["Benign"], 42, "Benign must stay known"),
            (frame, {}, ["absent"], 42, "Unknown class absent"),
            (frame, {}, "Arp Spoofing", 42, "not a string"),
            (frame, {"excluded_classes": ["Benign"]}, ["Arp Spoofing"], 42, "Exclusions"),
            (frame, {"excluded_classes": ["Arp Spoofing"]}, ["Arp Spoofing"], 42, "Exclusions"),
            (frame, {"label_column": "missing"}, ["Arp Spoofing"], 42, "label_column"),
            (frame, {"test_fraction_of_known": np.nan}, ["Arp Spoofing"], 42, "Fractions"),
            (frame, {"validation_fraction_of_known_pool": .9}, ["Arp Spoofing"], 42, "Fractions"),
            (frame, {"calibration_fraction_of_known_pool": 0}, ["Arp Spoofing"], 42, "Fractions"),
            (frame, {}, ["Arp Spoofing"], -1, "seed"),
            (frame, {}, ["Arp Spoofing"], 2**32 - 1, "seed"),
            (frame, {}, ["Arp Spoofing"], 4.2, "seed"),
            (frame[["traffic", "is_attack"]], {}, ["Arp Spoofing"], 42, "observable"),
            (frame.rename(columns={"proto": "duration"}), {}, ["Arp Spoofing"], 42, "unique column"),
        ]
        for data, changes, unknown, seed, message in cases:
            with self.subTest(changes=changes, unknown=unknown, seed=seed, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.split(data, config={**CONFIG, **changes}, unknown=unknown, seed=seed)
        frame.loc[0, "traffic"] = None
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            self.split(frame)

    def test_empty_exclusions_are_a_valid_empty_partition(self):
        frame = fixture()
        splits, audit = self.split(frame[frame["traffic"] != "BotNet DDoS"],
                                   config={**CONFIG, "excluded_classes": []})
        self.assertEqual(len(splits["excluded"]), 0)
        self.assertEqual(audit["group_counts"]["excluded"], 0)
        self.assertEqual(audit["row_fractions"]["excluded"], 0.0)

    def test_import_does_not_load_tensorflow_or_legacy_trainer(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(ROOT / "src")) + "); "
            "import iotids_supplement.splits; "
            "assert not any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules); "
            "assert 'iotids.paper_train' not in sys.modules; "
            "assert 'iotids.paper_protocol' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
