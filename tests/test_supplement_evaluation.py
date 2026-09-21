"""Calibration-contract tests. Synthetic values are not paper results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iotids_supplement.evaluation import (upper_threshold, above, fit_operating_point,
                                        apply_operating_point, no_source_relations)
from iotids_supplement.suite import establish_identity, mark_complete, verify_complete, summarize


class CalibrationTests(unittest.TestCase):
    def test_every_integer_budget_and_ties(self):
        rng = np.random.default_rng(13)
        for n in (1, 4, 29, 1000):
            scores = rng.integers(0, 20, n).astype(float)
            for budget in range(n + 1):
                threshold = upper_threshold(scores, budget)
                self.assertLessEqual(int(above(scores, threshold).sum()), budget)
    def test_empty_branch_disabled(self):
        self.assertIsNone(upper_threshold([], 0))
        self.assertFalse(above([999], None)[0])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            upper_threshold([np.nan], 1)
    def test_base_floor_infeasible(self):
        point = fit_operating_point([0, 0, 1], [.1, .2, .3], 0, .1)
        self.assertFalse(point["feasible"])
        with self.assertRaises(ValueError):
            apply_operating_point([0], [.5], point, 0, 3)
    def test_context_union_conservative_and_gated(self):
        rng = np.random.default_rng(37)
        pred = np.zeros(2000, dtype=int)
        pred[:3] = 1
        score = rng.normal(size=2000)
        context = rng.normal(size=2000)
        for target in (.005, .01, .05):
            point = fit_operating_point(pred, score, 0, target, context)
            result = apply_operating_point(pred, score, point, 0, 3, context)
            self.assertLessEqual(int((result != 0).sum()), int(target * len(pred)))
            self.assertEqual(int((result != 0).sum()), point["achieved_calibration_errors"])
            json.dumps(point, allow_nan=False)
    def test_context_cannot_reject_known_attack_label(self):
        point = {"feasible": True, "confidence_threshold": 9., "context_threshold": .1}
        np.testing.assert_array_equal(apply_operating_point([0, 1], [0., 0.], point, 0, 3, [1., 1.]), [3, 1])
    def test_no_source_control_removes_only_source_and_deduplicates(self):
        specs = no_source_relations()
        self.assertTrue(all("id.orig_h" not in s.columns and s.columns for s in specs))
        self.assertEqual(len(specs), len(set(s.columns for s in specs)))
        self.assertIn(("history",), [s.columns for s in specs])


class ResumeTests(unittest.TestCase):
    def test_identity_change_refused_and_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            establish_identity(directory, {"seed": 42})
            (directory / "metrics.json").write_text('{"ok":true}', encoding="utf-8")
            mark_complete(directory)
            self.assertTrue(verify_complete(directory))
            establish_identity(directory, {"seed": 42})
            self.assertTrue(verify_complete(directory))
            with self.assertRaises(ValueError):
                establish_identity(directory, {"seed": 62})
            (directory / "metrics.json").write_text('{}', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_complete(directory)

    def test_summary_never_averages_only_feasible_seeds(self):
        import pandas as pd
        from iotids.paper_protocol import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "training.json", {"scenarios": {"one": ["Unknown"]}, "seeds": [42, 52, 62, 72, 82]})
            output = root / "out"
            for seed in [42, 52, 62, 72, 82]:
                directory = output / "matched/legacy/one" / f"seed{seed}"
                good = seed != 62
                metrics = {"B_benign_false_alarm_rate": .01, "F_open_set_macro_f1": .5,
                           "U_unknown_recall": .4, "unknown_class_macro_recall": .4}
                atomic_json(directory / "metrics.json", {
                    "targets": {"0.010000": {"classwise_msp_context": {
                        "calibration": {"feasible": good}, "metrics": metrics if good else None}}}})
                mark_complete(directory)
            status = summarize(root, {"legacy_config": "training.json"}, output)
            self.assertFalse(status["complete"])
            aggregate = pd.read_csv(output / "aggregate_results.csv")
            self.assertEqual(aggregate.iloc[0].n_feasible, 4)
            self.assertFalse(bool(aggregate.iloc[0].valid_five_seed_mean))
            self.assertNotIn("F_open_set_macro_f1_mean_percent", aggregate.columns)


if __name__ == "__main__":
    unittest.main()
