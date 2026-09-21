"""Synthetic algebra/plumbing tests, not evidence of benchmark performance.

Run with Python 3.10 and NumPy 1.23 (no pytest or native LibMR required):
    python -B -m unittest discover -s tests -p test_supplement_baselines.py -v
Only the explicitly named integration test fits a real Weibull distribution.
Mocks below verify the LibMR call contract, never approximate its statistics.
"""

import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.distance import cosine, euclidean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iotids_supplement import baselines


def _train_fixture(k=2, n=32):
    rng = np.random.default_rng(701)
    labels = np.repeat(np.arange(k), n)
    logits = rng.normal(0.0, 0.4, (len(labels), k))
    logits[np.arange(len(labels)), labels] += np.linspace(4.0, 8.0, len(labels))
    return logits, labels


def _backend(*probabilities):
    """Recording mocks returning only explicitly supplied tail probabilities."""
    models = []
    for probability in probabilities:
        model = Mock(spec=["fit_high", "w_score", "is_valid"])
        model.is_valid = True
        model.w_score.return_value = probability
        models.append(model)
    return SimpleNamespace(MR=Mock(side_effect=models)), models


def _state_fixture():
    logits, labels = _train_fixture()
    backend, _ = _backend(0.0, 0.0)
    with patch.object(baselines, "_load_libmr", return_value=backend):
        return baselines.fit_openmax(logits, labels)


class EnergyTests(unittest.TestCase):
    def test_t1_raw_logsumexp_and_score_direction(self):
        logits = np.log([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        result = baselines.energy_score(logits)
        np.testing.assert_allclose(result, -np.log([6.0, 12.0]), rtol=1e-14)
        self.assertGreater(result[0], result[1])
        np.testing.assert_allclose(baselines.energy_score(logits + 50), result - 50)

    def test_stable_large_positive_negative_and_float32_logits(self):
        logits = np.array([[10000, 9999], [-10000, -10001]], dtype=np.float32)
        with np.errstate(all="raise"):
            result = baselines.energy_score(logits)
        np.testing.assert_allclose(result, [-10000 - np.log1p(np.exp(-1)),
                                          10000 - np.log1p(np.exp(-1))])
        self.assertEqual(result.dtype, np.float64)

    def test_extreme_finite_range(self):
        huge = np.finfo(np.float64).max
        with np.errstate(all="raise"):
            result = baselines.energy_score([[huge, -huge], [-huge, -huge]])
        np.testing.assert_array_equal(result, [-huge, huge])

    def test_empty_and_single_class(self):
        self.assertEqual(baselines.energy_score(np.empty((0, 3))).shape, (0,))
        np.testing.assert_array_equal(baselines.energy_score([[5], [-8]]), [-5, 8])

    def test_rejects_invalid_logits(self):
        for logits in ([1, 2], np.empty((2, 0)), [[np.nan]], [[np.inf]],
                       [[-np.inf]], [[1 + 2j]], [["3"]], [[True]], [[[1]]]):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                baselines.energy_score(logits)


class RecalibrationTests(unittest.TestCase):
    def test_rank_weights_scattered_by_logit_rank_not_class_id(self):
        logits = np.array([[2., 9., 5., 1.], [8., 3., 1., 6.]])
        np.testing.assert_allclose(baselines._rank_weights(logits, 3),
                                   [[1/3, 1, 2/3, 0], [1, 1/3, 0, 2/3]])

    def test_default_alpha_is_min_10_k_and_ties_are_deterministic(self):
        np.testing.assert_allclose(baselines._rank_weights(np.ones((1, 3)), 10),
                                   [[1, 2/3, 1/3]])
        weights = baselines._rank_weights(np.arange(12.)[None, :], 10)
        np.testing.assert_allclose(weights, [[0, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]])

    def test_provided_tail_probabilities_revise_logits_then_softmax(self):
        logits = np.array([[2., 9., 5., 1.]])
        tails = np.array([[.4, .5, .6, .9]])
        ids, probabilities = baselines._recalibrate_openmax(logits, tails, alpha=3)
        # Rank factors = [1/3, 1, 2/3, 0]; rejection = [.1333, .5, .4, 0].
        revised = np.array([2 * (1 - .4/3), 4.5, 3., 1.])
        combined = np.append(revised, 2 * .4/3 + 4.5 + 2.)
        expected = np.exp(combined - combined.max())
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities[0], expected, rtol=1e-14)
        np.testing.assert_array_equal(ids, [1])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_zero_rejection_adds_unknown_logit_zero_not_zero_probability(self):
        logits = np.log([[2., 3.]])
        ids, probabilities = baselines._recalibrate_openmax(logits, np.zeros((1, 2)))
        np.testing.assert_allclose(probabilities, [[2/6, 3/6, 1/6]])
        np.testing.assert_array_equal(ids, [1])

    def test_vectorized_recalibration_matches_scalar_reference(self):
        rng = np.random.default_rng(103)
        for k in (1, 3, 12):
            logits = rng.uniform(-10., 10., (8, k))
            tails = rng.uniform(0., 1., (8, k))
            for alpha in (1, 3, 10, 50):
                with self.subTest(k=k, alpha=alpha):
                    ids, probabilities = baselines._recalibrate_openmax(logits, tails, alpha)
                    expected_probabilities, expected_ids = [], []
                    for row, tail in zip(logits, tails):
                        limit = min(alpha, k)
                        ranked = sorted(range(k), key=lambda j: (-row[j], j))[:limit]
                        revised, positive_residuals = row.copy(), []
                        for rank, class_id in enumerate(ranked):
                            fraction = (limit - rank) / limit * tail[class_id]
                            revised[class_id] = row[class_id] * (1 - fraction)
                            positive_residuals.append(row[class_id] * fraction)
                        expected_ids.append(np.argmax(revised))
                        activations = [*revised, sum(positive_residuals)]
                        exponentials = np.exp(activations - np.max(activations))
                        expected_probabilities.append(exponentials / exponentials.sum())
                    np.testing.assert_array_equal(ids, expected_ids)
                    np.testing.assert_allclose(probabilities, expected_probabilities,
                                               rtol=1e-13, atol=1e-15)

    def test_signed_rejected_mass_includes_negative_residuals(self):
        logits = np.array([[4., -2., -6.]])
        _, probabilities = baselines._recalibrate_openmax(logits, np.ones((1, 3)))
        # Author convention: unknown = 4 - 4/3 - 2 = 2/3.
        combined = np.array([0., -2/3, -4., 2/3])
        expected = np.exp(combined - combined.max())
        np.testing.assert_allclose(probabilities[0], expected / expected.sum())

    def test_all_negative_activations_have_signed_unknown_logit(self):
        _, probabilities = baselines._recalibrate_openmax([[-2., -4.]], [[.5, .5]])
        expected = np.exp([-1., -3., -2.])
        np.testing.assert_allclose(probabilities[0], expected / expected.sum())

    def test_revised_known_prediction_can_change(self):
        ids, _ = baselines._recalibrate_openmax([[5., 4.]], [[1., 0.]], alpha=1)
        np.testing.assert_array_equal(ids, [1])

    def test_extreme_logits_and_known_argmax_when_probabilities_underflow(self):
        logits = np.array([[1e300, 2.5e300, 3e300], [-1e300, -2e300, -3e300],
                           [10000., 12000., 15000.], [0., 0., 0.]])
        with np.errstate(all="raise"):
            ids, probabilities = baselines._recalibrate_openmax(logits, np.ones_like(logits))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        np.testing.assert_array_equal(probabilities[0, :-1], [0., 0., 0.])
        self.assertEqual(ids[0], 1)  # argmax of zero probabilities would be 0.
        self.assertEqual(ids[1], 0)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_signed_unknown_sum_overflow_is_rejected(self):
        huge = np.finfo(np.float64).max
        with np.errstate(all="raise"), self.assertRaises(ValueError):
            baselines._recalibrate_openmax([[huge, huge]], [[1., 1.]])

    def test_empty_single_class_and_inputs_not_mutated(self):
        ids, probabilities = baselines._recalibrate_openmax(np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(probabilities.shape, (0, 3))
        logits, tails = np.array([[4.]]), np.array([[.25]])
        before_logits, before_tails = logits.copy(), tails.copy()
        ids, probabilities = baselines._recalibrate_openmax(logits, tails)
        expected = np.exp([0., -2.])
        np.testing.assert_allclose(probabilities[0], expected / expected.sum())
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_array_equal(logits, before_logits)
        np.testing.assert_array_equal(tails, before_tails)

    def test_rejects_invalid_tail_probabilities_and_alpha(self):
        for tails in ([[np.nan, 0]], [[np.inf, 0]], [[-.1, 0]], [[1.1, 0]], [[.5]]):
            with self.subTest(tails=tails), self.assertRaises(ValueError):
                baselines._recalibrate_openmax([[1, 2]], tails)
        for alpha in (0, -1, 1.5, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                baselines._recalibrate_openmax([[1, 2]], [[.5, .5]], alpha=alpha)


class DistanceTests(unittest.TestCase):
    def test_matches_author_eucos_on_nonzero_vectors(self):
        rng = np.random.default_rng(99)
        logits, mean = rng.normal(size=(20, 5)), rng.normal(size=5)
        expected = [euclidean(row, mean) / 200 + cosine(row, mean) for row in logits]
        np.testing.assert_allclose(baselines._eucos(logits, mean), expected, rtol=1e-14)

    def test_zero_vector_convention(self):
        np.testing.assert_allclose(baselines._eucos(np.array([[0., 0.], [3., 4.]]),
                                                   np.zeros(2)), [0., 1.025])
        np.testing.assert_allclose(baselines._eucos(np.zeros((1, 2)), np.array([3., 4.])),
                                   [1.025])

    def test_extreme_norms_and_tiny_nonzero_cosine_are_stable(self):
        with np.errstate(all="raise"):
            large = baselines._eucos(np.array([[-1e300, 0.], [1e300, 0.]]),
                                      np.array([1e300, 0.]))
            tiny = baselines._eucos(np.array([[0., 1e-300]]), np.array([1e-300, 0.]))
        np.testing.assert_allclose(large, [1e298, 0.])
        np.testing.assert_allclose(tiny, [1.])


class FitAndScoreContractTests(unittest.TestCase):
    def test_mavs_only_correct_training_logits_and_largest_20_distances(self):
        logits, labels = _train_fixture()
        # Deliberately misclassified TRAIN rows must not enter either MAV.
        logits = np.vstack((logits, [[-500., 500.], [500., -500.]]))
        labels = np.append(labels, [0, 1])
        before_logits, before_labels = logits.copy(), labels.copy()
        backend, models = _backend(.0, .0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            state = baselines.fit_openmax(logits, labels)
        self.assertEqual(state["alpha"], 2)
        self.assertEqual(state["tail_size"], 20)
        self.assertEqual(state["correct_counts"], [32, 32])
        for class_id in range(2):
            correct = logits[(labels == class_id) & (logits.argmax(axis=1) == class_id)]
            expected_mean = correct.mean(axis=0)
            expected_tail = sorted(euclidean(row, expected_mean)/200 + cosine(row, expected_mean)
                                   for row in correct)[-20:]
            np.testing.assert_allclose(state["means"][class_id], expected_mean, rtol=1e-14)
            np.testing.assert_allclose(state["tail_distances"][class_id], expected_tail, atol=1e-14)
            models[class_id].fit_high.assert_called_once_with(state["tail_distances"][class_id], 20)
            models[class_id].w_score.assert_not_called()
        self.assertEqual(json.loads(json.dumps(state, allow_nan=False)), state)
        np.testing.assert_array_equal(logits, before_logits)
        np.testing.assert_array_equal(labels, before_labels)

    def test_small_classes_use_available_tail_and_custom_alpha(self):
        logits, labels = _train_fixture(k=3, n=7)
        backend, models = _backend(0, 0, 0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            state = baselines.fit_openmax(logits, labels, alpha=1)
        self.assertEqual(state["alpha"], 1)
        self.assertEqual([len(tail) for tail in state["tail_distances"]], [7, 7, 7])
        for model, tail in zip(models, state["tail_distances"]):
            model.fit_high.assert_called_once_with(tail, 7)

    def test_custom_tail_size(self):
        logits, labels = _train_fixture()
        backend, _ = _backend(0, 0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            state = baselines.fit_openmax(logits, labels, tail_size=5)
        self.assertEqual(state["tail_size"], 5)
        self.assertEqual([len(tail) for tail in state["tail_distances"]], [5, 5])

    def test_fit_default_alpha_caps_at_ten_for_more_than_ten_classes(self):
        logits, labels = _train_fixture(k=12, n=7)
        backend, _ = _backend(*([0.] * 12))
        with patch.object(baselines, "_load_libmr", return_value=backend):
            state = baselines.fit_openmax(logits, labels)
        self.assertEqual(state["alpha"], 10)
        self.assertEqual(np.asarray(state["means"]).shape, (12, 12))

    def test_score_rebuilds_native_models_from_json_and_calls_w_score(self):
        state = json.loads(json.dumps(_state_fixture()))
        before_state = copy.deepcopy(state)
        queries = np.array([[5., 4.], [-1., 2.], [1e6, 2e6]])
        before_queries = queries.copy()
        backend, models = _backend(.9, .1)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            ids, unknown = baselines.score_openmax(queries, state)
        expected_ids, probabilities = baselines._recalibrate_openmax(
            queries, np.tile([.9, .1], (3, 1)), alpha=2)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(unknown, probabilities[:, -1])
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(unknown.dtype, np.float64)
        self.assertEqual(ids.shape, (3,))
        self.assertEqual(unknown.shape, (3,))
        for class_id, model in enumerate(models):
            model.fit_high.assert_called_once_with(state["tail_distances"][class_id], 20)
            passed_distances = [call.args[0] for call in model.w_score.call_args_list]
            expected = [euclidean(row, state["means"][class_id])/200 +
                        cosine(row, state["means"][class_id]) for row in queries]
            np.testing.assert_allclose(passed_distances, expected, atol=1e-14)
        self.assertEqual(state, before_state)
        np.testing.assert_array_equal(queries, before_queries)

    def test_empty_score_batch(self):
        state = _state_fixture()
        backend, models = _backend(0, 0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            ids, scores = baselines.score_openmax(np.empty((0, 2)), state)
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(scores.shape, (0,))
        for model in models:
            model.w_score.assert_not_called()

    def test_wrong_embedding_width_rejected(self):
        state = _state_fixture()
        with self.assertRaisesRegex(ValueError, "hidden embeddings"):
            baselines.score_openmax(np.ones((3, 8)), state)

    def test_rejects_unknown_noninteger_or_mismatched_training_labels(self):
        logits, labels = _train_fixture()
        invalid_labels = [np.append(labels[:-1], -1), np.append(labels[:-1], 2),
                          labels.astype(float), labels[:, None], labels[:-1],
                          np.ones(len(labels), dtype=bool)]
        for invalid in invalid_labels:
            with self.subTest(labels=invalid), self.assertRaises(ValueError):
                baselines.fit_openmax(logits, invalid)

    def test_rejects_empty_training_and_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            baselines.fit_openmax(np.empty((0, 2)), np.empty(0, dtype=int))
        logits, labels = _train_fixture()
        for config in ({"tail_size": 1}, {"tail_size": 0}, {"tail_size": 2.5},
                       {"tail_size": True}, {"alpha": 0}, {"alpha": 2.5}, {"alpha": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                baselines.fit_openmax(logits, labels, **config)

    def test_missing_correct_class_fails_without_model_fallback(self):
        logits, labels = _train_fixture()
        logits[labels == 1] = [8., 0.]
        backend, _ = _backend(0, 0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            with self.assertRaisesRegex(ValueError, "class 1 has 0 correctly classified TRAIN"):
                baselines.fit_openmax(logits, labels)
        backend.MR.assert_not_called()

    def test_degenerate_tail_is_not_jittered_or_replaced(self):
        backend, _ = _backend(0, 0)
        with patch.object(baselines, "_load_libmr", return_value=backend):
            with self.assertRaisesRegex(ValueError, "degenerate Weibull"):
                baselines.fit_openmax([[3., 1.]] * 4 + [[1., 3.]] * 4, [0] * 4 + [1] * 4)
        backend.MR.assert_not_called()

    def test_failed_or_invalid_native_fit_is_explicit(self):
        logits, labels = _train_fixture()
        for exception in (None, RuntimeError("native failure")):
            backend, models = _backend(0, 0)
            models[0].is_valid = False
            models[0].fit_high.side_effect = exception
            with patch.object(baselines, "_load_libmr", return_value=backend):
                with self.assertRaisesRegex(ValueError, "LibMR fit_high.*class 0"):
                    baselines.fit_openmax(logits, labels)

    def test_invalid_native_tail_scores_are_not_clipped(self):
        state = _state_fixture()
        for score in (np.nan, np.inf, -.01, 1.01):
            backend, _ = _backend(score, 0)
            with patch.object(baselines, "_load_libmr", return_value=backend):
                with self.assertRaisesRegex(ValueError, "LibMR w_score"):
                    baselines.score_openmax([[3., 1.]], state)

    def test_invalid_serialized_state_is_rejected(self):
        state = _state_fixture()
        changes = [{"backend": "scipy"}, {"activation_space": "embeddings"},
                   {"distance": "euclidean"}, {"euclidean_scale": 1},
                   {"rejected_mass": "signed"}, {"alpha": 3}, {"num_classes": 3},
                   {"means": [[np.nan, 0], [0, 1]]}, {"correct_counts": [2, 2]},
                   {"tail_distances": [[0., 0.], [0., 1.]]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                baselines.score_openmax([[1., 2.]], {**state, **change})
        for invalid in (None, {}, {key: value for key, value in state.items() if key != "means"}):
            with self.subTest(state=invalid), self.assertRaises(ValueError):
                baselines.score_openmax([[1., 2.]], invalid)


class OptionalDependencyTests(unittest.TestCase):
    def test_absent_libmr_is_actionable_for_fit_and_score_energy_still_works(self):
        logits, labels = _train_fixture()
        state = _state_fixture()
        with patch.dict(sys.modules, {"libmr": None}):
            # Reload proves importing the baseline module does not need LibMR.
            importlib.reload(baselines)
            self.assertTrue(np.isfinite(baselines.energy_score(logits)).all())
            for operation in (lambda: baselines.fit_openmax(logits, labels),
                              lambda: baselines.score_openmax(logits, state),
                              lambda: baselines.score_openmax(np.empty((0, 2)), state)):
                with self.assertRaisesRegex(ImportError, "python -m pip install libmr"):
                    operation()

    def test_dll_load_failure_is_actionable(self):
        with patch.object(baselines.importlib, "import_module", side_effect=OSError("missing DLL")):
            with self.assertRaisesRegex(ImportError, "No SciPy substitute") as raised:
                baselines._load_libmr()
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_incompatible_libmr_api_is_explicit(self):
        with patch.object(baselines.importlib, "import_module", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImportError, "LibMR"):
                baselines._load_libmr()
        logits, labels = _train_fixture()
        with patch.object(baselines, "_load_libmr", return_value=SimpleNamespace(MR=lambda: object())):
            with self.assertRaisesRegex(ImportError, "fit_high"):
                baselines.fit_openmax(logits, labels)

    def test_real_libmr_integration_when_installed(self):
        try:
            importlib.import_module("libmr")
        except (ImportError, OSError) as exc:
            self.skipTest(f"Genuine LibMR unavailable; algebra/contract tests are independent: {exc}")
        logits, labels = _train_fixture(k=3, n=40)
        state = baselines.fit_openmax(logits, labels)
        queries = np.vstack((logits[:5], np.zeros((1, 3)), [[1000., -1000., 500.]]))
        ids, scores = baselines.score_openmax(queries, state)
        restored_ids, restored_scores = baselines.score_openmax(queries, json.loads(json.dumps(state)))
        np.testing.assert_array_equal(ids, restored_ids)
        np.testing.assert_allclose(scores, restored_scores, rtol=1e-12, atol=1e-14)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        self.assertTrue(np.all((ids >= 0) & (ids < 3)))


if __name__ == "__main__":
    unittest.main()
