"""Supplemental Energy and LibMR OpenMax scoring; no threshold selection.

Inputs are finite (N, K) *raw class-output logits*, before softmax, with known
class IDs 0..K-1. They are not hidden embeddings or softmax probabilities.
Original OpenMax MAVs use the K-dimensional FC8 class activation space, not
FC7 embeddings. Here each row is one activation vector (no multi-crop average).
Shape checks cannot distinguish an embedding that happens to have K columns;
the caller must extract the classifier's actual pre-softmax output.

Primary sources (independent implementation of the algorithm):
* Bendale and Boult, Towards Open Set Deep Networks, CVPR 2016:
  https://arxiv.org/abs/1511.06233
* Author MAV/training selection, eucos, tail fit, and recalibration code:
  https://github.com/abhijitbendale/OSDN/blob/master/preprocessing/MAV_Compute.py
  https://github.com/abhijitbendale/OSDN/blob/master/openmax_utils.py
  https://github.com/abhijitbendale/OSDN/blob/master/evt_fitting.py
  https://github.com/abhijitbendale/OSDN/blob/master/compute_openmax.py
* LibMR fit_high/w_score: https://github.com/Vastlab/libMR
* Liu et al., Energy-based Out-of-distribution Detection, NeurIPS 2020:
  https://github.com/weitliu/energy_ood/blob/master/CIFAR/test.py

Known activations are z * (1 - rank_weight * w_score), and the unknown
activation is the sum of SIGNED rejected activations, as in the author code.
This is important because raw logits can be negative. This module does not
recalibrate already-softmaxed probabilities. It
uses a stable softmax over the K revised logits plus the unknown activation.

Fit ONLY on correctly classified TRAIN known examples. Defaults are tail20
and alpha=min(10, K). There is no test/unknown tuning or fitted threshold.
Energy needs only NumPy; OpenMax requires optional genuine LibMR. No SciPy
Weibull, surrogate distances, jitter, or fabricated fallback models are used.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

__all__ = ["fit_openmax", "score_openmax", "energy_score"]

_STATE_CONFIG = {
    "version": 1,
    "backend": "libmr",
    "activation_space": "class_logits",
    "distance": "eucos",
    "euclidean_scale": 200.0,
    "rejected_mass": "signed_author_convention",
    "zero_cosine": "both_zero_0_one_zero_1",
}
_LIBMR_HELP = (
    "OpenMax requires the optional genuine LibMR Python extension "
    "(libmr.MR.fit_high and libmr.MR.w_score). Install it in the active "
    "environment with `python -m pip install libmr`; if compilation or DLL "
    "loading fails, follow https://github.com/Vastlab/libMR to build the "
    "extension for your Python/compiler. No SciPy substitute is used. "
    "Energy scoring remains available without LibMR."
)


def _logit_array(values: Any, name: str = "logits") -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric raw class logits.")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, K), with K >= 1.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite logits.")
    return array


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _load_libmr() -> Any:
    try:
        module = importlib.import_module("libmr")
    except (ImportError, OSError) as exc:
        raise ImportError(_LIBMR_HELP) from exc
    if not callable(getattr(module, "MR", None)):
        raise ImportError(_LIBMR_HELP)
    return module


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Return float64 (N,) Energy = -logsumexp(raw logits), exactly T=1.

    Higher scores indicate more unknown-like samples. No normalization,
    centering, learned temperature, or threshold is applied. An empty (0, K)
    input returns an empty vector. Nonfinite or non-matrix inputs are rejected.
    """
    values = _logit_array(logits)
    # logaddexp never exponentiates an unshifted logit. Extreme differences
    # may overflow to infinity internally, representing a negligible term.
    with np.errstate(over="ignore", under="ignore"):
        return -np.logaddexp.reduce(values, axis=1)


def _eucos(logits: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Euclidean/200 + cosine, with overflow-safe norms and dot products.

    Cosine is undefined at zero in the original SciPy expression. Explicit
    extension: two zero vectors have cosine distance 0; exactly one has 1.
    No epsilon is added to nonzero norms, including tiny activation vectors.
    """
    # Divide before subtraction; hypot avoids overflow when squaring logits.
    with np.errstate(over="ignore", under="ignore"):
        euclidean = np.hypot.reduce(logits / 200.0 - mean / 200.0, axis=1)
        scales = np.max(np.abs(logits), axis=1, keepdims=True)
        scaled = logits / np.where(scales == 0.0, 1.0, scales)
        norms = np.hypot.reduce(scaled, axis=1, keepdims=True)
        unit = scaled / np.where(norms == 0.0, 1.0, norms)
        mean_scale = float(np.max(np.abs(mean)))
        scaled_mean = mean / (mean_scale if mean_scale else 1.0)
        mean_norm = float(np.hypot.reduce(scaled_mean))
        mean_unit = scaled_mean / (mean_norm if mean_norm else 1.0)
        cosine = 1.0 - np.clip(np.sum(unit * mean_unit, axis=1), -1.0, 1.0)
        if mean_scale == 0.0:
            cosine[scales[:, 0] == 0.0] = 0.0
        distances = euclidean + cosine
    if not np.isfinite(distances).all():
        raise ValueError("Eucos distances exceed the finite float64 range.")
    return distances


def _rank_weights(logits: np.ndarray, alpha: int) -> np.ndarray:
    alpha = min(_integer(alpha, "alpha"), logits.shape[1])
    # For a single activation vector, softmax and logits have the same rank.
    # Stable sort gives lower class IDs precedence for exactly tied logits.
    order = np.argsort(-logits, axis=1, kind="stable")[:, :alpha]
    weights = np.zeros_like(logits)
    weights[np.arange(len(logits))[:, None], order] = (
        np.arange(alpha, 0, -1, dtype=np.float64) / alpha
    )
    return weights


def _recalibrate_openmax(
    logits: np.ndarray, tail_weights: np.ndarray, alpha: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Pure recalibration using supplied LibMR outlier probabilities (N, K).

    This does NOT estimate Weibull probabilities and is not a replacement for
    LibMR. It permits testing the algebra independently of the native library.
    Returns revised known argmax IDs (N,) and full K+unknown probabilities.
    """
    values = _logit_array(logits)
    tails = _logit_array(tail_weights, "tail_weights")
    if tails.shape != values.shape or np.any((tails < 0.0) | (tails > 1.0)):
        raise ValueError("tail_weights must match logits and lie in [0, 1].")
    rejection = _rank_weights(values, alpha) * tails
    with np.errstate(over="ignore", under="ignore"):
        revised = values * (1.0 - rejection)
        # Multiplying directly avoids cancellation in z - revised(z).
        unknown = np.sum(values * rejection, axis=1)
    if not np.isfinite(unknown).all():
        raise ValueError("Signed rejected activation sum exceeds float64 range")
    # Do not argmax probabilities: all known probabilities can underflow to 0.
    known_ids = np.argmax(revised, axis=1).astype(np.int64, copy=False)
    probabilities = np.zeros((len(values), values.shape[1] + 1), dtype=np.float64)
    finite = np.isfinite(unknown)
    combined = np.column_stack((revised[finite], unknown[finite]))
    with np.errstate(over="ignore", under="ignore"):
        combined -= np.max(combined, axis=1, keepdims=True)
        exp_scores = np.exp(combined)
        probabilities[finite] = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return known_ids, probabilities


def _fit_tail_models(tails: list[list[float]], libmr: Any) -> list[Any]:
    models = []
    for class_id, tail in enumerate(tails):
        model = libmr.MR()
        if not all(callable(getattr(model, name, None)) for name in ("fit_high", "w_score")):
            raise ImportError(_LIBMR_HELP)
        try:
            model.fit_high(tail, len(tail))
        except Exception as exc:
            raise ValueError(f"LibMR fit_high failed for known class {class_id}.") from exc
        valid = getattr(model, "is_valid", True)
        if not (valid() if callable(valid) else valid):
            raise ValueError(
                f"LibMR fit_high produced an invalid model for known class {class_id}; "
                "check the correctly classified TRAIN logits and tail distances."
            )
        models.append(model)
    return models


def _validate_tail(tail: Any, class_id: int) -> list[float]:
    array = np.asarray(tail, dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) < 2
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(np.diff(array) < 0.0)
        or array[0] == array[-1]
    ):
        raise ValueError(
            f"Known class {class_id} needs at least two finite, nonnegative, "
            "nonconstant sorted tail distances from correctly classified TRAIN "
            "logits; a degenerate Weibull fit is not fabricated."
        )
    return array.tolist()


def fit_openmax(
    train_logits: np.ndarray, train_y: np.ndarray, tail_size: int = 20, alpha: int = 10
) -> dict:
    """Fit logit-space MAVs and genuine LibMR high-tail models on TRAIN only.

    train_logits: finite (N, K) pre-softmax known-class activation vectors.
    train_y: integer (N,) known IDs in 0..K-1; unknown labels are an error.
    tail_size: >=2, default 20; use min(tail_size, correct class count).
    alpha: >=1, default 10; the effective rank limit is min(alpha, K).

    Returns a JSON-serializable dict with fixed algorithm metadata, K, effective
    alpha, requested tail_size, (K, K) means, per-class correct_counts, and
    sorted tail_distances. No native objects or test data are stored. Models
    are actually fitted/validated here and rebuilt via fit_high when scoring.
    Every class must have sufficient correct examples and a nonconstant tail;
    otherwise ValueError is raised without substituting or perturbing data.
    Missing/unloadable LibMR raises an actionable ImportError.
    """
    values = _logit_array(train_logits, "train_logits")
    labels = np.asarray(train_y)
    k = values.shape[1]
    tail_size = _integer(tail_size, "tail_size", minimum=2)
    alpha = min(_integer(alpha, "alpha"), k)
    if len(values) == 0:
        raise ValueError("train_logits must contain TRAIN known examples.")
    if labels.shape != (len(values),) or labels.dtype.kind not in "iu":
        raise ValueError("train_y must be an integer (N,) array of known class IDs.")
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError("train_y must contain only known class IDs in 0..K-1.")
    libmr = _load_libmr()
    predicted = np.argmax(values, axis=1)
    means, tails, counts = [], [], []
    for class_id in range(k):
        correct = values[(labels == class_id) & (predicted == class_id)]
        if len(correct) < 2:
            raise ValueError(
                f"Known class {class_id} has {len(correct)} correctly classified "
                "TRAIN examples; at least two with a nonconstant distance tail "
                "are required. Do not fill this gap with validation/test data."
            )
        # Scale per coordinate to avoid overflow in summing large activations.
        scale = np.max(np.abs(correct), axis=0)
        scale[scale == 0.0] = 1.0
        with np.errstate(under="ignore"):
            mean = np.clip(np.mean(correct / scale, axis=0), -1.0, 1.0) * scale
        distances = np.sort(_eucos(correct, mean))[-tail_size:]
        means.append(mean.tolist())
        tails.append(_validate_tail(distances, class_id))
        counts.append(len(correct))
    _fit_tail_models(tails, libmr)
    return {
        **_STATE_CONFIG,
        "num_classes": k,
        "tail_size": tail_size,
        "alpha": alpha,
        "means": means,
        "tail_distances": tails,
        "correct_counts": counts,
    }


def _validate_state(state: dict) -> tuple[np.ndarray, list[list[float]], int]:
    if not isinstance(state, dict):
        raise ValueError("state must be the serializable dict returned by fit_openmax.")
    for key, expected in _STATE_CONFIG.items():
        if state.get(key) != expected:
            raise ValueError(f"Unsupported OpenMax state {key}; expected {expected!r}.")
    try:
        k = _integer(state["num_classes"], "state num_classes")
        alpha = _integer(state["alpha"], "state alpha")
        tail_size = _integer(state["tail_size"], "state tail_size", minimum=2)
        means = _logit_array(state["means"], "state means")
        tails = state["tail_distances"]
        counts = state["correct_counts"]
        if means.shape != (k, k) or alpha > k or len(tails) != k or len(counts) != k:
            raise ValueError("Inconsistent OpenMax state shapes/configuration.")
        validated = []
        for class_id, tail in enumerate(tails):
            count = _integer(counts[class_id], "state correct_counts", minimum=2)
            clean = _validate_tail(tail, class_id)
            if len(clean) != min(tail_size, count):
                raise ValueError("OpenMax state tail length disagrees with training counts.")
            validated.append(clean)
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete or malformed OpenMax state.") from exc
    return means, validated, alpha


def score_openmax(logits: np.ndarray, state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (known_pred_ids, unknown_scores), each shape (N,).

    logits must be raw (N, K) class activations in the same order/space as fit.
    known_pred_ids are int64 argmax IDs among the *revised known activations*,
    even when unknown is most probable; they never contain an unknown ID.
    unknown_scores are float64 P(unknown) from the K+1 softmax, in [0, 1],
    with larger values more unknown-like. No rejection threshold is applied.
    The JSON-compatible state is not mutated. Real LibMR models are rebuilt
    once per call using saved tails; no implicit process-global cache exists.
    Missing/unloadable LibMR raises ImportError, including for an empty batch.
    """
    values = _logit_array(logits)
    means, tails, alpha = _validate_state(state)
    if values.shape[1] != means.shape[1]:
        raise ValueError(
            "logits must have K columns matching the fitted class activation "
            "space; hidden embeddings are not OpenMax class logits."
        )
    models = _fit_tail_models(tails, _load_libmr())
    weights = np.empty_like(values)
    for class_id, (mean, model) in enumerate(zip(means, models)):
        distances = _eucos(values, mean)
        weights[:, class_id] = [model.w_score(float(distance)) for distance in distances]
        if not np.isfinite(weights[:, class_id]).all() or np.any(
            (weights[:, class_id] < 0.0) | (weights[:, class_id] > 1.0)
        ):
            raise ValueError(f"LibMR w_score for class {class_id} must be finite in [0, 1].")
    known_ids, probabilities = _recalibrate_openmax(values, weights, alpha)
    return known_ids, probabilities[:, -1]
