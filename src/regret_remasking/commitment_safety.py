"""Direct counterfactual commitment-safety prediction for PhaseLock Phase 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CHEAP_COMMITMENT_FEATURES = [
    "confidence",
    "margin",
    "entropy",
    "token_age",
    "trace_probability",
    "block_age_fraction",
]

FULL_COMMITMENT_FEATURES = CHEAP_COMMITMENT_FEATURES + [
    "max_hidden_drift",
    "attention_influence",
    "drift_available",
]


@dataclass(frozen=True)
class SafetyThreshold:
    threshold: float
    selected: int
    unsafe: int
    coverage: float
    observed_unsafe_rate: float
    unsafe_rate_upper95: float


def wilson_upper_bound(failures: int, total: int, z: float = 1.959963984540054) -> float:
    """Upper endpoint of a two-sided 95% Wilson interval for a failure rate."""
    if total < 0 or failures < 0 or failures > total:
        raise ValueError("failures and total must satisfy 0 <= failures <= total")
    if total == 0:
        return 1.0
    rate = failures / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return min(1.0, (center + radius) / denominator)


def select_safety_threshold(
    probabilities: Sequence[float] | np.ndarray,
    safe_labels: Sequence[int] | np.ndarray,
    *,
    max_observed_unsafe_rate: float = 0.0,
    minimum_selected: int = 8,
) -> SafetyThreshold:
    """Choose the highest-coverage threshold satisfying a preregistered risk cap.

    The threshold is chosen on a calibration split only. If no non-empty set
    satisfies the rule, a threshold above one disables learned extra commits;
    the ordinary denoising transfer schedule still guarantees progress.
    """
    scores = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(safe_labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("probabilities and labels must be aligned vectors")
    if not 0.0 <= max_observed_unsafe_rate <= 1.0:
        raise ValueError("unsafe-rate cap must be in [0, 1]")
    if minimum_selected < 1:
        raise ValueError("minimum_selected must be positive")
    clean = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[clean]
    labels = labels[clean]
    if not len(scores):
        return SafetyThreshold(1.000001, 0, 0, 0.0, 0.0, 1.0)

    candidates = np.unique(scores)
    candidates.sort()
    best: SafetyThreshold | None = None
    for threshold in candidates:
        selected = scores >= threshold
        count = int(selected.sum())
        if count < minimum_selected:
            continue
        unsafe = int((labels[selected] == 0).sum())
        unsafe_rate = unsafe / count
        if unsafe_rate > max_observed_unsafe_rate:
            continue
        record = SafetyThreshold(
            threshold=float(threshold),
            selected=count,
            unsafe=unsafe,
            coverage=float(count / len(scores)),
            observed_unsafe_rate=float(unsafe_rate),
            unsafe_rate_upper95=wilson_upper_bound(unsafe, count),
        )
        if best is None or record.selected > best.selected:
            best = record
    return best or SafetyThreshold(1.000001, 0, 0, 0.0, 0.0, 1.0)


class CommitmentSafetyScorer:
    """Load a calibrated direct-safety classifier and its deployment threshold."""

    def __init__(self, path: str | Path):
        from joblib import load

        bundle: dict[str, Any] = load(path)
        self.pipeline = bundle["pipeline"]
        self.calibrator = bundle.get("calibrator")
        self.features = list(bundle["features"])
        self.threshold = float(bundle["threshold"])
        self.target = str(bundle.get("target", "sequence_safe"))
        scaler = self.pipeline.steps[0][1]
        classifier = self.pipeline.steps[-1][1]
        base_weights = np.asarray(classifier.coef_[0], dtype=np.float64) / np.asarray(
            scaler.scale_, dtype=np.float64
        )
        base_bias = float(classifier.intercept_[0]) - float(
            np.dot(base_weights, np.asarray(scaler.mean_, dtype=np.float64))
        )
        if self.calibrator is None:
            calibration_weight = 1.0
            calibration_bias = 0.0
        else:
            calibration_weight = float(self.calibrator.coef_[0, 0])
            calibration_bias = float(self.calibrator.intercept_[0])
        self.linear_weights = base_weights * calibration_weight
        self.linear_bias = base_bias * calibration_weight + calibration_bias

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        raw = self.pipeline.decision_function(values).reshape(-1, 1)
        if self.calibrator is None:
            return 1.0 / (1.0 + np.exp(-raw[:, 0]))
        return self.calibrator.predict_proba(raw)[:, 1]

    def score_tensors(self, features: dict[str, Any]) -> Any:
        """Evaluate the calibrated linear model without a CPU synchronization."""
        import torch

        missing = [name for name in self.features if name not in features]
        if missing:
            raise ValueError(f"missing commitment-safety tensors: {missing}")
        shape = features[self.features[0]].shape
        if any(features[name].shape != shape for name in self.features):
            raise ValueError("commitment-safety feature tensors must align")
        matrix = torch.stack(
            [features[name].float() for name in self.features], dim=-1
        )
        weights = torch.as_tensor(
            self.linear_weights,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        return torch.sigmoid(matrix.matmul(weights) + float(self.linear_bias))
