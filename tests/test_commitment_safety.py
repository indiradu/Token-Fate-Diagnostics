from __future__ import annotations

import numpy as np

from regret_remasking.commitment_safety import (
    select_safety_threshold,
    wilson_upper_bound,
)


def test_threshold_maximizes_coverage_under_zero_failure_rule() -> None:
    probabilities = np.asarray([0.99, 0.95, 0.90, 0.85, 0.80])
    safe = np.asarray([1, 1, 1, 0, 1])

    selected = select_safety_threshold(
        probabilities,
        safe,
        max_observed_unsafe_rate=0.0,
        minimum_selected=2,
    )

    assert selected.threshold == 0.90
    assert selected.selected == 3
    assert selected.unsafe == 0
    assert selected.coverage == 0.6


def test_threshold_fails_closed_without_enough_safe_calibration_rows() -> None:
    selected = select_safety_threshold(
        [0.9, 0.8, 0.7],
        [1, 0, 1],
        max_observed_unsafe_rate=0.0,
        minimum_selected=3,
    )
    assert selected.threshold > 1.0
    assert selected.selected == 0


def test_wilson_upper_bound_is_conservative_for_zero_failures() -> None:
    assert 0.0 < wilson_upper_bound(0, 20) < 0.2
    assert wilson_upper_bound(0, 100) < wilson_upper_bound(0, 20)

