from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_lock_admission_bound import EOS_ID, oracle_curve, token_bounds


def test_oracle_curve_matches_hand_computation():
    """Reverse-triangle bound and its prefix minimum, worked by hand.

    drifts 0 -> 0.5 -> 0.2 -> 0.9 at steps 1..4. Locking at commit must
    reproduce the max drift (0.9); the best lock retaining half the horizon
    is step 2, bounding later movement at |0.9 - 0.5| = 0.4.
    """
    steps = np.array([1, 2, 3, 4])
    drifts = np.array([0.0, 0.5, 0.2, 0.9])
    budgets, prefix_lb = oracle_curve(steps, drifts)

    np.testing.assert_allclose(budgets, [1.0, 2 / 3, 1 / 3])
    np.testing.assert_allclose(prefix_lb, [0.9, 0.4, 0.4])


def test_bound_is_nonnegative_even_when_drift_decreases():
    """A monotonically settling trajectory still yields a positive bound."""
    steps = np.array([0, 1, 2, 3])
    drifts = np.array([0.0, 0.9, 0.5, 0.1])
    _, prefix_lb = oracle_curve(steps, drifts)
    assert (prefix_lb >= 0).all()
    # Locking at commit must see the full excursion up to 0.9.
    assert prefix_lb[0] == 0.9


def test_final_step_is_never_a_lock_candidate():
    """Locking at the last observed step would give a free zero bound."""
    steps = np.array([0, 1, 2])
    drifts = np.array([0.0, 0.3, 0.7])
    budgets, prefix_lb = oracle_curve(steps, drifts)
    assert budgets.size == steps.size - 1
    assert prefix_lb.min() > 0


def test_lock_at_commit_equals_max_drift_from_commit():
    """f = 1.0 must reproduce premise A's max-drift statistic exactly."""
    rows = []
    for step, drift in zip([2, 3, 4, 5], [0.2, 0.6, 0.45, 0.8]):
        rows.append(
            {
                "example_id": "ex1",
                "position": 70,
                "layer": 24,
                "base_step": 1,
                "global_step": step,
                "relative_l2_drift": drift,
            }
        )
    bounds = token_bounds(pd.DataFrame(rows))
    assert bounds.shape[0] == 1
    row = bounds.iloc[0]
    assert row["max_drift_from_commit"] == 0.8
    assert row["oracle_lb_f1.0"] == 0.8
    # Delaying the lock can only help, never hurt.
    assert row["oracle_lb_f0.5"] <= row["oracle_lb_f1.0"]
    assert row["oracle_lb_f0.1"] <= row["oracle_lb_f0.5"]


def test_eos_id_matches_llada_trace():
    from regret_remasking.llada_trace import EOS_ID as TRACE_EOS

    assert EOS_ID == TRACE_EOS


if __name__ == "__main__":
    test_oracle_curve_matches_hand_computation()
    test_bound_is_nonnegative_even_when_drift_decreases()
    test_final_step_is_never_a_lock_candidate()
    test_lock_at_commit_equals_max_drift_from_commit()
    test_eos_id_matches_llada_trace()
    print("ok")
