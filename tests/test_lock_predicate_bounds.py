from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_lock_predicate_audit import (
    RULES,
    evaluate_rule,
    rule_less_jsd_persistence,
    token_layer_geometry,
    token_posteriors,
)


def _synthetic_trajectory(seed: int = 0, n_steps: int = 8, dim: int = 16):
    """Build real hidden vectors, then the exact columns the audit would log.

    Returns (time_rows frame, hidden matrix including the base row at index 0).
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    rows = [base]
    for _ in range(n_steps):
        rows.append(rows[-1] + rng.normal(scale=0.3, size=dim))
    hidden = np.stack(rows)  # index 0 is the base row b

    base_norm = np.linalg.norm(base)
    records = []
    for j in range(1, hidden.shape[0]):
        records.append(
            {
                "example_id": "ex1",
                "position": 70,
                "layer": 24,
                "base_step": 1,
                "global_step": j + 1,
                "relative_l2_drift": np.linalg.norm(hidden[j] - base) / base_norm,
                "step_l2_drift": np.linalg.norm(hidden[j] - hidden[j - 1]) / np.linalg.norm(hidden[j - 1]),
                "row_norm": np.linalg.norm(hidden[j]),
                "base_norm": base_norm,
            }
        )
    return pd.DataFrame(records), hidden


def _posterior_frame(n_steps: int, kl: list[float] | None = None) -> pd.DataFrame:
    kl = kl if kl is not None else [1.0] * n_steps
    return pd.DataFrame(
        [
            {
                "example_id": "ex1",
                "position": 70,
                "global_step": j + 2,
                "committed_token": 42,
                "post_kl": kl[j],
                "post_jsd": kl[j] / 2,
                "post_confidence": 0.95,
                "raw_top1_matches_commit": 1,
            }
            for j in range(n_steps)
        ]
    )


def test_bounds_bracket_true_post_lock_movement():
    """The reported interval must contain the movement actually realized.

    This is the property the whole audit rests on: the lower bound may only
    convict and the upper bound may only acquit, so a bound on the wrong side
    of the truth would silently invert a conclusion.
    """
    n_steps = 8
    time_rows, hidden = _synthetic_trajectory(seed=3, n_steps=n_steps)
    geometry = token_layer_geometry(time_rows)
    geo_key = ("ex1", 70, 24)

    for lock_idx in range(n_steps - 1):
        # Lock at observed index lock_idx, i.e. hidden row lock_idx + 1.
        kl = [1.0] * n_steps
        kl[lock_idx] = 0.0
        post = _posterior_frame(n_steps, kl)
        per_token = evaluate_rule(
            token_posteriors(post), geometry, [24], RULES["surelock_kl"][0], tau=0.5
        )
        row = per_token.iloc[0]
        assert row["locked"] == 1

        h_lock = hidden[lock_idx + 1]
        true_max = max(
            np.linalg.norm(hidden[j] - h_lock) / np.linalg.norm(h_lock)
            for j in range(lock_idx + 2, hidden.shape[0])
        )
        lower = row["lower_bound_vs_reference"]
        upper = row["upper_bound_vs_reference"]
        assert lower <= true_max + 1e-9, f"lower bound exceeded truth at lock {lock_idx}"
        assert upper >= true_max - 1e-9, f"upper bound below truth at lock {lock_idx}"
        assert lower <= upper + 1e-9

    _ = geometry[geo_key]


def test_lock_at_first_step_reproduces_max_drift():
    """earliest_logged_lock: the bound must span the full excursion from its lock."""
    n_steps = 6
    time_rows, hidden = _synthetic_trajectory(seed=11, n_steps=n_steps)
    geometry = token_layer_geometry(time_rows)
    post = _posterior_frame(n_steps)
    per_token = evaluate_rule(
        token_posteriors(post), geometry, [24], RULES["earliest_logged_lock"][0], tau=0.0
    )
    row = per_token.iloc[0]
    h_lock = hidden[1]
    true_max = max(
        np.linalg.norm(hidden[j] - h_lock) / np.linalg.norm(h_lock) for j in range(2, hidden.shape[0])
    )
    assert row["lower_bound_vs_reference"] <= true_max + 1e-9 <= row["upper_bound_vs_reference"] + 1e-9
    assert row["frozen_steps"] == n_steps - 1


def test_unfired_rule_is_counted_but_not_scored():
    """A rule that never admits a lock saves nothing and must not be credited."""
    n_steps = 5
    time_rows, _ = _synthetic_trajectory(seed=5, n_steps=n_steps)
    geometry = token_layer_geometry(time_rows)
    post = _posterior_frame(n_steps, kl=[9.0] * n_steps)
    per_token = evaluate_rule(
        token_posteriors(post), geometry, [24], RULES["surelock_kl"][0], tau=0.001
    )
    row = per_token.iloc[0]
    assert row["locked"] == 0
    assert row["frozen_steps"] == 0
    assert not np.isfinite(row["lower_bound_vs_reference"])


def test_persistence_requires_consecutive_agreement():
    """LESS-style persistence must not fire on a single agreeing step."""
    post = {
        "global_step": np.arange(5),
        "post_jsd": np.zeros(5),
        "post_confidence": np.full(5, 0.99),
        "raw_top1_matches_commit": np.array([1, 0, 1, 1, 0]),
    }
    fires = rule_less_jsd_persistence(post, tau=1.0)
    # Runs of agreement are length 1 at idx 0, then 1,2 at idx 2,3.
    assert not fires[0]
    assert not fires[1]
    assert fires[3]
    assert not fires[4]


if __name__ == "__main__":
    test_bounds_bracket_true_post_lock_movement()
    test_lock_at_first_step_reproduces_max_drift()
    test_unfired_rule_is_counted_but_not_scored()
    test_persistence_requires_consecutive_agreement()
    print("ok")
