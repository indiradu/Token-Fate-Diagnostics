from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from regret_remasking.llada_trace import EOS_ID
from run_settlement_fate_audit import (
    choose_less_premise_candidates,
    choose_polestar_cache_proxy_candidates,
    choose_rule_lock_candidates,
    choose_surelock_premise_candidates,
    validate_polestar_event_retention,
)


def _row(
    position,
    base_step,
    lock_step,
    token=1000,
    semantic_safe=1,
    future=5,
    drift=0.4,
    commit_source="confidence_schedule",
):
    return {
        "example_id": "ex1",
        "dataset": "math500",
        "position": position,
        "relative_position": position - 100,
        "selected_step": base_step - 1,
        "base_step": base_step,
        "lock_step": lock_step,
        "selected_token": token,
        "final_token": token,
        "semantic_safe": semantic_safe,
        "selected_confidence": 0.9,
        "selected_entropy": 0.2,
        "selected_margin": 0.8,
        "selected_kl": 0.0,
        "future_steps_observed": future,
        "baseline_nfe": 64,
        "max_drift_mean": drift,
        "final_drift_mean": drift,
        "commit_source": commit_source,
        "selected_less_jsd": 0.01,
        "selected_less_persistence": 1,
    }


def test_pair_is_within_token_with_two_freeze_times():
    """Both arms must be the same position: that is what makes the pair tight."""
    df = pd.DataFrame([_row(110, base_step=3, lock_step=9)])
    out = choose_rule_lock_candidates(df, per_example=1)
    assert out.shape[0] == 2

    target = out[out.match_role == "target"].iloc[0]
    control = out[out.match_role == "control"].iloc[0]

    assert int(target["position"]) == int(control["position"]) == 110
    assert target["pair_id"] == control["pair_id"]
    # Target is the aggressive arm: freeze at commit against the commit-step row.
    assert int(target["freeze_step"]) == 3
    assert target["freeze_reference"] == "base"
    # Control is the gate's arm: freeze later, against the row cached there.
    assert int(control["freeze_step"]) == 9
    assert control["freeze_reference"] == "lock"
    assert int(control["lock_delay_steps"]) == 6


def test_zero_delay_tokens_are_dropped():
    """lock_step == base_step gives two identical arms and no contrast."""
    df = pd.DataFrame([_row(110, base_step=4, lock_step=4)])
    assert choose_rule_lock_candidates(df, per_example=3).empty


def test_tokens_the_rule_never_admitted_are_dropped():
    """No lock step means the gate never fired, so there is nothing to compare."""
    df = pd.DataFrame([_row(110, base_step=2, lock_step=float("nan"))])
    assert choose_rule_lock_candidates(df, per_example=3).empty


def test_semantically_unsafe_tokens_are_dropped():
    """The premise conditions on identity being fixed; keep that invariant."""
    df = pd.DataFrame([_row(110, base_step=2, lock_step=7, semantic_safe=0)])
    assert choose_rule_lock_candidates(df, per_example=3).empty


def test_eos_padding_excluded_when_requested():
    """Padding passes stability gates trivially while drifting ~2x body tokens."""
    df = pd.DataFrame(
        [
            _row(110, base_step=2, lock_step=7, token=EOS_ID),
            _row(111, base_step=2, lock_step=7, token=1234),
        ]
    )
    kept = choose_rule_lock_candidates(df, per_example=5, exclude_eos_eot=True)
    assert set(kept["position"].astype(int)) == {111}
    both = choose_rule_lock_candidates(df, per_example=5, exclude_eos_eot=False)
    assert set(both["position"].astype(int)) == {110, 111}


def test_sampling_is_seeded_and_not_drift_ranked():
    """Ranking candidates by drift would preselect the tokens most likely harmed.

    Selection must be reproducible for a given seed and must not simply return
    the highest-drift tokens.
    """
    rows = [_row(100 + i, base_step=2, lock_step=6, drift=i / 20.0) for i in range(20)]
    df = pd.DataFrame(rows)
    a = choose_rule_lock_candidates(df, per_example=4, seed=7)
    b = choose_rule_lock_candidates(df, per_example=4, seed=7)
    c = choose_rule_lock_candidates(df, per_example=4, seed=8)

    picked_a = sorted(a["position"].astype(int).unique())
    assert picked_a == sorted(b["position"].astype(int).unique())
    assert len(picked_a) == 4
    # The four highest-drift positions are 116-119; a seeded sample should not
    # coincide with them (this is what a drift-ranked selector would return).
    assert picked_a != [116, 117, 118, 119]
    assert sorted(c["position"].astype(int).unique()) != picked_a


def test_surelock_premise_candidate_is_yf_only_at_lock_step():
    """The clean premise mode compares SureLock YF against baseline-as-YC."""
    df = pd.DataFrame([_row(110, base_step=3, lock_step=9)])
    out = choose_surelock_premise_candidates(df, per_example=1)
    assert out.shape[0] == 1

    row = out.iloc[0]
    assert row["intervention_group"] == "surelock_freeze_yf"
    assert row["match_role"] == "yf"
    assert int(row["position"]) == 110
    assert int(row["freeze_step"]) == 9
    assert row["freeze_reference"] == "lock"
    assert int(row["lock_delay_steps"]) == 6
    assert bool(row["y0_equals_yc_by_design"])
    assert row["yc_source"] == "baseline_recomputed_reference"


def test_less_premise_uses_only_rule_accepts_and_first_post_commit_row():
    df = pd.DataFrame(
        [
            _row(110, base_step=4, lock_step=float("nan"), commit_source="less_accept"),
            _row(111, base_step=5, lock_step=float("nan"), commit_source="less_fallback"),
        ]
    )
    out = choose_less_premise_candidates(df, per_example=2)
    assert out.shape[0] == 1
    row = out.iloc[0]
    assert int(row["position"]) == 110
    assert row["intervention_group"] == "less_freeze_yf"
    assert int(row["freeze_step"]) == 4
    assert row["freeze_reference"] == "base"
    assert row["yc_source"] == "less_baseline_recomputed_reference"
    assert row["commit_source"] == "less_accept"


def test_polestar_proxy_pairs_stale_and_refreshed_rows_at_one_event():
    drift_df = pd.DataFrame([_row(110, base_step=4, lock_step=float("nan"))])
    event_df = pd.DataFrame(
        [
            {
                "event_id": "ex1:1:12:110",
                "example_id": "ex1",
                "dataset": "math500",
                "position": 110,
                "relative_position": 10,
                "selected_token": 1000,
                "block": 1,
                "cache_step": 8,
                "refresh_step": 12,
                "cache_age": 4,
                "future_steps_after_refresh": 51,
                "attention_kl": 0.25,
                "refresh_rank": 1,
                "refresh_pool_size": 32,
                "refresh_drift_mean": 0.4,
            }
        ]
    )
    out = choose_polestar_cache_proxy_candidates(event_df, drift_df, per_example=1)
    assert out.shape[0] == 2
    stale = out[out.proxy_arm == "stale"].iloc[0]
    refresh = out[out.proxy_arm == "refresh"].iloc[0]
    assert stale["pair_id"] == refresh["pair_id"] == "ex1:1:12:110"
    assert int(stale["position"]) == int(refresh["position"]) == 110
    assert int(stale["freeze_step"]) == int(refresh["freeze_step"]) == 12
    assert stale["freeze_reference"] == "polestar_cache_stale"
    assert refresh["freeze_reference"] == "polestar_cache_refresh"
    assert stale["proxy_kind"] == "token_level_per_head_attention_kl_single_refresh"


def test_polestar_proxy_top_kl_selection_does_not_use_harm_outcomes():
    drift_df = pd.DataFrame([_row(110, 4, float("nan")), _row(111, 5, float("nan"))])
    event_df = pd.DataFrame(
        [
            {
                "event_id": f"event-{position}",
                "example_id": "ex1",
                "dataset": "math500",
                "position": position,
                "relative_position": position - 100,
                "selected_token": 1000,
                "block": 1,
                "cache_step": 8,
                "refresh_step": 12,
                "cache_age": 4,
                "future_steps_after_refresh": 51,
                "attention_kl": attention_kl,
                "refresh_rank": rank,
                "refresh_pool_size": 32,
                "refresh_drift_mean": 0.4,
            }
            for position, attention_kl, rank in [(110, 0.1, 2), (111, 0.5, 1)]
        ]
    )
    out = choose_polestar_cache_proxy_candidates(
        event_df,
        drift_df,
        per_example=1,
        selection="top_kl",
    )
    assert set(out["position"].astype(int)) == {111}
    assert "non_target_token_changed" not in event_df


def test_polestar_top_kl_rejects_a_truncated_event_reservoir():
    assert validate_polestar_event_retention("random", 100, 32)
    assert not validate_polestar_event_retention("top_kl", 32, 32)
    try:
        validate_polestar_event_retention("top_kl", 100, 32)
    except ValueError as error:
        assert "retained 32 of 100" in str(error)
    else:
        raise AssertionError("top_kl must reject a truncated event reservoir")


if __name__ == "__main__":
    test_pair_is_within_token_with_two_freeze_times()
    test_zero_delay_tokens_are_dropped()
    test_tokens_the_rule_never_admitted_are_dropped()
    test_semantically_unsafe_tokens_are_dropped()
    test_eos_padding_excluded_when_requested()
    test_sampling_is_seeded_and_not_drift_ranked()
    test_surelock_premise_candidate_is_yf_only_at_lock_step()
    test_less_premise_uses_only_rule_accepts_and_first_post_commit_row()
    print("ok")
