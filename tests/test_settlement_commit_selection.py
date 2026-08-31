from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from regret_remasking.llada_trace import EOS_ID
from run_settlement_fate_audit import choose_commit_candidates


def _proposal_row(example_id, position, step, top_token, confidence):
    return {
        "example_id": example_id,
        "dataset": "math500",
        "position": position,
        "relative_position": position - 100,
        "global_step": step,
        "top_token": top_token,
        "confidence": confidence,
    }


def _drift_row(example_id, position, selected_step, token):
    return {
        "example_id": example_id,
        "position": position,
        "relative_position": position - 100,
        "selected_step": selected_step,
        "selected_token": token,
        "final_token": token,
        "semantic_safe": 1,
    }


def _build_frames():
    # Position 110: stable from step 0, committed at step 10 -> lead 10 target,
    # late control at step 9. Confidence 0.9 at the earliest stable step.
    # Position 111: stable only from step 8, committed at step 10 -> max lead 2,
    # below min_lead 4, so it must not be selected.
    # Position 112: EOS token, stable from step 0, committed at step 12 ->
    # excluded when exclude_eos_eot is on.
    # Position 113: flickers (stable at 0, unstable 1-5, stable 6+), committed
    # at step 12 -> eligible with lead 12; stable_run_fraction < 1.
    proposals = []
    for step in range(10):
        proposals.append(_proposal_row("a", 110, step, 555, 0.9))
    for step in range(10):
        proposals.append(_proposal_row("a", 111, step, 666 if step >= 8 else 42, 0.95))
    for step in range(12):
        proposals.append(_proposal_row("a", 112, step, EOS_ID, 0.99))
    for step in range(12):
        stable = step == 0 or step >= 6
        proposals.append(_proposal_row("a", 113, step, 777 if stable else 42, 0.7))
    drift = [
        _drift_row("a", 110, 10, 555),
        _drift_row("a", 111, 10, 666),
        _drift_row("a", 112, 12, EOS_ID),
        _drift_row("a", 113, 12, 777),
    ]
    return pd.DataFrame(proposals), pd.DataFrame(drift)


def test_commit_candidates_require_stability_lead_and_exclude_eos() -> None:
    proposal_df, drift_df = _build_frames()
    candidates = choose_commit_candidates(
        proposal_df,
        drift_df,
        per_example=5,
        min_lead=4,
        min_confidence=0.5,
        exclude_eos_eot=True,
    )
    positions = set(candidates["position"].astype(int))
    assert 111 not in positions, "lead below min_lead must be excluded"
    assert 112 not in positions, "EOS tokens must be excluded"
    assert positions == {110, 113}

    by_role = candidates.set_index(["position", "match_role"])
    target_110 = by_role.loc[(110, "target")]
    control_110 = by_role.loc[(110, "control")]
    assert int(target_110["forced_step"]) == 0
    assert int(target_110["commit_lead"]) == 10
    assert int(control_110["forced_step"]) == 9
    assert int(control_110["forced_step"]) > int(target_110["forced_step"])
    assert int(control_110["forced_step"]) < int(control_110["selected_step"])
    assert int(target_110["forced_token"]) == 555

    # Flickering position: earliest eligible stable step is 0 (stable, lead 12),
    # and the stable-run fraction over [0, 12) reflects the unstable middle.
    target_113 = by_role.loc[(113, "target")]
    assert int(target_113["forced_step"]) == 0
    assert 0.5 < float(target_113["stable_run_fraction"]) < 1.0

    # Pairs share pair_id and both roles are present for every pair.
    for _, pair in candidates.groupby("pair_id"):
        assert set(pair["match_role"]) == {"target", "control"}


def test_commit_candidates_rank_by_confidence_and_cap_per_example() -> None:
    proposal_df, drift_df = _build_frames()
    candidates = choose_commit_candidates(
        proposal_df,
        drift_df,
        per_example=1,
        min_lead=4,
        min_confidence=0.5,
        exclude_eos_eot=True,
    )
    # Only one pair allowed; position 110 (confidence 0.9) outranks 113 (0.7).
    assert set(candidates["position"].astype(int)) == {110}
    assert len(candidates) == 2


if __name__ == "__main__":
    test_commit_candidates_require_stability_lead_and_exclude_eos()
    test_commit_candidates_rank_by_confidence_and_cap_per_example()
    print("ok")
