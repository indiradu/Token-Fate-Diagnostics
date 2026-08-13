from __future__ import annotations

import pathlib
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_counterfactual_commit import (
    _strict_same_trajectory_high_confidence_control,
    choose_selector_candidates,
    counterfactual_token_effects,
)
from analyze_token_fate_superiority import causal_comparisons


def test_surface_selectors_choose_their_best_early_row_per_example() -> None:
    metadata = pd.DataFrame(
        [
            {"example_id": "a", "global_step": 0, "position": 10, "block_t_frac": 0.25, "confidence": 0.9, "entropy": 0.1, "margin": 0.8},
            {"example_id": "a", "global_step": 1, "position": 11, "block_t_frac": 0.50, "confidence": 0.2, "entropy": 2.0, "margin": 0.1},
            {"example_id": "a", "global_step": 2, "position": 12, "block_t_frac": 0.75, "confidence": 0.01, "entropy": 4.0, "margin": 0.0},
            {"example_id": "b", "global_step": 0, "position": 20, "block_t_frac": 0.25, "confidence": 0.7, "entropy": 0.3, "margin": 0.6},
            {"example_id": "b", "global_step": 1, "position": 21, "block_t_frac": 0.50, "confidence": 0.4, "entropy": 1.5, "margin": 0.2},
        ]
    )

    low_conf = choose_selector_candidates(metadata, "low_confidence", max_candidates=10)
    high_entropy = choose_selector_candidates(metadata, "high_entropy", max_candidates=10)
    low_margin = choose_selector_candidates(metadata, "low_margin", max_candidates=10)

    assert set(low_conf["example_id"]) == {"a", "b"}
    assert low_conf.set_index("example_id").loc["a", "position"] == 11
    assert high_entropy.set_index("example_id").loc["b", "position"] == 21
    assert low_margin.set_index("example_id").loc["a", "position"] == 11
    assert set(high_entropy["candidate_group"]) == {"high_entropy"}


def test_strict_same_trajectory_control_matches_same_step_confidence_and_position() -> None:
    rows = pd.DataFrame(
        [
            {"example_id": "a", "global_step": 5, "position": 110, "relative_position": 10, "confidence": 0.800, "candidate_score": 0.90, "candidate_row_id": "target"},
            {"example_id": "a", "global_step": 5, "position": 112, "relative_position": 12, "confidence": 0.803, "candidate_score": 0.10, "candidate_row_id": "good"},
            {"example_id": "a", "global_step": 4, "position": 111, "relative_position": 11, "confidence": 0.801, "candidate_score": 0.01, "candidate_row_id": "wrong_step"},
            {"example_id": "a", "global_step": 5, "position": 120, "relative_position": 20, "confidence": 0.801, "candidate_score": 0.01, "candidate_row_id": "wrong_position"},
            {"example_id": "a", "global_step": 5, "position": 111, "relative_position": 11, "confidence": 0.810, "candidate_score": 0.01, "candidate_row_id": "wrong_confidence"},
        ]
    )
    match = _strict_same_trajectory_high_confidence_control(
        rows.iloc[0],
        rows,
        used={"target"},
        confidence_caliper=0.005,
        relative_position_caliper=4,
    )

    assert match is not None
    assert match["candidate_row_id"] == "good"
    assert match["match_scope"] == "same_trajectory_same_step_strict"
    assert match["match_global_step_abs_diff"] == 0


def test_downstream_effects_exclude_the_forced_position() -> None:
    effects = counterfactual_token_effects(
        forced_final_tokens=[9, 2, 8, 4],
        baseline_final_tokens=[1, 2, 3, 4],
        target_relative_position=0,
    )

    assert effects["target_token_changed"]
    assert effects["non_target_token_changed"]
    assert effects["non_target_token_change_count"] == 1
    assert effects["suffix_token_changed"]
    assert effects["suffix_token_change_count"] == 1


def test_same_trajectory_causal_bootstrap_uses_candidate_pair_ids() -> None:
    rows = pd.DataFrame(
        [
            {
                "example_id": "a",
                "candidate_group": "learned_high_confidence",
                "candidate_row_id": "risk-a",
                "pair_id": "risk-a",
                "final_token_changed": True,
                "non_target_token_changed": True,
                "non_target_token_change_count": 2,
                "answer_correct_changed": False,
            },
            {
                "example_id": "b",
                "candidate_group": "learned_high_confidence",
                "candidate_row_id": "risk-b",
                "pair_id": "risk-b",
                "final_token_changed": True,
                "non_target_token_changed": True,
                "non_target_token_change_count": 1,
                "answer_correct_changed": True,
            },
            {
                "example_id": "a",
                "candidate_group": "matched_same_trajectory_high_conf_low_risk",
                "candidate_row_id": "control-a",
                "pair_id": "risk-a",
                "final_token_changed": False,
                "non_target_token_changed": False,
                "non_target_token_change_count": 0,
                "answer_correct_changed": False,
            },
            {
                "example_id": "b",
                "candidate_group": "matched_same_trajectory_high_conf_low_risk",
                "candidate_row_id": "control-b",
                "pair_id": "risk-b",
                "final_token_changed": False,
                "non_target_token_changed": False,
                "non_target_token_change_count": 0,
                "answer_correct_changed": False,
            },
        ]
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path = pathlib.Path(temp_dir) / "causal.csv"
        rows.to_csv(path, index=False)
        result = causal_comparisons(
            path,
            bootstrap_samples=100,
            seed=23,
            reference_group="learned_high_confidence",
            pairing="paired",
        )

    downstream = result[result["metric"].eq("non_target_token_changed")].iloc[0]
    assert downstream["comparison_design"] == "same_trajectory_paired"
    assert downstream["reference_examples"] == 2
    assert downstream["delta_learned_minus_comparison"] == 1.0


if __name__ == "__main__":
    test_surface_selectors_choose_their_best_early_row_per_example()
    test_strict_same_trajectory_control_matches_same_step_confidence_and_position()
    test_downstream_effects_exclude_the_forced_position()
    test_same_trajectory_causal_bootstrap_uses_candidate_pair_ids()
    print("counterfactual selector tests passed")
