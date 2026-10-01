from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fast_dllm_official import (
    add_derived_pair_columns,
    bootstrap_metrics,
    pair_counts,
    validate_run_artifact,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixed_full_correct": True,
                "fixed_full_nfe": 100,
                "fixed_full_input_token_positions": 1000,
                "fixed_full_latency_s": 4.0,
                "dual_parallel_correct": False,
                "dual_parallel_nfe": 40,
                "dual_parallel_input_token_positions": 200,
                "dual_parallel_latency_s": 1.0,
                "dual_parallel_output_changed_vs_fixed_full": "True",
                "dual_parallel_answer_value_changed_vs_fixed_full": "True",
            },
            {
                "fixed_full_correct": False,
                "fixed_full_nfe": 100,
                "fixed_full_input_token_positions": 1000,
                "fixed_full_latency_s": 4.0,
                "dual_parallel_correct": True,
                "dual_parallel_nfe": 60,
                "dual_parallel_input_token_positions": 400,
                "dual_parallel_latency_s": 2.0,
                "dual_parallel_output_changed_vs_fixed_full": "False",
                "dual_parallel_answer_value_changed_vs_fixed_full": "False",
            },
        ]
    )


def test_pair_counts_report_both_correctness_directions() -> None:
    counts = pair_counts(_rows(), "fixed_full", "dual_parallel")
    assert counts["output_changed"] == 1
    assert counts["answer_value_changed"] == 1
    assert counts["correct_to_wrong"] == 1
    assert counts["wrong_to_correct"] == 1
    assert counts["correctness_exact_sign_p"] == 1.0


def test_bootstrap_is_seeded_and_includes_work_metrics() -> None:
    first = bootstrap_metrics(_rows(), reps=100, seed=7)
    second = bootstrap_metrics(_rows(), reps=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    points = first.set_index("metric")["point"]
    assert points["relative_input_token_positions"] == 0.3
    assert points["wall_clock_speedup"] == 8 / 3


def test_missing_pair_columns_are_derived_from_saved_outputs() -> None:
    rows = pd.DataFrame(
        {
            "dataset": ["math500"],
            "fixed_full_token_ids": ["[1, 2]"],
            "prefix_fixed_token_ids": ["[1, 3]"],
            "parallel_full_token_ids": ["[1, 2]"],
            "prefix_parallel_token_ids": ["[1, 3]"],
            "dual_parallel_token_ids": ["[1, 4]"],
            "fixed_full_generation": [r"answer is \\boxed{2}"],
            "prefix_fixed_generation": [r"answer is \\boxed{3}"],
            "parallel_full_generation": [r"answer is \\boxed{2}"],
            "prefix_parallel_generation": [r"answer is \\boxed{3}"],
            "dual_parallel_generation": [r"answer is \\boxed{4}"],
        }
    )
    derived = add_derived_pair_columns(rows)
    assert bool(derived.loc[0, "prefix_parallel_output_changed_vs_prefix_fixed"]) is False
    assert bool(derived.loc[0, "prefix_parallel_answer_value_changed_vs_prefix_fixed"]) is False
    assert bool(derived.loc[0, "dual_parallel_output_changed_vs_prefix_parallel"]) is True


def test_pair_derivation_accepts_reduced_arm_sets() -> None:
    rows = pd.DataFrame(
        {
            "dataset": ["math500"],
            "fixed_full_token_ids": ["[1]"],
            "parallel_full_token_ids": ["[2]"],
            "fixed_full_generation": [r"\\boxed{1}"],
            "parallel_full_generation": [r"\\boxed{2}"],
        }
    )
    derived = add_derived_pair_columns(rows)
    assert bool(derived.loc[0, "parallel_full_output_changed_vs_fixed_full"]) is True


def test_run_validation_rejects_partial_artifacts() -> None:
    rows = pd.DataFrame(
        {
            "example_id": ["a"],
            "dataset": ["math500"],
            "fixed_full_token_ids": ["[1, 2]"],
            "fixed_full_generation": [r"\\boxed{2}"],
            "fixed_full_correct": [True],
            "fixed_full_nfe": [2],
            "fixed_full_forward_calls": [2],
            "fixed_full_final_mask_count": [0],
        }
    )
    config = {"limit": 2, "gen_length": 2, "arms": ["fixed_full"]}
    try:
        validate_run_artifact(rows, config)
    except ValueError as error:
        assert "partial run" in str(error)
    else:
        raise AssertionError("partial run was accepted")


def test_run_validation_accepts_complete_artifact() -> None:
    rows = pd.DataFrame(
        {
            "example_id": ["a"],
            "dataset": ["math500"],
            "fixed_full_token_ids": ["[1, 2]"],
            "fixed_full_generation": [r"\\boxed{2}"],
            "fixed_full_correct": [True],
            "fixed_full_nfe": [2],
            "fixed_full_forward_calls": [2],
            "fixed_full_final_mask_count": [0],
        }
    )
    config = {"limit": 1, "gen_length": 2, "arms": ["fixed_full"]}
    assert validate_run_artifact(rows, config)["total_outputs"] == 1
