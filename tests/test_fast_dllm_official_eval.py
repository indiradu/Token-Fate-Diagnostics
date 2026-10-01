from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fast_dllm_official_eval import (
    ARM_SPECS,
    comparison_metrics,
    resolve_steps,
    summarize,
    validate_schedule,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixed_full_correct": True,
                "fixed_full_nfe": 64,
                "fixed_full_input_token_positions": 6400,
                "fixed_full_latency_s": 4.0,
                "fixed_full_final_mask_count": 0,
                "dual_parallel_correct": False,
                "dual_parallel_nfe": 20,
                "dual_parallel_input_token_positions": 1600,
                "dual_parallel_latency_s": 1.0,
                "dual_parallel_final_mask_count": 0,
                "dual_parallel_output_changed_vs_fixed_full": True,
                "dual_parallel_answer_value_changed_vs_fixed_full": True,
            },
            {
                "fixed_full_correct": False,
                "fixed_full_nfe": 64,
                "fixed_full_input_token_positions": 6400,
                "fixed_full_latency_s": 4.0,
                "fixed_full_final_mask_count": 0,
                "dual_parallel_correct": True,
                "dual_parallel_nfe": 24,
                "dual_parallel_input_token_positions": 1920,
                "dual_parallel_latency_s": 2.0,
                "dual_parallel_final_mask_count": 0,
                "dual_parallel_output_changed_vs_fixed_full": True,
                "dual_parallel_answer_value_changed_vs_fixed_full": False,
            },
        ]
    )


def test_official_step_schedules_are_explicit() -> None:
    assert resolve_steps(ARM_SPECS["fixed_full"], 256, 32) == 256
    assert resolve_steps(ARM_SPECS["parallel_full"], 256, 32) == 8
    assert resolve_steps(ARM_SPECS["dual_parallel"], 256, 32) == 256


def test_schedule_validation() -> None:
    validate_schedule(256, 32)
    with pytest.raises(ValueError, match="divisible"):
        validate_schedule(65, 32)


def test_comparison_metrics_separate_output_answer_and_correctness() -> None:
    metrics = comparison_metrics(_rows(), "fixed_full", "dual_parallel")
    assert metrics["output_change_rate"] == 1.0
    assert metrics["answer_value_change_rate"] == 0.5
    assert metrics["reference_accuracy"] == 0.5
    assert metrics["candidate_accuracy"] == 0.5
    assert metrics["correct_to_wrong_rate"] == 0.5
    assert metrics["wrong_to_correct_rate"] == 0.5
    assert metrics["relative_input_token_positions"] == 0.275
    assert metrics["wall_clock_speedup"] == pytest.approx(8 / 3)


def test_summary_includes_arm_and_pair_views() -> None:
    output = summarize(_rows(), ["fixed_full", "dual_parallel"])
    assert output["examples"] == 2
    assert output["arms"]["dual_parallel"]["mean_nfe"] == 22
    assert output["comparisons"]["fixed_full->dual_parallel"]["accuracy_delta"] == 0
