from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_surelock_premise import (
    cluster_bootstrap_metrics,
    metric_functions,
    report_copy,
    validate_less_replay_prefix,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "example_id": "a",
                "non_target_token_changed": True,
                "non_target_token_change_count": 4,
                "extracted_answer_changed_regraded": True,
                "answer_value_changed_regraded": True,
                "baseline_correct_regraded": True,
                "modified_correct_regraded": False,
                "correctness_changed_regraded": True,
                "correct_to_wrong_regraded": True,
                "wrong_to_correct_regraded": False,
                "baseline_scoring_status_regraded": "passed",
                "modified_scoring_status_regraded": "wrong_answer",
            },
            {
                "example_id": "a",
                "non_target_token_changed": False,
                "non_target_token_change_count": 0,
                "extracted_answer_changed_regraded": False,
                "answer_value_changed_regraded": False,
                "baseline_correct_regraded": True,
                "modified_correct_regraded": True,
                "correctness_changed_regraded": False,
                "correct_to_wrong_regraded": False,
                "wrong_to_correct_regraded": False,
                "baseline_scoring_status_regraded": "passed",
                "modified_scoring_status_regraded": "passed",
            },
            {
                "example_id": "b",
                "non_target_token_changed": True,
                "non_target_token_change_count": 2,
                "extracted_answer_changed_regraded": True,
                "answer_value_changed_regraded": True,
                "baseline_correct_regraded": False,
                "modified_correct_regraded": True,
                "correctness_changed_regraded": True,
                "correct_to_wrong_regraded": False,
                "wrong_to_correct_regraded": True,
                "baseline_scoring_status_regraded": "wrong_answer",
                "modified_scoring_status_regraded": "passed",
            },
        ]
    )


def test_metric_surface_reports_accuracy_and_both_change_notions():
    metrics = {name: function(_rows()) for name, function in metric_functions().items()}
    assert metrics["extracted_answer_change_rate"] == 2 / 3
    assert metrics["answer_value_change_rate"] == 2 / 3
    assert metrics["original_accuracy"] == 2 / 3
    assert metrics["original_unique_example_accuracy"] == 0.5
    assert metrics["modified_accuracy"] == 2 / 3
    assert metrics["accuracy_delta"] == 0.0
    assert metrics["correctness_change_rate"] == 2 / 3
    assert metrics["correct_to_wrong_rate"] == 1 / 3
    assert metrics["wrong_to_correct_rate"] == 1 / 3


def test_cluster_bootstrap_is_seeded_and_bounded():
    first = cluster_bootstrap_metrics(_rows(), reps=200, seed=7)
    second = cluster_bootstrap_metrics(_rows(), reps=200, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["ci_low"].between(-1, 4).all()
    assert first["ci_high"].between(-1, 4).all()


def test_accuracy_excludes_sandbox_failures_but_counts_program_timeouts():
    rows = _rows()
    rows.loc[0, "baseline_scoring_status_regraded"] = "sandbox_error"
    rows.loc[2, "modified_scoring_status_regraded"] = "timeout"
    rows.loc[2, "modified_correct_regraded"] = False
    metrics = {name: function(rows) for name, function in metric_functions().items()}
    assert metrics["original_accuracy"] == 0.5
    assert metrics["modified_accuracy"] == 0.5
    assert metrics["original_scoring_infrastructure_failure_rate"] == 1 / 3
    assert metrics["modified_scoring_infrastructure_failure_rate"] == 0.0
    assert metrics["modified_execution_timeout_rate"] == 1 / 3


def test_less_replay_validation_accepts_explicit_true_values():
    validate_less_replay_prefix(pd.DataFrame({"pre_intervention_replay_valid": [True, "true", 1]}))


@pytest.mark.parametrize(
    ("value", "message"),
    [(False, "1 invalid"), ("not-checked", "1 malformed")],
)
def test_less_replay_validation_rejects_false_and_malformed_values(value, message):
    with pytest.raises(ValueError, match=message):
        validate_less_replay_prefix(
            pd.DataFrame({"pre_intervention_replay_valid": [True, value]})
        )


def test_report_copy_describes_the_actual_intervention_mode():
    less_title, less_description, less_heading = report_copy("less_premise")
    assert "LESS" in less_title
    assert "LESS-admitted" in less_description
    assert "Programs" in less_heading

    surelock_title, surelock_description, surelock_heading = report_copy("surelock_premise")
    assert "SureLock" in surelock_title
    assert "SureLock-style lock step" in surelock_description
    assert "Mathematically" in surelock_heading
