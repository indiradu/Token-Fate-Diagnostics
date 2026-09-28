import pytest

from analyze_phaselock_trace_semantic import select_policy, stable_rows


def _result(policy: str, gain: float, *, eligible: bool = True) -> dict:
    return {
        "policy": policy,
        "eligible": eligible,
        "estimated_online_gain_a3_vs_a0": {"mean": gain},
    }


def test_policy_selection_uses_speed_outside_tie_band() -> None:
    assert select_policy(
        [
            _result("guarded", 0.10),
            _result("balanced", 0.12),
            _result("open", 0.15),
        ]
    ) == "open"


def test_policy_selection_prefers_conservative_within_one_point() -> None:
    assert select_policy(
        [
            _result("guarded", 0.141),
            _result("balanced", 0.145),
            _result("open", 0.15),
        ]
    ) == "guarded"


def test_policy_selection_never_selects_ineligible_policy() -> None:
    assert select_policy(
        [
            _result("guarded", 0.10, eligible=False),
            _result("balanced", 0.12, eligible=False),
        ]
    ) is None


def test_guarded_surelite_must_clear_baseline_tie_band() -> None:
    baseline = _result("open_confidence", 0.20)
    candidate = _result("trace_surelite_union_d02_c80_p2", 0.209)
    assert select_policy([baseline, candidate]) == "open_confidence"

    candidate["estimated_online_gain_a3_vs_a0"]["mean"] = 0.211
    assert (
        select_policy([baseline, candidate])
        == "trace_surelite_union_d02_c80_p2"
    )


def test_stable_rows_deduplicates_timing_repeats() -> None:
    row = {
        "example_id": "x",
        "a0_correct": 1,
        "a1_correct": 1,
        "a2_correct": 1,
        "a3_correct": 1,
        "a1_exact_a0": 0,
        "a2_exact_a1": 1,
        "a3_exact_a2": 1,
        "a1_answer_changed_a0": 0,
        "a2_answer_changed_a1": 0,
        "a3_answer_changed_a2": 0,
        "a0_nfe": 128,
        "a1_nfe": 96,
        "a2_nfe": 96,
        "a3_nfe": 96,
    }
    assert stable_rows([row, {**row, "repeat": 1}]) == [row]


def test_stable_rows_rejects_non_deterministic_quality() -> None:
    row = {
        "example_id": "x",
        "a0_correct": 1,
        "a1_correct": 1,
        "a2_correct": 1,
        "a3_correct": 1,
        "a1_exact_a0": 0,
        "a2_exact_a1": 1,
        "a3_exact_a2": 1,
        "a1_answer_changed_a0": 0,
        "a2_answer_changed_a1": 0,
        "a3_answer_changed_a2": 0,
        "a0_nfe": 128,
        "a1_nfe": 96,
        "a2_nfe": 96,
        "a3_nfe": 96,
    }
    with pytest.raises(ValueError, match="non-deterministic replay"):
        stable_rows([row, {**row, "a1_correct": 0}])
