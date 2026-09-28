from benchmark_phaselock_matched_plans import (
    reference_replay_plan,
    semantic_replay_plan,
    transition_counts,
)


def test_plan_conversion_preserves_exact_semantic_identity() -> None:
    semantic = semantic_replay_plan({2: [(4, 99)], 0: [(1, 7)]})
    reference = reference_replay_plan({3: [4], 1: [1, 2]})
    assert list(semantic) == ["0", "2"]
    assert semantic["2"] == [
        {
            "relative_position": 4,
            "token_id": 99,
            "selection_source": "planner",
        }
    ]
    assert reference == {
        "1": [{"relative_position": 1}, {"relative_position": 2}],
        "3": [{"relative_position": 4}],
    }


def test_transition_counts_are_paired() -> None:
    result = transition_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert result == {
        "regressions": 1,
        "improvements": 1,
        "regression_rate": 0.25,
        "improvement_rate": 0.25,
    }
