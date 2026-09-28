from __future__ import annotations

import math

import torch

from regret_remasking.paper_baselines import (
    parse_example_indices,
    posterior_kl,
    surelock_decision,
)


def _log_probs(probabilities: list[list[list[float]]]) -> torch.Tensor:
    return torch.tensor(probabilities, dtype=torch.float32).log()


def test_posterior_kl_uses_current_to_previous_direction() -> None:
    current = _log_probs([[[0.8, 0.2]]])
    previous = _log_probs([[[0.5, 0.5]]])
    actual = posterior_kl(current, previous).item()
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_surelock_abstains_without_previous_posterior() -> None:
    current = _log_probs([[[0.9, 0.1], [0.6, 0.4]]])
    decision = surelock_decision(
        current_log_probs=current,
        previous_log_probs=None,
        candidate=torch.ones((1, 2), dtype=torch.bool),
        epsilon=5e-4,
    )
    assert not decision.selected.any()
    assert torch.isinf(decision.posterior_kl).all()


def test_surelock_combines_percentile_confidence_and_kl() -> None:
    previous = _log_probs(
        [[[0.90, 0.10], [0.80, 0.20], [0.70, 0.30], [0.60, 0.40]]]
    )
    current = _log_probs(
        [[[0.90, 0.10], [0.80, 0.20], [0.71, 0.29], [0.60, 0.40]]]
    )
    decision = surelock_decision(
        current_log_probs=current,
        previous_log_probs=previous,
        candidate=torch.ones((1, 4), dtype=torch.bool),
        epsilon=5e-4,
        confidence_percentile=50.0,
    )
    # The top-confidence half (positions 0 and 1) also pass the KL gate.
    assert torch.equal(
        decision.selected, torch.tensor([[True, True, False, False]])
    )


def test_surelock_respects_candidate_mask() -> None:
    posterior = _log_probs([[[0.99, 0.01], [0.90, 0.10], [0.80, 0.20]]])
    decision = surelock_decision(
        current_log_probs=posterior,
        previous_log_probs=posterior,
        candidate=torch.tensor([[False, True, True]]),
        epsilon=0.0,
        confidence_percentile=50.0,
    )
    assert torch.equal(decision.selected, torch.tensor([[False, True, False]]))


def test_matched_benchmark_index_parser_accepts_spans() -> None:
    assert parse_example_indices("3,7:9") == [3, 7, 8, 9]
