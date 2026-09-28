from __future__ import annotations

import pytest
import torch

from regret_remasking.phase_lock_semantic import (
    scalar_posterior_stability_candidates,
    select_future_stability_commits,
    select_sure_semantic_commits,
    update_token_identity_patience,
)


def test_future_stability_gate_applies_confidence_and_cap() -> None:
    probability = torch.tensor([[0.99, 0.98, 0.97, 0.40]])
    confidence = torch.tensor([[0.95, 0.85, 0.99, 0.99]])
    allowed = torch.tensor([[True, True, True, True]])

    selected = select_future_stability_commits(
        probability,
        confidence,
        allowed,
        probability_threshold=0.95,
        confidence_threshold=0.90,
        maximum_fraction=0.25,
    )

    assert selected.tolist() == [[True, False, False, False]]


def test_future_stability_gate_falls_back_per_example() -> None:
    probability = torch.tensor([[0.20, 0.30, 0.10], [0.90, 0.80, 0.70]])
    confidence = torch.zeros_like(probability)
    allowed = torch.tensor([[True, True, False], [False, True, True]])

    selected = select_future_stability_commits(
        probability,
        confidence,
        allowed,
        probability_threshold=0.95,
        confidence_threshold=0.90,
        maximum_fraction=0.25,
    )

    assert selected.tolist() == [[False, True, False], [False, True, False]]


def test_future_stability_gate_supports_confidence_fallback() -> None:
    probability = torch.tensor([[0.90, 0.80, 0.70]])
    confidence = torch.tensor([[0.10, 0.99, 0.50]])
    allowed = torch.tensor([[True, True, True]])

    selected = select_future_stability_commits(
        probability,
        confidence,
        allowed,
        probability_threshold=0.95,
        confidence_threshold=0.0,
        maximum_fraction=1.0,
        fallback_score="confidence",
    )

    assert selected.tolist() == [[False, True, False]]


def test_future_stability_gate_supports_joint_fallback() -> None:
    probability = torch.tensor([[0.90, 0.80, 0.70]])
    confidence = torch.tensor([[0.60, 0.90, 0.50]])
    allowed = torch.tensor([[True, True, True]])

    selected = select_future_stability_commits(
        probability,
        confidence,
        allowed,
        probability_threshold=0.95,
        confidence_threshold=0.0,
        maximum_fraction=1.0,
        fallback_score="joint",
    )

    assert selected.tolist() == [[False, True, False]]


@pytest.mark.parametrize("maximum_fraction", [0.0, 1.1])
def test_future_stability_gate_rejects_invalid_caps(maximum_fraction: float) -> None:
    values = torch.ones((1, 1))
    with pytest.raises(ValueError, match="maximum fraction"):
        select_future_stability_commits(
            values,
            values,
            values.bool(),
            probability_threshold=0.95,
            confidence_threshold=0.0,
            maximum_fraction=maximum_fraction,
        )


def test_future_stability_gate_rejects_unknown_fallback_score() -> None:
    values = torch.ones((1, 1))
    with pytest.raises(ValueError, match="fallback score"):
        select_future_stability_commits(
            values,
            values,
            values.bool(),
            probability_threshold=0.95,
            confidence_threshold=0.0,
            maximum_fraction=1.0,
            fallback_score="mystery",
        )


def test_token_identity_patience_resets_outside_current_domain() -> None:
    previous_count = torch.tensor([[2, 4, 1, 8]])
    previous_token = torch.tensor([[7, 8, 9, 10]])
    current_token = torch.tensor([[7, 3, 9, 10]])
    allowed = torch.tensor([[True, True, False, True]])

    count = update_token_identity_patience(
        previous_count, current_token, previous_token, allowed
    )

    assert count.tolist() == [[3, 1, 0, 9]]


def test_sure_semantic_adds_confidence_patience_and_cap() -> None:
    sure = torch.tensor([[True, True, True, False]])
    confidence = torch.tensor([[0.91, 0.99, 0.97, 0.80]])
    patience = torch.tensor([[2, 1, 3, 5]])
    allowed = torch.tensor([[True, True, True, True]])

    selected = select_sure_semantic_commits(
        sure,
        confidence,
        patience,
        allowed,
        confidence_threshold=0.90,
        minimum_token_patience=2,
        maximum_fraction=0.25,
    )

    assert selected.tolist() == [[False, False, True, False]]


def test_sure_semantic_confidence_fallback_guarantees_progress() -> None:
    selected = select_sure_semantic_commits(
        torch.zeros((1, 3), dtype=torch.bool),
        torch.tensor([[0.2, 0.8, 0.5]]),
        torch.ones((1, 3), dtype=torch.long),
        torch.tensor([[True, True, False]]),
        confidence_threshold=0.9,
        minimum_token_patience=2,
    )

    assert selected.tolist() == [[False, True, False]]


def test_scalar_posterior_stability_uses_percentile_and_delta() -> None:
    confidence = torch.tensor([[0.99, 0.95, 0.80, 0.70]])
    previous = torch.tensor([[0.98, 0.80, 0.79, 0.70]])
    allowed = torch.tensor([[True, True, True, False]])

    selected = scalar_posterior_stability_candidates(
        confidence,
        previous,
        allowed,
        maximum_confidence_delta=0.02,
        confidence_percentile=50.0,
    )

    assert selected.tolist() == [[True, False, False, False]]


def test_scalar_posterior_stability_needs_previous_observation() -> None:
    confidence = torch.tensor([[0.9, 0.8]])
    allowed = torch.tensor([[True, True]])

    selected = scalar_posterior_stability_candidates(
        confidence,
        None,
        allowed,
        maximum_confidence_delta=0.02,
    )

    assert not bool(selected.any())
