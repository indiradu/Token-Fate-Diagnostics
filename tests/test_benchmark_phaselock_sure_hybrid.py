from __future__ import annotations

import torch

from benchmark_phaselock_sure_hybrid import (
    _fallback_progress,
    sure_selected_from_stats,
)


def test_sure_selection_uses_candidate_relative_uncertainty_quantile() -> None:
    uncertainty = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    divergence = torch.tensor([[0.001, 0.001, 0.001, 0.001]])
    candidate = torch.tensor([[False, True, True, True]])
    selected = sure_selected_from_stats(
        uncertainty,
        divergence,
        candidate,
        epsilon=0.005,
        confidence_percentile=20.0,
    )
    assert selected.tolist() == [[False, True, False, False]]


def test_sure_selection_rejects_large_kl_and_fallback_makes_progress() -> None:
    uncertainty = torch.tensor([[0.1, 0.2, 0.3]])
    divergence = torch.tensor([[0.02, 0.02, 0.02]])
    candidate = torch.tensor([[True, True, False]])
    selected = sure_selected_from_stats(
        uncertainty,
        divergence,
        candidate,
        epsilon=0.005,
        confidence_percentile=20.0,
    )
    progressed = _fallback_progress(
        selected,
        candidate,
        confidence=torch.tensor([[0.8, 0.9, 1.0]]),
    )
    assert progressed.tolist() == [[False, True, False]]

