from __future__ import annotations

import torch

from benchmark_commitment_safety_policy import select_extra_commits


def test_extra_commit_selector_respects_threshold_and_cap() -> None:
    score = torch.tensor([[0.99, 0.95, 0.90, 0.20]])
    candidate = torch.tensor([[True, True, True, True]])
    selected = select_extra_commits(
        score, candidate, threshold=0.8, maximum_per_sample=2
    )
    assert selected.tolist() == [[True, True, False, False]]

