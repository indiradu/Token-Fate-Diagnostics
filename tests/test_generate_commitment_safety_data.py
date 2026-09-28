from __future__ import annotations

import torch

from generate_commitment_safety_data import (
    CounterfactualReplayIneligible,
    add_counterfactual_commit,
    ordinary_transfer_mask,
    select_stratified_candidates,
)


def test_counterfactual_commit_is_additive() -> None:
    confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    allowed = torch.tensor([[True, True, True, True]])
    ordinary = ordinary_transfer_mask(confidence, allowed, torch.tensor([2]))
    forced = add_counterfactual_commit(ordinary, allowed, [2])
    assert ordinary.tolist() == [[True, True, False, False]]
    assert forced.tolist() == [[True, True, True, False]]


def test_counterfactual_commit_rejects_a_naturally_transferred_target() -> None:
    ordinary = torch.tensor([[True, True, False, False]])
    allowed = torch.tensor([[True, True, True, True]])
    try:
        add_counterfactual_commit(ordinary, allowed, [1])
    except CounterfactualReplayIneligible as error:
        assert "already in the ordinary transfer set" in str(error)
    else:
        raise AssertionError("a null counterfactual treatment must be excluded")


def test_stratified_candidates_cover_proxy_range_and_positions() -> None:
    rows = []
    for index in range(20):
        value = index / 20.0
        rows.append(
            {
                "global_step": index,
                "relative_position": index,
                "confidence": value,
                "margin": value,
                "entropy": 1.0 - value,
                "token_age": index + 1,
                "trace_probability": value,
                "max_hidden_drift": 1.0 - value,
                "attention_influence": 1.0 - value,
            }
        )
    selected = select_stratified_candidates(rows, 5)
    assert len(selected) == 5
    assert len({row["relative_position"] for row in selected}) == 5
    proxies = [row["selection_proxy"] for row in selected]
    assert max(proxies) - min(proxies) > 0.8
