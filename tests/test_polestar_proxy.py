from __future__ import annotations

import math

import pytest
import torch

from regret_remasking.polestar_proxy import (
    PolestarCacheProxyConfig,
    attention_distribution_kl,
    local_prefix_bounds,
    select_high_drift_positions,
    should_trigger_refresh,
)


def test_default_config_matches_labeled_polestar_proxy_values() -> None:
    config = PolestarCacheProxyConfig()
    assert config.block_length == 32
    assert config.prefix_blocks == 2
    assert config.refresh_threshold == 3
    assert config.refresh_fraction == 0.5
    assert config.attention_layer == 32
    assert config.max_saved_events == 32
    assert config.seed == 23


def test_config_rejects_invalid_harness_values() -> None:
    with pytest.raises(ValueError, match="refresh_fraction"):
        PolestarCacheProxyConfig(refresh_fraction=0.0)
    with pytest.raises(ValueError, match="max_saved_events"):
        PolestarCacheProxyConfig(max_saved_events=0)


def test_attention_distribution_kl_zero_for_identical_probabilities() -> None:
    current = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
    got = attention_distribution_kl(current, current)
    assert got.shape == (1, 2)
    assert torch.all(torch.isfinite(got))
    assert torch.all(got >= 0)
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-7)


def test_attention_distribution_kl_positive_for_shifted_probabilities() -> None:
    current = torch.tensor([[[0.75, 0.25]]])
    previous = torch.tensor([[[0.25, 0.75]]])
    got = attention_distribution_kl(current, previous)
    expected = torch.tensor([[0.75 * math.log(3.0) + 0.25 * math.log(1.0 / 3.0)]])
    assert got.shape == (1, 1)
    assert torch.all(torch.isfinite(got))
    assert torch.all(got >= 0)
    assert torch.allclose(got, expected, atol=1e-7)


def test_attention_distribution_kl_averages_headwise_kl() -> None:
    current = torch.tensor([[[[0.75, 0.25]], [[0.5, 0.5]]]])
    previous = torch.tensor([[[[0.25, 0.75]], [[0.5, 0.5]]]])
    got = attention_distribution_kl(current, previous)
    shifted = 0.75 * math.log(3.0) + 0.25 * math.log(1.0 / 3.0)
    assert got.shape == (1, 1)
    assert torch.allclose(got, torch.tensor([[shifted / 2.0]]), atol=1e-7)


def test_attention_distribution_kl_rejects_shape_errors() -> None:
    current = torch.ones(1, 2, 3) / 3
    previous = torch.ones(1, 2, 4) / 4
    with pytest.raises(ValueError, match="matching shapes"):
        attention_distribution_kl(current, previous)

    with pytest.raises(ValueError, match="rank-3 or rank-4"):
        attention_distribution_kl(torch.ones(2, 3), torch.ones(2, 3))


def test_should_trigger_refresh_uses_strict_more_than_threshold() -> None:
    assert not should_trigger_refresh(
        decoded_this_step=1,
        decoded_since_refresh=2,
        threshold=3,
    )
    assert should_trigger_refresh(
        decoded_this_step=2,
        decoded_since_refresh=2,
        threshold=3,
    )
    assert should_trigger_refresh(
        decoded_this_step=4,
        decoded_since_refresh=0,
        threshold=3,
    )


def test_select_high_drift_positions_deterministic_top_half() -> None:
    scores = torch.tensor([[0.4, 0.9, 0.2], [0.9, 0.1, 0.8]])
    eligible = torch.tensor([[True, True, False], [True, True, True]])
    got = select_high_drift_positions(scores, eligible, fraction=0.5)
    expected = torch.tensor([[False, True, False], [True, False, True]])
    assert torch.equal(got, expected)


def test_select_high_drift_positions_empty_mask() -> None:
    scores = torch.tensor([[0.4, 0.9]])
    eligible = torch.tensor([[False, False]])
    got = select_high_drift_positions(scores, eligible, fraction=0.5)
    assert got.dtype == torch.bool
    assert not got.any()


def test_local_prefix_bounds_clip_to_prompt_boundary() -> None:
    assert local_prefix_bounds(
        prompt_len=10,
        block_start=20,
        block_length=8,
        prefix_blocks=2,
    ) == (10, 20)
    assert local_prefix_bounds(
        prompt_len=10,
        block_start=50,
        block_length=8,
        prefix_blocks=2,
    ) == (34, 50)
