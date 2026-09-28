from __future__ import annotations

from types import SimpleNamespace

import torch

from regret_remasking.phase_lock_influence import (
    SampledAttentionInfluenceCapture,
    multilayer_influence_ready,
    multilayer_max_drift,
    update_joint_patience,
)


def test_multilayer_max_drift_uses_worst_layer() -> None:
    current = {
        8: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        16: torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
    }
    previous = {
        8: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        16: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    }

    maximum, per_layer = multilayer_max_drift(current, previous)

    assert torch.allclose(maximum, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(per_layer[8], torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(per_layer[16], torch.tensor([[1.0, 0.0]]))


def test_joint_gate_requires_drift_and_influence_for_consecutive_steps() -> None:
    candidate = torch.tensor([[True, True, True]])
    count = torch.tensor([[3, 3, 3]])
    drift = torch.tensor([[0.001, 0.010, 0.001]])
    influence = torch.tensor([[0.4, 0.4, 0.8]])

    count = update_joint_patience(
        count,
        candidate,
        drift,
        influence,
        drift_threshold=0.005,
        influence_threshold=0.5,
    )
    ready = multilayer_influence_ready(
        candidate,
        drift,
        influence,
        age=torch.tensor([[2, 2, 2]]),
        joint_count=count,
        drift_threshold=0.005,
        influence_threshold=0.5,
        patience=4,
        min_age=2,
    )

    assert torch.equal(count, torch.tensor([[4, 0, 0]]))
    assert torch.equal(ready, torch.tensor([[True, False, False]]))


class _UniformAttentionBlock:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_heads=2,
            effective_n_kv_heads=2,
            rope=False,
        )
        self.q_norm = None
        self.k_norm = None

    @staticmethod
    def _cast_attn_bias(bias: torch.Tensor, _dtype: torch.dtype) -> torch.Tensor:
        return bias

    @staticmethod
    def attention(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return q


def test_attention_influence_is_one_for_uniform_attention() -> None:
    block = _UniformAttentionBlock()
    capture = SampledAttentionInfluenceCapture(block, max_queries=2)
    capture.install()
    try:
        capture.set_query_mask(torch.tensor([[True, False, True, False]]))
        q = torch.zeros((1, 4, 4))
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        output = block.attention(q, k, v)
        assert torch.equal(output, q)
        assert capture.latest is not None
        assert torch.allclose(capture.latest, torch.ones((1, 4)))
    finally:
        capture.remove()
