from __future__ import annotations

import copy
import math

import pytest
import torch

from regret_remasking.paper_reference import PaperReferenceRuntime


def _same_bound_method(left, right) -> bool:
    return getattr(left, "__self__", None) is getattr(right, "__self__", None) and getattr(left, "__func__", None) is getattr(
        right, "__func__", None
    )


class ToyConfig:
    n_heads = 1
    effective_n_kv_heads = 1


class CountingLinear(torch.nn.Linear):
    def __init__(self, dim: int) -> None:
        super().__init__(dim, dim, bias=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.config = ToyConfig()
        self.attn_norm = torch.nn.Identity()
        self.ff_norm = torch.nn.Identity()
        self.dropout = torch.nn.Identity()
        self.act = torch.nn.Tanh()
        self.q_proj = CountingLinear(dim)
        self.k_proj = CountingLinear(dim)
        self.v_proj = CountingLinear(dim)
        self.attn_out = torch.nn.Linear(dim, dim, bias=False)
        self.ff_proj = torch.nn.Linear(dim, dim, bias=False)
        self.ff_out = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = None
        self.k_norm = None
        self.attention_query_rows: list[int] = []
        self.attention_key_cols: list[int] = []
        self.attention_masks: list[torch.Tensor | None] = []
        self.attention_k_values: list[torch.Tensor] = []
        self.attention_v_values: list[torch.Tensor] = []

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.attention_query_rows.append(int(q.shape[2]))
        self.attention_key_cols.append(int(k.shape[2]))
        self.attention_masks.append(None if attn_mask is None else attn_mask.detach().clone())
        self.attention_k_values.append(k.detach().clone())
        self.attention_v_values.append(v.detach().clone())
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past=None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if layer_past is not None or use_cache:
            raise RuntimeError("toy block does not support cache")
        q = self.q_proj(self.attn_norm(x))
        k = self.k_proj(self.attn_norm(x))
        v = self.v_proj(self.attn_norm(x))
        bsz, seq_len, dim = q.shape
        q = q.view(bsz, seq_len, 1, dim).transpose(1, 2)
        k = k.view(bsz, seq_len, 1, dim).transpose(1, 2)
        v = v.view(bsz, seq_len, 1, dim).transpose(1, 2)
        att = self._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        att = att.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = x + self.dropout(self.attn_out(att))
        residual = x
        ff = self.act(self.ff_proj(self.ff_norm(x)))
        return residual + self.dropout(self.ff_out(ff)), None


class FakeRotary:
    def __init__(self) -> None:
        self.config = type("Config", (), {"rope_full_precision": False})()
        self.positions: list[torch.Tensor] = []

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(seq_len, dtype=torch.float32, device=device).view(1, 1, seq_len, 1)
        return values, torch.zeros_like(values)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, _pos_cos: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        self.positions.append(pos_sin[0, 0, :, 0].detach().cpu().to(dtype=torch.long))
        return tensor + (pos_sin.to(dtype=tensor.dtype) * 0.0)


class RotaryToyBlock(ToyBlock):
    def __init__(self, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.rotary_emb = FakeRotary()


class CountingHead(torch.nn.Linear):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__(dim, vocab, bias=False)
        self.input_rows: list[int] = []
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_rows.append(int(x.shape[1]))
        self.inputs.append(x.detach().clone())
        return super().forward(x)


class ToyModel(torch.nn.Module):
    def __init__(self, *, layers: int = 2, dim: int = 4, vocab: int = 7, block_cls=ToyBlock) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([block_cls(dim) for _ in range(layers)])
        self.lm_head = CountingHead(dim, vocab)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x, _cache = block(x, attention_bias=attention_bias)
        return self.lm_head(x)


class LLaDAToyModel(torch.nn.Module):
    def __init__(self, *, layers: int = 1, dim: int = 4, vocab: int = 7) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList([ToyBlock(dim) for _ in range(layers)])
        self.model.transformer.ff_out = CountingHead(dim, vocab)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.model.transformer.blocks:
            x, _cache = block(x, attention_bias=attention_bias)
        return self.model.transformer.ff_out(x)


class BadSparseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.lm_head = CountingHead(4, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.blocks[0](x))


def _model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel()
    model.eval()
    return model


def _rotary_model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel(block_cls=RotaryToyBlock)
    model.eval()
    return model


def _input(seed: int = 23) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 5, 4)


def test_no_lock_sparse_matches_native_and_skips_compact_head_gather() -> None:
    x = _input()
    baseline = _model()
    sparse = copy.deepcopy(baseline)
    expected = baseline(x)

    runtime = PaperReferenceRuntime(sparse, mode="sparse", total_len=x.shape[1])
    runtime.install()
    try:
        actual = sparse(x)
    finally:
        runtime.remove()

    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual.shape == expected.shape
    assert sparse.lm_head.input_rows[-1] == x.shape[1]
    assert runtime.last_head_rows == x.shape[1]
    assert runtime.last_attention_query_rows == len(runtime.blocks) * x.shape[1]
    assert runtime.last_output_hidden is not None
    assert runtime.last_output_hidden.shape == sparse.lm_head.inputs[-1].shape
    assert torch.allclose(runtime.last_output_hidden, sparse.lm_head.inputs[-1])
    runtime.begin_forward()
    assert runtime.last_output_hidden is None
    assert torch.equal(runtime.active_positions.cpu(), torch.arange(x.shape[1]))


def test_sparse_uses_compact_attention_queries_head_rows_and_all_kv_columns() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    sparse = _model()
    runtime = PaperReferenceRuntime(sparse, mode="sparse", total_len=x0.shape[1], capture_diagnostics=True)
    runtime.install()
    try:
        sparse(x0)
        runtime.commit([2])
        logits = sparse(x1)
    finally:
        runtime.remove()

    assert logits.shape[:2] == (1, 4)
    assert torch.equal(runtime.output_positions.cpu(), torch.tensor([0, 1, 3, 4]))
    assert all(block.attention_query_rows[-1] == 4 for block in sparse.blocks)
    assert all(block.attention_key_cols[-1] == 5 for block in sparse.blocks)
    assert sparse.lm_head.input_rows[-1] == 4
    assert runtime.last_output_hidden is not None
    assert runtime.last_output_hidden.shape == (1, 4, x0.shape[-1])
    assert torch.allclose(runtime.last_output_hidden, sparse.lm_head.inputs[-1])
    assert runtime.last_attention_query_rows == 2 * 4
    assert runtime.last_projection_rows == 2 * 4
    assert runtime.last_ffn_rows == 2 * 4
    assert runtime.last_head_rows == 4
    assert runtime.last_work_counts["attention_query_rows"] == 2 * 4
    assert runtime.last_work_counts["head_rows"] == 4
    assert runtime.frozen_logits is not None
    assert runtime.frozen_logits.shape == (1, 5, 7)


def test_dense_and_sparse_compact_logits_match_fp32_after_lock() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    dense = _model()
    sparse = copy.deepcopy(dense)

    dense_runtime = PaperReferenceRuntime(dense, mode="dense", total_len=x0.shape[1], capture_diagnostics=True)
    sparse_runtime = PaperReferenceRuntime(sparse, mode="sparse", total_len=x0.shape[1], capture_diagnostics=True)
    dense_runtime.install()
    sparse_runtime.install()
    try:
        dense(x0)
        sparse(x0)
        dense_runtime.commit([2])
        sparse_runtime.commit([2])
        dense_logits = dense(x1)
        sparse_logits = sparse(x1)
    finally:
        dense_runtime.remove()
        sparse_runtime.remove()

    assert dense_logits.shape == sparse_logits.shape == (1, 4, 7)
    assert torch.equal(dense_runtime.output_positions.cpu(), sparse_runtime.output_positions.cpu())
    assert torch.allclose(dense_logits, sparse_logits, atol=1e-6)
    assert dense.lm_head.input_rows[-1] == 5
    assert sparse.lm_head.input_rows[-1] == 4
    assert dense_runtime.last_output_hidden is not None
    assert sparse_runtime.last_output_hidden is not None
    assert dense_runtime.last_output_hidden.shape == sparse_runtime.last_output_hidden.shape == (1, 4, x0.shape[-1])
    assert torch.allclose(dense_runtime.last_output_hidden, dense.lm_head.inputs[-1].index_select(1, dense_runtime.output_positions))
    assert torch.allclose(sparse_runtime.last_output_hidden, sparse.lm_head.inputs[-1])
    assert dense_runtime.last_head_rows == 5
    assert sparse_runtime.last_head_rows == 4
    assert set(dense_runtime.frozen_k) == {0, 1}
    assert set(dense_runtime.frozen_v) == {0, 1}
    assert torch.allclose(dense_runtime.last_k[0][:, :, 2, :], dense_runtime.frozen_k[0][:, :, 2, :])
    assert torch.allclose(dense_runtime.last_v[0][:, :, 2, :], dense_runtime.frozen_v[0][:, :, 2, :])


def test_dense_captures_native_projected_kv_without_duplicate_projection_calls() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = x1[:, 2, :] + 3.0
    model = _model()
    runtime = PaperReferenceRuntime(model, mode="dense", total_len=x0.shape[1], capture_diagnostics=True)
    runtime.install()
    try:
        model(x0)
        first_k_calls = [block.k_proj.calls for block in model.blocks]
        first_v_calls = [block.v_proj.calls for block in model.blocks]
        runtime.commit([2])
        model(x1)
    finally:
        runtime.remove()

    assert first_k_calls == [1, 1]
    assert first_v_calls == [1, 1]
    assert [block.k_proj.calls for block in model.blocks] == [2, 2]
    assert [block.v_proj.calls for block in model.blocks] == [2, 2]
    assert runtime.last_projection_rows == len(model.blocks) * x0.shape[1]


def test_dense_attention_consumes_frozen_kv_cache_after_sentinel_perturbation() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = x1[:, 2, :] + 1.0
    model = _model()
    runtime = PaperReferenceRuntime(model, mode="dense", total_len=x0.shape[1], capture_diagnostics=True)
    runtime.install()
    try:
        model(x0)
        runtime.commit([2])
        sentinel = torch.full_like(runtime.frozen_v[0][:, :, 2, :], 123.0)
        runtime.frozen_v[0][:, :, 2, :] = sentinel
        model(x1)
    finally:
        runtime.remove()

    assert torch.allclose(model.blocks[0].attention_v_values[-1][:, :, 2, :], sentinel)
    assert torch.allclose(runtime.last_v[0][:, :, 2, :], sentinel)


def test_sparse_noncontiguous_active_positions_keep_absolute_rope_and_bias_rows() -> None:
    x0 = _input()
    x1 = x0.clone()
    model = _rotary_model()
    runtime = PaperReferenceRuntime(model, mode="sparse", total_len=x0.shape[1])
    bias_values = torch.arange(x0.shape[1] * x0.shape[1], dtype=x0.dtype).view(1, 1, x0.shape[1], x0.shape[1])
    runtime.install()
    try:
        model(x0, attention_bias=bias_values)
        runtime.commit([1, 3])
        logits = model(x1, attention_bias=bias_values)
    finally:
        runtime.remove()

    expected_positions = torch.tensor([0, 2, 4])
    assert logits.shape == (1, 3, 7)
    assert torch.equal(runtime.output_positions.cpu(), expected_positions)
    for block in model.blocks:
        assert torch.equal(block.rotary_emb.positions[-2], expected_positions)
        assert torch.equal(block.rotary_emb.positions[-1], expected_positions)
        assert block.attention_masks[-1] is not None
        assert torch.equal(block.attention_masks[-1], bias_values[:, :, expected_positions, :])


def test_frozen_logits_and_layer_snapshots_remain_constant_across_repeated_commits() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 1, :] = x1[:, 1, :] + 2.0
    model = _model()
    runtime = PaperReferenceRuntime(model, mode="sparse", total_len=x0.shape[1], capture_diagnostics=True)
    runtime.install()
    try:
        model(x0)
        runtime.commit([1])
        assert torch.equal(runtime.output_positions.cpu(), torch.arange(x0.shape[1]))
        assert torch.equal(runtime.active_positions.cpu(), torch.tensor([0, 2, 3, 4]))
        frozen_input = runtime.frozen_inputs[0][:, 1, :].clone()
        frozen_logit = runtime.frozen_logits[:, 1, :].clone()
        model(x1)
        runtime.commit([1, 3])
    finally:
        runtime.remove()

    assert torch.allclose(runtime.frozen_inputs[0][:, 1, :], frozen_input)
    assert torch.allclose(runtime.frozen_logits[:, 1, :], frozen_logit)
    assert bool(runtime.locked[0, 1])
    assert bool(runtime.locked[0, 3])
    assert torch.equal(runtime.active_positions.cpu(), torch.tensor([0, 2, 4]))


def test_zero_and_all_locks_have_stable_compact_api() -> None:
    x = _input()
    model = _model()
    runtime = PaperReferenceRuntime(model, mode="sparse", total_len=x.shape[1])
    runtime.install()
    try:
        no_lock_logits = model(x)
        runtime.commit([])
        still_no_lock_logits = model(x)
        runtime.commit([0, 1, 2, 3, 4])
        all_locked_logits = model(x)
    finally:
        runtime.remove()

    assert torch.allclose(no_lock_logits, still_no_lock_logits, atol=1e-6)
    assert all_locked_logits.shape == (1, 0, 7)
    assert runtime.last_attention_query_rows == 0
    assert runtime.last_projection_rows == 0
    assert runtime.last_ffn_rows == 0
    assert runtime.last_head_rows == 0
    assert runtime.output_positions.numel() == 0


def test_llada_ff_out_head_is_patched_before_generic_lm_head() -> None:
    x = _input()
    model = LLaDAToyModel()
    runtime = PaperReferenceRuntime(model, mode="sparse", total_len=x.shape[1])
    assert runtime.output_head is model.model.transformer.ff_out
    runtime.install()
    try:
        model(x)
        runtime.commit([0, 4])
        logits = model(x)
    finally:
        runtime.remove()

    assert logits.shape == (1, 3, 7)
    assert runtime.last_output_hidden is not None
    assert torch.allclose(runtime.last_output_hidden, model.model.transformer.ff_out.inputs[-1])
    assert model.model.transformer.ff_out.input_rows[-1] == 3


def test_remove_restores_sparse_block_and_output_head_after_exception_checks() -> None:
    bad = BadSparseModel()
    original_block_forward = bad.blocks[0].forward
    original_head_forward = bad.lm_head.forward
    runtime = PaperReferenceRuntime(bad, mode="sparse", total_len=5)
    with pytest.raises(TypeError):
        runtime.install()
    assert _same_bound_method(bad.blocks[0].forward, original_block_forward)
    assert _same_bound_method(bad.lm_head.forward, original_head_forward)

    model = _model()
    runtime = PaperReferenceRuntime(model, mode="sparse", total_len=5)
    original_block_forward = model.blocks[0].forward
    original_head_forward = model.lm_head.forward
    runtime.install()
    try:
        with pytest.raises(RuntimeError, match="batch size 1"):
            model(torch.randn(2, 5, 4))
        with pytest.raises(RuntimeError, match="past KV cache"):
            model.blocks[0](torch.randn(1, 5, 4), layer_past=object())
    finally:
        runtime.remove()
    assert _same_bound_method(model.blocks[0].forward, original_block_forward)
    assert _same_bound_method(model.lm_head.forward, original_head_forward)

    dense_model = _model()
    dense_runtime = PaperReferenceRuntime(dense_model, mode="dense", total_len=5)
    original_attention = dense_model.blocks[0]._scaled_dot_product_attention
    dense_runtime.install()
    dense_runtime.remove()
    assert _same_bound_method(dense_model.blocks[0]._scaled_dot_product_attention, original_attention)
