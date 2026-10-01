from __future__ import annotations

import copy
import math

import pytest
import torch

from regret_remasking.locking_reference import ReferenceRuntime


def _same_bound_method(left, right) -> bool:
    return getattr(left, "__self__", None) is getattr(right, "__self__", None) and getattr(left, "__func__", None) is getattr(
        right, "__func__", None
    )


class ToyConfig:
    n_heads = 1
    effective_n_kv_heads = 1


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.config = ToyConfig()
        self.attn_norm = torch.nn.Identity()
        self.ff_norm = torch.nn.Identity()
        self.dropout = torch.nn.Identity()
        self.act = torch.nn.Tanh()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.attn_out = torch.nn.Linear(dim, dim, bias=False)
        self.ff_proj = torch.nn.Linear(dim, dim, bias=False)
        self.ff_out = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = None
        self.k_norm = None

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


class NativeCastToyBlock(ToyBlock):
    def __init__(self, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.cast_dtypes: list[torch.dtype] = []
        self.attn_mask_dtypes: list[torch.dtype | None] = []

    def _cast_attn_bias(self, attention_bias: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        self.cast_dtypes.append(dtype)
        return attention_bias.to(dtype=dtype)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.attn_mask_dtypes.append(None if attn_mask is None else attn_mask.dtype)
        return super()._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

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
        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(attention_bias[:, :, :seq_len, :seq_len], k.dtype)
        att = self._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        att = att.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = x + self.dropout(self.attn_out(att))
        residual = x
        ff = self.act(self.ff_proj(self.ff_norm(x)))
        return residual + self.dropout(self.ff_out(ff)), None


class ShapeSensitiveToyBlock(ToyBlock):
    def __init__(self, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.attention_query_rows: list[int] = []
        self.attention_query_strides: list[tuple[int, ...]] = []

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.attention_query_rows.append(int(q.shape[2]))
        self.attention_query_strides.append(q.stride())
        out = super()._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if q.shape[2] != k.shape[2]:
            out = out + 0.125
        return out


class StrideSensitiveToyBlock(ToyBlock):
    def __init__(self, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.native_k_stride: tuple[int, ...] | None = None
        self.native_v_stride: tuple[int, ...] | None = None
        self.attention_k_strides: list[tuple[int, ...]] = []
        self.attention_v_strides: list[tuple[int, ...]] = []

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.attention_k_strides.append(k.stride())
        self.attention_v_strides.append(v.stride())
        if q.shape[2] == k.shape[2] and self.native_k_stride is None:
            self.native_k_stride = k.stride()
            self.native_v_stride = v.stride()
        out = super()._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if self.native_k_stride is not None and (k.stride() != self.native_k_stride or v.stride() != self.native_v_stride):
            out = out + 0.125
        return out


class ToyModel(torch.nn.Module):
    def __init__(self, *, layers: int = 2, dim: int = 4, vocab: int = 7, block_cls=ToyBlock) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([block_cls(dim) for _ in range(layers)])
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x, _cache = block(x, attention_bias=attention_bias)
        return self.lm_head(x)


class BadSparseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


def _model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel()
    model.eval()
    return model


def _shape_sensitive_model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel(block_cls=ShapeSensitiveToyBlock)
    model.eval()
    return model


def _native_cast_model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel(block_cls=NativeCastToyBlock)
    model.eval()
    return model


def _stride_sensitive_model(seed: int = 11) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel(block_cls=StrideSensitiveToyBlock)
    model.eval()
    return model


def _input(seed: int = 23) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 5, 4)


def test_no_lock_dense_and_sparse_match_unpatched_model() -> None:
    x = _input()
    baseline = _model()
    dense = copy.deepcopy(baseline)
    sparse = copy.deepcopy(baseline)

    expected = baseline(x)
    dense_runtime = ReferenceRuntime(dense, mode="dense", total_len=x.shape[1])
    sparse_runtime = ReferenceRuntime(sparse, mode="sparse", total_len=x.shape[1])
    dense_runtime.install()
    sparse_runtime.install()
    try:
        dense_logits = dense(x)
        sparse_logits = sparse(x)
    finally:
        dense_runtime.remove()
        sparse_runtime.remove()

    assert torch.allclose(dense_logits, expected, atol=1e-6)
    assert torch.allclose(sparse_logits, expected, atol=1e-6)
    assert set(dense_runtime.last_inputs) == {0, 1}
    assert set(sparse_runtime.last_k) == {0, 1}


def test_dense_and_sparse_active_logits_match_after_controlled_freeze() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    dense = _model()
    sparse = copy.deepcopy(dense)

    dense_runtime = ReferenceRuntime(dense, mode="dense", total_len=x0.shape[1])
    sparse_runtime = ReferenceRuntime(sparse, mode="sparse", total_len=x0.shape[1])
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

    active = torch.tensor([0, 1, 3, 4])
    assert bool(dense_runtime.locked[0, 2])
    assert torch.allclose(dense_runtime.frozen_inputs[0][:, 2, :], dense_runtime.last_inputs[0][:, 2, :])
    assert torch.allclose(dense_logits[:, active, :], sparse_logits[:, active, :], atol=1e-6)


def test_sparse_full_attention_layout_preserves_native_query_shape_and_active_row_savings() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    dense = _shape_sensitive_model()
    sparse_full = copy.deepcopy(dense)
    sparse_compact = copy.deepcopy(dense)

    dense_runtime = ReferenceRuntime(dense, mode="dense", total_len=x0.shape[1])
    full_runtime = ReferenceRuntime(sparse_full, mode="sparse", total_len=x0.shape[1], attention_layout="full")
    compact_runtime = ReferenceRuntime(sparse_compact, mode="sparse", total_len=x0.shape[1], attention_layout="compact")
    dense_runtime.install()
    full_runtime.install()
    compact_runtime.install()
    try:
        dense(x0)
        sparse_full(x0)
        sparse_compact(x0)
        dense_runtime.commit([2])
        full_runtime.commit([2])
        compact_runtime.commit([2])
        dense_logits = dense(x1)
        full_logits = sparse_full(x1)
        compact_logits = sparse_compact(x1)
    finally:
        dense_runtime.remove()
        full_runtime.remove()
        compact_runtime.remove()

    active = torch.tensor([0, 1, 3, 4])
    seq_len = x0.shape[1]
    active_count = int(active.numel())
    assert full_runtime.attention_layout == "full"
    assert compact_runtime.attention_layout == "compact"
    assert torch.allclose(dense_logits[:, active, :], full_logits[:, active, :], atol=1e-6)
    assert not torch.allclose(dense_logits[:, active, :], compact_logits[:, active, :], atol=1e-6)
    assert full_runtime.last_attention_query_rows == seq_len
    assert full_runtime.last_projection_rows == active_count
    assert full_runtime.last_ffn_rows == active_count
    assert compact_runtime.last_attention_query_rows == active_count
    assert compact_runtime.last_projection_rows == active_count
    assert compact_runtime.last_ffn_rows == active_count
    assert full_runtime.native_q_strides[0] == sparse_full.blocks[0].attention_query_strides[0]
    assert sparse_full.blocks[0].attention_query_rows[-1] == seq_len
    assert sparse_compact.blocks[0].attention_query_rows[-1] == active_count


def test_sparse_attention_bias_is_cast_like_native_attention() -> None:
    x = _input().to(dtype=torch.bfloat16)
    attention_bias = torch.zeros((1, 1, x.shape[1], x.shape[1]), dtype=torch.float32)
    baseline = _native_cast_model().to(dtype=torch.bfloat16)
    sparse = copy.deepcopy(baseline)

    expected = baseline(x, attention_bias=attention_bias)
    runtime = ReferenceRuntime(sparse, mode="sparse", total_len=x.shape[1])
    runtime.install()
    try:
        actual = sparse(x, attention_bias=attention_bias)
    finally:
        runtime.remove()

    assert torch.allclose(actual, expected, atol=1e-2)
    assert baseline.blocks[0].cast_dtypes[-1] == torch.bfloat16
    assert sparse.blocks[0].cast_dtypes[-1] == torch.bfloat16
    assert baseline.blocks[0].attn_mask_dtypes[-1] == torch.bfloat16
    assert sparse.blocks[0].attn_mask_dtypes[-1] == torch.bfloat16


def test_sparse_full_attention_layout_preserves_native_kv_strides() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    dense = _stride_sensitive_model()
    sparse_full = copy.deepcopy(dense)

    dense_runtime = ReferenceRuntime(dense, mode="dense", total_len=x0.shape[1])
    full_runtime = ReferenceRuntime(sparse_full, mode="sparse", total_len=x0.shape[1], attention_layout="full")
    dense_runtime.install()
    full_runtime.install()
    try:
        dense(x0)
        sparse_full(x0)
        dense_runtime.commit([2])
        full_runtime.commit([2])
        dense_logits = dense(x1)
        full_logits = sparse_full(x1)
    finally:
        dense_runtime.remove()
        full_runtime.remove()

    active = torch.tensor([0, 1, 3, 4])
    assert torch.allclose(dense_logits[:, active, :], full_logits[:, active, :], atol=1e-6)
    assert full_runtime.native_k_strides[0] == sparse_full.blocks[0].attention_k_strides[-1]
    assert full_runtime.native_v_strides[0] == sparse_full.blocks[0].attention_v_strides[-1]


def test_sparse_active_rows_still_attend_to_frozen_position_values() -> None:
    x0 = _input()
    x1 = x0.clone()
    sparse = _model()
    runtime = ReferenceRuntime(sparse, mode="sparse", total_len=x0.shape[1])
    runtime.install()
    try:
        sparse(x0)
        runtime.commit([0])
        normal = sparse(x1)
        for layer in range(len(runtime.blocks)):
            runtime.frozen_v[layer][:, :, 0, :] = runtime.frozen_v[layer][:, :, 0, :] + 10.0
        perturbed = sparse(x1)
    finally:
        runtime.remove()

    assert not torch.allclose(normal[:, 1:, :], perturbed[:, 1:, :])


def test_snapshots_are_immutable_after_source_tensor_mutation() -> None:
    x = _input()
    model = _model()
    runtime = ReferenceRuntime(model, mode="sparse", total_len=x.shape[1])
    runtime.install()
    try:
        model(x)
        stored = runtime.last_inputs[0].clone()
        x.add_(99.0)
    finally:
        runtime.remove()

    assert torch.allclose(runtime.last_inputs[0], stored)


def test_sparse_capture_diagnostics_false_keeps_transient_refs_but_freezes_immutable_cache() -> None:
    x = _input()
    model = _model()
    runtime = ReferenceRuntime(model, mode="sparse", total_len=x.shape[1], capture_diagnostics=False)
    runtime.install()
    try:
        model(x)
        assert runtime.last_inputs[0].data_ptr() == x.data_ptr()
        runtime.commit([1])
        frozen_k = runtime.frozen_k[0].clone()
        runtime.last_k[0].add_(99.0)
    finally:
        runtime.remove()

    assert torch.allclose(runtime.frozen_k[0], frozen_k)


def test_remove_restores_hooks_and_wrappers() -> None:
    x0 = _input()
    x1 = x0.clone()
    x1[:, 1, :] = x1[:, 1, :] + 5.0
    model = _model()
    baseline = copy.deepcopy(model)
    runtime = ReferenceRuntime(model, mode="dense", total_len=x0.shape[1])
    original_forward = model.blocks[0].forward
    runtime.install()
    model(x0)
    runtime.commit([1])
    frozen_logits = model(x1)
    runtime.remove()
    restored_logits = model(x1)
    expected_logits = baseline(x1)

    assert _same_bound_method(model.blocks[0].forward, original_forward)
    assert not torch.allclose(frozen_logits, expected_logits)
    assert torch.allclose(restored_logits, expected_logits, atol=1e-6)


def test_sparse_rejects_cache_batch_and_unsupported_architecture_cleanly() -> None:
    bad = BadSparseModel()
    original_forward = bad.blocks[0].forward
    runtime = ReferenceRuntime(bad, mode="sparse", total_len=5)
    with pytest.raises(TypeError):
        runtime.install()
    assert _same_bound_method(bad.blocks[0].forward, original_forward)

    model = _model()
    runtime = ReferenceRuntime(model, mode="sparse", total_len=5)
    runtime.install()
    try:
        with pytest.raises(RuntimeError, match="batch size 1"):
            model(torch.randn(2, 5, 4))
        with pytest.raises(RuntimeError, match="past KV cache"):
            model.blocks[0](torch.randn(1, 5, 4), layer_past=object())
    finally:
        runtime.remove()


if __name__ == "__main__":
    test_no_lock_dense_and_sparse_match_unpatched_model()
    test_dense_and_sparse_active_logits_match_after_controlled_freeze()
    test_sparse_active_rows_still_attend_to_frozen_position_values()
    test_snapshots_are_immutable_after_source_tensor_mutation()
    test_remove_restores_hooks_and_wrappers()
    test_sparse_rejects_cache_batch_and_unsupported_architecture_cleanly()
    print("locking reference tests passed")
