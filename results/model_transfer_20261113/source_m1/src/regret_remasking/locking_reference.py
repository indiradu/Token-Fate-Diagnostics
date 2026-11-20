"""Reference backends for PhaseLock representation-freeze experiments.

The implementation is intentionally adapted from
``c43be08:scripts/run_four_arm_audit.py``.  That historical script named the
same two mechanisms A2/A3; this module exposes the narrower experiment-v2 API:
``dense`` freezes full block inputs/outputs, while ``sparse`` reuses frozen
K/V rows and computes query/feed-forward rows only for active positions.
"""

from __future__ import annotations

from typing import Any

import torch


def transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find the model's transformer blocks across common LLaDA wrappers."""

    def maybe_blocks(module: Any) -> list[torch.nn.Module] | None:
        if module is None:
            return None
        if hasattr(module, "blocks"):
            blocks = list(module.blocks)
            if blocks:
                return blocks
        if hasattr(module, "block_groups"):
            grouped_blocks: list[torch.nn.Module] = []
            for group in module.block_groups:
                grouped_blocks.extend(list(group))
            if grouped_blocks:
                return grouped_blocks
        for attr in ("layers", "h"):
            layers = getattr(module, attr, None)
            if layers is not None:
                blocks = list(layers)
                if blocks:
                    return blocks
        return None

    candidates: list[Any] = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.extend([inner, getattr(inner, "transformer", None), getattr(inner, "model", None)])
    candidates.extend([getattr(model, "transformer", None), getattr(model, "backbone", None)])
    for candidate in candidates:
        blocks = maybe_blocks(candidate)
        if blocks is not None:
            return blocks
    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 1:
            blocks = list(module)
            names = " ".join(block.__class__.__name__.lower() for block in blocks[:4])
            if any(key in names for key in ("block", "layer", "decoder")):
                return blocks
    raise AttributeError("could not find transformer blocks")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _clone_detached(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def _cast_attention_bias(
    block: torch.nn.Module,
    attention_bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_bias is None:
        return None
    cast = getattr(block, "_cast_attn_bias", None)
    if cast is not None:
        return cast(attention_bias, dtype)
    return attention_bias.to(dtype=dtype)


def _apply_rope_at_positions(
    rotary: Any,
    tensor: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Apply the model's RoPE at arbitrary absolute sequence positions."""

    if getattr(rotary.config, "rope_full_precision", False):
        working = tensor.float()
    else:
        working = tensor
    device_type = tensor.device.type if tensor.device.type != "cpu" else "cpu"
    with torch.autocast(device_type, enabled=False):
        pos_sin, pos_cos = rotary.get_rotary_embedding(seq_len, tensor.device)
        pos_sin = pos_sin[:, :, positions, :].type_as(working)
        pos_cos = pos_cos[:, :, positions, :].type_as(working)
        working = rotary.apply_rotary_pos_emb(pos_sin, pos_cos, working)
    return working.type_as(tensor)


def _require_sparse_block(block: torch.nn.Module) -> None:
    required = (
        "attn_norm",
        "ff_norm",
        "attn_out",
        "ff_proj",
        "ff_out",
        "act",
        "dropout",
        "_scaled_dot_product_attention",
        "config",
    )
    missing = [name for name in required if not hasattr(block, name)]
    has_fused = hasattr(block, "att_proj") and hasattr(block, "fused_dims")
    has_split = all(hasattr(block, name) for name in ("q_proj", "k_proj", "v_proj"))
    if missing or not (has_fused or has_split):
        detail = ", ".join(missing) if missing else "attention projections"
        raise TypeError(f"unsupported sparse ReferenceRuntime block architecture: missing {detail}")
    config = getattr(block, "config")
    for name in ("n_heads", "effective_n_kv_heads"):
        if not hasattr(config, name):
            raise TypeError(f"unsupported sparse ReferenceRuntime block config: missing {name}")


def _project_kv(
    block: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_normed = block.attn_norm(x)
    if hasattr(block, "att_proj"):
        _q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
    else:
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)
    dtype = k.dtype
    if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
        k = block.k_norm(k).to(dtype=dtype)
    head_dim = k.shape[-1] // block.config.effective_n_kv_heads
    k = k.view(x.shape[0], x.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(x.shape[0], x.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
    rotary = getattr(block, "rotary_emb", None)
    if rotary is not None:
        k = _apply_rope_at_positions(rotary, k, positions, seq_len)
    return k, v


class ReferenceRuntime:
    """Install dense or sparse representation locking on a LLaDA-style model.

    Args:
        model: Model containing transformer blocks.
        mode: ``"dense"`` freezes block input/output rows while still executing
            dense blocks. ``"sparse"`` computes only active rows and reuses
            frozen K/V rows for full-sequence attention.
        total_len: Sequence length for the run.
        sparse_backend: ``"reference"`` allocates fresh K/V workspaces each
            layer. ``"packed"`` reuses the same full-length K/V buffers.
        attention_layout: Sparse attention query layout. ``"full"`` preserves
            the native full-sequence query tensor shape/stride while
            ``"compact"`` uses active query rows only for diagnostics.
        capture_diagnostics: When true, clone full last-input/output/K/V
            snapshots for validation. Decoder paths may set this false to keep
            only transient references needed by ``commit``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "dense",
        total_len: int = 0,
        sparse_backend: str = "reference",
        attention_layout: str = "full",
        capture_diagnostics: bool = True,
    ) -> None:
        if mode not in {"dense", "sparse"}:
            raise ValueError(f"unsupported reference mode: {mode}")
        if sparse_backend not in {"reference", "packed"}:
            raise ValueError(f"unsupported sparse backend: {sparse_backend}")
        if attention_layout not in {"full", "compact"}:
            raise ValueError(f"unsupported sparse attention layout: {attention_layout}")
        if total_len <= 0:
            raise ValueError("total_len must be positive")
        self.model = model
        self.mode = mode
        self.sparse_backend = sparse_backend
        self.attention_layout = attention_layout
        self.capture_diagnostics = bool(capture_diagnostics)
        self.blocks = transformer_blocks(model)
        self.total_len = int(total_len)
        self.device = _model_device(model)
        self.locked = torch.zeros((1, self.total_len), dtype=torch.bool, device=self.device)
        self.frozen_inputs: dict[int, torch.Tensor] = {}
        self.frozen_outputs: dict[int, torch.Tensor] = {}
        self.frozen_k: dict[int, torch.Tensor] = {}
        self.frozen_v: dict[int, torch.Tensor] = {}
        self.last_inputs: dict[int, torch.Tensor] = {}
        self.last_outputs: dict[int, torch.Tensor] = {}
        self.last_k: dict[int, torch.Tensor] = {}
        self.last_v: dict[int, torch.Tensor] = {}
        self.packed_k: dict[int, torch.Tensor] = {}
        self.packed_v: dict[int, torch.Tensor] = {}
        self.native_q_shapes: dict[int, torch.Size] = {}
        self.native_q_strides: dict[int, tuple[int, ...]] = {}
        self.native_k_shapes: dict[int, torch.Size] = {}
        self.native_k_strides: dict[int, tuple[int, ...]] = {}
        self.native_v_shapes: dict[int, torch.Size] = {}
        self.native_v_strides: dict[int, tuple[int, ...]] = {}
        self.last_attention_query_rows = 0
        self.last_projection_rows = 0
        self.last_ffn_rows = 0
        self.total_attention_query_rows = 0
        self.total_projection_rows = 0
        self.total_ffn_rows = 0
        self.handles: list[Any] = []
        self.original_forwards: dict[torch.nn.Module, Any] = {}

    def install(self) -> None:
        """Install hooks/wrappers, rolling back if any layer cannot be patched."""

        if self.handles or self.original_forwards:
            raise RuntimeError("ReferenceRuntime is already installed")
        try:
            if self.mode == "dense":
                for layer, block in enumerate(self.blocks):
                    self.handles.append(block.register_forward_pre_hook(self._make_pre_hook(layer), with_kwargs=True))
                    self.handles.append(block.register_forward_hook(self._make_post_hook(layer)))
            else:
                for block in self.blocks:
                    _require_sparse_block(block)
                for layer, block in enumerate(self.blocks):
                    original = block.forward
                    self.original_forwards[block] = original

                    def wrapped(
                        x: torch.Tensor,
                        attention_bias: torch.Tensor | None = None,
                        layer_past: Any = None,
                        use_cache: bool = False,
                        *,
                        _block: torch.nn.Module = block,
                        _layer: int = layer,
                    ) -> Any:
                        return self._sparse_forward(
                            _block,
                            _layer,
                            x,
                            attention_bias=attention_bias,
                            layer_past=layer_past,
                            use_cache=use_cache,
                        )

                    block.forward = wrapped  # type: ignore[method-assign]
        except Exception:
            self.remove()
            raise

    def remove(self) -> None:
        """Remove all installed hooks/wrappers."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, original in self.original_forwards.items():
            block.forward = original  # type: ignore[method-assign]
        self.original_forwards.clear()

    def _store_kv_snapshot(self, block: torch.nn.Module, layer: int, x: torch.Tensor) -> None:
        if not self.capture_diagnostics:
            return
        try:
            positions = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
            k, v = _project_kv(block, x, positions, x.shape[1])
        except (AttributeError, TypeError, RuntimeError):
            return
        self.last_k[layer] = _clone_detached(k)
        self.last_v[layer] = _clone_detached(v)

    def _make_pre_hook(self, layer: int):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            x = args[0]
            self._validate_batch_and_length(x)
            if bool(self.locked.any()):
                if layer not in self.frozen_inputs:
                    raise RuntimeError(f"missing frozen dense inputs for layer {layer}")
                x = x.clone()
                x[self.locked] = self.frozen_inputs[layer][self.locked]
            self.last_inputs[layer] = self._capture_tensor(x)
            self._store_kv_snapshot(module, layer, x)
            return (x, *args[1:]), kwargs

        return hook

    def _make_post_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            x, cache = self._split_block_output(output)
            if bool(self.locked.any()):
                if layer not in self.frozen_outputs:
                    raise RuntimeError(f"missing frozen dense outputs for layer {layer}")
                x = x.clone()
                x[self.locked] = self.frozen_outputs[layer][self.locked]
            self.last_outputs[layer] = self._capture_tensor(x)
            return x, cache

        return hook

    def _sparse_forward(
        self,
        block: torch.nn.Module,
        layer: int,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if layer_past is not None or use_cache:
            raise RuntimeError("sparse ReferenceRuntime does not support external past KV cache")
        self._validate_batch_and_length(x)

        if bool(self.locked.any()):
            if layer not in self.frozen_inputs:
                raise RuntimeError(f"missing frozen sparse inputs for layer {layer}")
            x_work = x.clone()
            x_work[self.locked] = self.frozen_inputs[layer][self.locked]
        else:
            x_work = x
        self.last_inputs[layer] = self._capture_tensor(x_work)
        active = ~self.locked[0]
        active_positions = torch.nonzero(active, as_tuple=False).flatten()
        locked_positions = torch.nonzero(self.locked[0], as_tuple=False).flatten()
        self._record_sparse_rows(attention_queries=0, projected=0, ffn=0)
        if active_positions.numel() == 0:
            if layer not in self.frozen_outputs:
                raise RuntimeError(f"all rows are locked but layer {layer} has no frozen output")
            out = self.frozen_outputs[layer].clone()
            self.last_outputs[layer] = self._capture_tensor(out)
            self.last_k[layer] = self._capture_tensor(self.frozen_k[layer])
            self.last_v[layer] = self._capture_tensor(self.frozen_v[layer])
            return out, None

        seq_len = x.shape[1]
        x_active = x_work.index_select(1, active_positions)
        active_count = int(active_positions.numel())
        x_normed = block.attn_norm(x_active)
        if hasattr(block, "att_proj"):
            q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
        else:
            q = block.q_proj(x_normed)
            k = block.k_proj(x_normed)
            v = block.v_proj(x_normed)
        dtype = k.dtype
        if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
            q = block.q_norm(q).to(dtype=dtype)
            k = block.k_norm(k).to(dtype=dtype)

        head_dim = q.shape[-1] // block.config.n_heads
        q = q.view(1, q.shape[1], block.config.n_heads, head_dim).transpose(1, 2)
        k = k.view(1, k.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(1, v.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        rotary = getattr(block, "rotary_emb", None)
        if rotary is not None:
            q = _apply_rope_at_positions(rotary, q, active_positions, seq_len)
            k = _apply_rope_at_positions(rotary, k, active_positions, seq_len)
        if active_count == seq_len:
            self.native_q_shapes[layer] = q.shape
            self.native_q_strides[layer] = q.stride()
            self.native_k_shapes[layer] = k.shape
            self.native_k_strides[layer] = k.stride()
            self.native_v_shapes[layer] = v.shape
            self.native_v_strides[layer] = v.stride()

        full_k, full_v = self._kv_workspace(block, layer, seq_len, k)
        full_k.index_copy_(2, active_positions, k)
        full_v.index_copy_(2, active_positions, v)
        if locked_positions.numel() > 0:
            if layer not in self.frozen_k or layer not in self.frozen_v:
                raise RuntimeError(f"missing frozen sparse K/V for layer {layer}")
            full_k.index_copy_(2, locked_positions, self.frozen_k[layer].index_select(2, locked_positions))
            full_v.index_copy_(2, locked_positions, self.frozen_v[layer].index_select(2, locked_positions))

        if self.attention_layout == "compact":
            attention_q = q
            row_bias = None
            if attention_bias is not None:
                row_bias = attention_bias[:, :, active_positions, :seq_len]
                row_bias = _cast_attention_bias(block, row_bias, full_k.dtype)
            att = block._scaled_dot_product_attention(attention_q, full_k, full_v, attn_mask=row_bias)
            att = att.transpose(1, 2).contiguous().view(1, active_count, -1)
            attention_query_rows = active_count
        else:
            attention_q = self._full_query_workspace(layer, q, seq_len)
            attention_q.zero_()
            attention_q.index_copy_(2, active_positions, q)
            row_bias = None
            if attention_bias is not None:
                row_bias = attention_bias[:, :, :seq_len, :seq_len]
                row_bias = _cast_attention_bias(block, row_bias, full_k.dtype)
            att = block._scaled_dot_product_attention(attention_q, full_k, full_v, attn_mask=row_bias)
            att = att.index_select(2, active_positions)
            att = att.transpose(1, 2).contiguous().view(1, active_count, -1)
            attention_query_rows = seq_len
        self._record_sparse_rows(attention_queries=attention_query_rows, projected=active_count, ffn=active_count)
        active_out = x_active + block.dropout(block.attn_out(att))

        residual = active_out
        ff = block.ff_norm(active_out)
        if hasattr(block, "up_proj"):
            ff, ff_up = block.ff_proj(ff), block.up_proj(ff)
            ff = block.act(ff) * ff_up
        else:
            ff = block.act(block.ff_proj(ff))
        active_out = residual + block.dropout(block.ff_out(ff))

        out = x_work.clone()
        out[:, active_positions, :] = active_out
        if locked_positions.numel() > 0:
            out[:, locked_positions, :] = self.frozen_outputs[layer][:, locked_positions, :]
        self.last_outputs[layer] = self._capture_tensor(out)
        self.last_k[layer] = self._capture_tensor(full_k)
        self.last_v[layer] = self._capture_tensor(full_v)
        return out, None

    def _full_query_workspace(self, layer: int, active_q: torch.Tensor, seq_len: int) -> torch.Tensor:
        if layer not in self.native_q_strides:
            raise RuntimeError(f"missing native full-query layout for layer {layer}")
        expected_shape = (active_q.shape[0], active_q.shape[1], seq_len, active_q.shape[-1])
        native_shape = self.native_q_shapes[layer]
        if tuple(native_shape) != expected_shape:
            raise RuntimeError(
                f"native full-query shape for layer {layer} changed from {tuple(native_shape)} to {expected_shape}"
            )
        return torch.empty_strided(
            expected_shape,
            self.native_q_strides[layer],
            device=active_q.device,
            dtype=active_q.dtype,
        )

    def _kv_workspace(
        self,
        block: torch.nn.Module,
        layer: int,
        seq_len: int,
        active_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (1, block.config.effective_n_kv_heads, seq_len, active_k.shape[-1])
        expected_stride = None
        if self.attention_layout == "full" and layer in self.native_k_strides and layer in self.native_v_strides:
            if tuple(self.native_k_shapes[layer]) != expected_shape or tuple(self.native_v_shapes[layer]) != expected_shape:
                raise RuntimeError(f"native full-K/V shape for layer {layer} changed")
            expected_stride = (self.native_k_strides[layer], self.native_v_strides[layer])
        if self.sparse_backend == "packed":
            full_k = self.packed_k.get(layer)
            full_v = self.packed_v.get(layer)
            stride_mismatch = (
                expected_stride is not None
                and full_k is not None
                and full_v is not None
                and (full_k.stride() != expected_stride[0] or full_v.stride() != expected_stride[1])
            )
            if full_k is None or tuple(full_k.shape) != expected_shape or full_k.dtype != active_k.dtype or stride_mismatch:
                if expected_stride is None:
                    full_k = torch.empty(expected_shape, device=active_k.device, dtype=active_k.dtype)
                    full_v = torch.empty_like(full_k)
                else:
                    full_k = torch.empty_strided(
                        expected_shape,
                        expected_stride[0],
                        device=active_k.device,
                        dtype=active_k.dtype,
                    )
                    full_v = torch.empty_strided(
                        expected_shape,
                        expected_stride[1],
                        device=active_k.device,
                        dtype=active_k.dtype,
                    )
                self.packed_k[layer] = full_k
                self.packed_v[layer] = full_v
            return full_k, full_v
        if expected_stride is not None:
            full_k = torch.empty_strided(expected_shape, expected_stride[0], device=active_k.device, dtype=active_k.dtype)
            full_v = torch.empty_strided(expected_shape, expected_stride[1], device=active_k.device, dtype=active_k.dtype)
            return full_k, full_v
        full_k = torch.empty(expected_shape, device=active_k.device, dtype=active_k.dtype)
        return full_k, torch.empty_like(full_k)

    def _capture_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.capture_diagnostics:
            return _clone_detached(tensor)
        return tensor.detach()

    def _record_sparse_rows(self, attention_queries: int, projected: int, ffn: int) -> None:
        self.last_attention_query_rows = int(attention_queries)
        self.last_projection_rows = int(projected)
        self.last_ffn_rows = int(ffn)
        self.total_attention_query_rows += int(attention_queries)
        self.total_projection_rows += int(projected)
        self.total_ffn_rows += int(ffn)

    def commit(self, positions: list[int]) -> None:
        """Freeze the supplied absolute sequence positions from last forward."""

        if not positions:
            return
        if any(position < 0 or position >= self.total_len for position in positions):
            raise IndexError("commit positions must be within total_len")
        position_tensor = torch.tensor(sorted(set(positions)), dtype=torch.long, device=self.device)
        for layer in range(len(self.blocks)):
            if layer not in self.last_inputs or layer not in self.last_outputs:
                raise RuntimeError(f"missing current hidden state for layer {layer}")
            if layer not in self.frozen_inputs:
                self.frozen_inputs[layer] = torch.zeros_like(self.last_inputs[layer])
                self.frozen_outputs[layer] = torch.zeros_like(self.last_outputs[layer])
            self.frozen_inputs[layer][:, position_tensor, :] = self.last_inputs[layer][:, position_tensor, :]
            self.frozen_outputs[layer][:, position_tensor, :] = self.last_outputs[layer][:, position_tensor, :]
            if self.mode == "sparse":
                if layer not in self.last_k or layer not in self.last_v:
                    raise RuntimeError(f"missing current K/V state for layer {layer}")
                if layer not in self.frozen_k:
                    self.frozen_k[layer] = torch.zeros_like(self.last_k[layer])
                    self.frozen_v[layer] = torch.zeros_like(self.last_v[layer])
                self.frozen_k[layer][:, :, position_tensor, :] = self.last_k[layer][:, :, position_tensor, :]
                self.frozen_v[layer][:, :, position_tensor, :] = self.last_v[layer][:, :, position_tensor, :]
        self.locked[:, position_tensor] = True

    def _validate_batch_and_length(self, x: torch.Tensor) -> None:
        if x.shape[0] != 1:
            raise RuntimeError("ReferenceRuntime currently requires batch size 1")
        if x.shape[1] != self.total_len:
            raise RuntimeError(f"expected sequence length {self.total_len}, got {x.shape[1]}")

    @staticmethod
    def _split_block_output(output: Any) -> tuple[torch.Tensor, Any]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise RuntimeError("ReferenceRuntime expects block output as tensor or (tensor, cache)")
            return output[0], output[1]
        return output, None
