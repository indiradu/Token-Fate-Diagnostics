"""Compact reference backend for paper-style representation locking.

``PaperReferenceRuntime`` is the E1 backend used to validate real row removal:
locked rows keep immutable per-layer inputs, outputs, K/V, and logits, while
unlocked rows execute compact attention/feed-forward rows and a compact vocab
head.  The dense mode keeps the same public compact-logit API but runs full
rows as a quality-control reference under the same frozen layer states.
"""

from __future__ import annotations

from typing import Any

import torch

from regret_remasking.locking_reference import (
    _apply_rope_at_positions,
    _cast_attention_bias,
    _clone_detached,
    _model_device,
    _require_sparse_block,
    transformer_blocks,
)


def _find_output_head(model: torch.nn.Module) -> torch.nn.Module:
    """Find the LLaDA vocab head, with a generic toy-model fallback."""

    paths = (
        ("model", "transformer", "ff_out"),
        ("transformer", "ff_out"),
        ("model", "ff_out"),
        ("lm_head",),
        ("model", "lm_head"),
    )
    for path in paths:
        module: Any = model
        for attr in path:
            module = getattr(module, attr, None)
            if module is None:
                break
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError("could not find LLaDA ff_out or generic lm_head output module")


class PaperReferenceRuntime:
    """Install dense or sparse paper-locking execution on a LLaDA-style model.

    The model output is compact after locks are present: returned logits
    correspond to ``output_positions``/``active_positions``.  With no locks the
    sparse path preserves the native full-sequence output and avoids an extra
    gather.  ``frozen_logits`` stores committed-position diagnostics and is not
    rebuilt on each forward.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "sparse",
        total_len: int = 0,
        capture_diagnostics: bool = False,
    ) -> None:
        if mode not in {"dense", "sparse"}:
            raise ValueError(f"unsupported paper reference mode: {mode}")
        if total_len <= 0:
            raise ValueError("total_len must be positive")
        self.model = model
        self.mode = mode
        self.total_len = int(total_len)
        self.capture_diagnostics = bool(capture_diagnostics)
        self.blocks = transformer_blocks(model)
        self.output_head = _find_output_head(model)
        self.device = _model_device(model)

        self.locked = torch.zeros((1, self.total_len), dtype=torch.bool, device=self.device)
        self.active_positions = torch.arange(self.total_len, dtype=torch.long, device=self.device)
        self.locked_positions = torch.empty(0, dtype=torch.long, device=self.device)
        self.output_positions = self.active_positions
        self._has_locks = False
        self._active_count = self.total_len
        self._locked_count = 0

        self.frozen_inputs: dict[int, torch.Tensor] = {}
        self.frozen_outputs: dict[int, torch.Tensor] = {}
        self.frozen_k: dict[int, torch.Tensor] = {}
        self.frozen_v: dict[int, torch.Tensor] = {}
        self.frozen_logits: torch.Tensor | None = None
        self.last_inputs: dict[int, torch.Tensor] = {}
        self.last_outputs: dict[int, torch.Tensor] = {}
        self.last_k: dict[int, torch.Tensor] = {}
        self.last_v: dict[int, torch.Tensor] = {}
        self.last_logits: torch.Tensor | None = None
        self.last_output_hidden: torch.Tensor | None = None
        self.last_output_positions = self.active_positions
        self.packed_k: dict[int, torch.Tensor] = {}
        self.packed_v: dict[int, torch.Tensor] = {}

        self.layer_attention_query_rows: dict[int, int] = {}
        self.layer_projection_rows: dict[int, int] = {}
        self.layer_ffn_rows: dict[int, int] = {}
        self.last_attention_query_rows = 0
        self.last_projection_rows = 0
        self.last_ffn_rows = 0
        self.last_head_rows = 0
        self.total_attention_query_rows = 0
        self.total_projection_rows = 0
        self.total_ffn_rows = 0
        self.total_head_rows = 0

        self.handles: list[Any] = []
        self.original_forwards: dict[torch.nn.Module, Any] = {}
        self.original_attentions: dict[torch.nn.Module, Any] = {}
        self.original_head_forward: Any | None = None

    def install(self) -> None:
        """Install wrappers, rolling back if any patch cannot be completed."""

        if self.handles or self.original_forwards or self.original_head_forward is not None:
            raise RuntimeError("PaperReferenceRuntime is already installed")
        try:
            if self.mode == "dense":
                for layer, block in enumerate(self.blocks):
                    original_attention = block._scaled_dot_product_attention
                    self.original_attentions[block] = original_attention

                    def wrapped_attention(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        *args: Any,
                        _block: torch.nn.Module = block,
                        _layer: int = layer,
                        _original_attention: Any = original_attention,
                        **kwargs: Any,
                    ) -> torch.Tensor:
                        return self._dense_attention(
                            _block,
                            _layer,
                            _original_attention,
                            q,
                            k,
                            v,
                            *args,
                            **kwargs,
                        )

                    block._scaled_dot_product_attention = wrapped_attention  # type: ignore[method-assign]
                    self.handles.append(block.register_forward_pre_hook(self._make_dense_pre_hook(layer), with_kwargs=True))
                    self.handles.append(block.register_forward_hook(self._make_dense_post_hook(layer)))
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
            self.original_head_forward = self.output_head.forward
            self.output_head.forward = self._head_forward  # type: ignore[method-assign]
        except Exception:
            self.remove()
            raise

    def remove(self) -> None:
        """Remove all installed hooks and method patches."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, original in self.original_forwards.items():
            block.forward = original  # type: ignore[method-assign]
        self.original_forwards.clear()
        for block, original in self.original_attentions.items():
            block._scaled_dot_product_attention = original  # type: ignore[method-assign]
        self.original_attentions.clear()
        if self.original_head_forward is not None:
            self.output_head.forward = self.original_head_forward  # type: ignore[method-assign]
            self.original_head_forward = None

    def begin_forward(self) -> None:
        """Reset per-forward diagnostics before a model call."""

        self.layer_attention_query_rows.clear()
        self.layer_projection_rows.clear()
        self.layer_ffn_rows.clear()
        self.last_attention_query_rows = 0
        self.last_projection_rows = 0
        self.last_ffn_rows = 0
        self.last_head_rows = 0
        self.output_positions = self.active_positions
        self.last_output_hidden = None

    @property
    def last_work_counts(self) -> dict[str, Any]:
        """Actual row work from the most recent forward."""

        return {
            "attention_query_rows": self.last_attention_query_rows,
            "projection_rows": self.last_projection_rows,
            "ffn_rows": self.last_ffn_rows,
            "head_rows": self.last_head_rows,
            "layer_attention_query_rows": dict(self.layer_attention_query_rows),
            "layer_projection_rows": dict(self.layer_projection_rows),
            "layer_ffn_rows": dict(self.layer_ffn_rows),
        }

    @property
    def total_work_counts(self) -> dict[str, int]:
        """Cumulative actual row work since installation."""

        return {
            "attention_query_rows": self.total_attention_query_rows,
            "projection_rows": self.total_projection_rows,
            "ffn_rows": self.total_ffn_rows,
            "head_rows": self.total_head_rows,
        }

    def commit(self, absolute_positions: list[int] | torch.Tensor) -> None:
        """Freeze supplied absolute sequence positions from the most recent forward."""

        if isinstance(absolute_positions, torch.Tensor):
            positions_list = [int(pos) for pos in absolute_positions.detach().cpu().flatten().tolist()]
        else:
            positions_list = [int(pos) for pos in absolute_positions]
        if not positions_list:
            return
        if any(position < 0 or position >= self.total_len for position in positions_list):
            raise IndexError("commit positions must be within total_len")
        requested = torch.tensor(sorted(set(positions_list)), dtype=torch.long, device=self.device)
        requested_mask = torch.zeros(self.total_len, dtype=torch.bool, device=self.device)
        requested_mask.index_fill_(0, requested, True)
        new_mask = requested_mask & ~self.locked[0]
        position_tensor = torch.nonzero(new_mask, as_tuple=False).flatten()
        if int(position_tensor.numel()) == 0:
            return

        for layer in range(len(self.blocks)):
            if layer not in self.last_inputs or layer not in self.last_outputs:
                raise RuntimeError(f"missing current hidden state for layer {layer}")
            if layer not in self.frozen_inputs:
                self.frozen_inputs[layer] = torch.zeros_like(self.last_inputs[layer])
                self.frozen_outputs[layer] = torch.zeros_like(self.last_outputs[layer])
            self.frozen_inputs[layer][:, position_tensor, :] = self.last_inputs[layer][:, position_tensor, :]
            self.frozen_outputs[layer][:, position_tensor, :] = self.last_outputs[layer][:, position_tensor, :]
            if layer in self.last_k and layer in self.last_v:
                if layer not in self.frozen_k:
                    self.frozen_k[layer] = torch.zeros_like(self.last_k[layer])
                    self.frozen_v[layer] = torch.zeros_like(self.last_v[layer])
                self.frozen_k[layer][:, :, position_tensor, :] = self.last_k[layer][:, :, position_tensor, :]
                self.frozen_v[layer][:, :, position_tensor, :] = self.last_v[layer][:, :, position_tensor, :]
            elif self.mode == "sparse":
                raise RuntimeError(f"missing current K/V state for layer {layer}")

        self._commit_logits(position_tensor)
        self.locked[:, position_tensor] = True
        self.locked_positions = torch.nonzero(self.locked[0], as_tuple=False).flatten()
        self.active_positions = torch.nonzero(~self.locked[0], as_tuple=False).flatten()
        self._locked_count = int(self.locked_positions.numel())
        self._active_count = int(self.active_positions.numel())
        self._has_locks = self._locked_count > 0

    def _commit_logits(self, position_tensor: torch.Tensor) -> None:
        if self.last_logits is None:
            raise RuntimeError("missing current logits for commit")
        output_positions = self.last_output_positions
        if self.frozen_logits is None:
            shape = (1, self.total_len, self.last_logits.shape[-1])
            self.frozen_logits = torch.zeros(shape, dtype=self.last_logits.dtype, device=self.last_logits.device)
        compact_indices = torch.searchsorted(output_positions, position_tensor)
        valid = compact_indices < output_positions.numel()
        if bool(valid.all()):
            valid = output_positions.index_select(0, compact_indices) == position_tensor
        if not bool(valid.all()):
            missing = position_tensor[~valid].detach().cpu().tolist()
            raise RuntimeError(f"last compact logits do not contain positions {missing}")
        self.frozen_logits[:, position_tensor, :] = self.last_logits.index_select(1, compact_indices)

    def _make_dense_pre_hook(self, layer: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            x = args[0]
            if layer == 0:
                self.begin_forward()
            self._validate_batch_and_length(x)
            x_work = x
            if self._has_locks:
                if layer not in self.frozen_inputs:
                    raise RuntimeError(f"missing frozen dense inputs for layer {layer}")
                x_work = x.clone()
                x_work[:, self.locked_positions, :] = self.frozen_inputs[layer][:, self.locked_positions, :]
            self.last_inputs[layer] = self._capture_tensor(x_work)
            self._record_layer_rows(layer, attention_queries=self.total_len, projected=self.total_len, ffn=self.total_len)
            return (x_work, *args[1:]), kwargs

        return hook

    def _dense_attention(
        self,
        _block: torch.nn.Module,
        layer: int,
        original_attention: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._has_locks and layer in self.frozen_k and layer in self.frozen_v:
            k = k.clone()
            v = v.clone()
            k[:, :, self.locked_positions, :] = self.frozen_k[layer][:, :, self.locked_positions, :]
            v[:, :, self.locked_positions, :] = self.frozen_v[layer][:, :, self.locked_positions, :]
        self.last_k[layer] = self._capture_tensor(k)
        self.last_v[layer] = self._capture_tensor(v)
        return original_attention(q, k, v, *args, **kwargs)

    def _make_dense_post_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            x, cache = self._split_block_output(output)
            if self._has_locks:
                if layer not in self.frozen_outputs:
                    raise RuntimeError(f"missing frozen dense outputs for layer {layer}")
                x = x.clone()
                x[:, self.locked_positions, :] = self.frozen_outputs[layer][:, self.locked_positions, :]
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
        if layer == 0:
            self.begin_forward()
        if layer_past is not None or use_cache:
            raise RuntimeError("sparse PaperReferenceRuntime does not support external past KV cache")
        self._validate_batch_and_length(x)

        x_work = x
        if self._has_locks:
            if layer not in self.frozen_inputs:
                raise RuntimeError(f"missing frozen sparse inputs for layer {layer}")
            x_work = x.clone()
            x_work[:, self.locked_positions, :] = self.frozen_inputs[layer][:, self.locked_positions, :]
        self.last_inputs[layer] = self._capture_tensor(x_work)

        active_positions = self.active_positions
        locked_positions = self.locked_positions
        active_count = self._active_count
        if active_count == 0:
            if layer not in self.frozen_outputs:
                raise RuntimeError(f"all rows are locked but layer {layer} has no frozen output")
            out = self.frozen_outputs[layer].clone()
            self.last_outputs[layer] = self._capture_tensor(out)
            if layer in self.frozen_k and layer in self.frozen_v:
                self.last_k[layer] = self._capture_tensor(self.frozen_k[layer])
                self.last_v[layer] = self._capture_tensor(self.frozen_v[layer])
            self._record_layer_rows(layer, attention_queries=0, projected=0, ffn=0)
            return out, None

        seq_len = x.shape[1]
        x_active = x_work if not self._has_locks else x_work.index_select(1, active_positions)
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

        full_k, full_v = self._kv_workspace(block, layer, seq_len, k)
        if self._has_locks:
            full_k.index_copy_(2, active_positions, k)
            full_v.index_copy_(2, active_positions, v)
            if layer not in self.frozen_k or layer not in self.frozen_v:
                raise RuntimeError(f"missing frozen sparse K/V for layer {layer}")
            full_k.index_copy_(2, locked_positions, self.frozen_k[layer].index_select(2, locked_positions))
            full_v.index_copy_(2, locked_positions, self.frozen_v[layer].index_select(2, locked_positions))
        else:
            full_k = k
            full_v = v

        row_bias = None
        if attention_bias is not None:
            if self._has_locks:
                row_bias = attention_bias[:, :, active_positions, :seq_len]
            else:
                row_bias = attention_bias[:, :, :seq_len, :seq_len]
            row_bias = _cast_attention_bias(block, row_bias, full_k.dtype)
        att = block._scaled_dot_product_attention(q, full_k, full_v, attn_mask=row_bias)
        att = att.transpose(1, 2).contiguous().view(1, active_count, -1)
        self._record_layer_rows(
            layer,
            attention_queries=active_count,
            projected=active_count,
            ffn=active_count,
        )

        active_out = x_active + block.dropout(block.attn_out(att))
        residual = active_out
        ff = block.ff_norm(active_out)
        if hasattr(block, "up_proj"):
            ff, ff_up = block.ff_proj(ff), block.up_proj(ff)
            ff = block.act(ff) * ff_up
        else:
            ff = block.act(block.ff_proj(ff))
        active_out = residual + block.dropout(block.ff_out(ff))

        if self._has_locks:
            out = x_work.clone()
            out[:, active_positions, :] = active_out
            out[:, locked_positions, :] = self.frozen_outputs[layer][:, locked_positions, :]
        else:
            out = active_out
        self.last_outputs[layer] = self._capture_tensor(out)
        self.last_k[layer] = self._capture_tensor(full_k)
        self.last_v[layer] = self._capture_tensor(full_v)
        return out, None

    def _head_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.original_head_forward is None:
            raise RuntimeError("output head was not installed")
        self._validate_batch_and_length(x)
        if self.mode == "dense":
            hidden = x if not self._has_locks else x.index_select(1, self.active_positions)
            self.last_output_hidden = self._capture_tensor(hidden)
            logits_full = self.original_head_forward(x, *args, **kwargs)
            self._record_head_rows(self.total_len)
            logits = logits_full if not self._has_locks else logits_full.index_select(1, self.active_positions)
        else:
            if self._has_locks:
                x_head = x.index_select(1, self.active_positions)
                self.last_output_hidden = self._capture_tensor(x_head)
                logits = self.original_head_forward(x_head, *args, **kwargs)
                self._record_head_rows(self._active_count)
            else:
                self.last_output_hidden = self._capture_tensor(x)
                logits = self.original_head_forward(x, *args, **kwargs)
                self._record_head_rows(self.total_len)
        self.output_positions = self.active_positions
        self.last_output_positions = self.output_positions.detach().clone()
        self.last_logits = self._capture_tensor(logits)
        return logits

    def _kv_workspace(
        self,
        block: torch.nn.Module,
        layer: int,
        seq_len: int,
        active_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (1, block.config.effective_n_kv_heads, seq_len, active_k.shape[-1])
        full_k = self.packed_k.get(layer)
        full_v = self.packed_v.get(layer)
        if full_k is None or tuple(full_k.shape) != expected_shape or full_k.dtype != active_k.dtype:
            full_k = torch.empty(expected_shape, device=active_k.device, dtype=active_k.dtype)
            full_v = torch.empty_like(full_k)
            self.packed_k[layer] = full_k
            self.packed_v[layer] = full_v
        return full_k, full_v

    def _capture_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.capture_diagnostics:
            return _clone_detached(tensor)
        return tensor.detach()

    def _record_layer_rows(self, layer: int, attention_queries: int, projected: int, ffn: int) -> None:
        attention_queries = int(attention_queries)
        projected = int(projected)
        ffn = int(ffn)
        self.layer_attention_query_rows[layer] = attention_queries
        self.layer_projection_rows[layer] = projected
        self.layer_ffn_rows[layer] = ffn
        self.last_attention_query_rows += attention_queries
        self.last_projection_rows += projected
        self.last_ffn_rows += ffn
        self.total_attention_query_rows += attention_queries
        self.total_projection_rows += projected
        self.total_ffn_rows += ffn

    def _record_head_rows(self, rows: int) -> None:
        rows = int(rows)
        self.last_head_rows = rows
        self.total_head_rows += rows

    def _validate_batch_and_length(self, x: torch.Tensor) -> None:
        if x.shape[0] != 1:
            raise RuntimeError("PaperReferenceRuntime currently requires batch size 1")
        if x.shape[1] != self.total_len:
            raise RuntimeError(f"expected sequence length {self.total_len}, got {x.shape[1]}")

    @staticmethod
    def _split_block_output(output: Any) -> tuple[torch.Tensor, Any]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise RuntimeError("PaperReferenceRuntime expects block output as tensor or (tensor, cache)")
            return output[0], output[1]
        return output, None
