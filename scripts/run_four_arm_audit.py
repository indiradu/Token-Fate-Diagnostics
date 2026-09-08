#!/usr/bin/env python3
"""Run the four-arm PhaseLock intervention audit on identical examples.

The audit deliberately separates the interventions:

* A0: stock dense LLaDA decoding.
* A1: high-confidence token identities are committed early, while every
  hidden state is still recomputed by the dense model.
* A2: replay the exact A1 commitment plan, but freeze committed positions at
  every transformer block. Their hidden states, and therefore their derived
  attention references, remain fixed while active positions continue updating.
* A3: replay A1 again with the A2 references, but compute only active query
  and feed-forward rows; committed rows reuse their cached K/V and hidden
  outputs.

A2/A3 are implemented for the bidirectional RoPE LLaDA MDM architecture used
by this repository. The output records make the operational definitions
explicit; A3 is row-sparse computation over committed generated positions,
not a claim that the entire attention kernel becomes sparse automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from regret_remasking.data import build_prompt, load_examples, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    add_gumbel_noise,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)


def transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    def maybe_blocks(module: Any) -> list[torch.nn.Module] | None:
        if module is None:
            return None
        if hasattr(module, "blocks"):
            blocks = list(module.blocks)
            if blocks:
                return blocks
        if hasattr(module, "block_groups"):
            blocks: list[torch.nn.Module] = []
            for group in module.block_groups:
                blocks.extend(list(group))
            if blocks:
                return blocks
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
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 4:
            blocks = list(module)
            names = " ".join(block.__class__.__name__.lower() for block in blocks[:4])
            if any(key in names for key in ("block", "layer", "decoder")):
                return blocks
    raise AttributeError("could not find LLaDA transformer blocks")


def _apply_rope_at_positions(
    rotary: Any,
    tensor: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Apply the model's RoPE at arbitrary sequence positions."""
    if getattr(rotary.config, "rope_full_precision", False):
        working = tensor.float()
    else:
        working = tensor
    with torch.autocast(tensor.device.type, enabled=False):
        pos_sin, pos_cos = rotary.get_rotary_embedding(seq_len, tensor.device)
        pos_sin = pos_sin[:, :, positions, :].type_as(working)
        pos_cos = pos_cos[:, :, positions, :].type_as(working)
        working = rotary.apply_rotary_pos_emb(pos_sin, pos_cos, working)
    return working.type_as(tensor)


class ReferenceRuntime:
    """Install A2 dense freezing or A3 row-sparse block execution."""

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        total_len: int,
        sparse_backend: str = "reference",
    ):
        if mode not in {"A2", "A3"}:
            raise ValueError(f"unsupported reference mode: {mode}")
        if sparse_backend not in {"reference", "packed"}:
            raise ValueError(f"unsupported sparse backend: {sparse_backend}")
        self.model = model
        self.mode = mode
        self.sparse_backend = sparse_backend
        self.blocks = transformer_blocks(model)
        self.total_len = total_len
        self.device = model_device(model)
        self.locked = torch.zeros((1, total_len), dtype=torch.bool, device=self.device)
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
        self.handles: list[Any] = []
        self.original_forwards: dict[torch.nn.Module, Any] = {}

    def install(self) -> None:
        if self.mode == "A2":
            for layer, block in enumerate(self.blocks):
                self.handles.append(
                    block.register_forward_pre_hook(self._make_pre_hook(layer), with_kwargs=True)
                )
                self.handles.append(
                    block.register_forward_hook(self._make_post_hook(layer))
                )
        else:
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

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, original in self.original_forwards.items():
            block.forward = original  # type: ignore[method-assign]
        self.original_forwards.clear()

    def _make_pre_hook(self, layer: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            x = args[0]
            self.last_inputs[layer] = x.detach()
            if not bool(self.locked.any()):
                return None
            x = x.clone()
            x[self.locked] = self.frozen_inputs[layer][self.locked]
            return (x, *args[1:]), kwargs

        return hook

    def _make_post_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            x, cache = output
            self.last_outputs[layer] = x.detach()
            if not bool(self.locked.any()):
                return output
            x = x.clone()
            x[self.locked] = self.frozen_outputs[layer][self.locked]
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
            raise RuntimeError("A3 row-sparse audit does not support a past KV cache")
        if x.shape[0] != 1:
            raise RuntimeError("the four-arm audit currently requires batch size 1")

        if bool(self.locked.any()):
            x_work = x.clone()
            x_work[self.locked] = self.frozen_inputs[layer][self.locked]
        else:
            x_work = x
        self.last_inputs[layer] = x_work.detach()
        active = ~self.locked[0]
        active_positions = torch.nonzero(active, as_tuple=False).flatten()
        locked_positions = torch.nonzero(self.locked[0], as_tuple=False).flatten()
        seq_len = x.shape[1]
        x_active = x_work.index_select(1, active_positions)

        x_normed = block.attn_norm(x_active)
        if hasattr(block, "att_proj"):
            q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
        else:
            q = block.q_proj(x_normed)
            k = block.k_proj(x_normed)
            v = block.v_proj(x_normed)
        if block.q_norm is not None and block.k_norm is not None:
            q = block.q_norm(q)
            k = block.k_norm(k)

        head_dim = q.shape[-1] // block.config.n_heads
        q = q.view(1, q.shape[1], block.config.n_heads, head_dim).transpose(1, 2)
        k = k.view(1, k.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(1, v.shape[1], block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        rotary = getattr(block, "rotary_emb", None)
        if rotary is not None:
            q = _apply_rope_at_positions(rotary, q, active_positions, seq_len)
            k = _apply_rope_at_positions(rotary, k, active_positions, seq_len)

        if self.sparse_backend == "packed":
            # Keep a persistent full-length K/V workspace.  Active rows are
            # packed into the projection above and copied into the workspace;
            # locked rows remain in-place from their prior snapshot.  This
            # removes the per-layer empty+two-scatter allocation in the
            # reference backend while preserving the exact attention layout.
            full_k = self.packed_k.get(layer)
            full_v = self.packed_v.get(layer)
            expected_shape = (1, block.config.effective_n_kv_heads, seq_len, head_dim)
            if full_k is None or tuple(full_k.shape) != expected_shape:
                full_k = torch.empty(expected_shape, device=x.device, dtype=k.dtype)
                full_v = torch.empty_like(full_k)
                self.packed_k[layer] = full_k
                self.packed_v[layer] = full_v
            full_k.index_copy_(2, active_positions, k)
            full_v.index_copy_(2, active_positions, v)
        else:
            full_k = torch.empty(
                (1, block.config.effective_n_kv_heads, seq_len, head_dim), device=x.device, dtype=k.dtype
            )
            full_v = torch.empty_like(full_k)
            full_k[:, :, active_positions, :] = k
            full_v[:, :, active_positions, :] = v
            if locked_positions.numel() > 0:
                full_k[:, :, locked_positions, :] = self.frozen_k[layer][:, :, locked_positions, :]
                full_v[:, :, locked_positions, :] = self.frozen_v[layer][:, :, locked_positions, :]

        if rotary is not None and locked_positions.numel() == 0:
            # The active assignment above already contains the complete rotated K.
            pass
        row_bias = None
        if attention_bias is not None:
            row_bias = attention_bias[:, :, active_positions, :seq_len]
        att = block._scaled_dot_product_attention(q, full_k, full_v, attn_mask=row_bias)
        att = att.transpose(1, 2).contiguous().view(1, active_positions.numel(), -1)
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
        self.last_outputs[layer] = out.detach()
        self.last_k[layer] = full_k.detach()
        self.last_v[layer] = full_v.detach()
        return out, None

    def commit(self, positions: list[int]) -> None:
        if not positions:
            return
        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        for layer in range(len(self.blocks)):
            if layer not in self.last_inputs or layer not in self.last_outputs:
                raise RuntimeError(f"missing current hidden state for layer {layer}")
            if layer not in self.frozen_inputs:
                hidden_dim = self.last_inputs[layer].shape[-1]
                self.frozen_inputs[layer] = torch.zeros(
                    (1, self.total_len, hidden_dim), device=self.device, dtype=self.last_inputs[layer].dtype
                )
                self.frozen_outputs[layer] = torch.zeros_like(self.frozen_inputs[layer])
            self.frozen_inputs[layer][:, position_tensor, :] = self.last_inputs[layer][:, position_tensor, :]
            self.frozen_outputs[layer][:, position_tensor, :] = self.last_outputs[layer][:, position_tensor, :]
            if self.mode == "A3":
                if layer not in self.last_k or layer not in self.last_v:
                    raise RuntimeError(f"missing current K/V state for layer {layer}")
                if layer not in self.frozen_k:
                    self.frozen_k[layer] = torch.zeros_like(self.last_k[layer])
                    self.frozen_v[layer] = torch.zeros_like(self.last_v[layer])
                self.frozen_k[layer][:, :, position_tensor, :] = self.last_k[layer][:, :, position_tensor, :]
                self.frozen_v[layer][:, :, position_tensor, :] = self.last_v[layer][:, :, position_tensor, :]
        self.locked[:, position_tensor] = True


def _topk_transfer(
    allowed: torch.Tensor,
    x0_probability: torch.Tensor,
    base_k: int,
    lock_fraction: float,
) -> torch.Tensor:
    available = int(allowed.sum().item())
    if available <= 0:
        return torch.zeros_like(allowed)
    k = base_k
    if lock_fraction > 0.0:
        k = max(k, int(math.ceil(lock_fraction * available)))
    k = min(k, available)
    transfer = torch.zeros_like(allowed)
    if k > 0:
        _, indices = torch.topk(torch.where(allowed, x0_probability, torch.full_like(x0_probability, -torch.inf)), k=k)
        transfer[indices] = True
    return transfer


def _jsonable_plan(plan: dict[int, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {str(step): events for step, events in sorted(plan.items())}


@torch.no_grad()
def decode_one(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
    arm: str,
    lock_fraction: float,
    replay_plan: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    device = model_device(model)
    input_ids, attention_mask = prepare_prompts(
        tokenizer,
        [build_prompt(example.question, example.dataset)],
        device,
    )
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError("the four-arm audit requires one example per forward")
    total_len = prompt_len + config.gen_length
    x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], dim=-1
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if config.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of generation blocks")
    if config.gen_length % config.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    runtime = ReferenceRuntime(model, arm, total_len) if arm in {"A2", "A3"} else None
    if runtime is not None:
        runtime.install()
    plan: dict[int, list[dict[str, Any]]] = {}
    commit_events: list[dict[str, Any]] = []
    nfe = 0
    masked_token_forwards = 0
    started = time.perf_counter()
    try:
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_idx * steps_per_block + step_in_block
                mask_index = x == config.mask_id
                masked_token_forwards += int(mask_index[:, prompt_len:].sum().item())
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                nfe += 1
                logits_with_noise = add_gumbel_noise(logits, config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probabilities = log_probs.exp()
                x0_probability = probabilities.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False

                if arm in {"A2", "A3"}:
                    if replay_plan is None:
                        raise ValueError(f"{arm} requires a replay plan")
                    events = replay_plan.get(global_step, [])
                    transfer = torch.zeros_like(allowed)
                    for event in events:
                        pos = int(event["position"])
                        token_id = int(event["token_id"])
                        if not bool(allowed[0, pos]):
                            raise RuntimeError(
                                f"replay plan position {pos} is not masked/allowed at step {global_step} for {arm}"
                            )
                        x0[0, pos] = token_id
                        transfer[0, pos] = True
                else:
                    base_k = int(num_transfer_tokens[0, step_in_block].item())
                    transfer = _topk_transfer(
                        allowed[0], x0_probability[0], base_k, lock_fraction if arm == "A1" else 0.0
                    ).unsqueeze(0)
                    events = []
                    if arm == "A1":
                        for pos in torch.nonzero(transfer[0], as_tuple=False).flatten().tolist():
                            events.append(
                                {
                                    "position": int(pos),
                                    "relative_position": int(pos - prompt_len),
                                    "token_id": int(x0[0, pos].item()),
                                    "confidence": float(x0_probability[0, pos].item()),
                                    "global_step": global_step,
                                    "block": block_idx,
                                    "step_in_block": step_in_block,
                                }
                            )
                        plan[global_step] = events

                selected_positions = torch.nonzero(transfer[0], as_tuple=False).flatten().tolist()
                for pos in selected_positions:
                    event = next((item for item in events if int(item["position"]) == pos), None)
                    if event is None:
                        event = {
                            "position": int(pos),
                            "relative_position": int(pos - prompt_len),
                            "token_id": int(x0[0, pos].item()),
                            "global_step": global_step,
                            "block": block_idx,
                            "step_in_block": step_in_block,
                        }
                    commit_events.append(dict(event))
                x[transfer] = x0[transfer]
                if runtime is not None:
                    runtime.commit(selected_positions)
                del outputs, logits, logits_with_noise, log_probs, probabilities
    finally:
        if runtime is not None:
            runtime.remove()

    if arm != "A1":
        plan = replay_plan or {}
    final_tokens = x[:, prompt_len:].detach().cpu()
    text = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)[0].strip()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    record = {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "arm": arm,
        "generation": text,
        "correct": bool(score_generation(text, example)),
        "token_ids": final_tokens[0].tolist(),
        "gold_answer": example.gold_answer,
        "nfe": nfe,
        "masked_token_forwards": masked_token_forwards,
        "latency_s": latency_s,
        "commit_events": commit_events,
        "prompt_len": prompt_len,
        "gen_length": config.gen_length,
    }
    return record, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A0/A1/A2/A3 PhaseLock intervention audit")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--lock-fraction", type=float, default=0.06)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    examples = load_examples(args.dataset, args.split, args.limit, offset=args.offset)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    mask_id = infer_mask_token_id(tokenizer)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=mask_id,
        temperature=0.0,
    )
    arms = ("A0", "A1", "A2", "A3")
    generation_handles = {arm: (output_dir / f"{arm}.jsonl").open("w", encoding="utf-8") for arm in arms}
    plan_path = output_dir / "A1_commit_plans.jsonl"
    plan_handle = plan_path.open("w", encoding="utf-8")
    summary: dict[str, Any] = {"dataset": args.dataset, "examples": len(examples), "arms": {}}
    try:
        for index, example in enumerate(examples, start=1):
            print(f"[four-arm] {index}/{len(examples)} {example.example_id}", flush=True)
            records: dict[str, dict[str, Any]] = {}
            plan: dict[int, list[dict[str, Any]]] | None = None
            for arm in arms:
                record, generated_plan = decode_one(
                    model,
                    tokenizer,
                    example,
                    config,
                    arm,
                    args.lock_fraction,
                    replay_plan=plan,
                )
                if arm == "A1":
                    plan = generated_plan
                    plan_handle.write(
                        json.dumps({"example_id": example.example_id, "plan": _jsonable_plan(plan)}) + "\n"
                    )
                    plan_handle.flush()
                records[arm] = record
                generation_handles[arm].write(json.dumps(record) + "\n")
                generation_handles[arm].flush()
            for arm in arms:
                summary.setdefault("arms", {}).setdefault(arm, {"examples": 0, "correct": 0, "nfe": 0, "latency_s": 0.0})
                summary["arms"][arm]["examples"] += 1
                summary["arms"][arm]["correct"] += int(records[arm]["correct"])
                summary["arms"][arm]["nfe"] += int(records[arm]["nfe"])
                summary["arms"][arm]["latency_s"] += float(records[arm]["latency_s"])
    finally:
        for handle in generation_handles.values():
            handle.close()
        plan_handle.close()
    for arm, stats in summary["arms"].items():
        stats["accuracy"] = stats["correct"] / stats["examples"] if stats["examples"] else None
        stats["mean_nfe"] = stats["nfe"] / stats["examples"] if stats["examples"] else None
        stats["mean_latency_s"] = stats["latency_s"] / stats["examples"] if stats["examples"] else None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
