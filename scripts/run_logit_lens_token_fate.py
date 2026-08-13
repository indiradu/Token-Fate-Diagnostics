#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from regret_remasking import FEATURE_NAMES
from regret_remasking.data import build_prompt, load_examples, score_generation
from regret_remasking.features import entropy_from_log_probs, local_window_average, safe_kl
from regret_remasking.llada_trace import (
    DecodeConfig,
    add_gumbel_noise,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)


def parse_layers(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def transformer_core(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", model)


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
    inner = transformer_core(model)
    candidates.extend([inner, getattr(inner, "transformer", None), getattr(inner, "model", None)])
    nested = getattr(inner, "model", None)
    if nested is not None:
        candidates.append(getattr(nested, "transformer", None))
    candidates.extend([getattr(model, "transformer", None), getattr(model, "backbone", None)])

    for candidate in candidates:
        blocks = maybe_blocks(candidate)
        if blocks is not None:
            return blocks

    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 4:
            blocks = list(module)
            class_names = " ".join(block.__class__.__name__.lower() for block in blocks[:4])
            if any(key in class_names for key in ("block", "layer", "decoder")):
                return blocks

    raise AttributeError("could not find LLaDA/iLLaDA transformer blocks")


def resolve_hook_layers(model: torch.nn.Module, layers: list[int]) -> list[int]:
    n_layers = len(transformer_blocks(model))
    resolved = []
    for layer in layers:
        item = layer if layer > 0 else n_layers + layer + 1
        if item <= 0 or item > n_layers:
            raise ValueError(f"layer {layer} resolves to {item}, expected 1..{n_layers}")
        resolved.append(item)
    return resolved


def register_layer_hooks(model: torch.nn.Module, layers: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    blocks = transformer_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captured[layer_idx] = output[0] if isinstance(output, tuple) else output

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    return captured, handles


def patch_attention_capture(model: torch.nn.Module, layer: int) -> tuple[dict[str, torch.Tensor], Any]:
    block = transformer_blocks(model)[layer - 1]
    original_attention = block.attention
    captured: dict[str, torch.Tensor] = {}

    def wrapped_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: Any | None = None,
        use_cache: bool = False,
    ):
        with torch.no_grad():
            batch_size, query_len, channels = q.size()
            dtype = k.dtype
            q_work = q
            k_work = k
            if block.q_norm is not None and block.k_norm is not None:
                q_work = block.q_norm(q_work).to(dtype=dtype)
                k_work = block.k_norm(k_work).to(dtype=dtype)
            q_heads = q_work.view(
                batch_size,
                query_len,
                block.config.n_heads,
                channels // block.config.n_heads,
            ).transpose(1, 2)
            k_heads = k_work.view(
                batch_size,
                query_len,
                block.config.effective_n_kv_heads,
                channels // block.config.n_heads,
            ).transpose(1, 2)
            if layer_past is not None:
                past_key, _past_value = layer_past
                k_heads = torch.cat((past_key, k_heads), dim=-2)
            key_len = k_heads.shape[-2]
            if block.config.rope:
                q_heads, k_heads = block.rotary_emb(q_heads, k_heads)
            if q_heads.size(1) != k_heads.size(1):
                k_heads = k_heads.repeat_interleave(q_heads.size(1) // k_heads.size(1), dim=1)
            scores = torch.matmul(q_heads.float(), k_heads.float().transpose(-2, -1))
            scores = scores / math.sqrt(float(q_heads.shape[-1]))
            if attention_bias is not None:
                bias = block._cast_attn_bias(
                    attention_bias[:, :, key_len - query_len : key_len, :key_len],
                    dtype,
                )
                scores = scores + bias.float()
            captured["weights"] = torch.softmax(scores, dim=-1).mean(dim=1).detach()
        return original_attention(
            q,
            k,
            v,
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
        )

    block.attention = wrapped_attention
    return captured, original_attention


def restore_attention(model: torch.nn.Module, layer: int, original_attention: Any) -> None:
    transformer_blocks(model)[layer - 1].attention = original_attention


def lm_head_logits(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    inner = transformer_core(model)
    transformer = inner.transformer
    x = transformer.ln_f(hidden)
    if inner.config.weight_tying:
        weight = transformer.wte.weight
        logits = F.linear(x.to(dtype=weight.dtype), weight, None)
    else:
        weight = transformer.ff_out.weight
        logits = transformer.ff_out(x.to(dtype=weight.dtype))
    if inner.config.scale_logits:
        logits = logits * (1 / math.sqrt(inner.config.d_model))
    return logits


def tensor_value(feature_map: dict[str, torch.Tensor], name: str, batch: int, pos: int) -> float:
    return float(feature_map[name][batch, pos].detach().cpu())


def select_positions(
    allowed: torch.Tensor,
    score: torch.Tensor,
    max_rows_per_step: int,
) -> torch.Tensor:
    positions = torch.nonzero(allowed, as_tuple=False)
    if max_rows_per_step <= 0 or positions.shape[0] <= max_rows_per_step:
        return positions
    values = score[positions[:, 0], positions[:, 1]]
    _, order = torch.topk(values, k=max_rows_per_step)
    return positions[order]


@torch.no_grad()
def confidence_decode_final(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
) -> dict[str, Any]:
    device = model_device(model)
    input_ids, attention_mask = prepare_prompts(tokenizer, [build_prompt(example.question, example.dataset)], device)
    _, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length
    x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    nfe = 0
    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * config.block_length
        block_end = prompt_len + (block_idx + 1) * config.block_length
        block_mask_index = x[:, block_start:block_end] == config.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_in_block in range(steps_per_block):
            mask_index = x == config.mask_id
            logits = model(x, attention_mask=attention_mask).logits
            nfe += 1
            logits_with_noise = add_gumbel_noise(logits, config.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            probs = torch.softmax(logits.float(), dim=-1)
            x0_p = probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            allowed = mask_index.clone()
            allowed[:, :prompt_len] = False
            allowed[:, block_end:] = False
            x0 = torch.where(mask_index, x0, x)
            score = torch.full_like(x0_p, fill_value=-torch.inf)
            score[allowed] = x0_p[allowed]
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            k = int(num_transfer_tokens[0, step_in_block].item())
            if k > 0:
                _, select_index = torch.topk(score[0], k=k)
                transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]
            del logits, logits_with_noise, probs, x0_p, score
    final_tokens = x[:, prompt_len:].detach().cpu()
    generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return {
        "prompt_len": prompt_len,
        "final_tokens": final_tokens,
        "generation": generation,
        "correct": score_generation(generation, example),
        "nfe": nfe,
    }


@torch.no_grad()
def collect_lens_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
    layers: list[int],
    final_tokens: torch.Tensor,
    max_rows_per_step: int,
    attention_layer: int | None,
) -> tuple[list[dict[str, Any]], int]:
    device = model_device(model)
    captured, handles = register_layer_hooks(model, layers)
    captured_attention: dict[str, torch.Tensor] | None = None
    original_attention = None
    if attention_layer is not None:
        captured_attention, original_attention = patch_attention_capture(model, attention_layer)
    try:
        input_ids, attention_mask = prepare_prompts(tokenizer, [build_prompt(example.question, example.dataset)], device)
        _, prompt_len = input_ids.shape
        total_len = prompt_len + config.gen_length
        x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids.clone()
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )
        final_tokens_device = final_tokens.to(device)
        num_blocks = config.gen_length // config.block_length
        steps_per_block = config.steps // num_blocks
        prev_log_probs: torch.Tensor | None = None
        prev_top1 = torch.full_like(x, fill_value=-1)
        runlength = torch.zeros_like(x, dtype=torch.float32)
        rows: list[dict[str, Any]] = []
        nfe = 0

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_in_block in range(steps_per_block):
                global_step = block_idx * steps_per_block + step_in_block
                mask_index = x == config.mask_id
                captured.clear()
                if captured_attention is not None:
                    captured_attention.clear()
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                nfe += 1
                logits_with_noise = add_gumbel_noise(logits, config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                x0_p = probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                top_probs, _ = probs.topk(k=2, dim=-1)
                confidence = top_probs[:, :, 0]
                margin = top_probs[:, :, 0] - top_probs[:, :, 1]
                entropy = entropy_from_log_probs(log_probs)
                kl = safe_kl(log_probs, prev_log_probs)
                top1_flip = ((prev_top1 >= 0) & (prev_top1 != x0)).float()
                runlength = torch.where(prev_top1 == x0, runlength + 1.0, torch.ones_like(runlength))
                local_mask_ratio = local_window_average(mask_index.float(), config.cpv_window)
                context_volatility = local_window_average(kl, config.cpv_window)

                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                x0 = torch.where(mask_index, x0, x)
                score = torch.full_like(confidence, fill_value=-torch.inf)
                score[allowed] = x0_p[allowed]
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                k = int(num_transfer_tokens[0, step_in_block].item())
                if k > 0:
                    _, select_index = torch.topk(score[0], k=k)
                    transfer_index[0, select_index] = True

                feature_map = {
                    "confidence": confidence,
                    "entropy": entropy,
                    "margin": margin,
                    "kl": kl,
                    "top1_flip": top1_flip,
                    "runlength": runlength,
                    "t_frac": torch.full_like(confidence, (global_step + 1) / config.steps),
                    "local_mask_ratio": local_mask_ratio,
                    "context_volatility": context_volatility,
                }
                positions = select_positions(allowed, score, max_rows_per_step)
                if positions.numel() > 0:
                    rel_positions = positions[:, 1] - prompt_len
                    surface_tokens = x0[positions[:, 0], positions[:, 1]]
                    target_tokens = final_tokens_device[positions[:, 0], rel_positions]
                    attn_feature_values: dict[str, torch.Tensor] = {}
                    if captured_attention is not None and "weights" in captured_attention:
                        weights = captured_attention["weights"]
                        attn_feature_values = {
                            "attn_context_volatility": (weights * kl.float()).sum(dim=-1),
                            "attn_mask_ratio": (weights * mask_index.float()).sum(dim=-1),
                            "attn_local_mask_ratio": (weights * local_mask_ratio.float()).sum(dim=-1),
                        }
                    for layer in layers:
                        if layer not in captured:
                            raise RuntimeError(f"missing hidden capture for layer {layer}")
                        hidden = captured[layer][positions[:, 0], positions[:, 1]].float()
                        lens_logits = lm_head_logits(model, hidden).float()
                        lens_top_values, lens_top_ids = torch.topk(lens_logits, k=10, dim=-1)
                        final_logits = lens_logits.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
                        surface_logits = lens_logits.gather(1, surface_tokens.unsqueeze(1)).squeeze(1)
                        final_rank = (lens_logits > final_logits.unsqueeze(1)).sum(dim=1) + 1
                        surface_rank = (lens_logits > surface_logits.unsqueeze(1)).sum(dim=1) + 1
                        for row_idx, (batch_idx, pos) in enumerate(positions.tolist()):
                            rel_pos = int(pos - prompt_len)
                            surface_token = int(surface_tokens[row_idx].detach().cpu())
                            final_token = int(target_tokens[row_idx].detach().cpu())
                            row: dict[str, Any] = {
                                "example_id": example.example_id,
                                "dataset": example.dataset,
                                "layer": layer,
                                "global_step": global_step,
                                "block": block_idx,
                                "step_in_block": step_in_block,
                                "block_t_frac": (step_in_block + 1) / steps_per_block,
                                "position": pos,
                                "relative_position": rel_pos,
                                "top_token": surface_token,
                                "final_token": final_token,
                                "trace_regret": int(surface_token != final_token),
                                "selected": bool(transfer_index[batch_idx, pos].detach().cpu()),
                                "lens_top1_token": int(lens_top_ids[row_idx, 0].detach().cpu()),
                                "lens_top1_matches_final": int(int(lens_top_ids[row_idx, 0].detach().cpu()) == final_token),
                                "lens_top1_matches_surface": int(int(lens_top_ids[row_idx, 0].detach().cpu()) == surface_token),
                                "final_rank": int(final_rank[row_idx].detach().cpu()),
                                "surface_rank": int(surface_rank[row_idx].detach().cpu()),
                                "final_in_top10": int((lens_top_ids[row_idx] == final_token).any().detach().cpu()),
                                "surface_in_top10": int((lens_top_ids[row_idx] == surface_token).any().detach().cpu()),
                                "final_logit_minus_surface_logit": float(
                                    (final_logits[row_idx] - surface_logits[row_idx]).detach().cpu()
                                ),
                                "lens_margin": float((lens_top_values[row_idx, 0] - lens_top_values[row_idx, 1]).detach().cpu()),
                            }
                            for name in FEATURE_NAMES:
                                row[name] = tensor_value(feature_map, name, batch_idx, pos)
                            if attn_feature_values:
                                row["attention_layer"] = attention_layer
                                for name, values in attn_feature_values.items():
                                    row[name] = float(values[batch_idx, pos].detach().cpu())
                            rows.append(row)
                        del lens_logits

                x[transfer_index] = x0[transfer_index]
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del outputs, logits, logits_with_noise, log_probs, probs, score
        return rows, nfe
    finally:
        for handle in handles:
            handle.remove()
        if attention_layer is not None and original_attention is not None:
            restore_attention(model, attention_layer, original_attention)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = rows.copy()
    df["early"] = df["block_t_frac"] <= 0.5
    layer_summary = (
        df.groupby("layer")
        .agg(
            rows=("trace_regret", "size"),
            trace_regret_rate=("trace_regret", "mean"),
            lens_top1_final_rate=("lens_top1_matches_final", "mean"),
            lens_top1_surface_rate=("lens_top1_matches_surface", "mean"),
            final_top10_rate=("final_in_top10", "mean"),
            surface_top10_rate=("surface_in_top10", "mean"),
            mean_final_rank=("final_rank", "mean"),
            median_final_rank=("final_rank", "median"),
            mean_surface_rank=("surface_rank", "mean"),
            mean_final_minus_surface_logit=("final_logit_minus_surface_logit", "mean"),
        )
        .reset_index()
    )
    slice_summary = (
        df.groupby(["layer", "early", "trace_regret"])
        .agg(
            rows=("trace_regret", "size"),
            lens_top1_final_rate=("lens_top1_matches_final", "mean"),
            final_top10_rate=("final_in_top10", "mean"),
            mean_final_rank=("final_rank", "mean"),
            mean_final_minus_surface_logit=("final_logit_minus_surface_logit", "mean"),
        )
        .reset_index()
    )
    return layer_summary, slice_summary


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a layerwise logit-lens token-fate diagnostic.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--cpv-window", type=int, default=4)
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument("--attention-layer", type=int, default=32)
    parser.add_argument("--max-rows-per-step", type=int, default=8)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    examples = load_examples(args.dataset, args.split, args.limit, offset=args.offset)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    mask_id = infer_mask_token_id(tokenizer)
    layers = resolve_hook_layers(model, parse_layers(args.layers))
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        cpv_window=args.cpv_window,
        mask_id=mask_id,
    )
    if config.gen_length % config.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    if config.steps % (config.gen_length // config.block_length) != 0:
        raise ValueError("steps must be divisible by gen_length / block_length")

    start = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    total_nfe = 0
    for idx, example in enumerate(examples, start=1):
        print(f"[logit-lens] {idx}/{len(examples)} {example.example_id}", flush=True)
        baseline = confidence_decode_final(model, tokenizer, example, config)
        lens_rows, lens_nfe = collect_lens_rows(
            model,
            tokenizer,
            example,
            config,
            layers,
            baseline["final_tokens"],
            args.max_rows_per_step,
            args.attention_layer,
        )
        all_rows.extend(lens_rows)
        total_nfe += int(baseline["nfe"]) + int(lens_nfe)
        generation_rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "gold_answer": example.gold_answer,
                "generation": baseline["generation"],
                "correct": bool(baseline["correct"]),
            }
        )
        pd.DataFrame(all_rows).to_csv(output_dir / "logit_lens_rows.csv", index=False)
        pd.DataFrame(generation_rows).to_json(output_dir / "generations.jsonl", orient="records", lines=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_df = pd.DataFrame(all_rows)
    layer_summary, slice_summary = summarize(rows_df)
    layer_summary.to_csv(output_dir / "layer_summary.csv", index=False)
    slice_summary.to_csv(output_dir / "slice_summary.csv", index=False)
    summary = {
        "examples": len(examples),
        "rows": int(len(rows_df)),
        "layers": layers,
        "nfe": int(total_nfe),
        "latency_s": time.perf_counter() - start,
        "accuracy": float(np.mean([row["correct"] for row in generation_rows])) if generation_rows else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Logit-Lens Token-Fate Summary", "", "```json", json.dumps(summary, indent=2), "```", ""]
    if not layer_summary.empty:
        lines.append("## Layer Summary")
        lines.append(table_text(layer_summary))
        lines.append("")
    if not slice_summary.empty:
        lines.append("## Early/Regret Slices")
        lines.append(table_text(slice_summary))
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
