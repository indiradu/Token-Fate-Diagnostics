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
from regret_remasking.features import entropy_from_log_probs, local_window_average, safe_jsd, safe_kl
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


def resolve_layer(layer: int, hidden_states: tuple[torch.Tensor, ...]) -> int:
    resolved = layer if layer >= 0 else len(hidden_states) + layer
    if resolved < 0 or resolved >= len(hidden_states):
        raise ValueError(f"layer {layer} resolved to {resolved}, but hidden_states has {len(hidden_states)} entries")
    return resolved


def make_projection(hidden_dim: int, projection_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(hidden_dim, projection_dim, generator=generator, dtype=torch.float32)
    matrix = matrix / math.sqrt(float(projection_dim))
    return matrix.to(device)


def transformer_blocks(model: torch.nn.Module):
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

    raise AttributeError("could not find LLaDA/iLLaDA transformer blocks for hidden hooks")


def register_layer_hooks(model: torch.nn.Module, layers: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    blocks = transformer_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        if layer <= 0 or layer > len(blocks):
            raise ValueError(f"hook capture supports layers 1..{len(blocks)}, got {layer}")

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = tensor

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    return captured, handles


def tensor_value(feature_map: dict[str, torch.Tensor], name: str, batch: int, pos: int) -> float:
    return float(feature_map[name][batch, pos].detach().cpu())


@torch.no_grad()
def collect_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    examples = load_examples(
        args.dataset,
        args.split,
        args.limit,
        offset=args.offset,
        jsonl_path=args.jsonl_path,
    )
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    device = model_device(model)
    mask_id = infer_mask_token_id(tokenizer)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        cpv_window=args.cpv_window,
        mask_id=mask_id,
    )
    if config.gen_length % config.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if steps_per_block < 1 or config.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by gen_length / block_length")

    requested_layers = parse_layers(args.layers)
    if args.capture_mode == "hooks":
        requested_layers = [layer if layer > 0 else args.n_layers + layer + 1 for layer in requested_layers]
    resolved_layers: list[int] | None = None
    captured_layers: dict[int, torch.Tensor] | None = None
    hook_handles: list[Any] = []
    if args.capture_mode == "hooks":
        captured_layers, hook_handles = register_layer_hooks(model, requested_layers)
        resolved_layers = requested_layers
        (output_dir / "layers.json").write_text(
            json.dumps(
                {
                    "requested_layers": requested_layers,
                    "resolved_layers": resolved_layers,
                    "capture_mode": args.capture_mode,
                    "projection_dim": args.projection_dim,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    projections: dict[int, torch.Tensor] = {}
    hidden_chunks: dict[int, list[np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    nfe = 0
    start_time = time.perf_counter()

    for example_idx, example in enumerate(examples, start=1):
        print(f"[probe-data] {example_idx}/{len(examples)} {example.example_id}", flush=True)
        input_ids, attention_mask = prepare_prompts(tokenizer, [build_prompt(example.question, example.dataset)], device)
        batch_size, prompt_len = input_ids.shape
        total_len = prompt_len + config.gen_length
        x = torch.full((batch_size, total_len), config.mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids.clone()
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, config.gen_length), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

        prev_log_probs: torch.Tensor | None = None
        prev_top1 = torch.full_like(x, fill_value=-1)
        runlength = torch.zeros_like(x, dtype=torch.float32)
        example_row_start = len(rows)

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_in_block in range(steps_per_block):
                global_step = block_idx * steps_per_block + step_in_block
                mask_index = x == config.mask_id
                if captured_layers is not None:
                    captured_layers.clear()
                    outputs = model(x, attention_mask=attention_mask)
                    hidden_lookup = captured_layers
                else:
                    outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = tuple(outputs.hidden_states)
                    if resolved_layers is None:
                        resolved_layers = [resolve_layer(layer, hidden_states) for layer in requested_layers]
                    hidden_lookup = {layer: hidden_states[layer] for layer in resolved_layers}
                logits = outputs.logits
                nfe += 1

                if resolved_layers is None:
                    raise RuntimeError("resolved_layers was not initialized")
                if not projections:
                    for layer in resolved_layers:
                        if layer not in hidden_lookup:
                            raise RuntimeError(f"missing hidden capture for layer {layer}")
                        hidden_dim = int(hidden_lookup[layer].shape[-1])
                        projections[layer] = make_projection(
                            hidden_dim,
                            args.projection_dim,
                            seed=args.seed + layer,
                            device=device,
                        )
                        hidden_chunks[layer] = []
                        np.save(output_dir / f"projection_layer_{layer}.npy", projections[layer].detach().cpu().numpy())
                    (output_dir / "layers.json").write_text(
                        json.dumps(
                            {
                            "requested_layers": requested_layers,
                            "resolved_layers": resolved_layers,
                            "capture_mode": args.capture_mode,
                            "projection_dim": args.projection_dim,
                        },
                        indent=2,
                        ),
                        encoding="utf-8",
                    )

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
                jsd = safe_jsd(log_probs, prev_log_probs)
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
                feature_map = {
                    "confidence": confidence,
                    "entropy": entropy,
                    "margin": margin,
                    "kl": kl,
                    "jsd": jsd,
                    "top1_flip": top1_flip,
                    "runlength": runlength,
                    "t_frac": torch.full_like(confidence, (global_step + 1) / config.steps),
                    "local_mask_ratio": local_mask_ratio,
                    "context_volatility": context_volatility,
                }

                positions = torch.nonzero(allowed, as_tuple=False)
                if args.max_rows_per_step > 0 and positions.shape[0] > args.max_rows_per_step:
                    generator = torch.Generator(device=positions.device)
                    generator.manual_seed(args.seed + global_step + example_idx * 1009)
                    keep = torch.randperm(positions.shape[0], generator=generator, device=positions.device)[
                        : args.max_rows_per_step
                    ]
                    positions = positions[keep]

                if positions.numel() > 0:
                    for layer in resolved_layers or []:
                        projected = hidden_lookup[layer][positions[:, 0], positions[:, 1]].float() @ projections[layer]
                        hidden_chunks[layer].append(projected.detach().cpu().numpy().astype(np.float32))
                    for batch_idx, pos in positions.tolist():
                        row: dict[str, Any] = {
                            "example_id": example.example_id,
                            "dataset": example.dataset,
                            "global_step": global_step,
                            "block": block_idx,
                            "step_in_block": step_in_block,
                            "block_t_frac": (step_in_block + 1) / steps_per_block,
                            "position": pos,
                            "relative_position": pos - prompt_len,
                            "top_token": int(x0[batch_idx, pos].detach().cpu()),
                            "score": float(score[batch_idx, pos].detach().cpu()),
                            "selected": False,
                            "mask_token_id": config.mask_id,
                        }
                        for name in FEATURE_NAMES:
                            row[name] = tensor_value(feature_map, name, batch_idx, pos)
                        row["jsd"] = tensor_value(feature_map, "jsd", batch_idx, pos)
                        rows.append(row)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for batch_idx in range(batch_size):
                    k = int(num_transfer_tokens[batch_idx, step_in_block].item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(score[batch_idx], k=k)
                    transfer_index[batch_idx, select_index] = True
                selected_positions = set(tuple(item) for item in torch.nonzero(transfer_index, as_tuple=False).tolist())
                for row in rows[example_row_start:]:
                    if (0, int(row["position"])) in selected_positions and int(row["global_step"]) == global_step:
                        row["selected"] = True

                x[transfer_index] = x0[transfer_index]
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del outputs, logits, logits_with_noise, log_probs, probs, hidden_lookup
                if torch.cuda.is_available() and args.empty_cache:
                    torch.cuda.empty_cache()

        final_tokens = x[:, prompt_len:].detach().cpu()
        for row in rows[example_row_start:]:
            rel_pos = int(row["relative_position"])
            final_token = int(final_tokens[0, rel_pos])
            row["final_token"] = final_token
            row["trace_regret"] = int(int(row["top_token"]) != final_token)

        text = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        generation_rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "gold_answer": example.gold_answer,
                "generation": text,
                "correct": score_generation(text, example),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(generation_rows).to_json(output_dir / "generations.jsonl", orient="records", lines=True)
    for layer, chunks in hidden_chunks.items():
        hidden = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, args.projection_dim), dtype=np.float32)
        np.save(output_dir / f"hidden_layer_{layer}.npy", hidden)

    summary = {
        "examples": len(examples),
        "rows": len(rows),
        "nfe": nfe,
        "latency_s": time.perf_counter() - start_time,
        "accuracy": float(np.mean([row["correct"] for row in generation_rows])) if generation_rows else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for handle in hook_handles:
        handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect token-fate hidden-state probe data.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--jsonl-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--cpv-window", type=int, default=4)
    parser.add_argument("--layers", default="-1,-8,-16")
    parser.add_argument("--capture-mode", choices=["hidden_states", "hooks"], default="hidden_states")
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--max-rows-per-step", type=int, default=0)
    parser.add_argument("--empty-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    collect_dataset(parse_args())


if __name__ == "__main__":
    main()
