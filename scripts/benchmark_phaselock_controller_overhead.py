#!/usr/bin/env python3
"""Paired timing of the frozen PhaseLock P2 controller on dense A1 decoding."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_phase_lock_fused_replay import _examples, _load_plans
from regret_remasking.data import build_prompt
from regret_remasking.llada_trace import (
    DecodeConfig,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices
from regret_remasking.phase_lock import (
    select_representation_lock,
    select_semantic_transfer,
)
from regret_remasking.phase_lock_influence import (
    SampledAttentionInfluenceCapture,
    multilayer_influence_ready,
    multilayer_max_drift,
    update_joint_patience,
)
from run_four_arm_audit import transformer_blocks


def _install_capture(
    model: torch.nn.Module, layers: list[int]
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    """Capture requested transformer outputs without depending on the CLI runner."""
    blocks = transformer_blocks(model)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("representation layers must be non-empty and unique")
    if any(layer < 1 or layer > len(blocks) for layer in layers):
        raise ValueError(f"representation layers must be in 1..{len(blocks)}")
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer in layers:
        def make_hook(layer_number: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured[layer_number] = output[0] if isinstance(output, tuple) else output

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    return captured, handles


@torch.inference_mode()
def decode_planner(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: DecodeConfig,
    *,
    with_phase2: bool,
    representation_layers: list[int],
    influence_layer: int,
    drift_threshold: float,
    influence_threshold: float,
    patience: int,
    min_age: int,
    lock_fraction: float,
    max_queries: int,
) -> dict[str, Any]:
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset) for example in examples]
    input_ids, prompt_attention = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length
    x = torch.full((batch_size, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attention_mask = torch.cat(
        [
            prompt_attention,
            torch.ones((batch_size, config.gen_length), dtype=prompt_attention.dtype, device=device),
        ],
        dim=-1,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    capture: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    influence: SampledAttentionInfluenceCapture | None = None
    if with_phase2:
        capture, handles = _install_capture(model, representation_layers)
        blocks = transformer_blocks(model)
        influence = SampledAttentionInfluenceCapture(
            blocks[influence_layer - 1], max_queries=max_queries
        )
        influence.install()
    semantic = torch.zeros_like(x, dtype=torch.bool)
    reference = torch.zeros_like(x, dtype=torch.bool)
    age = torch.zeros_like(x, dtype=torch.long)
    joint_count = torch.zeros_like(x, dtype=torch.long)
    previous_hidden: dict[int, torch.Tensor] | None = None
    semantic_plans: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(batch_size)
    ]
    reference_plans: list[dict[int, list[int]]] = [{} for _ in range(batch_size)]
    nfe = 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = block_start + config.block_length
            schedule = get_num_transfer_tokens(
                x[:, block_start:block_end] == config.mask_id, steps_per_block
            )
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                mask_index = x == config.mask_id
                if with_phase2:
                    capture.clear()
                    if influence is None:
                        raise RuntimeError("missing PhaseLock influence capture")
                    unresolved = mask_index.clone()
                    unresolved[:, :prompt_len] = False
                    influence.set_query_mask(unresolved)
                logits = model(x, attention_mask=attention_mask).logits
                nfe += 1
                proposals = torch.argmax(logits, dim=-1)
                max_logits = logits.amax(dim=-1)
                confidence = torch.exp(
                    max_logits.float() - torch.logsumexp(logits.float(), dim=-1)
                )
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                available = allowed.sum(dim=1).to(dtype=torch.long)
                base_k = torch.minimum(available, schedule[:, step_in_block])
                total_k = base_k.clone()
                eligible = torch.zeros_like(allowed)
                if step_in_block < 2:
                    requested = torch.ceil(0.06 * available.float()).to(dtype=torch.long)
                    total_k = torch.minimum(available, torch.maximum(base_k, requested))
                    eligible = allowed & (confidence >= 0.95)
                transfer, _scheduled, _optional = select_semantic_transfer(
                    allowed,
                    confidence,
                    confidence,
                    base_k,
                    total_k,
                    accelerated_eligible=eligible,
                )
                x[transfer] = proposals[transfer]
                selected_mask = transfer & ~semantic
                semantic |= transfer
                for batch_index in range(batch_size):
                    semantic_plans[batch_index][global_step] = [
                        (
                            int(position - prompt_len),
                            int(proposals[batch_index, position].item()),
                        )
                        for position in torch.nonzero(
                            transfer[batch_index], as_tuple=False
                        ).flatten().tolist()
                    ]

                if with_phase2:
                    if set(capture) != set(representation_layers):
                        raise RuntimeError("PhaseLock planner did not capture every layer")
                    current_hidden = {
                        layer: capture[layer].detach() for layer in representation_layers
                    }
                    if previous_hidden is None:
                        drift = torch.zeros_like(x, dtype=torch.float32)
                    else:
                        drift, _per_layer = multilayer_max_drift(
                            current_hidden, previous_hidden
                        )
                    if influence is None or influence.latest is None:
                        raise RuntimeError("PhaseLock influence score was not captured")
                    influence_score = influence.latest
                    old_semantic = semantic & ~selected_mask
                    age[old_semantic] += 1
                    age[selected_mask] = 0
                    joint_count[selected_mask] = 0
                    joint_count = update_joint_patience(
                        joint_count,
                        old_semantic,
                        drift,
                        influence_score,
                        drift_threshold=drift_threshold,
                        influence_threshold=influence_threshold,
                    )
                    candidate = semantic & ~reference
                    ready = multilayer_influence_ready(
                        candidate,
                        drift,
                        influence_score,
                        age,
                        joint_count,
                        drift_threshold=drift_threshold,
                        influence_threshold=influence_threshold,
                        patience=patience,
                        min_age=min_age,
                    )
                    risk = drift * (1.0 + influence_score)
                    selected_reference = select_representation_lock(
                        candidate,
                        ready,
                        risk,
                        lock_fraction,
                    )
                    reference |= selected_reference
                    for batch_index in range(batch_size):
                        reference_plans[batch_index][global_step] = [
                            int(position - prompt_len)
                            for position in torch.nonzero(
                                selected_reference[batch_index], as_tuple=False
                            ).flatten().tolist()
                        ]
                    previous_hidden = current_hidden
                del logits, confidence, max_logits
    finally:
        for handle in handles:
            handle.remove()
        if influence is not None:
            influence.remove()
    torch.cuda.synchronize(device)
    final_tokens = x[:, prompt_len:].detach().cpu()
    return {
        "latency_s": time.perf_counter() - started,
        "token_ids": final_tokens.tolist(),
        "nfe": nfe,
        "semantic_plans": semantic_plans,
        "reference_plans": reference_plans,
        "reference_lock_count": int(reference.sum().item()),
    }


def _source_signature(plan: dict[str, list[dict[str, Any]]], *, semantic: bool) -> dict[int, list[Any]]:
    signature: dict[int, list[Any]] = {}
    for step, events in plan.items():
        if semantic:
            signature[int(step)] = [
                (int(event["relative_position"]), int(event["token_id"]))
                for event in events
            ]
        else:
            signature[int(step)] = [int(event["relative_position"]) for event in events]
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--representation-label", default="mli_ultra")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()
    if args.batch_size < 2 or args.timing_repeats < 4:
        raise ValueError("use batch >=2 and at least four timing repeats")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    source_plans = _load_plans(args.plan_dir, args.representation_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, tokenizer = load_llada("GSAI-ML/LLaDA-8B-Instruct", device="cuda", dtype=args.dtype)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
        temperature=0.0,
    )
    common = {
        "representation_layers": [8, 16, 24],
        "influence_layer": 24,
        "drift_threshold": 0.002,
        "influence_threshold": 0.5,
        "patience": 6,
        "min_age": 4,
        "lock_fraction": 0.04,
        "max_queries": 8,
    }
    rows: list[dict[str, Any]] = []
    plan_mismatches: list[dict[str, Any]] = []
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(f"[phase-controller] batch={batch_id} size={len(batch)}", flush=True)
        for _ in range(args.timing_warmups):
            decode_planner(model, tokenizer, batch, config, with_phase2=False, **common)
            decode_planner(model, tokenizer, batch, config, with_phase2=True, **common)
        for repeat in range(args.timing_repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            records = {
                enabled: decode_planner(
                    model, tokenizer, batch, config, with_phase2=enabled, **common
                )
                for enabled in order
            }
            p1, planner = records[False], records[True]
            rows.append(
                {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "repeat": repeat,
                    "batch_size": len(batch),
                    "p1_latency_s": p1["latency_s"],
                    "planner_latency_s": planner["latency_s"],
                    "controller_overhead_s": planner["latency_s"] - p1["latency_s"],
                    "planner_exact_p1": int(planner["token_ids"] == p1["token_ids"]),
                    "reference_lock_count": planner["reference_lock_count"],
                }
            )
            if repeat == 0:
                for local_index, example in enumerate(batch):
                    source = source_plans[example.example_id]
                    semantic_match = planner["semantic_plans"][local_index] == _source_signature(
                        source["semantic_plan"], semantic=True
                    )
                    reference_match = planner["reference_plans"][local_index] == _source_signature(
                        source["reference_plan"], semantic=False
                    )
                    if not semantic_match or not reference_match:
                        plan_mismatches.append(
                            {
                                "example_id": example.example_id,
                                "semantic_match": semantic_match,
                                "reference_match": reference_match,
                            }
                        )
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch_id"]), []).append(row)
    medians = [
        {
            "p1_latency_s": float(np.median([row["p1_latency_s"] for row in group])),
            "planner_latency_s": float(np.median([row["planner_latency_s"] for row in group])),
        }
        for _, group in sorted(by_batch.items())
    ]
    p1_total = sum(row["p1_latency_s"] for row in medians)
    planner_total = sum(row["planner_latency_s"] for row in medians)
    summary = {
        "examples": len(examples),
        "batches": len(medians),
        "planner_overhead_fraction": planner_total / p1_total - 1.0,
        "planner_overhead_s_per_batch": (planner_total - p1_total) / len(medians),
        "plan_mismatch_count": len(plan_mismatches),
        "planner_exact_p1_rate": float(np.mean([row["planner_exact_p1"] for row in rows])),
        "mean_reference_lock_count": float(np.mean([row["reference_lock_count"] for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output_dir / "plan_mismatches.json").write_text(json.dumps(plan_mismatches, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "plan_dir": str(args.plan_dir), "output_dir": str(args.output_dir), **common}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
