#!/usr/bin/env python3
"""Run the staged PhaseLock algorithm on LLaDA.

This driver separates three irreversible events:

1. semantic_commit: replace a mask with a token identity;
2. reference_lock: freeze the hidden/K/V state of an already committed slot;
3. compute_lock: execute only active query/FFN rows while reusing frozen K/V.

The semantic plan is produced once with dense computation and replayed for the
reference and compute arms. The reference plan is then replayed by the compute
arm. This nested replay makes A1/A2 and A2/A3 comparisons interventionally
matched instead of allowing a later arm to choose a different trajectory.
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
from regret_remasking.phase_lock import (
    PhaseLockPolicy,
    cosine_distance,
    representation_gate_preset,
    representation_ready,
    select_semantic_transfer,
    semantic_priority,
    update_top1_runlength,
)

# Reuse the architecture-specific A2/A3 implementation that was validated by
# the earlier four-arm audit. Keeping one backend avoids silently changing what
# “reference frozen” means between the audit and the PhaseLock sweep.
from run_four_arm_audit import ReferenceRuntime, transformer_blocks


def _feature_map(
    logits: torch.Tensor,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
    prev_top1: torch.Tensor,
    runlength: torch.Tensor,
    cpv_window: int,
    t_frac: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    x0 = torch.argmax(logits, dim=-1)
    x0_probability = probabilities.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    top_probs, _ = probabilities.topk(k=2, dim=-1)
    confidence = top_probs[:, :, 0]
    margin = top_probs[:, :, 0] - top_probs[:, :, 1]
    entropy = (-(probabilities * log_probs).sum(dim=-1)).clamp_min(0.0)
    if prev_log_probs is None:
        kl = torch.zeros(log_probs.shape[:-1], dtype=torch.float32, device=logits.device)
    else:
        kl = (probabilities * (log_probs - prev_log_probs.float())).sum(dim=-1).clamp_min(0.0)
    top1_flip = ((prev_top1 >= 0) & (prev_top1 != x0)).float()
    runlength = update_top1_runlength(prev_top1, x0, runlength)
    # The local helpers in the repository require a 2-D tensor. The window is
    # intentionally computed over the full sequence, matching the trace code.
    from regret_remasking.features import local_window_average

    context_volatility = local_window_average(kl, cpv_window)
    local_mask_ratio = local_window_average(mask_index.float(), cpv_window)
    return (
        {
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "kl": kl,
            "top1_flip": top1_flip,
            "runlength": runlength,
            "t_frac": torch.full_like(confidence, t_frac),
            "local_mask_ratio": local_mask_ratio,
            "context_volatility": context_volatility,
        },
        x0,
        x0_probability,
        log_probs,
        runlength,
    )


def _predict_regret(
    scorer: Any | None,
    feature_map: dict[str, torch.Tensor],
    allowed: torch.Tensor,
    t_frac: float,
) -> torch.Tensor | None:
    if scorer is None:
        return None
    positions = torch.nonzero(allowed, as_tuple=False)
    if positions.numel() == 0:
        return torch.zeros_like(feature_map["confidence"])
    rows: list[list[float]] = []
    for batch, pos in positions.tolist():
        values: list[float] = []
        for name in scorer.features:
            if name == "t_frac":
                values.append(t_frac)
            else:
                values.append(float(feature_map[name][batch, pos].detach().cpu()))
        rows.append(values)
    scores = scorer.score_matrix(np.asarray(rows, dtype=np.float32))
    result = torch.zeros_like(feature_map["confidence"])
    for (batch, pos), score in zip(positions.tolist(), scores):
        result[batch, pos] = float(score)
    return result


def _install_capture(model: torch.nn.Module, layer: int) -> tuple[dict[str, torch.Tensor], Any]:
    blocks = transformer_blocks(model)
    if layer < 1 or layer > len(blocks):
        raise ValueError(f"representation layer must be in 1..{len(blocks)}")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    return captured, blocks[layer - 1].register_forward_hook(hook)


def _json_plan(plan: dict[int, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {str(step): events for step, events in sorted(plan.items())}


@torch.no_grad()
def decode_one(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
    policy: PhaseLockPolicy,
    arm: str,
    scorer: Any | None,
    semantic_plan: dict[int, list[dict[str, Any]]] | None = None,
    reference_plan: dict[int, list[dict[str, Any]]] | None = None,
    ledger_handle: Any | None = None,
    representation_layer: int = 24,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset)]
    input_ids, attention_mask = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError("PhaseLock currently requires one example per forward")
    total_len = prompt_len + config.gen_length
    x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], dim=-1
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if config.steps % num_blocks or config.gen_length % config.block_length:
        raise ValueError("steps/gen_length must be divisible by the number of blocks")

    mode = "dense"
    if arm.startswith("P3") or "_row_sparse" in arm:
        mode = "A3"
    elif arm.endswith("_kv_cache"):
        mode = "A2"
    sparse_backend = "packed" if arm.endswith("_row_sparse_packed") else "reference"
    runtime = (
        ReferenceRuntime(model, mode, total_len, sparse_backend=sparse_backend)
        if mode in {"A2", "A3"}
        else None
    )
    capture, capture_handle = _install_capture(model, representation_layer)
    if runtime is not None:
        # Runtime hooks/wrappers are installed first; capture therefore sees
        # the effective block output that the next layer receives.
        runtime.install()
    semantic_events: list[dict[str, Any]] = []
    reference_events: list[dict[str, Any]] = []
    generated_semantic_plan: dict[int, list[dict[str, Any]]] = {}
    generated_reference_plan: dict[int, list[dict[str, Any]]] = {}
    semantic_positions = torch.zeros((1, total_len), dtype=torch.bool, device=device)
    reference_positions = torch.zeros((1, total_len), dtype=torch.bool, device=device)
    semantic_age = torch.zeros((1, total_len), dtype=torch.long, device=device)
    below_count = torch.zeros((1, total_len), dtype=torch.long, device=device)
    previous_hidden: torch.Tensor | None = None
    previous_rep_drift = torch.zeros((1, total_len), dtype=torch.float32, device=device)
    prev_log_probs: torch.Tensor | None = None
    prev_top1 = torch.full_like(x, -1)
    runlength = torch.zeros_like(x, dtype=torch.float32)
    nfe = 0
    masked_token_forwards = 0
    start = time.perf_counter()

    def write_event(event: dict[str, Any]) -> None:
        if ledger_handle is not None:
            ledger_handle.write(json.dumps(event, ensure_ascii=True) + "\n")

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
                capture.clear()
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                nfe += 1
                feature_map, x0, x0_probability, log_probs, runlength = _feature_map(
                    logits,
                    x,
                    mask_index,
                    prev_log_probs,
                    prev_top1,
                    runlength,
                    config.cpv_window,
                    float(global_step + 1) / float(config.steps),
                )
                t_frac = float(global_step + 1) / float(config.steps)
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                needs_regret = policy.semantic in {
                    "token_fate",
                    "trace",
                    "fate",
                    "token_fate_only",
                    "trace_regret",
                }
                predicted_regret = _predict_regret(scorer, feature_map, allowed, t_frac) if needs_regret else None
                priority = semantic_priority(
                    policy.semantic,
                    feature_map["confidence"],
                    feature_map["margin"],
                    feature_map["entropy"],
                    feature_map["kl"],
                    feature_map["runlength"],
                    feature_map["top1_flip"],
                    predicted_regret=predicted_regret,
                    beta_kl=policy.beta_kl,
                    beta_fate=policy.beta_fate,
                )
                transfer = torch.zeros_like(allowed)
                sem_events_this_step: list[dict[str, Any]] = []
                if semantic_plan is not None:
                    for event in semantic_plan.get(global_step, []):
                        pos = int(event["position"])
                        if not bool(allowed[0, pos]):
                            raise RuntimeError(f"semantic replay position {pos} is unavailable at step {global_step}")
                        transfer[0, pos] = True
                        x0[0, pos] = int(event["token_id"])
                        sem_events_this_step.append(dict(event))
                else:
                    available = int(allowed[0].sum().item())
                    base_k = int(num_transfer_tokens[0, step_in_block].item())
                    k = min(available, max(base_k, int(math.ceil(policy.semantic_lock_fraction * available))))
                    if k > 0:
                        # A posterior difference does not exist on the first
                        # denoising step.  Use the baseline confidence ordering
                        # for that step rather than treating all-zero KL as
                        # meaningful evidence.
                        effective_priority = priority
                        selector_fallback = None
                        if policy.semantic in {"posterior_kl", "kl_only"} and prev_log_probs is None:
                            effective_priority = feature_map["confidence"]
                            selector_fallback = "confidence_no_posterior_history"
                        accelerated_eligible = allowed.clone()
                        if policy.semantic in {
                            "capped_confidence",
                            "gated_confidence",
                            "capped_persistence",
                            "gated_persistence",
                        }:
                            accelerated_eligible &= (
                                feature_map["confidence"] >= policy.semantic_min_confidence
                            )
                            accelerated_eligible &= (
                                feature_map["runlength"] >= policy.semantic_min_runlength
                            )
                            if policy.semantic_optional_steps_per_block >= 0:
                                accelerated_eligible &= (
                                    step_in_block < policy.semantic_optional_steps_per_block
                                )
                        transfer, scheduled_mask, accelerated_mask = select_semantic_transfer(
                            allowed,
                            feature_map["confidence"],
                            effective_priority,
                            base_k,
                            k,
                            accelerated_eligible=accelerated_eligible,
                        )
                        indices = torch.nonzero(transfer[0], as_tuple=False).flatten()
                        for pos in indices.tolist():
                            selection_source = "accelerated" if bool(accelerated_mask[0, pos]) else "scheduled"
                            event = {
                                "position": int(pos),
                                "relative_position": int(pos - prompt_len),
                                "token_id": int(x0[0, pos].item()),
                                "confidence": float(feature_map["confidence"][0, pos].detach().cpu()),
                                "priority": float(effective_priority[0, pos].detach().cpu()),
                                "selection_source": selection_source,
                                "selector_fallback": selector_fallback if selection_source == "accelerated" else None,
                                "scheduled_k": int(scheduled_mask[0].sum().item()),
                                "accelerated_k": int(accelerated_mask[0].sum().item()),
                                "semantic_min_confidence": policy.semantic_min_confidence,
                                "semantic_min_runlength": policy.semantic_min_runlength,
                                "semantic_optional_steps_per_block": policy.semantic_optional_steps_per_block,
                                "global_step": global_step,
                                "block": block_idx,
                                "step_in_block": step_in_block,
                            }
                            sem_events_this_step.append(event)
                    generated_semantic_plan[global_step] = sem_events_this_step

                for event in sem_events_this_step:
                    pos = int(event["position"])
                    event = {
                        **event,
                        "example_id": example.example_id,
                        "arm": arm,
                        "phase": "denoise_transfer" if arm == "A0" else "semantic_commit",
                        "semantic_policy": policy.semantic,
                        "confidence_online": float(feature_map["confidence"][0, pos].detach().cpu()),
                        "margin_online": float(feature_map["margin"][0, pos].detach().cpu()),
                        "entropy_online": float(feature_map["entropy"][0, pos].detach().cpu()),
                        "kl_online": float(feature_map["kl"][0, pos].detach().cpu()),
                        "runlength_online": float(feature_map["runlength"][0, pos].detach().cpu()),
                        "top1_flip_online": float(feature_map["top1_flip"][0, pos].detach().cpu()),
                        "context_volatility_online": float(feature_map["context_volatility"][0, pos].detach().cpu()),
                        "predicted_regret": None if predicted_regret is None else float(predicted_regret[0, pos].detach().cpu()),
                    }
                    semantic_events.append(event)
                    write_event(event)
                selected_positions = torch.nonzero(transfer[0], as_tuple=False).flatten().tolist()
                if selected_positions:
                    x[transfer] = x0[transfer]
                    semantic_positions[0, selected_positions] = True
                    semantic_age[0, selected_positions] = 0
                    below_count[0, selected_positions] = 0

                # The representation test is evaluated after semantic transfer
                # but before the next denoising step. Newly committed positions
                # get a reference snapshot and cannot pass on the same sample.
                current_hidden = capture.get("hidden")
                if current_hidden is None:
                    raise RuntimeError("representation hook did not capture a hidden state")
                hidden = current_hidden.detach()
                if previous_hidden is None:
                    drift = torch.zeros((1, total_len), dtype=torch.float32, device=device)
                else:
                    drift = cosine_distance(hidden, previous_hidden)
                previous_rep_drift = drift
                selected_mask = torch.zeros_like(semantic_positions)
                if selected_positions:
                    selected_mask[0, selected_positions] = True
                old_positions = semantic_positions & ~selected_mask
                semantic_age[old_positions] += 1
                for pos in torch.nonzero(old_positions[0], as_tuple=False).flatten().tolist():
                    if float(drift[0, pos].detach().cpu()) <= policy.representation_threshold:
                        below_count[0, pos] += 1
                    else:
                        below_count[0, pos] = 0

                if runtime is not None and reference_plan is not None:
                    ref_selected: list[int] = []
                    ref_events_this_step = []
                    for event in reference_plan.get(global_step, []):
                        pos = int(event["position"])
                        ref_selected.append(pos)
                        ref_events_this_step.append(dict(event))
                elif runtime is not None:
                    candidate = semantic_positions & ~reference_positions
                    ready = representation_ready(
                        policy.representation,
                        drift,
                        feature_map["confidence"],
                        feature_map["kl"],
                        feature_map["runlength"],
                        semantic_age,
                        below_count,
                        threshold=policy.representation_threshold,
                        patience=policy.representation_patience,
                        min_age=policy.representation_min_age,
                    )
                    ready &= candidate
                    ref_selected = torch.nonzero(ready[0], as_tuple=False).flatten().tolist()
                    ref_events_this_step = []
                    for pos in ref_selected:
                        ref_events_this_step.append(
                            {
                                "position": int(pos),
                                "relative_position": int(pos - prompt_len),
                                "global_step": global_step,
                                "block": block_idx,
                                "step_in_block": step_in_block,
                            }
                        )
                    generated_reference_plan[global_step] = ref_events_this_step
                else:
                    ref_selected = []
                    ref_events_this_step = []

                if runtime is not None and ref_selected:
                    runtime.commit(ref_selected)
                    reference_positions[0, ref_selected] = True
                for event in ref_events_this_step:
                    pos = int(event["position"])
                    event = {
                        **event,
                        "example_id": example.example_id,
                        "arm": arm,
                        "phase": "reference_lock",
                        "representation_policy": policy.representation,
                        "representation_layer": representation_layer,
                        "representation_drift": float(drift[0, pos].detach().cpu()),
                        "representation_threshold": policy.representation_threshold,
                        "below_threshold_count": int(below_count[0, pos].detach().cpu()),
                        "semantic_age": int(semantic_age[0, pos].detach().cpu()),
                        "confidence_online": float(feature_map["confidence"][0, pos].detach().cpu()),
                        "kl_online": float(feature_map["kl"][0, pos].detach().cpu()),
                    }
                    reference_events.append(event)
                    write_event(event)

                # Every currently committed but unfrozen position gets a compact
                # representation audit row. This is the key evidence for premise
                # A/B/C and lets later analysis compare drift against answer harm.
                rep_candidates = semantic_positions & ~reference_positions
                for pos in torch.nonzero(rep_candidates[0], as_tuple=False).flatten().tolist():
                    write_event(
                        {
                            "example_id": example.example_id,
                            "arm": arm,
                            "phase": "representation_observation",
                            "semantic_policy": policy.semantic,
                            "global_step": global_step,
                            "position": int(pos),
                            "relative_position": int(pos - prompt_len),
                            "representation_policy": policy.representation,
                            "representation_layer": representation_layer,
                            "representation_drift": float(drift[0, pos].detach().cpu()),
                            "representation_threshold": policy.representation_threshold,
                            "below_threshold_count": int(below_count[0, pos].detach().cpu()),
                            "semantic_age": int(semantic_age[0, pos].detach().cpu()),
                            "confidence_online": float(feature_map["confidence"][0, pos].detach().cpu()),
                            "kl_online": float(feature_map["kl"][0, pos].detach().cpu()),
                            "compute_policy": policy.compute,
                        }
                    )
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                previous_hidden = hidden
                del outputs, logits, log_probs
    finally:
        capture_handle.remove()
        if runtime is not None:
            runtime.remove()

    final_tokens = x[:, prompt_len:].detach().cpu()
    text = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)[0].strip()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    record = {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "arm": arm,
        "semantic_policy": policy.semantic,
        "representation_policy": policy.representation,
        "compute_policy": policy.compute,
        "generation": text,
        "correct": bool(score_generation(text, example)),
        "token_ids": final_tokens[0].tolist(),
        "gold_answer": example.gold_answer,
        "nfe": nfe,
        "masked_token_forwards": masked_token_forwards,
        "latency_s": time.perf_counter() - start,
        "semantic_commit_count": 0 if arm == "A0" else len(semantic_events),
        "accelerated_commit_count": 0 if arm == "A0" else sum(
            event.get("selection_source") == "accelerated" for event in semantic_events
        ),
        "reference_lock_count": len(reference_events),
        "prompt_len": prompt_len,
        "gen_length": config.gen_length,
    }
    if semantic_plan is not None:
        generated_semantic_plan = semantic_plan
    if reference_plan is not None:
        generated_reference_plan = reference_plan
    return record, generated_semantic_plan, generated_reference_plan


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0}
    return {
        "examples": len(records),
        "accuracy": sum(bool(r["correct"]) for r in records) / len(records),
        "mean_nfe": sum(float(r["nfe"]) for r in records) / len(records),
        "mean_latency_s": sum(float(r["latency_s"]) for r in records) / len(records),
        "mean_semantic_commit_count": sum(float(r["semantic_commit_count"]) for r in records) / len(records),
        "mean_accelerated_commit_count": sum(float(r.get("accelerated_commit_count", 0)) for r in records) / len(records),
        "mean_reference_lock_count": sum(float(r["reference_lock_count"]) for r in records) / len(records),
    }


@torch.no_grad()
def _warmup_model(model: torch.nn.Module, tokenizer: Any, example: Any, config: DecodeConfig) -> None:
    """Warm the first CUDA forward so arm timing is not order-dependent."""
    device = model_device(model)
    input_ids, attention_mask = prepare_prompts(
        tokenizer,
        [build_prompt(example.question, example.dataset)],
        device,
    )
    prompt_len = input_ids.shape[1]
    x = torch.full(
        (1, prompt_len + config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], dim=-1
    )
    _ = model(x, attention_mask=attention_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del x, input_ids, attention_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged PhaseLock semantic/reference/compute experiments")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "humaneval", "human_eval", "human-eval"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--semantic-policies", default="confidence,posterior,consensus")
    parser.add_argument("--representation-policies", default="drift")
    parser.add_argument("--compute-policies", default="row_sparse")
    parser.add_argument("--semantic-lock-fraction", type=float, default=0.06)
    parser.add_argument("--representation-layer", type=int, default=24)
    parser.add_argument(
        "--representation-gate",
        choices=["custom", "strict", "default", "loose", "very_loose"],
        default="custom",
        help="Named drift gate; custom uses the explicit threshold/patience/min-age values.",
    )
    parser.add_argument("--representation-threshold", type=float, default=0.02)
    parser.add_argument("--representation-patience", type=int, default=2)
    parser.add_argument("--representation-min-age", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--beta-fate", type=float, default=2.0)
    parser.add_argument("--semantic-min-confidence", type=float, default=0.0)
    parser.add_argument("--semantic-min-runlength", type=int, default=1)
    parser.add_argument("--semantic-optional-steps-per-block", type=int, default=-1)
    parser.add_argument("--regret-model-path", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.representation_gate != "custom":
        gate = representation_gate_preset(args.representation_gate)
        args.representation_threshold = float(gate["threshold"])
        args.representation_patience = int(gate["patience"])
        args.representation_min_age = int(gate["min_age"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    examples = load_examples(args.dataset, args.split, args.limit, offset=args.offset)
    if not examples:
        raise ValueError("no examples loaded")
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    mask_id = infer_mask_token_id(tokenizer)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=mask_id,
        temperature=0.0,
        cpv_window=4,
    )
    _warmup_model(model, tokenizer, examples[0], config)
    scorer = None
    if args.regret_model_path:
        from regret_remasking.regret_model import RegretScorer

        scorer = RegretScorer(args.regret_model_path)
    semantic_policies = [item.strip() for item in args.semantic_policies.split(",") if item.strip()]
    representation_policies = [item.strip() for item in args.representation_policies.split(",") if item.strip()]
    compute_policies = [item.strip() for item in args.compute_policies.split(",") if item.strip()]
    ledger = (output_dir / "phase_ledger.jsonl").open("w", encoding="utf-8")
    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    plans_path = output_dir / "phase_plans.jsonl"
    try:
        for index, example in enumerate(examples, start=1):
            print(f"[phase-lock] {index}/{len(examples)} {example.example_id}", flush=True)
            # A0 is the ordinary confidence sampler: no extra early semantic
            # lock and no representation/compute intervention.
            base_policy = PhaseLockPolicy(semantic="confidence", representation="always", compute="dense", semantic_lock_fraction=0.0)
            a0, _, _ = decode_one(model, tokenizer, example, config, base_policy, "A0", scorer, ledger_handle=ledger, representation_layer=args.representation_layer)
            records_by_arm.setdefault("A0", []).append(a0)
            _append_jsonl(output_dir / "A0.jsonl", a0)

            for semantic_name in semantic_policies:
                p = PhaseLockPolicy(
                    semantic=semantic_name,
                    representation="none",
                    compute="dense",
                    semantic_lock_fraction=args.semantic_lock_fraction,
                    beta_kl=args.beta_kl,
                    beta_fate=args.beta_fate,
                    semantic_min_confidence=args.semantic_min_confidence,
                    semantic_min_runlength=args.semantic_min_runlength,
                    semantic_optional_steps_per_block=args.semantic_optional_steps_per_block,
                    representation_threshold=args.representation_threshold,
                    representation_patience=args.representation_patience,
                    representation_min_age=args.representation_min_age,
                )
                arm1 = f"P1_{semantic_name}"
                a1, semantic_plan, _ = decode_one(model, tokenizer, example, config, p, arm1, scorer, ledger_handle=ledger, representation_layer=args.representation_layer)
                records_by_arm.setdefault(arm1, []).append(a1)
                _append_jsonl(output_dir / f"{arm1}.jsonl", a1)
                for representation_name in representation_policies:
                    if any(name not in {"dense", "kv_cache", "row_sparse", "row_sparse_packed"} for name in compute_policies):
                        raise ValueError("compute policies must be dense, kv_cache, row_sparse, and/or row_sparse_packed")
                    p2 = PhaseLockPolicy(
                        semantic=semantic_name,
                        representation=representation_name,
                        compute="kv_cache",
                        semantic_lock_fraction=args.semantic_lock_fraction,
                        beta_kl=args.beta_kl,
                        beta_fate=args.beta_fate,
                        semantic_min_confidence=args.semantic_min_confidence,
                        semantic_min_runlength=args.semantic_min_runlength,
                        semantic_optional_steps_per_block=args.semantic_optional_steps_per_block,
                        representation_threshold=args.representation_threshold,
                        representation_patience=args.representation_patience,
                        representation_min_age=args.representation_min_age,
                    )
                    # A2 is the reference-lock arm. It is required whenever
                    # either KV reuse or row-sparse compute is requested because
                    # A3 must replay the exact A2 reference plan.
                    reference_plan: dict[int, list[dict[str, Any]]] = {}
                    wants_row_sparse = any(name in {"row_sparse", "row_sparse_packed"} for name in compute_policies)
                    if "kv_cache" in compute_policies or wants_row_sparse:
                        arm2 = f"P2_{semantic_name}_{representation_name}_kv_cache"
                        a2, _, reference_plan = decode_one(
                            model, tokenizer, example, config, p2, arm2, scorer,
                            semantic_plan=semantic_plan, ledger_handle=ledger,
                            representation_layer=args.representation_layer,
                        )
                        records_by_arm.setdefault(arm2, []).append(a2)
                        _append_jsonl(output_dir / f"{arm2}.jsonl", a2)
                    for compute_name in [name for name in compute_policies if name in {"row_sparse", "row_sparse_packed"}]:
                        p3 = PhaseLockPolicy(
                            semantic=semantic_name,
                            representation=representation_name,
                            compute=compute_name,
                            semantic_lock_fraction=args.semantic_lock_fraction,
                            beta_kl=args.beta_kl,
                            beta_fate=args.beta_fate,
                            semantic_min_confidence=args.semantic_min_confidence,
                            semantic_min_runlength=args.semantic_min_runlength,
                            semantic_optional_steps_per_block=args.semantic_optional_steps_per_block,
                            representation_threshold=args.representation_threshold,
                            representation_patience=args.representation_patience,
                            representation_min_age=args.representation_min_age,
                        )
                        arm3 = f"P3_{semantic_name}_{representation_name}_{compute_name}"
                        a3, _, _ = decode_one(
                            model, tokenizer, example, config, p3, arm3, scorer,
                            semantic_plan=semantic_plan, reference_plan=reference_plan,
                            ledger_handle=ledger, representation_layer=args.representation_layer,
                        )
                        records_by_arm.setdefault(arm3, []).append(a3)
                        _append_jsonl(output_dir / f"{arm3}.jsonl", a3)
                    _append_jsonl(
                        plans_path,
                        {
                            "example_id": example.example_id,
                            "semantic_policy": semantic_name,
                            "representation_policy": representation_name,
                            "compute_policies_requested": compute_policies,
                            "semantic_plan": _json_plan(semantic_plan),
                            "reference_plan": _json_plan(reference_plan),
                        },
                    )
    finally:
        ledger.close()
    summary = {
        "dataset": args.dataset,
        "examples": len(examples),
        "arms": {arm: _summary(records) for arm, records in sorted(records_by_arm.items())},
        "interpretation": {
            "phase1": "P1 vs A0 tests semantic commitment under matched model and schedule settings.",
            "phase2": "P2 with the same semantic plan tests whether representation freezing adds harm.",
            "phase3": "P3 reuses the P2 plan and tests output equivalence and measured runtime for row removal.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
