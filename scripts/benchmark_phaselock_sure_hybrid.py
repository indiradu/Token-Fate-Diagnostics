#!/usr/bin/env python3
"""H0--H4 audit of a true SureLock/PhaseLock hybrid.

H0 is dense LLaDA. H1 is published-style SureLock. H2 retains H1's immediate
row-removal executor while adding full-Sure semantic commitment. H3 replaces
H2's reference decision with PhaseLock's multi-layer/influence safety veto and
keeps execution dense. H4 replays H3's exact semantic/reference plans through
the independently profitability-gated fused executor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_phase_lock_batched_online import _answer_changed, _global_plan, decode_batch
from benchmark_phase_lock_fused_replay import _examples
from benchmark_phaselock_matched_plans import (
    _bootstrap_gain,
    reference_replay_plan,
    semantic_replay_plan,
)
from benchmark_surelock_vs_phaselock import decode_surelock
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices, posterior_kl
from regret_remasking.phase_lock import select_representation_lock
from regret_remasking.phase_lock_influence import (
    SampledAttentionInfluenceCapture,
    multilayer_influence_ready,
    multilayer_max_drift,
    update_joint_patience,
)
from regret_remasking.phase_lock_sparse import (
    BucketedReferenceRuntime,
    FusedLatencyTable,
)
from run_four_arm_audit import transformer_blocks


ARMS = ("h0", "h1", "h2", "h3", "h4")
TIMED_METHODS = ("h0", "h1", "h2", "h3", "h3_replay", "h4")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sure_selected_from_stats(
    uncertainty: torch.Tensor,
    divergence: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float,
    confidence_percentile: float,
) -> torch.Tensor:
    """Apply SureLock's candidate-relative uncertainty and KL thresholds."""
    if not (
        uncertainty.shape == divergence.shape == candidate.shape
        and uncertainty.ndim == 2
        and candidate.dtype == torch.bool
    ):
        raise ValueError("Sure statistics and candidate mask must align")
    selected = torch.zeros_like(candidate)
    quantile = confidence_percentile / 100.0
    for batch_index in range(candidate.shape[0]):
        values = uncertainty[batch_index][candidate[batch_index]]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.float(), quantile)
        selected[batch_index] = (
            candidate[batch_index]
            & (uncertainty[batch_index] <= threshold)
            & (divergence[batch_index] <= epsilon)
        )
    return selected


def _fallback_progress(
    selected: torch.Tensor,
    allowed: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    output = selected.clone()
    for batch_index in range(allowed.shape[0]):
        positions = torch.nonzero(allowed[batch_index], as_tuple=False).flatten()
        if positions.numel() == 0 or bool(output[batch_index].any()):
            continue
        scores = confidence[batch_index].index_select(0, positions)
        output[batch_index, positions[int(torch.argmax(scores).item())]] = True
    return output


def guard_semantic_candidates(
    selected: torch.Tensor, allowed: torch.Tensor, confidence: torch.Tensor,
    *, min_confidence: float = 0.0, cap_fraction: float = 1.0,
) -> torch.Tensor:
    """Guard optional Sure commits; the caller separately guarantees progress."""
    if not 0 <= min_confidence <= 1 or not 0 < cap_fraction <= 1:
        raise ValueError("invalid confidence floor or per-update cap")
    guarded = selected & allowed & (confidence >= min_confidence)
    if cap_fraction == 1.0:
        return guarded
    for b in range(allowed.shape[0]):
        limit = max(1, math.ceil(cap_fraction * int(allowed[b].sum().item())))
        positions = torch.nonzero(guarded[b], as_tuple=False).flatten()
        if positions.numel() > limit:
            keep = positions[torch.topk(confidence[b, positions], limit).indices]
            guarded[b] = False
            guarded[b, keep] = True
    return guarded


def speed_reference_gate(
    proposed: torch.Tensor, age: torch.Tensor, *, prompt_len: int,
    policy: str, min_age: int, drift: torch.Tensor | None = None,
    count: torch.Tensor | None = None, threshold: float = 0.01, patience: int = 1,
) -> torch.Tensor:
    """Keep H2's prompt eligibility; gate only generated reference snapshots.

    age=0: token changed after this forward; age=1: first post-commit forward.
    A drift decision requires age>=2 (two post-commit observations).
    """
    if policy not in {"sure", "delay", "drift", "live", "generated_only"}:
        raise ValueError("unknown speed reference policy")
    if min_age < 1 or patience < 1 or threshold < 0:
        raise ValueError("invalid reference gate parameters")
    result = proposed.clone()
    if policy == "live":
        return torch.zeros_like(result)
    if policy == "generated_only":
        result[:, :prompt_len] = False
    if policy == "delay":
        result[:, prompt_len:] &= age[:, prompt_len:] >= min_age
    if policy == "drift":
        if drift is None or count is None:
            raise ValueError("drift gate needs fresh observations and patience")
        result[:, prompt_len:] &= (
            (age[:, prompt_len:] >= max(2, min_age))
            & (drift[:, prompt_len:] <= threshold)
            & (count[:, prompt_len:] >= patience)
        )
    return result


@torch.inference_mode()
def decode_hybrid(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: DecodeConfig,
    *,
    mode: str,
    epsilon: float,
    confidence_percentile: float,
    bucket_width_multiple: int,
    representation_layers: list[int],
    influence_layer: int,
    drift_threshold: float,
    influence_threshold: float,
    representation_patience: int,
    representation_min_age: int,
    representation_lock_fraction: float,
    max_queries: int,
    speed_reference_policy: str = "sure",
    refresh_interval: int = 0,
    semantic_min_confidence: float = 0.0,
    semantic_cap_fraction: float = 1.0,
) -> dict[str, Any]:
    """Run H2 or H3 while sharing one full-posterior KL per update."""
    if mode not in {"h2", "h3"}:
        raise ValueError("hybrid mode must be h2 or h3")
    if refresh_interval < 0:
        raise ValueError("refresh interval must be nonnegative")
    if mode != "h2" and (speed_reference_policy != "sure" or refresh_interval
                         or semantic_min_confidence or semantic_cap_fraction != 1.0):
        raise ValueError("new speed controls apply to H2 only")
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset) for example in examples]
    input_ids, attention_mask = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length
    x = torch.full((batch_size, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (batch_size, config.gen_length),
                dtype=attention_mask.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if config.gen_length % config.block_length or config.steps % num_blocks:
        raise ValueError("schedule must divide evenly into generation blocks")

    blocks = transformer_blocks(model)
    for layer in set(representation_layers + [influence_layer]):
        if layer < 1 or layer > len(blocks):
            raise ValueError("representation/influence layer is out of range")
    runtime = BucketedReferenceRuntime(
        blocks,
        batch_size=batch_size,
        total_len=total_len,
        fusion_mode="exact",
    )
    runtime.install_dense()
    influence: SampledAttentionInfluenceCapture | None = None
    if mode == "h3":
        influence = SampledAttentionInfluenceCapture(
            blocks[influence_layer - 1], max_queries=max_queries
        )
        influence.install()

    previous_log_probs: torch.Tensor | None = None
    semantic = torch.zeros_like(x, dtype=torch.bool)
    age = torch.zeros_like(x, dtype=torch.long)
    joint_count = torch.zeros_like(x, dtype=torch.long)
    previous_hidden: dict[int, torch.Tensor] | None = None
    semantic_plans: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(batch_size)
    ]
    reference_plans: list[dict[int, list[int]]] = [
        {} for _ in range(batch_size)
    ]
    nfe = 0
    semantic_commit_count = 0
    reference_lock_count = 0
    removed_row_token_forwards = 0
    first_compute_step: int | None = None
    controller_latency_s = 0.0
    decision_trace: list[dict[str, Any]] = []
    refresh_count = 0

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = block_start + config.block_length
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                if refresh_interval and nfe and nfe % refresh_interval == 0 and runtime.locked_count:
                    # Releasing representations never releases semantic identities.
                    # Start with a genuinely live full pass, not cached zero drift.
                    runtime.remove()
                    runtime = BucketedReferenceRuntime(
                        blocks, batch_size=batch_size, total_len=total_len,
                        fusion_mode="exact",
                    )
                    runtime.install_dense()
                    previous_log_probs = None
                    previous_hidden = None
                    joint_count.zero_()
                    refresh_count += 1
                if runtime.compute_active:
                    removed_row_token_forwards += runtime.locked_count
                mask_before = x == config.mask_id
                if influence is not None:
                    unresolved = mask_before.clone()
                    unresolved[:, :prompt_len] = False
                    influence.set_query_mask(unresolved)

                logits = model(x, attention_mask=attention_mask).logits
                nfe += 1
                torch.cuda.synchronize(device)
                controller_started = time.perf_counter()
                log_probs = F.log_softmax(logits.float(), dim=-1)
                proposals = torch.argmax(logits, dim=-1)
                confidence = log_probs.amax(dim=-1).exp()
                uncertainty = 1.0 - confidence
                divergence = (
                    torch.full_like(uncertainty, torch.inf)
                    if previous_log_probs is None
                    else posterior_kl(log_probs, previous_log_probs)
                )

                allowed = mask_before.clone()
                allowed[:, :block_start] = False
                allowed[:, block_end:] = False
                semantic_selected = sure_selected_from_stats(
                    uncertainty,
                    divergence,
                    allowed,
                    epsilon=epsilon,
                    confidence_percentile=confidence_percentile,
                )
                if semantic_min_confidence or semantic_cap_fraction != 1.0:
                    semantic_selected = guard_semantic_candidates(
                        semantic_selected, allowed, confidence,
                        min_confidence=semantic_min_confidence,
                        cap_fraction=semantic_cap_fraction,
                    )
                semantic_selected = _fallback_progress(
                    semantic_selected, allowed, confidence
                )
                x[semantic_selected] = proposals[semantic_selected]
                selected_mask = semantic_selected & ~semantic
                semantic |= semantic_selected
                semantic_commit_count += int(semantic_selected.sum().item())
                for batch_index in range(batch_size):
                    events = [
                        (
                            int(position - prompt_len),
                            int(proposals[batch_index, position].item()),
                        )
                        for position in torch.nonzero(
                            semantic_selected[batch_index], as_tuple=False
                        ).flatten().tolist()
                    ]
                    if events:
                        semantic_plans[batch_index][global_step] = events

                reference_candidate = (x != config.mask_id) & ~runtime.locked
                sure_reference = sure_selected_from_stats(
                    uncertainty,
                    divergence,
                    reference_candidate,
                    epsilon=epsilon,
                    confidence_percentile=confidence_percentile,
                )

                if mode == "h2":
                    selected_reference = sure_reference
                    if speed_reference_policy != "sure":
                        old_semantic = semantic & ~selected_mask
                        age[old_semantic] += 1
                        age[selected_mask] = 0
                        drift = None
                        if speed_reference_policy == "drift":
                            current_hidden = {
                                layer: runtime.last_outputs[layer - 1].detach().clone()
                                for layer in representation_layers
                            }
                            drift = torch.full_like(x, torch.inf, dtype=torch.float32)
                            if previous_hidden is not None:
                                drift, _ = multilayer_max_drift(current_hidden, previous_hidden)
                            fresh = old_semantic & ~runtime.locked & (age >= 2)
                            joint_count = torch.where(
                                fresh & (drift <= drift_threshold), joint_count + 1,
                                torch.zeros_like(joint_count),
                            )
                            previous_hidden = current_hidden
                        selected_reference = speed_reference_gate(
                            sure_reference, age, prompt_len=prompt_len,
                            policy=speed_reference_policy, min_age=representation_min_age,
                            drift=drift, count=joint_count, threshold=drift_threshold,
                            patience=representation_patience,
                        )
                else:
                    if influence is None or influence.latest is None:
                        raise RuntimeError("Phase-2 influence score was not captured")
                    current_hidden = {
                        layer: runtime.last_outputs[layer - 1].detach()
                        for layer in representation_layers
                    }
                    if previous_hidden is None:
                        drift = torch.zeros_like(x, dtype=torch.float32)
                    else:
                        drift, _ = multilayer_max_drift(current_hidden, previous_hidden)
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
                    generated_candidate = semantic & ~runtime.locked
                    ready = multilayer_influence_ready(
                        generated_candidate,
                        drift,
                        influence_score,
                        age,
                        joint_count,
                        drift_threshold=drift_threshold,
                        influence_threshold=influence_threshold,
                        patience=representation_patience,
                        min_age=representation_min_age,
                    )
                    veto_survivor = generated_candidate & sure_reference & ready
                    risk = drift * (1.0 + influence_score)
                    selected_reference = select_representation_lock(
                        generated_candidate,
                        veto_survivor,
                        risk,
                        representation_lock_fraction,
                    )
                    previous_hidden = current_hidden

                newly_locked = [
                    (int(batch_index), int(position))
                    for batch_index, position in torch.nonzero(
                        selected_reference, as_tuple=False
                    ).tolist()
                ]
                if newly_locked:
                    runtime.commit(newly_locked)
                    reference_lock_count += len(newly_locked)
                    if mode == "h2":
                        plans = _global_plan(
                            runtime.active_counts, total_len, bucket_width_multiple
                        )
                        if runtime.compute_active:
                            runtime.set_compute_plan(plans)
                        else:
                            runtime.activate_compute(plans)
                            if first_compute_step is None:
                                first_compute_step = global_step
                    else:
                        for batch_index in range(batch_size):
                            positions = [
                                int(position - prompt_len)
                                for position in torch.nonzero(
                                    selected_reference[batch_index], as_tuple=False
                                ).flatten().tolist()
                            ]
                            if positions:
                                reference_plans[batch_index][global_step] = positions

                decision_trace.append(
                    {
                        "step": global_step,
                        "semantic_commits": int(semantic_selected.sum().item()),
                        "sure_reference_proposals": int(sure_reference.sum().item()),
                        "reference_locks": len(newly_locked),
                        "total_reference_locks": runtime.locked_count,
                        "compute_active": runtime.compute_active,
                        "prompt_locks": int(selected_reference[:, :prompt_len].sum().item()),
                        "same_update_reference_locks": int((selected_reference & selected_mask).sum().item()),
                        "active_counts": list(runtime.active_counts),
                        "refresh_count": refresh_count,
                    }
                )
                previous_log_probs = log_probs.detach()
                torch.cuda.synchronize(device)
                controller_latency_s += time.perf_counter() - controller_started
                del logits, log_probs, proposals, confidence, uncertainty, divergence
    finally:
        if influence is not None:
            influence.remove()
        runtime.remove()

    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    return {
        "arm": mode,
        "token_ids": final_tokens.tolist(),
        "generations": [value.strip() for value in generations],
        "latency_s": latency_s,
        "controller_latency_s": controller_latency_s,
        "nfe": nfe,
        "semantic_commit_count": semantic_commit_count,
        "reference_lock_count": reference_lock_count,
        "removed_row_token_forwards": removed_row_token_forwards,
        "switch_step": first_compute_step,
        "semantic_plans": semantic_plans,
        "reference_plans": reference_plans,
        "decision_trace": decision_trace,
        "prompt_len": prompt_len,
        "total_len": total_len,
        "refresh_count": refresh_count,
        "speed_reference_policy": speed_reference_policy,
    }


def _plan_signature(record: dict[str, Any]) -> tuple[Any, Any]:
    return record["semantic_plans"], record["reference_plans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--latency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--sure-epsilon", type=float, default=0.005)
    parser.add_argument("--sure-confidence-percentile", type=float, default=20.0)
    parser.add_argument("--representation-layers", default="8,16,24")
    parser.add_argument("--influence-layer", type=int, default=24)
    parser.add_argument("--drift-threshold", type=float, default=0.002)
    parser.add_argument("--influence-threshold", type=float, default=0.5)
    parser.add_argument("--representation-patience", type=int, default=6)
    parser.add_argument("--representation-min-age", type=int, default=4)
    parser.add_argument("--representation-lock-fraction", type=float, default=0.04)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--compute-min-locked-fraction", type=float, default=0.20)
    parser.add_argument("--compute-min-remaining-steps", type=int, default=8)
    parser.add_argument("--compute-min-projected-saved-rows", type=int, default=1500)
    parser.add_argument("--compute-min-expected-gain", type=float, default=0.03)
    parser.add_argument("--bucket-width-multiple", type=int, default=16)
    parser.add_argument("--max-bucket-size", type=int, default=8)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=251)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.timing_warmups < 0 or args.timing_repeats < 1:
        raise ValueError("use batch >=2, non-negative warmups, and positive repeats")
    representation_layers = [
        int(value) for value in args.representation_layers.split(",") if value
    ]
    if not representation_layers:
        raise ValueError("provide at least one representation layer")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.latency_table.open(encoding="utf-8") as handle:
        latency_table = FusedLatencyTable.from_rows(json.load(handle))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
        temperature=0.0,
    )
    hybrid_kwargs = {
        "epsilon": args.sure_epsilon,
        "confidence_percentile": args.sure_confidence_percentile,
        "bucket_width_multiple": args.bucket_width_multiple,
        "representation_layers": representation_layers,
        "influence_layer": args.influence_layer,
        "drift_threshold": args.drift_threshold,
        "influence_threshold": args.influence_threshold,
        "representation_patience": args.representation_patience,
        "representation_min_age": args.representation_min_age,
        "representation_lock_fraction": args.representation_lock_fraction,
        "max_queries": args.max_queries,
    }
    replay_common = {
        "semantic_lock_fraction": 0.06,
        "semantic_min_confidence": 0.95,
        "semantic_optional_steps_per_block": 2,
        "representation_policy": "capped_drift",
        "representation_delay_updates": 1,
        "representation_layer": representation_layers[0],
        "representation_drift_threshold": args.drift_threshold,
        "representation_patience": args.representation_patience,
        "representation_min_age": args.representation_min_age,
        "representation_lock_fraction": args.representation_lock_fraction,
        "compute_min_locked_fraction": args.compute_min_locked_fraction,
        "compute_min_remaining_steps": args.compute_min_remaining_steps,
        "compute_min_projected_saved_rows": args.compute_min_projected_saved_rows,
        "compute_min_expected_gain": args.compute_min_expected_gain,
        "bucket_width_multiple": args.bucket_width_multiple,
        "max_bucket_size": args.max_bucket_size,
        "force_semantic_plan_tokens": True,
    }

    rows: list[dict[str, Any]] = []
    timing_batches: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], int] = {}

    def score(text: str, example: Any) -> int:
        key = (str(example.example_id), text)
        if key not in score_cache:
            score_cache[key] = int(bool(score_generation(text, example)))
        return score_cache[key]

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(f"[sure-hybrid] batch={batch_id} size={len(batch)}", flush=True)
        fixed_h3 = decode_hybrid(
            model, tokenizer, batch, config, mode="h3", **hybrid_kwargs
        )
        replay = {
            "semantic_plans": [
                semantic_replay_plan(plan) for plan in fixed_h3["semantic_plans"]
            ],
            "reference_plans": [
                reference_replay_plan(plan) for plan in fixed_h3["reference_plans"]
            ],
        }
        for local_index, example in enumerate(batch):
            plans.append(
                {
                    "example_id": example.example_id,
                    "semantic_plan": replay["semantic_plans"][local_index],
                    "reference_plan": replay["reference_plans"][local_index],
                }
            )

        def run(method: str) -> dict[str, Any]:
            if method == "h0":
                result = decode_surelock(
                    model,
                    tokenizer,
                    batch,
                    config,
                    epsilon=None,
                    confidence_percentile=args.sure_confidence_percentile,
                    bucket_width_multiple=args.bucket_width_multiple,
                )
                result["reference_lock_count"] = 0
                result["switch_step"] = None
                return result
            if method == "h1":
                result = decode_surelock(
                    model,
                    tokenizer,
                    batch,
                    config,
                    epsilon=args.sure_epsilon,
                    confidence_percentile=args.sure_confidence_percentile,
                    bucket_width_multiple=args.bucket_width_multiple,
                )
                result["reference_lock_count"] = result["lock_count"]
                result["switch_step"] = result["first_lock_step"]
                return result
            if method in {"h2", "h3"}:
                result = decode_hybrid(
                    model, tokenizer, batch, config, mode=method, **hybrid_kwargs
                )
                if method == "h3" and _plan_signature(result) != _plan_signature(fixed_h3):
                    raise RuntimeError("H3 controller produced a non-deterministic plan")
                return result
            arm = "A2" if method == "h3_replay" else "A3"
            return decode_batch(
                model,
                tokenizer,
                batch,
                config,
                arm,
                latency_table,
                **replay_common,
                **replay,
            )

        for warmup in range(args.timing_warmups):
            shift = warmup % len(TIMED_METHODS)
            for method in TIMED_METHODS[shift:] + TIMED_METHODS[:shift]:
                run(method)
        for repeat in range(args.timing_repeats):
            shift = repeat % len(TIMED_METHODS)
            order = TIMED_METHODS[shift:] + TIMED_METHODS[:shift]
            records = {method: run(method) for method in order}
            if records["h3"]["token_ids"] != records["h3_replay"]["token_ids"]:
                raise RuntimeError("H3 online output differs from exact dense replay")
            if records["h3_replay"]["token_ids"] != records["h4"]["token_ids"]:
                raise RuntimeError("H4 compute output differs from H3 representation replay")
            controller_overhead = max(
                0.0,
                float(records["h3"]["latency_s"])
                - float(records["h3_replay"]["latency_s"]),
            )
            h4_estimated_online = float(records["h4"]["latency_s"]) + controller_overhead
            timing_batches.append(
                {
                    "batch_id": batch_id,
                    "repeat": repeat,
                    "execution_order": "_then_".join(order),
                    **{
                        f"{method}_latency_s": float(records[method]["latency_s"])
                        for method in TIMED_METHODS
                    },
                    "h3_controller_overhead_s": controller_overhead,
                    "h4_estimated_online_latency_s": h4_estimated_online,
                    "h4_compute_switched": int(records["h4"].get("switch_step") is not None),
                    "h4_removed_row_token_forwards": int(
                        records["h4"].get("removed_row_token_forwards", 0)
                    ),
                }
            )
            correctness = {
                arm: [
                    score(text, example)
                    for text, example in zip(records[arm]["generations"], batch)
                ]
                for arm in ARMS
            }
            for local_index, example in enumerate(batch):
                row: dict[str, Any] = {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "repeat": repeat,
                    "example_id": example.example_id,
                }
                for arm in ARMS:
                    row[f"{arm}_correct"] = correctness[arm][local_index]
                    row[f"{arm}_exact_h0"] = int(
                        records[arm]["token_ids"][local_index]
                        == records["h0"]["token_ids"][local_index]
                    )
                    row[f"{arm}_answer_changed_h0"] = int(
                        _answer_changed(
                            args.dataset,
                            records[arm]["generations"][local_index],
                            records["h0"]["generations"][local_index],
                        )
                    )
                    row[f"{arm}_nfe"] = int(records[arm]["nfe"])
                for left, right in zip(ARMS, ARMS[1:]):
                    row[f"{right}_exact_{left}"] = int(
                        records[right]["token_ids"][local_index]
                        == records[left]["token_ids"][local_index]
                    )
                    row[f"{right}_answer_changed_{left}"] = int(
                        _answer_changed(
                            args.dataset,
                            records[right]["generations"][local_index],
                            records[left]["generations"][local_index],
                        )
                    )
                rows.append(row)
                if any(
                    correctness[right][local_index] != correctness[left][local_index]
                    for left, right in zip(ARMS, ARMS[1:])
                ):
                    failures.append(
                        {
                            **row,
                            "outputs": {
                                arm: records[arm]["generations"][local_index]
                                for arm in ARMS
                            },
                        }
                    )

    stable: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable.setdefault(str(row["example_id"]), row)
    stable_rows = list(stable.values())
    medians: list[dict[str, float]] = []
    for batch_id in sorted({int(row["batch_id"]) for row in timing_batches}):
        group = [row for row in timing_batches if int(row["batch_id"]) == batch_id]
        medians.append(
            {
                key: float(np.median([float(row[key]) for row in group]))
                for key in (
                    "h0_latency_s",
                    "h1_latency_s",
                    "h2_latency_s",
                    "h3_latency_s",
                    "h3_replay_latency_s",
                    "h4_latency_s",
                    "h3_controller_overhead_s",
                    "h4_estimated_online_latency_s",
                )
            }
        )
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "examples": len(stable_rows),
        "timing_pairs": len(medians),
        "accuracy": {
            arm: float(np.mean([row[f"{arm}_correct"] for row in stable_rows]))
            for arm in ARMS
        },
        "mean_nfe": {
            arm: float(np.mean([row[f"{arm}_nfe"] for row in stable_rows]))
            for arm in ARMS
        },
        "transitions": {},
        "runtime_gain_vs_h0": {},
        "mean_h3_controller_overhead_s": float(
            np.mean([row["h3_controller_overhead_s"] for row in medians])
        ),
        "h4_compute_switch_rate": float(
            np.mean([row["h4_compute_switched"] for row in timing_batches])
        ),
        "mean_h4_removed_row_token_forwards": float(
            np.mean([row["h4_removed_row_token_forwards"] for row in timing_batches])
        ),
    }
    for left, right in zip(ARMS, ARMS[1:]):
        regressions = sum(
            row[f"{left}_correct"] == 1 and row[f"{right}_correct"] == 0
            for row in stable_rows
        )
        improvements = sum(
            row[f"{left}_correct"] == 0 and row[f"{right}_correct"] == 1
            for row in stable_rows
        )
        summary["transitions"][f"{left}_to_{right}"] = {
            "exact_rate": float(
                np.mean([row[f"{right}_exact_{left}"] for row in stable_rows])
            ),
            "answer_change_rate": float(
                np.mean(
                    [row[f"{right}_answer_changed_{left}"] for row in stable_rows]
                )
            ),
            "regressions": int(regressions),
            "improvements": int(improvements),
        }
    for offset, arm in enumerate(ARMS[1:]):
        candidate_key = (
            "h4_estimated_online_latency_s" if arm == "h4" else f"{arm}_latency_s"
        )
        gain, interval = _bootstrap_gain(
            medians,
            candidate_key,
            "h0_latency_s",
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        summary["runtime_gain_vs_h0"][arm] = {
            "mean": gain,
            "ci95": interval,
        }
    h4_compute_gain, h4_compute_interval = _bootstrap_gain(
        medians,
        "h4_latency_s",
        "h3_replay_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 20,
    )
    summary["h4_executor_gain_vs_h3_replay"] = {
        "mean": h4_compute_gain,
        "ci95": h4_compute_interval,
    }

    config_row = {
        **vars(args),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "arms": {
            "h0": "dense LLaDA",
            "h1": "published SureLock unchanged",
            "h2": "SureLock executor plus full-Sure semantic commitment",
            "h3": "H2 semantic rule plus Sure reference proposals vetoed by PhaseLock P2; dense execution",
            "h4": "exact H3 plans plus PhaseLock profitability-gated fused execution",
        },
        "h4_timing": "executor replay plus paired H3-minus-H3-replay controller overhead",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_row, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "timing_batches.json").write_text(
        json.dumps(timing_batches, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "plans.json").write_text(
        json.dumps(plans, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "safety_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "rows.csv", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
