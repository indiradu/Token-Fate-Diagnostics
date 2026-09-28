#!/usr/bin/env python3
"""Evaluate Trace-assisted PhaseLock while keeping all three lock decisions separate.

Phase 1 uses a learned future-stability score plus an LM-confidence veto and
per-update cap.  Phase 2 remains the frozen multi-layer drift plus attention-
influence gate.  Phase 3 remains the calibrated fused-row profitability gate.
Plans are generated on the exact serving batch and replayed with exact token
identities to audit A1 -> A2 -> A3 independently.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_phase_lock_batched_online import _answer_changed, decode_batch
from benchmark_commitment_safety_policy import select_extra_commits
from benchmark_phase_lock_fused_replay import _examples
from benchmark_phaselock_matched_plans import (
    _bootstrap_gain,
    reference_replay_plan,
    semantic_replay_plan,
    transition_counts,
)
from benchmark_surelock_vs_phaselock import decode_surelock
from benchmark_tracelock_vs_phaselock import _install_hooks
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.commitment_safety import CommitmentSafetyScorer
from regret_remasking.llada_trace import (
    DecodeConfig,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices, surelock_decision
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
from regret_remasking.phase_lock_semantic import (
    scalar_posterior_stability_candidates,
    select_future_stability_commits,
    select_sure_semantic_commits,
    update_token_identity_patience,
)
from regret_remasking.phase_lock_sparse import FusedLatencyTable
from regret_remasking.tracelock_baseline import (
    TraceLockConfig,
    TraceLockController,
    pack_left_padded_controller_batch,
    soft_crop_window,
)
from run_four_arm_audit import transformer_blocks


POLICIES = {
    "capped_confidence": {
        "proposal_mode": "capped_confidence",
        "semantic_lock_fraction": 0.06,
        "confidence_threshold": 0.95,
        "optional_steps_per_block": 2,
    },
    "direct_safety_full": {
        "proposal_mode": "direct_safety_full",
        "maximum_extra_per_update": 2,
    },
    "guarded": {
        "probability_threshold": 0.95,
        "confidence_threshold": 0.90,
        "maximum_fraction": 0.25,
    },
    "balanced": {
        "probability_threshold": 0.95,
        "confidence_threshold": 0.80,
        "maximum_fraction": 0.50,
    },
    "open": {
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "probability",
    },
    "open_confidence": {
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
    },
    "open_joint": {
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "joint",
    },
    "sure_raw": {
        "proposal_mode": "sure",
        "sure_epsilon": 0.005,
        "sure_confidence_percentile": 20.0,
        "confidence_threshold": 0.0,
        "minimum_token_patience": 1,
        "maximum_fraction": 1.0,
    },
    "sure_conf90": {
        "proposal_mode": "sure",
        "sure_epsilon": 0.005,
        "sure_confidence_percentile": 20.0,
        "confidence_threshold": 0.90,
        "minimum_token_patience": 1,
        "maximum_fraction": 1.0,
    },
    "sure_persist2_conf80": {
        "proposal_mode": "sure",
        "sure_epsilon": 0.005,
        "sure_confidence_percentile": 20.0,
        "confidence_threshold": 0.80,
        "minimum_token_patience": 2,
        "maximum_fraction": 1.0,
    },
    "trace_sure_union": {
        "proposal_mode": "trace_sure_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.90,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "sure_epsilon": 0.005,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.90,
        "minimum_token_patience": 2,
    },
    "surelite_p2_d02": {
        "proposal_mode": "surelite",
        "confidence_delta": 0.02,
        "sure_confidence_percentile": 20.0,
        "confidence_threshold": 0.0,
        "minimum_token_patience": 2,
        "maximum_fraction": 1.0,
    },
    "surelite_p2_d05": {
        "proposal_mode": "surelite",
        "confidence_delta": 0.05,
        "sure_confidence_percentile": 20.0,
        "confidence_threshold": 0.0,
        "minimum_token_patience": 2,
        "maximum_fraction": 1.0,
    },
    "trace_surelite_union_d02": {
        "proposal_mode": "trace_surelite_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "confidence_delta": 0.02,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.0,
        "minimum_token_patience": 2,
    },
    "trace_surelite_union_d05": {
        "proposal_mode": "trace_surelite_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "confidence_delta": 0.05,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.0,
        "minimum_token_patience": 2,
    },
    "trace_surelite_union_d02_c80_p2": {
        "proposal_mode": "trace_surelite_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "confidence_delta": 0.02,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.80,
        "minimum_token_patience": 2,
    },
    "trace_surelite_union_d02_c90_p2": {
        "proposal_mode": "trace_surelite_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "confidence_delta": 0.02,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.90,
        "minimum_token_patience": 2,
    },
    "trace_surelite_union_d02_c80_p3": {
        "proposal_mode": "trace_surelite_union",
        "probability_threshold": 0.95,
        "confidence_threshold": 0.0,
        "maximum_fraction": 1.0,
        "fallback_score": "confidence",
        "confidence_delta": 0.02,
        "sure_confidence_percentile": 20.0,
        "sure_confidence_threshold": 0.80,
        "minimum_token_patience": 3,
    },
}

for _policy in ("guarded", "balanced"):
    POLICIES[_policy]["fallback_score"] = "probability"


def _transition(
    reference: list[int], candidate: list[int]
) -> dict[str, float | int]:
    return transition_counts(reference, candidate)


def phase1_window_mask(
    controller_active: torch.Tensor,
    packed_prompt_lengths: torch.Tensor,
    *,
    model_prompt_width: int,
    generation_length: int,
    window_width: int,
) -> tuple[torch.Tensor, list[int]]:
    """Map each packed TraceLock soft crop back to model coordinates."""
    if controller_active.ndim != 2 or controller_active.dtype != torch.bool:
        raise ValueError("controller_active must be a two-dimensional bool tensor")
    if packed_prompt_lengths.shape != (controller_active.shape[0],):
        raise ValueError("packed prompt lengths must have one entry per batch row")
    if controller_active.shape[1] != model_prompt_width + generation_length:
        raise ValueError("controller width must equal model prompt plus generation")
    selected_domain = torch.zeros_like(controller_active)
    crop_ends: list[int] = []
    for batch_index in range(controller_active.shape[0]):
        actual_prompt = int(packed_prompt_lengths[batch_index].item())
        start, end = soft_crop_window(
            controller_active[batch_index],
            prompt_length=actual_prompt,
            window_width=window_width,
        )
        packed_window = controller_active[batch_index].clone()
        packed_window[:start] = False
        packed_window[end:] = False
        selected_domain[batch_index, model_prompt_width:] = packed_window[
            actual_prompt : actual_prompt + generation_length
        ]
        crop_ends.append(end)
    return selected_domain, crop_ends


@torch.inference_mode()
def decode_trace_semantic_planner(
    model: torch.nn.Module,
    tokenizer: Any,
    controller: TraceLockController,
    projection: torch.Tensor,
    trace_layers: list[int],
    examples: list[Any],
    config: DecodeConfig,
    *,
    policy: dict[str, Any],
    with_phase2: bool,
    representation_layers: list[int],
    influence_layer: int,
    drift_threshold: float,
    influence_threshold: float,
    patience: int,
    min_age: int,
    lock_fraction: float,
    max_queries: int,
    window_width: int,
    safety_scorer: CommitmentSafetyScorer | None = None,
) -> dict[str, Any]:
    """Generate Phase-1 and optional Phase-2 plans in one dense trajectory."""
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset) for example in examples]
    input_ids, prompt_attention = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length
    x = torch.full((batch_size, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    model_attention = torch.cat(
        [
            prompt_attention,
            torch.ones(
                (batch_size, config.gen_length),
                dtype=prompt_attention.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if config.gen_length % config.block_length or config.steps % num_blocks:
        raise ValueError("generation schedule must divide evenly into blocks")

    proposal_mode = str(policy.get("proposal_mode", "trace"))
    if proposal_mode not in {
        "capped_confidence",
        "direct_safety_full",
        "trace",
        "sure",
        "surelite",
        "trace_sure_union",
        "trace_surelite_union",
    }:
        raise ValueError(f"unknown Phase-1 proposal mode {proposal_mode!r}")
    uses_trace = proposal_mode in {
        "direct_safety_full",
        "trace",
        "trace_sure_union",
        "trace_surelite_union",
    }
    uses_sure = proposal_mode in {"sure", "trace_sure_union"}
    uses_surelite = proposal_mode in {"surelite", "trace_surelite_union"}
    uses_capped_confidence = proposal_mode == "capped_confidence"
    uses_direct_safety = proposal_mode == "direct_safety_full"
    uses_schedule = uses_capped_confidence or uses_direct_safety
    if uses_direct_safety and safety_scorer is None:
        raise ValueError("direct_safety_full requires a trained safety scorer")
    capture_layers = sorted(
        (set(trace_layers) if uses_trace else set())
        | (
            set(representation_layers)
            if with_phase2 or uses_direct_safety
            else set()
        )
    )
    captures, handles = _install_hooks(model, capture_layers)
    influence: SampledAttentionInfluenceCapture | None = None
    if with_phase2 or uses_direct_safety:
        influence = SampledAttentionInfluenceCapture(
            transformer_blocks(model)[influence_layer - 1],
            max_queries=max_queries,
        )
        influence.install()

    semantic = torch.zeros_like(x, dtype=torch.bool)
    reference = torch.zeros_like(x, dtype=torch.bool)
    age = torch.zeros_like(x, dtype=torch.long)
    joint_count = torch.zeros_like(x, dtype=torch.long)
    previous_hidden: dict[int, torch.Tensor] | None = None
    previous_log_probs: torch.Tensor | None = None
    previous_confidence: torch.Tensor | None = None
    previous_tokens: torch.Tensor | None = None
    token_identity_count = torch.zeros_like(x, dtype=torch.long)
    semantic_plans: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(batch_size)
    ]
    reference_plans: list[dict[int, list[int]]] = [
        {} for _ in range(batch_size)
    ]
    nfe = 0
    semantic_controller_latency_s = 0.0
    semantic_commit_count = 0
    fallback_commit_count = 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = block_start + config.block_length
            if uses_schedule:
                initial_mask = x[:, block_start:block_end] == config.mask_id
                transfers = get_num_transfer_tokens(initial_mask, steps_per_block)
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                mask_index = x == config.mask_id
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, :block_start] = False
                allowed[:, block_end:] = False
                captures.clear()
                if with_phase2 or uses_direct_safety:
                    if influence is None:
                        raise RuntimeError("missing influence capture")
                    unresolved = mask_index.clone()
                    unresolved[:, :prompt_len] = False
                    influence.set_query_mask(unresolved)
                logits = model(x, attention_mask=model_attention).logits
                nfe += 1
                if set(captures) != set(capture_layers):
                    raise RuntimeError("planner did not capture every requested layer")
                proposals = torch.argmax(logits, dim=-1)

                torch.cuda.synchronize(device)
                controller_started = time.perf_counter()
                log_probs: torch.Tensor | None = None
                if uses_sure or uses_direct_safety:
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    confidence = log_probs.amax(dim=-1).exp()
                else:
                    max_logits = logits.amax(dim=-1)
                    confidence = torch.exp(
                        max_logits.float() - torch.logsumexp(logits.float(), dim=-1)
                    )

                token_identity_count = update_token_identity_patience(
                    token_identity_count,
                    proposals,
                    previous_tokens,
                    allowed,
                )
                selected = torch.zeros_like(allowed)
                eligible = torch.zeros_like(allowed)
                current_hidden: dict[int, torch.Tensor] | None = None
                drift: torch.Tensor | None = None
                drift_available: torch.Tensor | None = None
                if uses_trace:
                    projected = torch.stack(
                        [
                            captures[layer].float().matmul(projection)
                            for layer in trace_layers
                        ],
                        dim=2,
                    )
                    (
                        controller_projected,
                        controller_attention,
                        controller_active,
                        packed_prompt_lengths,
                    ) = pack_left_padded_controller_batch(
                        projected,
                        prompt_attention,
                        allowed,
                        generation_length=config.gen_length,
                    )
                    window_allowed, crop_ends = phase1_window_mask(
                        controller_active,
                        packed_prompt_lengths,
                        model_prompt_width=prompt_len,
                        generation_length=config.gen_length,
                        window_width=window_width,
                    )
                    for batch_index, end in enumerate(crop_ends):
                        controller_attention[batch_index, end:] = False
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        controller_output = controller(
                            controller_projected, controller_attention
                        )
                    probability = torch.zeros_like(confidence)
                    for batch_index in range(batch_size):
                        actual_prompt = int(packed_prompt_lengths[batch_index].item())
                        probability[batch_index, prompt_len:] = controller_output[
                            "probability"
                        ][
                            batch_index,
                            actual_prompt : actual_prompt + config.gen_length,
                        ].float()
                    if not uses_direct_safety:
                        trace_eligible = (
                            window_allowed
                            & (probability > float(policy["probability_threshold"]))
                            & (confidence >= float(policy["confidence_threshold"]))
                        )
                        trace_selected = select_future_stability_commits(
                            probability,
                            confidence,
                            window_allowed,
                            probability_threshold=float(
                                policy["probability_threshold"]
                            ),
                            confidence_threshold=float(
                                policy["confidence_threshold"]
                            ),
                            maximum_fraction=float(policy["maximum_fraction"]),
                            fallback=proposal_mode == "trace",
                            fallback_score=str(policy["fallback_score"]),
                        )
                        selected |= trace_selected
                        eligible |= trace_eligible

                if uses_sure:
                    if log_probs is None:
                        raise RuntimeError("Sure proposal mode requires log probabilities")
                    sure = surelock_decision(
                        current_log_probs=log_probs,
                        previous_log_probs=previous_log_probs,
                        candidate=allowed,
                        epsilon=float(policy["sure_epsilon"]),
                        confidence_percentile=float(
                            policy["sure_confidence_percentile"]
                        ),
                    )
                    sure_confidence_threshold = float(
                        policy.get(
                            "sure_confidence_threshold",
                            policy["confidence_threshold"],
                        )
                    )
                    sure_selected = select_sure_semantic_commits(
                        sure.selected,
                        confidence,
                        token_identity_count,
                        allowed,
                        confidence_threshold=sure_confidence_threshold,
                        minimum_token_patience=int(policy["minimum_token_patience"]),
                        maximum_fraction=float(policy["maximum_fraction"]),
                        fallback=proposal_mode == "sure",
                    )
                    sure_eligible = (
                        sure.selected
                        & (confidence >= sure_confidence_threshold)
                        & (
                            token_identity_count
                            >= int(policy["minimum_token_patience"])
                        )
                    )
                    selected |= sure_selected
                    eligible |= sure_eligible

                if uses_surelite:
                    surelite_candidate = scalar_posterior_stability_candidates(
                        confidence,
                        previous_confidence,
                        allowed,
                        maximum_confidence_delta=float(policy["confidence_delta"]),
                        confidence_percentile=float(
                            policy["sure_confidence_percentile"]
                        ),
                    )
                    surelite_confidence_threshold = float(
                        policy.get(
                            "sure_confidence_threshold",
                            policy["confidence_threshold"],
                        )
                    )
                    surelite_selected = select_sure_semantic_commits(
                        surelite_candidate,
                        confidence,
                        token_identity_count,
                        allowed,
                        confidence_threshold=surelite_confidence_threshold,
                        minimum_token_patience=int(policy["minimum_token_patience"]),
                        maximum_fraction=float(policy["maximum_fraction"]),
                        fallback=proposal_mode == "surelite",
                    )
                    surelite_eligible = (
                        surelite_candidate
                        & (confidence >= surelite_confidence_threshold)
                        & (
                            token_identity_count
                            >= int(policy["minimum_token_patience"])
                        )
                    )
                    selected |= surelite_selected
                    eligible |= surelite_eligible

                if uses_capped_confidence:
                    available = allowed.sum(dim=1).to(dtype=torch.long)
                    base_k = torch.minimum(
                        available, transfers[:, step_in_block]
                    )
                    total_k = base_k.clone()
                    accelerated_eligible = torch.zeros_like(allowed)
                    if step_in_block < int(policy["optional_steps_per_block"]):
                        requested = torch.ceil(
                            float(policy["semantic_lock_fraction"])
                            * available.float()
                        ).to(dtype=torch.long)
                        total_k = torch.minimum(
                            available, torch.maximum(base_k, requested)
                        )
                        accelerated_eligible = allowed & (
                            confidence >= float(policy["confidence_threshold"])
                        )
                    capped_selected, _scheduled, accelerated = (
                        select_semantic_transfer(
                            allowed,
                            confidence,
                            confidence,
                            base_k,
                            total_k,
                            accelerated_eligible=accelerated_eligible,
                        )
                    )
                    selected |= capped_selected
                    eligible |= accelerated

                if uses_direct_safety:
                    if safety_scorer is None or log_probs is None:
                        raise RuntimeError("missing direct-safety scorer state")
                    ordinary, _scheduled, _accelerated = select_semantic_transfer(
                        allowed,
                        confidence,
                        confidence,
                        transfers[:, step_in_block],
                        transfers[:, step_in_block],
                    )
                    top = torch.topk(log_probs, k=2, dim=-1).values.exp()
                    margin = top[:, :, 0] - top[:, :, 1]
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                    current_hidden = {
                        layer: captures[layer].detach()
                        for layer in representation_layers
                    }
                    if previous_hidden is None:
                        drift = torch.zeros_like(confidence)
                        drift_available = torch.zeros_like(confidence)
                    else:
                        drift, _ = multilayer_max_drift(
                            current_hidden, previous_hidden
                        )
                        drift_available = torch.ones_like(confidence)
                    if influence is None or influence.latest is None:
                        raise RuntimeError("missing full-safety attention influence")
                    feature_map = {
                        "confidence": confidence,
                        "margin": margin,
                        "entropy": entropy,
                        "token_age": token_identity_count.float(),
                        "trace_probability": probability,
                        "block_age_fraction": torch.full_like(
                            confidence,
                            (step_in_block + 1) / steps_per_block,
                        ),
                        "max_hidden_drift": drift,
                        "attention_influence": influence.latest,
                        "drift_available": drift_available,
                    }
                    safety_score = safety_scorer.score_tensors(feature_map)
                    extra_domain = allowed & ~ordinary & window_allowed
                    extra = select_extra_commits(
                        safety_score,
                        extra_domain,
                        threshold=safety_scorer.threshold,
                        maximum_per_sample=int(
                            policy["maximum_extra_per_update"]
                        ),
                    )
                    selected |= ordinary | extra
                    eligible |= extra_domain & (
                        safety_score >= safety_scorer.threshold
                    )

                if proposal_mode in {
                    "trace_sure_union",
                    "trace_surelite_union",
                }:
                    fallback_selected = select_sure_semantic_commits(
                        torch.zeros_like(allowed),
                        confidence,
                        token_identity_count,
                        allowed,
                        confidence_threshold=1.0,
                        minimum_token_patience=1,
                        fallback=True,
                    )
                    missing = allowed.any(dim=1) & ~selected.any(dim=1)
                    selected[missing] = fallback_selected[missing]
                fallback_commit_count += int((selected & ~eligible).sum().item())
                torch.cuda.synchronize(device)
                semantic_controller_latency_s += time.perf_counter() - controller_started

                x[selected] = proposals[selected]
                selected_mask = selected & ~semantic
                semantic |= selected
                semantic_commit_count += int(selected.sum().item())
                for batch_index in range(batch_size):
                    events = [
                        (
                            int(position - prompt_len),
                            int(proposals[batch_index, position].item()),
                        )
                        for position in torch.nonzero(
                            selected[batch_index], as_tuple=False
                        ).flatten().tolist()
                    ]
                    if events:
                        semantic_plans[batch_index][global_step] = events

                if with_phase2:
                    if influence is None or influence.latest is None:
                        raise RuntimeError("influence score was not captured")
                    if current_hidden is None:
                        current_hidden = {
                            layer: captures[layer].detach()
                            for layer in representation_layers
                        }
                    if drift is None:
                        if previous_hidden is None:
                            drift = torch.zeros_like(x, dtype=torch.float32)
                        else:
                            drift, _per_layer = multilayer_max_drift(
                                current_hidden, previous_hidden
                            )
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
                        positions = [
                            int(position - prompt_len)
                            for position in torch.nonzero(
                                selected_reference[batch_index], as_tuple=False
                            ).flatten().tolist()
                        ]
                        if positions:
                            reference_plans[batch_index][global_step] = positions
                    previous_hidden = current_hidden
                elif uses_direct_safety:
                    if current_hidden is None:
                        raise RuntimeError("missing full-safety hidden state")
                    previous_hidden = current_hidden
                if uses_sure:
                    if log_probs is None:
                        raise RuntimeError("missing Sure posterior state")
                    previous_log_probs = log_probs.detach()
                previous_confidence = confidence.detach()
                previous_tokens = proposals.detach()
                del logits
                if uses_trace:
                    del projected, controller_projected, controller_output
    finally:
        for handle in handles:
            handle.remove()
        if influence is not None:
            influence.remove()
    torch.cuda.synchronize(device)
    final_tokens = x[:, prompt_len:].detach().cpu()
    return {
        "latency_s": time.perf_counter() - started,
        "semantic_controller_latency_s": semantic_controller_latency_s,
        "token_ids": final_tokens.tolist(),
        "nfe": nfe,
        "semantic_plans": semantic_plans,
        "reference_plans": reference_plans,
        "semantic_commit_count": semantic_commit_count,
        "fallback_commit_count": fallback_commit_count,
        "reference_lock_count": int(reference.sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path)
    parser.add_argument("--trace-layers", default="30,31,32")
    parser.add_argument("--window-width", type=int, default=32)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--latency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=251)
    parser.add_argument("--compute-min-locked-fraction", type=float, default=0.20)
    parser.add_argument("--compute-min-remaining-steps", type=int, default=8)
    parser.add_argument("--compute-min-projected-saved-rows", type=int, default=1500)
    parser.add_argument("--compute-min-expected-gain", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.timing_warmups < 0 or args.timing_repeats < 1:
        raise ValueError("use batch >=2, non-negative warmups, and positive repeats")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.latency_table.open(encoding="utf-8") as handle:
        latency_table = FusedLatencyTable.from_rows(json.load(handle))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller_config = TraceLockConfig(**checkpoint["config"])
    controller = TraceLockController(controller_config)
    controller.load_state_dict(checkpoint["model"])
    projection = torch.from_numpy(np.load(args.projection)).float()
    trace_layers = [int(value) for value in args.trace_layers.split(",") if value]
    if len(trace_layers) != 3:
        raise ValueError("exactly three TraceLock layers are required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    device = model_device(model)
    controller = controller.to(device).eval()
    projection = projection.to(device)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
        temperature=0.0,
    )
    policy = POLICIES[args.policy]
    safety_scorer: CommitmentSafetyScorer | None = None
    if args.policy == "direct_safety_full":
        if args.predictor_dir is None:
            raise ValueError("direct_safety_full requires --predictor-dir")
        safety_scorer = CommitmentSafetyScorer(
            args.predictor_dir / "commitment_safety_full.joblib"
        )
    phase2 = {
        "representation_layers": [8, 16, 24],
        "influence_layer": 24,
        "drift_threshold": 0.002,
        "influence_threshold": 0.5,
        "patience": 6,
        "min_age": 4,
        "lock_fraction": 0.04,
        "max_queries": 8,
        "window_width": args.window_width,
    }
    replay_common = {
        "semantic_lock_fraction": 0.06,
        "semantic_min_confidence": 0.95,
        "semantic_optional_steps_per_block": 2,
        "representation_policy": "capped_drift",
        "representation_delay_updates": 1,
        "representation_layer": 8,
        "representation_drift_threshold": 0.005,
        "representation_patience": 4,
        "representation_min_age": 1,
        "representation_lock_fraction": 0.06,
        "compute_min_locked_fraction": args.compute_min_locked_fraction,
        "compute_min_remaining_steps": args.compute_min_remaining_steps,
        "compute_min_projected_saved_rows": args.compute_min_projected_saved_rows,
        "compute_min_expected_gain": args.compute_min_expected_gain,
        "bucket_width_multiple": 16,
        "max_bucket_size": 8,
        "force_semantic_plan_tokens": True,
    }
    rows: list[dict[str, Any]] = []
    timing_batches: list[dict[str, float]] = []
    plans: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], int] = {}

    def score(text: str, example: Any) -> int:
        key = (str(example.example_id), text)
        if key not in score_cache:
            score_cache[key] = int(bool(score_generation(text, example)))
        return score_cache[key]

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(
            f"[phaselock-trace-semantic] policy={args.policy} "
            f"batch={batch_id} size={len(batch)}",
            flush=True,
        )
        planner_p1 = decode_trace_semantic_planner(
            model,
            tokenizer,
            controller,
            projection,
            trace_layers,
            batch,
            config,
            policy=policy,
            with_phase2=False,
            safety_scorer=safety_scorer,
            **phase2,
        )
        planner = decode_trace_semantic_planner(
            model,
            tokenizer,
            controller,
            projection,
            trace_layers,
            batch,
            config,
            policy=policy,
            with_phase2=True,
            safety_scorer=safety_scorer,
            **phase2,
        )
        if planner_p1["token_ids"] != planner["token_ids"]:
            raise RuntimeError("Phase-2 observation changed Phase-1 token output")
        planner_tokens = planner["token_ids"]
        planner_generations = [
            value.strip()
            for value in tokenizer.batch_decode(planner_tokens, skip_special_tokens=True)
        ]
        replay = {
            "semantic_plans": [
                semantic_replay_plan(plan) for plan in planner["semantic_plans"]
            ],
            "reference_plans": [
                reference_replay_plan(plan) for plan in planner["reference_plans"]
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

        def run(arm: str) -> dict[str, Any]:
            if arm == "A0":
                return decode_surelock(
                    model,
                    tokenizer,
                    batch,
                    config,
                    epsilon=None,
                    confidence_percentile=20.0,
                    bucket_width_multiple=16,
                )
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

        orders = (("A0", "A2", "A3"), ("A2", "A3", "A0"), ("A3", "A0", "A2"))
        for warmup in range(args.timing_warmups):
            for arm in orders[warmup % len(orders)]:
                run(arm)
        for repeat in range(args.timing_repeats):
            order = orders[repeat % len(orders)]
            records = {arm: run(arm) for arm in order}
            overhead = float(planner["semantic_controller_latency_s"]) + max(
                0.0, float(planner["latency_s"]) - float(planner_p1["latency_s"])
            )
            timing_batches.append(
                {
                    "batch_id": float(batch_id),
                    "repeat": float(repeat),
                    "a0_latency_s": float(records["A0"]["latency_s"]),
                    "a2_latency_s": float(records["A2"]["latency_s"]),
                    "a3_latency_s": float(records["A3"]["latency_s"]),
                    "estimated_online_a3_latency_s": float(
                        records["A3"]["latency_s"] + overhead
                    ),
                    "controller_overhead_s": overhead,
                    "removed_row_token_forwards": float(
                        records["A3"]["removed_row_token_forwards"]
                    ),
                    "compute_switched": float(records["A3"]["switch_step"] is not None),
                }
            )
            for local_index, example in enumerate(batch):
                a0_text = records["A0"]["generations"][local_index]
                a1_text = planner_generations[local_index]
                a2_text = records["A2"]["generations"][local_index]
                a3_text = records["A3"]["generations"][local_index]
                rows.append(
                    {
                        "dataset": args.dataset,
                        "policy": args.policy,
                        "batch_id": batch_id,
                        "repeat": repeat,
                        "example_id": example.example_id,
                        "a0_correct": score(a0_text, example),
                        "a1_correct": score(a1_text, example),
                        "a2_correct": score(a2_text, example),
                        "a3_correct": score(a3_text, example),
                        "a1_exact_a0": int(
                            planner_tokens[local_index] == records["A0"]["token_ids"][local_index]
                        ),
                        "a2_exact_a1": int(
                            records["A2"]["token_ids"][local_index] == planner_tokens[local_index]
                        ),
                        "a3_exact_a2": int(
                            records["A3"]["token_ids"][local_index]
                            == records["A2"]["token_ids"][local_index]
                        ),
                        "a1_answer_changed_a0": int(
                            _answer_changed(args.dataset, a1_text, a0_text)
                        ),
                        "a2_answer_changed_a1": int(
                            _answer_changed(args.dataset, a2_text, a1_text)
                        ),
                        "a3_answer_changed_a2": int(
                            _answer_changed(args.dataset, a3_text, a2_text)
                        ),
                        "a0_nfe": records["A0"]["nfe"],
                        "a1_nfe": planner["nfe"],
                        "a2_nfe": records["A2"]["nfe"],
                        "a3_nfe": records["A3"]["nfe"],
                        "a3_reference_lock_count": records["A3"]["reference_lock_count"],
                        "a3_removed_row_token_forwards": records["A3"]["removed_row_token_forwards"],
                        "a3_switch_step": records["A3"]["switch_step"],
                        "a3_switch_backend": records["A3"]["switch_backend"],
                    }
                )

    stable_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable_by_id.setdefault(str(row["example_id"]), row)
    stable = list(stable_by_id.values())
    correctness = {
        arm: [int(row[f"{arm}_correct"]) for row in stable]
        for arm in ("a0", "a1", "a2", "a3")
    }
    medians: list[dict[str, float]] = []
    for batch_id in sorted({int(row["batch_id"]) for row in timing_batches}):
        group = [row for row in timing_batches if int(row["batch_id"]) == batch_id]
        medians.append(
            {
                key: float(np.median([row[key] for row in group]))
                for key in (
                    "a0_latency_s",
                    "a2_latency_s",
                    "a3_latency_s",
                    "estimated_online_a3_latency_s",
                    "controller_overhead_s",
                )
            }
        )
    a3_gain, a3_interval = _bootstrap_gain(
        medians,
        "a3_latency_s",
        "a0_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    online_gain, online_interval = _bootstrap_gain(
        medians,
        "estimated_online_a3_latency_s",
        "a0_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    a3_vs_a2, a3_vs_a2_interval = _bootstrap_gain(
        medians,
        "a3_latency_s",
        "a2_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )
    summary = {
        "policy": args.policy,
        "examples": len(stable),
        "batches": len(medians),
        "accuracy": {
            arm: float(np.mean(values)) for arm, values in correctness.items()
        },
        "a0_to_a1": {
            "exact_rate": float(np.mean([row["a1_exact_a0"] for row in stable])),
            "answer_change_rate": float(
                np.mean([row["a1_answer_changed_a0"] for row in stable])
            ),
            **_transition(correctness["a0"], correctness["a1"]),
        },
        "a1_to_a2": {
            "exact_rate": float(np.mean([row["a2_exact_a1"] for row in stable])),
            "answer_change_rate": float(
                np.mean([row["a2_answer_changed_a1"] for row in stable])
            ),
            **_transition(correctness["a1"], correctness["a2"]),
        },
        "a2_to_a3": {
            "exact_rate": float(np.mean([row["a3_exact_a2"] for row in stable])),
            "answer_change_rate": float(
                np.mean([row["a3_answer_changed_a2"] for row in stable])
            ),
            **_transition(correctness["a2"], correctness["a3"]),
        },
        "mean_nfe": {
            arm: float(np.mean([row[f"{arm}_nfe"] for row in stable]))
            for arm in ("a0", "a1", "a2", "a3")
        },
        "executor_runtime_gain_a3_vs_a0": a3_gain,
        "executor_runtime_gain_a3_vs_a0_ci95": a3_interval,
        "estimated_online_runtime_gain_vs_a0": online_gain,
        "estimated_online_runtime_gain_vs_a0_ci95": online_interval,
        "compute_runtime_gain_a3_vs_a2": a3_vs_a2,
        "compute_runtime_gain_a3_vs_a2_ci95": a3_vs_a2_interval,
        "mean_controller_overhead_s_per_batch": float(
            np.mean([row["controller_overhead_s"] for row in medians])
        ),
        "mean_reference_locks_per_batch": float(
            np.mean([row["a3_reference_lock_count"] for row in rows])
        ),
        "mean_removed_row_token_forwards_per_batch": float(
            np.mean([row["removed_row_token_forwards"] for row in timing_batches])
        ),
        "compute_switch_rate": float(
            np.mean([row["compute_switched"] for row in timing_batches])
        ),
    }
    config_payload = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "projection": str(args.projection),
        "predictor_dir": (
            str(args.predictor_dir) if args.predictor_dir is not None else None
        ),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "semantic_policy": policy,
        "phase2_policy": phase2,
        "phase3_policy": {
            key: replay_common[key]
            for key in replay_common
            if key.startswith("compute_")
        },
        "timing_scope": {
            "executor": "paired A0/A2/A3 replay",
            "estimated_online": "A3 replay plus measured semantic controller and incremental Phase-2 overhead",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "timing_batches.json").write_text(
        json.dumps(timing_batches, indent=2), encoding="utf-8"
    )
    (args.output_dir / "plans.json").write_text(
        json.dumps(plans, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
