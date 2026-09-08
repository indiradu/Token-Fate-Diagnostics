#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from regret_remasking import FEATURE_NAMES
from regret_remasking.data import (
    build_prompt,
    generated_answers_equivalent,
    load_examples,
    normalize_answer,
    score_generation,
    score_generation_with_status,
)
from regret_remasking.features import entropy_from_log_probs, local_window_average, safe_jsd, safe_kl
from regret_remasking.llada_trace import (
    EOS_ID,
    EOT_ID,
    DecodeConfig,
    add_gumbel_noise,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.less_sampling import LessConfig, LessDecision, LessSelectionState
from regret_remasking.polestar_proxy import (
    PolestarCacheProxyConfig,
    attention_distribution_kl,
    local_prefix_bounds,
    select_high_drift_positions,
    should_trigger_refresh,
)
from run_counterfactual_commit import counterfactual_token_effects
from run_logit_lens_token_fate import (
    parse_layers,
    patch_attention_capture,
    register_layer_hooks,
    resolve_hook_layers,
    restore_attention,
)


@dataclass
class CommittedToken:
    example_id: str
    dataset: str
    position: int
    relative_position: int
    selected_step: int
    selected_block_t_frac: float
    selected_token: int
    selected_confidence: float
    selected_entropy: float
    selected_margin: float
    selected_kl: float
    commit_source: str = "confidence_schedule"
    selected_less_jsd: float = float("nan")
    selected_less_persistence: bool = False
    # First forward after selection, when the position is no longer masked.
    base_step: int | None = None
    base_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    # Previous observed step's row, for consecutive-step movement. Drift from
    # base alone cannot bound movement after a *delayed* lock (see
    # scripts/analyze_lock_admission_bound.py); the step-local deltas can.
    prev_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    # Phase 2: the step a SureLock-style gate would admit this token, and the
    # row it would cache *there*. Freezing at the rule's step but serving the
    # commit-step row would misrepresent the method -- it caches at its own
    # lock step, not retroactively.
    lock_step: int | None = None
    lock_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    max_drift_by_layer: dict[int, float] = field(default_factory=dict)
    final_drift_by_layer: dict[int, float] = field(default_factory=dict)
    future_steps_observed: int = 0
    final_token: int | None = None


@dataclass
class PolestarProxyRefreshEvent:
    event_id: str
    example_id: str
    dataset: str
    position: int
    relative_position: int
    selected_token: int
    block: int
    cache_step: int
    refresh_step: int
    cache_age: int
    attention_kl: float
    refresh_rank: int
    refresh_pool_size: int
    cache_hidden: dict[int, torch.Tensor]
    refresh_hidden: dict[int, torch.Tensor]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def tensor_value(feature_map: dict[str, torch.Tensor], name: str, batch: int, pos: int) -> float:
    return float(feature_map[name][batch, pos].detach().cpu())


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append per-example rows to a CSV, writing the header only once.

    Replaces rewriting the whole accumulated frame on every example, which was
    quadratic in example count and is what forced --skip-drift-time-rows on
    long generations.
    """
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def relative_l2(current: torch.Tensor, base: torch.Tensor, eps: float = 1e-6) -> float:
    current_f = current.detach().float().cpu()
    base_f = base.detach().float().cpu()
    return float(torch.linalg.vector_norm(current_f - base_f) / torch.clamp(torch.linalg.vector_norm(base_f), min=eps))


def decode_step_features(
    logits: torch.Tensor,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
    prev_top1: torch.Tensor,
    runlength: torch.Tensor,
    global_step: int,
    config: DecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
    next_runlength = torch.where(prev_top1 == x0, runlength + 1.0, torch.ones_like(runlength))
    local_mask_ratio = local_window_average(mask_index.float(), config.cpv_window)
    context_volatility = local_window_average(kl, config.cpv_window)
    feature_map = {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "kl": kl,
        "jsd": jsd,
        "top1_flip": top1_flip,
        "runlength": next_runlength,
        "t_frac": torch.full_like(confidence, (global_step + 1) / config.steps),
        "local_mask_ratio": local_mask_ratio,
        "context_volatility": context_volatility,
    }
    return x0, log_probs, next_runlength, feature_map


def choose_step_transfer(
    confidence: torch.Tensor,
    log_probs: torch.Tensor,
    allowed: torch.Tensor,
    default_k: int,
    less_state: LessSelectionState | None = None,
) -> tuple[torch.Tensor, LessDecision | None]:
    if less_state is not None:
        decision = less_state.select(log_probs.exp(), allowed, default_k)
        return decision.transfer_index, decision

    transfer_index = torch.zeros_like(allowed, dtype=torch.bool)
    if default_k > 0:
        score = torch.full_like(confidence, fill_value=-torch.inf)
        score[allowed] = confidence[allowed]
        _, selected = torch.topk(score[0], k=default_k)
        transfer_index[0, selected] = True
    return transfer_index, None


def replay_prefix_matches(
    baseline_trace: dict[int, dict[str, torch.Tensor]],
    global_step: int,
    freeze_step: int,
    input_tokens: torch.Tensor,
    raw_top1: torch.Tensor,
    transfer_index: torch.Tensor,
) -> bool:
    expected = baseline_trace.get(global_step)
    if expected is None:
        return False
    if not torch.equal(input_tokens.detach().cpu(), expected["input_tokens"]):
        return False
    if global_step >= freeze_step:
        return True
    return torch.equal(raw_top1.detach().cpu(), expected["raw_top1"]) and torch.equal(
        transfer_index.detach().cpu(), expected["transfer_index"]
    )


def less_replay_prefix_matches(
    baseline_trace: dict[int, dict[str, torch.Tensor]],
    global_step: int,
    freeze_step: int,
    input_tokens: torch.Tensor,
    raw_top1: torch.Tensor,
    transfer_index: torch.Tensor,
) -> bool:
    """Compatibility alias for the generic exact-prefix replay check."""
    return replay_prefix_matches(
        baseline_trace,
        global_step,
        freeze_step,
        input_tokens,
        raw_top1,
        transfer_index,
    )


def hidden_snapshot(
    captured: dict[int, torch.Tensor],
    layers: list[int],
    position: int,
) -> dict[int, torch.Tensor]:
    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"missing hidden capture for layers {missing}")
    return {
        layer: captured[layer][0, position].detach().float().cpu().clone()
        for layer in layers
    }


def initialize_polestar_proxy_cache(
    committed: dict[int, CommittedToken],
    captured: dict[int, torch.Tensor],
    layers: list[int],
    prefix_start: int,
    block_start: int,
    global_step: int,
) -> dict[int, tuple[int, dict[int, torch.Tensor]]]:
    return {
        position: (global_step, hidden_snapshot(captured, layers, position))
        for position in committed
        if prefix_start <= position < block_start
    }


def record_polestar_proxy_step(
    *,
    committed: dict[int, CommittedToken],
    captured: dict[int, torch.Tensor],
    layers: list[int],
    current_attention: torch.Tensor,
    previous_attention: torch.Tensor,
    cache_state: dict[int, tuple[int, dict[int, torch.Tensor]]],
    refresh_due: bool,
    refresh_fraction: float,
    block_idx: int,
    global_step: int,
    max_saved_events: int,
    seed: int,
    rows: list[dict[str, Any]],
    events: list[PolestarProxyRefreshEvent],
) -> None:
    attention_kl = attention_distribution_kl(current_attention, previous_attention)
    eligible = torch.zeros_like(attention_kl, dtype=torch.bool)
    if cache_state:
        positions = torch.as_tensor(list(cache_state), device=eligible.device, dtype=torch.long)
        eligible[0, positions] = True
    selected = (
        select_high_drift_positions(attention_kl, eligible, refresh_fraction)
        if refresh_due
        else torch.zeros_like(eligible)
    )
    selected_positions = torch.nonzero(selected[0], as_tuple=False).flatten()
    rank_by_position: dict[int, int] = {}
    if selected_positions.numel() > 0:
        ordered = selected_positions[
            torch.argsort(attention_kl[0, selected_positions], descending=True, stable=True)
        ]
        rank_by_position = {
            int(position): rank
            for rank, position in enumerate(ordered.detach().cpu().tolist(), start=1)
        }

    pool_size = int(eligible.sum().detach().cpu())
    for position, (cache_step, cache_hidden) in list(cache_state.items()):
        token = committed[position]
        score = float(attention_kl[0, position].detach().cpu())
        is_selected = bool(selected[0, position].detach().cpu())
        rows.append(
            {
                "example_id": token.example_id,
                "dataset": token.dataset,
                "block": block_idx,
                "global_step": global_step,
                "position": position,
                "relative_position": token.relative_position,
                "selected_token": token.selected_token,
                "cache_step": cache_step,
                "cache_age": global_step - cache_step,
                "attention_kl": score,
                "refresh_due": refresh_due,
                "selected_for_refresh": is_selected,
            }
        )
        if not is_selected:
            continue
        refresh_hidden = hidden_snapshot(captured, layers, position)
        event_id = f"{token.example_id}:{block_idx}:{global_step}:{position}"
        events.append(
            PolestarProxyRefreshEvent(
                event_id=event_id,
                example_id=token.example_id,
                dataset=token.dataset,
                position=position,
                relative_position=token.relative_position,
                selected_token=token.selected_token,
                block=block_idx,
                cache_step=cache_step,
                refresh_step=global_step,
                cache_age=global_step - cache_step,
                attention_kl=score,
                refresh_rank=rank_by_position[position],
                refresh_pool_size=pool_size,
                cache_hidden=cache_hidden,
                refresh_hidden=refresh_hidden,
            )
        )
        cache_state[position] = (global_step, refresh_hidden)
    if len(events) > max_saved_events:
        events.sort(
            key=lambda event: hashlib.sha256(
                f"{seed}:{event.event_id}".encode("utf-8")
            ).digest()
        )
        del events[max_saved_events:]


def update_committed_drift(
    committed: dict[int, CommittedToken],
    captured: dict[int, torch.Tensor],
    layers: list[int],
    global_step: int,
    time_rows: list[dict[str, Any]] | None,
    feature_map: dict[str, torch.Tensor] | None = None,
    raw_top1: torch.Tensor | None = None,
    post_rows: list[dict[str, Any]] | None = None,
    lock_rule_tau: float | None = None,
) -> None:
    """Advance post-commit drift for every committed token.

    `feature_map` / `raw_top1` / `post_rows` are optional so the existing
    drift-only callers and `tests/test_drift_update.py` keep working. When
    supplied, the position's *post-commit* posterior is logged: these signals
    are computed at every position on every step but were previously discarded
    for unmasked positions, which is what prior work's lock-admission rules
    (SureLock-style adjacent-step KL, LESS-style JSD + persistence) need.
    `raw_top1` must be the argmax *before* the `torch.where(mask_index, ...)`
    overwrite, otherwise it is just the committed token read back.
    """
    pending = [token for token in committed.values() if global_step > token.selected_step]
    tracked: list[CommittedToken] = []
    for token in pending:
        if token.base_step is None:
            token.base_step = global_step
            for layer in layers:
                row = captured[layer][0, token.position].detach().float().cpu()
                token.base_hidden[layer] = row
                token.prev_hidden[layer] = row
                token.max_drift_by_layer[layer] = 0.0
                token.final_drift_by_layer[layer] = 0.0
        else:
            tracked.append(token)
    if not tracked:
        return
    for token in tracked:
        token.future_steps_observed += 1

    if lock_rule_tau is not None and feature_map is not None:
        # SureLock-style admission: adjacent-step posterior KL at or below tau.
        # tau=0 means the posterior is numerically identical between steps.
        for token in tracked:
            if token.lock_step is not None:
                continue
            if tensor_value(feature_map, "kl", 0, token.position) <= lock_rule_tau:
                token.lock_step = global_step
                for layer in layers:
                    token.lock_hidden[layer] = (
                        captured[layer][0, token.position].detach().float().cpu()
                    )

    if post_rows is not None and feature_map is not None:
        for token in tracked:
            pos = token.position
            raw_token = int(raw_top1[0, pos].detach().cpu()) if raw_top1 is not None else -1
            post_rows.append(
                {
                    "example_id": token.example_id,
                    "dataset": token.dataset,
                    "position": pos,
                    "relative_position": token.relative_position,
                    "selected_step": token.selected_step,
                    "base_step": token.base_step,
                    "global_step": global_step,
                    "committed_token": token.selected_token,
                    "raw_top1": raw_token,
                    "raw_top1_matches_commit": int(raw_token == token.selected_token),
                    "post_confidence": tensor_value(feature_map, "confidence", 0, pos),
                    "post_entropy": tensor_value(feature_map, "entropy", 0, pos),
                    "post_margin": tensor_value(feature_map, "margin", 0, pos),
                    "post_kl": tensor_value(feature_map, "kl", 0, pos),
                    "post_jsd": tensor_value(feature_map, "jsd", 0, pos),
                }
            )
    # One device sync per layer per step instead of one per token; the math
    # matches relative_l2 row-wise (float32 on CPU, eps-clamped base norm).
    position_index = torch.as_tensor(
        [token.position for token in tracked],
        dtype=torch.long,
        device=captured[layers[0]].device,
    )
    for layer in layers:
        current = captured[layer][0].index_select(0, position_index).detach().float().cpu()
        base = torch.stack([token.base_hidden[layer] for token in tracked])
        diff_norm = torch.linalg.vector_norm(current - base, dim=1)
        base_norm = torch.clamp(torch.linalg.vector_norm(base, dim=1), min=1e-6)
        drifts = (diff_norm / base_norm).tolist()
        # Step-local movement, normalized by the previous row. Summing these
        # upper-bounds movement after a lock at any step, which drift-from-base
        # cannot do.
        prev = torch.stack([token.prev_hidden[layer] for token in tracked])
        step_norm = torch.linalg.vector_norm(current - prev, dim=1)
        prev_norm = torch.clamp(torch.linalg.vector_norm(prev, dim=1), min=1e-6)
        step_drifts = (step_norm / prev_norm).tolist()
        # Absolute row norms, so the consecutive-step deltas can be renormalized
        # to any lock step's own scale. Without them the triangle-inequality
        # upper bound is only exact when the norms happen to be constant.
        row_norms = torch.linalg.vector_norm(current, dim=1).tolist()
        # ||b|| itself is never logged as a row (no time row is written at
        # base_step), but every bound has to be expressed in ||b|| units to be
        # comparable with relative_l2_drift, so carry it explicitly.
        base_norms = base_norm.tolist()
        for idx, (token, drift) in enumerate(zip(tracked, drifts)):
            drift = float(drift)
            token.final_drift_by_layer[layer] = drift
            token.max_drift_by_layer[layer] = max(token.max_drift_by_layer.get(layer, 0.0), drift)
            token.prev_hidden[layer] = current[idx].clone()
            if time_rows is not None:
                time_rows.append(
                    {
                        "example_id": token.example_id,
                        "dataset": token.dataset,
                        "position": token.position,
                        "relative_position": token.relative_position,
                        "selected_step": token.selected_step,
                        "base_step": token.base_step,
                        "global_step": global_step,
                        "layer": layer,
                        "relative_l2_drift": drift,
                        "step_l2_drift": float(step_drifts[idx]),
                        "row_norm": float(row_norms[idx]),
                        "base_norm": float(base_norms[idx]),
                    }
                )


@torch.no_grad()
def collect_baseline_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
    layers: list[int],
    collect_time_rows: bool = True,
    lock_rule_tau: float | None = None,
    decode_policy: str = "confidence",
    less_config: LessConfig | None = None,
    polestar_proxy_config: PolestarCacheProxyConfig | None = None,
) -> dict[str, Any]:
    device = model_device(model)
    captured, handles = register_layer_hooks(model, layers)
    captured_attention: dict[str, torch.Tensor] | None = None
    original_attention = None
    if polestar_proxy_config is not None:
        captured_attention, original_attention = patch_attention_capture(
            model,
            polestar_proxy_config.attention_layer,
            capture_per_head=True,
        )
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
        num_blocks = config.gen_length // config.block_length
        steps_per_block = config.steps // num_blocks
        prev_log_probs: torch.Tensor | None = None
        prev_top1 = torch.full_like(x, fill_value=-1)
        runlength = torch.zeros_like(x, dtype=torch.float32)
        committed: dict[int, CommittedToken] = {}
        time_rows: list[dict[str, Any]] = []
        post_rows: list[dict[str, Any]] = []
        proposal_rows: list[dict[str, Any]] = []
        decode_trace: dict[int, dict[str, torch.Tensor]] = {}
        polestar_attention_rows: list[dict[str, Any]] = []
        polestar_events: list[PolestarProxyRefreshEvent] = []
        nfe = 0
        if decode_policy not in {"confidence", "less"}:
            raise ValueError(f"unsupported decode policy: {decode_policy}")
        less_state = (
            LessSelectionState(total_len, device, less_config or LessConfig())
            if decode_policy == "less"
            else None
        )
        if less_state is not None and config.temperature != 0:
            raise ValueError("released-code LESS audit currently requires temperature=0")
        trace_enabled = less_state is not None or polestar_proxy_config is not None

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            if less_state is not None:
                less_state.reset_range(block_start, block_end)
            previous_attention: torch.Tensor | None = None
            polestar_cache_state: dict[int, tuple[int, dict[int, torch.Tensor]]] = {}
            decoded_since_refresh = 0
            refresh_due = False
            if polestar_proxy_config is not None:
                prefix_start, _ = local_prefix_bounds(
                    prompt_len,
                    block_start,
                    config.block_length,
                    polestar_proxy_config.prefix_blocks,
                )

            for step_in_block in range(steps_per_block):
                mask_index = x == config.mask_id
                active_block_mask = mask_index.clone()
                active_block_mask[:, :block_start] = False
                active_block_mask[:, block_end:] = False
                if less_state is not None and not active_block_mask.any():
                    break
                global_step = nfe if less_state is not None else block_idx * steps_per_block + step_in_block
                input_tokens = x.detach().cpu().clone() if trace_enabled else None
                captured.clear()
                if captured_attention is not None:
                    captured_attention.clear()
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                nfe += 1
                x0, log_probs, runlength, feature_map = decode_step_features(
                    logits,
                    x,
                    mask_index,
                    prev_log_probs,
                    prev_top1,
                    runlength,
                    global_step,
                    config,
                )
                confidence = feature_map["confidence"]
                raw_top1 = x0.detach().cpu().clone() if trace_enabled else None
                # Logged here rather than before decode_step_features so the
                # posterior features exist, and before the torch.where below so
                # raw_top1 is the true argmax rather than the committed token
                # read back. decode_step_features neither reads `captured` nor
                # mutates `committed`, so this reorder cannot change decoding.
                update_committed_drift(
                    committed,
                    captured,
                    layers,
                    global_step,
                    time_rows if collect_time_rows else None,
                    feature_map=feature_map,
                    raw_top1=x0,
                    post_rows=post_rows if collect_time_rows else None,
                    lock_rule_tau=lock_rule_tau,
                )
                current_attention = None
                if captured_attention is not None:
                    current_attention = captured_attention.get("weights_by_head")
                    if current_attention is None:
                        raise RuntimeError("Polestar proxy attention capture is missing")
                    if step_in_block == 0:
                        polestar_cache_state = initialize_polestar_proxy_cache(
                            committed,
                            captured,
                            layers,
                            prefix_start,
                            block_start,
                            global_step,
                        )
                    elif previous_attention is not None:
                        record_polestar_proxy_step(
                            committed=committed,
                            captured=captured,
                            layers=layers,
                            current_attention=current_attention,
                            previous_attention=previous_attention,
                            cache_state=polestar_cache_state,
                            refresh_due=refresh_due,
                            refresh_fraction=polestar_proxy_config.refresh_fraction,
                            block_idx=block_idx,
                            global_step=global_step,
                            max_saved_events=polestar_proxy_config.max_saved_events,
                            seed=polestar_proxy_config.seed,
                            rows=polestar_attention_rows,
                            events=polestar_events,
                        )
                        if refresh_due:
                            decoded_since_refresh = 0
                            refresh_due = False
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                x0 = torch.where(mask_index, x0, x)

                # Pre-transfer proposal ledger: what each still-masked, in-block
                # position proposes at this step. Used offline to label
                # observational stability S_i,t = 1[top1_i,t == final_i].
                allowed_positions = torch.nonzero(allowed[0], as_tuple=False).flatten()
                if allowed_positions.numel() > 0:
                    proposal_tokens = x0[0, allowed_positions].detach().cpu().tolist()
                    proposal_confidences = confidence[0, allowed_positions].detach().cpu().tolist()
                    for pos, top_token, conf in zip(
                        allowed_positions.tolist(), proposal_tokens, proposal_confidences
                    ):
                        proposal_rows.append(
                            {
                                "example_id": example.example_id,
                                "dataset": example.dataset,
                                "position": int(pos),
                                "relative_position": int(pos - prompt_len),
                                "global_step": global_step,
                                "top_token": int(top_token),
                                "confidence": float(conf),
                            }
                        )

                k = int(num_transfer_tokens[0, step_in_block].item())
                transfer_index, less_decision = choose_step_transfer(
                    confidence,
                    log_probs,
                    allowed,
                    k,
                    less_state=less_state,
                )
                if trace_enabled:
                    decode_trace[global_step] = {
                        "input_tokens": input_tokens,
                        "raw_top1": raw_top1,
                        "transfer_index": transfer_index.detach().cpu().clone(),
                    }

                for pos in torch.nonzero(transfer_index[0], as_tuple=False).flatten().tolist():
                    if int(pos) in committed:
                        continue
                    committed[int(pos)] = CommittedToken(
                        example_id=example.example_id,
                        dataset=example.dataset,
                        position=int(pos),
                        relative_position=int(pos - prompt_len),
                        selected_step=global_step,
                        selected_block_t_frac=(step_in_block + 1) / steps_per_block,
                        selected_token=int(x0[0, pos].detach().cpu()),
                        selected_confidence=tensor_value(feature_map, "confidence", 0, int(pos)),
                        selected_entropy=tensor_value(feature_map, "entropy", 0, int(pos)),
                        selected_margin=tensor_value(feature_map, "margin", 0, int(pos)),
                        selected_kl=tensor_value(feature_map, "kl", 0, int(pos)),
                        commit_source=(
                            "less_accept"
                            if less_decision is not None
                            and bool(less_decision.accepted_index[0, pos].detach().cpu())
                            else "less_fallback"
                            if less_decision is not None
                            else "confidence_schedule"
                        ),
                        selected_less_jsd=(
                            float(less_decision.topk_jsd[0, pos].detach().cpu())
                            if less_decision is not None
                            else float("nan")
                        ),
                        selected_less_persistence=(
                            bool(less_decision.persistence[0, pos].detach().cpu())
                            if less_decision is not None
                            else False
                        ),
                    )

                x[transfer_index] = x0[transfer_index]
                if polestar_proxy_config is not None:
                    decoded_this_step = int(transfer_index.sum().detach().cpu())
                    refresh_due = should_trigger_refresh(
                        decoded_this_step,
                        decoded_since_refresh,
                        polestar_proxy_config.refresh_threshold,
                    )
                    decoded_since_refresh += decoded_this_step
                    if current_attention is not None:
                        previous_attention = current_attention.detach()
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del outputs, logits, log_probs

        final_tokens = x[:, prompt_len:].detach().cpu()
        generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        correct, scoring_status = score_generation_with_status(generation, example)
        for token in committed.values():
            token.final_token = int(final_tokens[0, token.relative_position])
        return {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "prompt_len": prompt_len,
            "final_tokens": final_tokens[0].numpy().astype(np.int64, copy=False),
            "generation": generation,
            "correct": correct,
            "scoring_status": scoring_status,
            "nfe": nfe,
            "decode_policy": decode_policy,
            "committed": list(committed.values()),
            "time_rows": time_rows,
            "post_rows": post_rows,
            "proposal_rows": proposal_rows,
            "decode_trace": decode_trace,
            "polestar_attention_rows": polestar_attention_rows,
            "polestar_events": polestar_events,
        }
    finally:
        for handle in handles:
            handle.remove()
        if polestar_proxy_config is not None and original_attention is not None:
            restore_attention(model, polestar_proxy_config.attention_layer, original_attention)


def committed_rows(baseline: dict[str, Any], layers: list[int]) -> list[dict[str, Any]]:
    rows = []
    for token in baseline["committed"]:
        max_values = [token.max_drift_by_layer.get(layer, np.nan) for layer in layers]
        final_values = [token.final_drift_by_layer.get(layer, np.nan) for layer in layers]
        row: dict[str, Any] = {
            "example_id": token.example_id,
            "dataset": token.dataset,
            "position": token.position,
            "relative_position": token.relative_position,
            "selected_step": token.selected_step,
            "base_step": token.base_step,
            "selected_block_t_frac": token.selected_block_t_frac,
            "selected_token": token.selected_token,
            "final_token": token.final_token,
            "semantic_safe": int(token.final_token == token.selected_token),
            "selected_confidence": token.selected_confidence,
            "selected_entropy": token.selected_entropy,
            "selected_margin": token.selected_margin,
            "selected_kl": token.selected_kl,
            "commit_source": token.commit_source,
            "selected_less_jsd": token.selected_less_jsd,
            "selected_less_persistence": int(token.selected_less_persistence),
            "baseline_nfe": int(baseline["nfe"]),
            "future_steps_observed": token.future_steps_observed,
            "lock_step": token.lock_step if token.lock_step is not None else np.nan,
            "baseline_correct": bool(baseline["correct"]),
        }
        for layer in layers:
            row[f"max_drift_layer_{layer}"] = token.max_drift_by_layer.get(layer, np.nan)
            row[f"final_drift_layer_{layer}"] = token.final_drift_by_layer.get(layer, np.nan)
        row["max_drift_mean"] = float(np.nanmean(max_values)) if np.isfinite(max_values).any() else np.nan
        row["final_drift_mean"] = float(np.nanmean(final_values)) if np.isfinite(final_values).any() else np.nan
        rows.append(row)
    return rows


def polestar_proxy_event_rows(
    baseline: dict[str, Any],
    layers: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for event in baseline.get("polestar_events", []):
        refresh_drifts = [
            relative_l2(event.refresh_hidden[layer], event.cache_hidden[layer])
            for layer in layers
        ]
        rows.append(
            {
                "event_id": event.event_id,
                "example_id": event.example_id,
                "dataset": event.dataset,
                "position": event.position,
                "relative_position": event.relative_position,
                "selected_token": event.selected_token,
                "block": event.block,
                "cache_step": event.cache_step,
                "refresh_step": event.refresh_step,
                "cache_age": event.cache_age,
                "future_steps_after_refresh": int(baseline["nfe"] - event.refresh_step - 1),
                "attention_kl": event.attention_kl,
                "refresh_rank": event.refresh_rank,
                "refresh_pool_size": event.refresh_pool_size,
                "refresh_drift_mean": float(np.mean(refresh_drifts)),
            }
        )
    return rows


def choose_polestar_cache_proxy_candidates(
    event_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    per_example: int,
    exclude_eos_eot: bool = True,
    seed: int = 23,
    selection: str = "random",
) -> pd.DataFrame:
    """Sample paired stale-vs-refresh events selected by the online proxy."""
    if event_df.empty or drift_df.empty:
        return pd.DataFrame()
    token_columns = [
        "example_id",
        "position",
        "selected_step",
        "base_step",
        "final_token",
        "semantic_safe",
        "selected_confidence",
        "selected_entropy",
        "selected_margin",
        "selected_kl",
        "future_steps_observed",
        "baseline_nfe",
        "max_drift_mean",
        "final_drift_mean",
    ]
    eligible = event_df.merge(
        drift_df[token_columns],
        on=["example_id", "position"],
        how="inner",
        validate="many_to_one",
    )
    eligible = eligible[
        eligible["semantic_safe"].eq(1)
        & (eligible["future_steps_after_refresh"] > 0)
    ].copy()
    if exclude_eos_eot:
        eligible = eligible[~eligible["selected_token"].isin([EOS_ID, EOT_ID])].copy()
    if eligible.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for example_id, group in eligible.groupby("example_id", sort=False):
        group = group.sort_values(
            ["refresh_step", "position", "event_id"], kind="mergesort"
        ).reset_index(drop=True)
        take = min(per_example, len(group))
        if selection == "top_kl":
            picks = (
                group.sort_values(
                    ["attention_kl", "refresh_step", "position"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
                .head(take)
                .index.tolist()
            )
        elif selection == "random":
            picks = sorted(rng.choice(len(group), size=take, replace=False).tolist())
        else:
            raise ValueError(f"unsupported Polestar proxy candidate selection: {selection}")
        for rank, idx in enumerate(picks, start=1):
            event = group.iloc[idx].copy()
            shared = event.to_dict()
            shared.update(
                {
                    "candidate_rank": rank,
                    "pair_id": str(event["event_id"]),
                    "freeze_step": int(event["refresh_step"]),
                    "lock_step": int(event["refresh_step"]),
                    "lock_delay_steps": int(event["cache_age"]),
                    "match_distance": 0.0,
                    "yc_source": "full_recompute_confidence_baseline",
                    "y0_equals_yc_by_design": True,
                    "proxy_kind": "token_level_per_head_attention_kl_single_refresh",
                }
            )
            rows.append(
                {
                    **shared,
                    "intervention_group": "polestar_proxy_stale",
                    "match_role": "target",
                    "proxy_arm": "stale",
                    "freeze_reference": "polestar_cache_stale",
                }
            )
            rows.append(
                {
                    **shared,
                    "intervention_group": "polestar_proxy_refresh",
                    "match_role": "control",
                    "proxy_arm": "refresh",
                    "freeze_reference": "polestar_cache_refresh",
                }
            )
    return pd.DataFrame(rows)


def validate_polestar_event_retention(
    selection: str,
    selected_event_count: int,
    retained_event_count: int,
) -> bool:
    truncated = retained_event_count < selected_event_count
    if selection == "top_kl" and truncated:
        raise ValueError(
            "top_kl requires every selected refresh event to be retained; "
            f"retained {retained_event_count} of {selected_event_count}"
        )
    return truncated


def choose_drift_candidates(
    drift_df: pd.DataFrame,
    per_example: int,
    exclude_eos_eot: bool = False,
) -> pd.DataFrame:
    eligible = drift_df[
        drift_df["semantic_safe"].eq(1)
        & drift_df["base_step"].notna()
        & drift_df["max_drift_mean"].notna()
        & (drift_df["future_steps_observed"] > 0)
    ].copy()
    if exclude_eos_eot:
        # EOS/EOT rows drift far more than body tokens and flood the high-drift
        # arm while being inert under the clamp; excluding them keeps the
        # high-vs-low contrast about answer-body tokens.
        eligible = eligible[~eligible["selected_token"].isin([EOS_ID, EOT_ID])].copy()
    if eligible.empty:
        return pd.DataFrame()
    rows = []
    for example_id, group in eligible.groupby("example_id", sort=False):
        high = group.sort_values("max_drift_mean", ascending=False, kind="mergesort").head(per_example).copy()
        controls_used: set[int] = set()
        match_features = ["selected_step", "relative_position", "selected_confidence", "future_steps_observed"]
        scales = group[match_features].astype(float).std().replace(0.0, 1.0).fillna(1.0)
        scales["selected_confidence"] = max(float(scales["selected_confidence"]), 0.05)
        scales["selected_step"] = max(float(scales["selected_step"]), 1.0)
        scales["relative_position"] = max(float(scales["relative_position"]), 1.0)
        scales["future_steps_observed"] = max(float(scales["future_steps_observed"]), 1.0)
        drift_median = float(group["max_drift_mean"].median())
        for rank, (_idx, target) in enumerate(high.iterrows(), start=1):
            pool = group[(group["position"] != target["position"]) & ~group["position"].astype(int).isin(controls_used)].copy()
            low_pool = pool[pool["max_drift_mean"] <= drift_median].copy()
            if low_pool.empty:
                low_pool = pool[pool["max_drift_mean"] < float(target["max_drift_mean"])].copy()
            if low_pool.empty:
                low_pool = pool
            if low_pool.empty:
                continue

            target_values = target[match_features].astype(float)
            distances = []
            for _, candidate in low_pool.iterrows():
                candidate_values = candidate[match_features].astype(float)
                distance = float(
                    np.linalg.norm(((candidate_values - target_values) / scales[match_features]).to_numpy(dtype=float))
                )
                distances.append(distance)
            low_pool = low_pool.copy()
            low_pool["match_distance"] = distances
            low_pool = low_pool.sort_values(
                ["match_distance", "max_drift_mean", "selected_step", "relative_position"],
                ascending=[True, True, True, True],
                kind="mergesort",
            )
            control = low_pool.iloc[0].copy()
            controls_used.add(int(control["position"]))
            pair_id = f"{example_id}:{int(target['position'])}:{int(target['selected_step'])}"

            target_row = target.copy()
            target_row["intervention_group"] = "semantic_safe_high_drift"
            target_row["match_role"] = "target"
            target_row["candidate_rank"] = rank
            target_row["pair_id"] = pair_id
            target_row["matched_control_position"] = int(control["position"])
            target_row["matched_control_relative_position"] = int(control["relative_position"])
            target_row["matched_control_selected_step"] = int(control["selected_step"])
            target_row["match_distance"] = float(control["match_distance"])

            control["intervention_group"] = "semantic_safe_matched_low_drift"
            control["match_role"] = "control"
            control["candidate_rank"] = rank
            control["pair_id"] = pair_id
            control["matched_to_position"] = int(target["position"])
            control["matched_to_relative_position"] = int(target["relative_position"])
            control["matched_to_selected_step"] = int(target["selected_step"])
            control["match_distance"] = float(control["match_distance"])
            rows.extend([target_row.to_frame().T, control.to_frame().T])
    if not rows:
        return pd.DataFrame()
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates.drop_duplicates(
        ["example_id", "position", "intervention_group"],
        keep="first",
    )
    return candidates.reset_index(drop=True)


def choose_rule_lock_candidates(
    drift_df: pd.DataFrame,
    per_example: int,
    exclude_eos_eot: bool = False,
    seed: int = 23,
) -> pd.DataFrame:
    """Phase 2 primary arm: the same token frozen at two different times.

    Target arm freezes at the commit step (the aggressive baseline); control arm
    freezes at the step a SureLock-style gate would admit it. Because both arms
    are the *same position in the same trajectory*, the pair needs no matching at
    all -- it is a within-token contrast, which is tighter than the across-token
    matching `choose_drift_candidates` has to do. The paired delta is then
    exactly "harm the gate's delay avoided".

    Tokens are sampled per example with a fixed seed rather than ranked by
    drift: ranking by drift would preselect the tokens most likely to be harmed
    and inflate both arms.
    """
    eligible = drift_df[
        drift_df["semantic_safe"].eq(1)
        & drift_df["base_step"].notna()
        & drift_df["lock_step"].notna()
        & drift_df["max_drift_mean"].notna()
        & (drift_df["future_steps_observed"] > 0)
    ].copy()
    if exclude_eos_eot:
        eligible = eligible[~eligible["selected_token"].isin([EOS_ID, EOT_ID])].copy()
    # A delay of zero steps gives two identical arms and no contrast.
    eligible = eligible[eligible["lock_step"].astype(float) > eligible["base_step"].astype(float)]
    if eligible.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for example_id, group in eligible.groupby("example_id", sort=False):
        group = group.sort_values("position", kind="mergesort")
        take = min(per_example, group.shape[0])
        picks = rng.choice(group.shape[0], size=take, replace=False)
        for rank, idx in enumerate(sorted(picks.tolist()), start=1):
            token_row = group.iloc[idx]
            pair_id = f"{example_id}:{int(token_row['position'])}:rulelock"
            common = {
                "candidate_rank": rank,
                "pair_id": pair_id,
                "match_distance": 0.0,
                "lock_delay_steps": int(float(token_row["lock_step"]) - float(token_row["base_step"])),
            }

            target = token_row.copy()
            target["intervention_group"] = "freeze_at_commit"
            target["match_role"] = "target"
            target["freeze_step"] = int(float(token_row["base_step"]))
            target["freeze_reference"] = "base"
            for key, value in common.items():
                target[key] = value

            control = token_row.copy()
            control["intervention_group"] = "freeze_at_rule_lock"
            control["match_role"] = "control"
            control["freeze_step"] = int(float(token_row["lock_step"]))
            control["freeze_reference"] = "lock"
            for key, value in common.items():
                control[key] = value

            rows.extend([target.to_frame().T, control.to_frame().T])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def choose_surelock_premise_candidates(
    drift_df: pd.DataFrame,
    per_example: int,
    exclude_eos_eot: bool = False,
    seed: int = 23,
) -> pd.DataFrame:
    """Select SureLock-admitted tokens for the clean Y0/YC/YF premise test.

    For post-commit tokens, the baseline trajectory is already the token-only
    commit arm: token identity is fixed, but every row is still recomputed.
    This selector therefore produces only the YF intervention rows: freeze the
    same token's reference at the SureLock-style lock step using the row cached
    at that step, then compare the replay output with the baseline-as-YC.
    """
    eligible = drift_df[
        drift_df["semantic_safe"].eq(1)
        & drift_df["base_step"].notna()
        & drift_df["lock_step"].notna()
        & drift_df["max_drift_mean"].notna()
        & (drift_df["future_steps_observed"] > 0)
    ].copy()
    if exclude_eos_eot:
        eligible = eligible[~eligible["selected_token"].isin([EOS_ID, EOT_ID])].copy()
    # The current online logger can only admit after the first post-commit row
    # has been captured. Keep the invariant explicit for readable candidate CSVs.
    eligible = eligible[eligible["lock_step"].astype(float) > eligible["base_step"].astype(float)]
    if eligible.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for example_id, group in eligible.groupby("example_id", sort=False):
        group = group.sort_values("position", kind="mergesort")
        take = min(per_example, group.shape[0])
        picks = rng.choice(group.shape[0], size=take, replace=False)
        for rank, idx in enumerate(sorted(picks.tolist()), start=1):
            token_row = group.iloc[idx].copy()
            token_row["intervention_group"] = "surelock_freeze_yf"
            token_row["match_role"] = "yf"
            token_row["candidate_rank"] = rank
            token_row["pair_id"] = f"{example_id}:{int(token_row['position'])}:surelock_premise"
            token_row["match_distance"] = 0.0
            token_row["lock_delay_steps"] = int(
                float(token_row["lock_step"]) - float(token_row["base_step"])
            )
            token_row["freeze_step"] = int(float(token_row["lock_step"]))
            token_row["freeze_reference"] = "lock"
            token_row["yc_source"] = "baseline_recomputed_reference"
            token_row["y0_equals_yc_by_design"] = True
            rows.append(token_row.to_frame().T)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def choose_less_premise_candidates(
    drift_df: pd.DataFrame,
    per_example: int,
    exclude_eos_eot: bool = False,
    seed: int = 23,
) -> pd.DataFrame:
    """Select genuine LESS admissions for a YC-versus-reference-freeze stress test."""
    eligible = drift_df[
        drift_df["semantic_safe"].eq(1)
        & drift_df["commit_source"].eq("less_accept")
        & drift_df["base_step"].notna()
        & drift_df["max_drift_mean"].notna()
        & (drift_df["future_steps_observed"] > 0)
    ].copy()
    if exclude_eos_eot:
        eligible = eligible[~eligible["selected_token"].isin([EOS_ID, EOT_ID])].copy()
    if eligible.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for example_id, group in eligible.groupby("example_id", sort=False):
        group = group.sort_values("position", kind="mergesort")
        take = min(per_example, group.shape[0])
        picks = rng.choice(group.shape[0], size=take, replace=False)
        for rank, idx in enumerate(sorted(picks.tolist()), start=1):
            token_row = group.iloc[idx].copy()
            token_row["intervention_group"] = "less_freeze_yf"
            token_row["match_role"] = "yf"
            token_row["candidate_rank"] = rank
            token_row["pair_id"] = f"{example_id}:{int(token_row['position'])}:less_premise"
            token_row["match_distance"] = 0.0
            token_row["lock_delay_steps"] = 0
            token_row["freeze_step"] = int(float(token_row["base_step"]))
            token_row["freeze_reference"] = "base"
            token_row["yc_source"] = "less_baseline_recomputed_reference"
            token_row["y0_equals_yc_by_design"] = True
            rows.append(token_row.to_frame().T)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def register_row_clamp_hooks(
    model: torch.nn.Module,
    layers: list[int],
    position: int,
    cached_hidden: dict[int, torch.Tensor],
    freeze_step: int,
) -> tuple[dict[str, int], list[Any]]:
    from run_logit_lens_token_fate import transformer_blocks

    blocks = transformer_blocks(model)
    state = {"global_step": -1}
    handles = []
    for layer in layers:
        cached = cached_hidden[layer]

        def make_hook(layer_idx: int, cached_tensor: torch.Tensor):
            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
                if state["global_step"] < freeze_step:
                    return None
                tensor = output[0] if isinstance(output, tuple) else output
                patched = tensor.clone()
                patched[0, position, :] = cached_tensor.to(device=tensor.device, dtype=tensor.dtype)
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer, cached)))
    return state, handles


@torch.no_grad()
def clamp_intervention_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    candidate: pd.Series,
    baseline: dict[str, Any],
    token: CommittedToken,
    config: DecodeConfig,
    layers: list[int],
    decode_policy: str = "confidence",
    less_config: LessConfig | None = None,
    reference_override: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    device = model_device(model)
    # Default is the historical behavior: freeze at the commit step against the
    # commit-step row. The rule-lock arm overrides both together -- a later
    # freeze step must be paired with the row cached at that step.
    if "freeze_step" in candidate and pd.notna(candidate["freeze_step"]):
        freeze_step = int(candidate["freeze_step"])
    else:
        freeze_step = int(candidate["base_step"])
    reference = reference_override if reference_override is not None else token.base_hidden
    if reference_override is None and str(candidate.get("freeze_reference", "base")) == "lock":
        if not token.lock_hidden:
            raise ValueError(f"no cached lock row for position {token.position}")
        reference = token.lock_hidden
    state, handles = register_row_clamp_hooks(model, layers, token.position, reference, freeze_step)
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
        num_blocks = config.gen_length // config.block_length
        steps_per_block = config.steps // num_blocks
        prev_log_probs: torch.Tensor | None = None
        prev_top1 = torch.full_like(x, fill_value=-1)
        runlength = torch.zeros_like(x, dtype=torch.float32)
        nfe = 0
        selected_token_matches = False
        pre_intervention_replay_valid = True
        pre_intervention_steps_compared = 0
        less_state = (
            LessSelectionState(total_len, device, less_config or LessConfig())
            if decode_policy == "less"
            else None
        )
        if less_state is not None and config.temperature != 0:
            raise ValueError("released-code LESS audit currently requires temperature=0")
        baseline_trace = baseline.get("decode_trace", {})
        trace_enabled = bool(baseline_trace)

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            if less_state is not None:
                less_state.reset_range(block_start, block_end)
            for step_in_block in range(steps_per_block):
                mask_index = x == config.mask_id
                active_block_mask = mask_index.clone()
                active_block_mask[:, :block_start] = False
                active_block_mask[:, block_end:] = False
                if less_state is not None and not active_block_mask.any():
                    break
                global_step = nfe if less_state is not None else block_idx * steps_per_block + step_in_block
                state["global_step"] = global_step
                input_tokens = x.detach().cpu().clone() if trace_enabled else None
                logits = model(x, attention_mask=attention_mask).logits
                nfe += 1
                x0, log_probs, runlength, feature_map = decode_step_features(
                    logits,
                    x,
                    mask_index,
                    prev_log_probs,
                    prev_top1,
                    runlength,
                    global_step,
                    config,
                )
                confidence = feature_map["confidence"]
                raw_top1 = x0.detach().cpu().clone() if trace_enabled else None
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                x0 = torch.where(mask_index, x0, x)
                k = int(num_transfer_tokens[0, step_in_block].item())
                transfer_index, _ = choose_step_transfer(
                    confidence,
                    log_probs,
                    allowed,
                    k,
                    less_state=less_state,
                )
                if trace_enabled and global_step <= freeze_step:
                    pre_intervention_replay_valid &= replay_prefix_matches(
                        baseline_trace,
                        global_step,
                        freeze_step,
                        input_tokens,
                        raw_top1,
                        transfer_index,
                    )
                    pre_intervention_steps_compared += 1
                if global_step == token.selected_step:
                    selected_token_matches = bool(
                        transfer_index[0, token.position].detach().cpu()
                        and int(x0[0, token.position].detach().cpu()) == token.selected_token
                    )
                x[transfer_index] = x0[transfer_index]
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del logits, log_probs

        final_tokens = x[:, prompt_len:].detach().cpu()[0].numpy().astype(np.int64, copy=False)
        baseline_tokens = np.asarray(baseline["final_tokens"], dtype=np.int64)
        effects = counterfactual_token_effects(
            final_tokens,
            baseline_tokens,
            token.relative_position,
        )
        changed_positions = np.flatnonzero(final_tokens != baseline_tokens).astype(int).tolist()
        non_target_positions = [pos for pos in changed_positions if pos != token.relative_position]
        generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        baseline_normalized_answer = normalize_answer(
            str(baseline["generation"]), example.dataset
        )
        clamped_normalized_answer = normalize_answer(generation, example.dataset)
        clamped_correct, clamped_scoring_status = score_generation_with_status(generation, example)
        answer_value_changed = not generated_answers_equivalent(
            str(baseline["generation"]), generation, example.dataset
        )
        return {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "intervention_group": candidate["intervention_group"],
            "match_role": candidate.get("match_role", ""),
            "pair_id": candidate.get("pair_id", ""),
            "candidate_rank": int(candidate["candidate_rank"]),
            "match_distance": float(candidate["match_distance"]) if "match_distance" in candidate else np.nan,
            "position": token.position,
            "relative_position": token.relative_position,
            "selected_step": token.selected_step,
            "base_step": token.base_step,
            "freeze_step": freeze_step,
            "freeze_reference": str(candidate.get("freeze_reference", "base")),
            "lock_step": token.lock_step if token.lock_step is not None else np.nan,
            "lock_delay_steps": int(candidate["lock_delay_steps"]) if "lock_delay_steps" in candidate and pd.notna(candidate.get("lock_delay_steps")) else np.nan,
            "selected_token": token.selected_token,
            "selected_confidence": token.selected_confidence,
            "selected_entropy": token.selected_entropy,
            "selected_margin": token.selected_margin,
            "selected_kl": token.selected_kl,
            "commit_source": token.commit_source,
            "selected_less_jsd": token.selected_less_jsd,
            "selected_less_persistence": token.selected_less_persistence,
            "future_steps_observed": token.future_steps_observed,
            "baseline_final_token": int(baseline["final_tokens"][token.relative_position]),
            "clamped_final_token": int(final_tokens[token.relative_position]),
            "selected_token_matches_replay": selected_token_matches,
            "pre_intervention_replay_valid": (
                pre_intervention_replay_valid if trace_enabled else np.nan
            ),
            "pre_intervention_steps_compared": (
                pre_intervention_steps_compared if trace_enabled else np.nan
            ),
            "baseline_correct": bool(baseline["correct"]),
            "baseline_scoring_status": baseline.get("scoring_status", "unknown"),
            "baseline_generation": baseline["generation"],
            "clamped_generation": generation,
            "baseline_normalized_answer": baseline_normalized_answer,
            "clamped_normalized_answer": clamped_normalized_answer,
            "normalized_answer_changed": baseline_normalized_answer != clamped_normalized_answer,
            "answer_value_changed": answer_value_changed,
            "clamped_correct": clamped_correct,
            "clamped_scoring_status": clamped_scoring_status,
            "answer_correct_changed": clamped_correct != bool(baseline["correct"]),
            "nfe": nfe,
            "decode_policy": decode_policy,
            "max_drift_mean": float(candidate["max_drift_mean"]),
            "final_drift_mean": float(candidate["final_drift_mean"]),
            "changed_relative_positions": json.dumps(changed_positions),
            "non_target_changed_relative_positions": json.dumps(non_target_positions),
            "polestar_event_id": candidate.get("event_id", ""),
            "polestar_proxy_arm": candidate.get("proxy_arm", ""),
            "polestar_cache_step": candidate.get("cache_step", np.nan),
            "polestar_refresh_step": candidate.get("refresh_step", np.nan),
            "polestar_cache_age": candidate.get("cache_age", np.nan),
            "polestar_attention_kl": candidate.get("attention_kl", np.nan),
            "polestar_refresh_rank": candidate.get("refresh_rank", np.nan),
            "polestar_refresh_pool_size": candidate.get("refresh_pool_size", np.nan),
            "polestar_refresh_drift_mean": candidate.get("refresh_drift_mean", np.nan),
            **effects,
        }
    finally:
        for handle in handles:
            handle.remove()


def choose_commit_candidates(
    proposal_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    per_example: int,
    min_lead: int,
    min_confidence: float,
    exclude_eos_eot: bool = True,
) -> pd.DataFrame:
    """Select observationally stable (S=1) token-steps for early forced commits.

    For each position the target is the earliest stable step with commit lead
    >= min_lead; the paired control is the same position forced at the latest
    stable step before its natural commit (schedule-perturbation floor).
    Conditioning on S uses final tokens by design: hypothesis A measures
    P(commit harm | S=1); this is offline fate decomposition, not a selector.
    """
    if proposal_df.empty or drift_df.empty:
        return pd.DataFrame()
    commits = drift_df[
        ["example_id", "position", "relative_position", "selected_step", "selected_token", "final_token"]
    ].copy()
    merged = proposal_df.merge(commits, on=["example_id", "position", "relative_position"], how="inner")
    merged = merged[merged["global_step"] < merged["selected_step"]].copy()
    merged["stable"] = merged["top_token"] == merged["final_token"]
    merged["lead"] = merged["selected_step"] - merged["global_step"]
    rows: list[dict[str, Any]] = []
    for example_id, group in merged.groupby("example_id", sort=False):
        ranked: list[dict[str, Any]] = []
        for position, pos_group in group.groupby("position", sort=False):
            pos_group = pos_group.sort_values("global_step", kind="mergesort")
            final_token = int(pos_group["final_token"].iloc[0])
            if exclude_eos_eot and final_token in (EOS_ID, EOT_ID):
                continue
            stable_rows = pos_group[pos_group["stable"]]
            eligible = stable_rows[
                (stable_rows["lead"] >= min_lead) & (stable_rows["confidence"] >= min_confidence)
            ]
            if eligible.empty:
                continue
            early = eligible.iloc[0]
            late_pool = stable_rows[stable_rows["global_step"] > int(early["global_step"])]
            if late_pool.empty:
                continue
            late = late_pool.iloc[-1]
            window = pos_group[pos_group["global_step"] >= int(early["global_step"])]
            ranked.append(
                {
                    "example_id": example_id,
                    "dataset": str(pos_group["dataset"].iloc[0]),
                    "position": int(position),
                    "relative_position": int(pos_group["relative_position"].iloc[0]),
                    "selected_step": int(pos_group["selected_step"].iloc[0]),
                    "selected_token": int(pos_group["selected_token"].iloc[0]),
                    "final_token": final_token,
                    "forced_token": final_token,
                    "is_eos_eot": final_token in (EOS_ID, EOT_ID),
                    "early_step": int(early["global_step"]),
                    "early_confidence": float(early["confidence"]),
                    "early_lead": int(early["lead"]),
                    "late_step": int(late["global_step"]),
                    "late_confidence": float(late["confidence"]),
                    "late_lead": int(late["lead"]),
                    "stable_run_fraction": float(window["stable"].mean()),
                }
            )
        ranked.sort(key=lambda item: (-item["early_confidence"], item["position"]))
        for rank, item in enumerate(ranked[:per_example], start=1):
            pair_id = f"{item['example_id']}:{item['position']}:{item['early_step']}"
            shared = {
                key: item[key]
                for key in (
                    "example_id",
                    "dataset",
                    "position",
                    "relative_position",
                    "selected_step",
                    "selected_token",
                    "final_token",
                    "forced_token",
                    "is_eos_eot",
                    "stable_run_fraction",
                )
            }
            rows.append(
                {
                    **shared,
                    "intervention_group": "commit_early_stable",
                    "match_role": "target",
                    "candidate_rank": rank,
                    "pair_id": pair_id,
                    "forced_step": item["early_step"],
                    "forced_step_confidence": item["early_confidence"],
                    "commit_lead": item["early_lead"],
                }
            )
            rows.append(
                {
                    **shared,
                    "intervention_group": "commit_late_stable_control",
                    "match_role": "control",
                    "candidate_rank": rank,
                    "pair_id": pair_id,
                    "forced_step": item["late_step"],
                    "forced_step_confidence": item["late_confidence"],
                    "commit_lead": item["late_lead"],
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def commit_intervention_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    candidate: pd.Series,
    baseline: dict[str, Any],
    config: DecodeConfig,
) -> dict[str, Any]:
    """Replay the decode, force-committing one stable token identity early.

    The forced position replaces one of the step's scheduled transfers
    (score=+inf), mirroring force_commit_decode in run_counterfactual_commit;
    representation updates continue everywhere (A1 arm: semantic lock only).
    """
    device = model_device(model)
    forced_step = int(candidate["forced_step"])
    forced_pos = int(candidate["position"])
    forced_token = int(candidate["forced_token"])
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
    prev_log_probs: torch.Tensor | None = None
    prev_top1 = torch.full_like(x, fill_value=-1)
    runlength = torch.zeros_like(x, dtype=torch.float32)
    nfe = 0
    forced_applied = False
    replay_top1_matches = False

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * config.block_length
        block_end = prompt_len + (block_idx + 1) * config.block_length
        block_mask_index = x[:, block_start:block_end] == config.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_in_block in range(steps_per_block):
            global_step = block_idx * steps_per_block + step_in_block
            mask_index = x == config.mask_id
            logits = model(x, attention_mask=attention_mask).logits
            nfe += 1
            x0, log_probs, runlength, feature_map = decode_step_features(
                logits,
                x,
                mask_index,
                prev_log_probs,
                prev_top1,
                runlength,
                global_step,
                config,
            )
            confidence = feature_map["confidence"]
            allowed = mask_index.clone()
            allowed[:, :prompt_len] = False
            allowed[:, block_end:] = False
            x0 = torch.where(mask_index, x0, x)
            score = torch.full_like(confidence, fill_value=-torch.inf)
            score[allowed] = confidence[allowed]

            if global_step == forced_step and bool(allowed[0, forced_pos].detach().cpu()):
                replay_top1_matches = int(x0[0, forced_pos].detach().cpu()) == forced_token
                x0[0, forced_pos] = forced_token
                score[0, forced_pos] = torch.inf
                forced_applied = True

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            k = int(num_transfer_tokens[0, step_in_block].item())
            if k > 0:
                _, select_index = torch.topk(score[0], k=k)
                transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]
            prev_log_probs = log_probs.detach()
            prev_top1 = x0.detach()
            del logits, log_probs, score

    final_tokens = x[:, prompt_len:].detach().cpu()[0].numpy().astype(np.int64, copy=False)
    baseline_tokens = np.asarray(baseline["final_tokens"], dtype=np.int64)
    rel_pos = int(candidate["relative_position"])
    effects = counterfactual_token_effects(final_tokens, baseline_tokens, rel_pos)
    changed_positions = np.flatnonzero(final_tokens != baseline_tokens).astype(int).tolist()
    non_target_positions = [pos for pos in changed_positions if pos != rel_pos]
    generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
    baseline_normalized_answer = normalize_answer(
        str(baseline["generation"]), example.dataset
    )
    forced_normalized_answer = normalize_answer(generation, example.dataset)
    forced_correct = score_generation(generation, example)
    return {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "intervention_group": candidate["intervention_group"],
        "match_role": candidate["match_role"],
        "pair_id": candidate["pair_id"],
        "candidate_rank": int(candidate["candidate_rank"]),
        "position": forced_pos,
        "relative_position": rel_pos,
        "selected_step": int(candidate["selected_step"]),
        "forced_step": forced_step,
        "commit_lead": int(candidate["commit_lead"]),
        "forced_token": forced_token,
        "forced_step_confidence": float(candidate["forced_step_confidence"]),
        "stable_run_fraction": float(candidate["stable_run_fraction"]),
        "is_eos_eot": bool(candidate["is_eos_eot"]),
        "forced_applied": forced_applied,
        "replay_top1_matches": replay_top1_matches,
        "baseline_final_token": int(baseline_tokens[rel_pos]),
        "forced_final_token": int(final_tokens[rel_pos]),
        "baseline_correct": bool(baseline["correct"]),
        "baseline_generation": baseline["generation"],
        "forced_generation": generation,
        "baseline_normalized_answer": baseline_normalized_answer,
        "forced_normalized_answer": forced_normalized_answer,
        "normalized_answer_changed": baseline_normalized_answer != forced_normalized_answer,
        "forced_correct": forced_correct,
        "answer_correct_changed": forced_correct != bool(baseline["correct"]),
        "nfe": nfe,
        "changed_relative_positions": json.dumps(changed_positions),
        "non_target_changed_relative_positions": json.dumps(non_target_positions),
        **effects,
    }


def write_commit_summary(
    output_dir: Path,
    args: argparse.Namespace,
    drift_df: pd.DataFrame,
    commit_df: pd.DataFrame,
    started_at: float,
) -> None:
    summary: dict[str, Any] = {
        "intervention_mode": "commit",
        "examples": int(drift_df["example_id"].nunique()) if not drift_df.empty else 0,
        "committed_tokens": int(len(drift_df)),
        "semantic_safe_rate": float(drift_df["semantic_safe"].mean()) if len(drift_df) else None,
        "interventions": int(len(commit_df)),
        "latency_s": time.perf_counter() - started_at,
        "config": vars(args),
    }
    group_summary = pd.DataFrame()
    pair_summary = pd.DataFrame()
    if not commit_df.empty:
        group_summary = (
            commit_df.groupby("intervention_group")
            .agg(
                candidates=("example_id", "size"),
                mean_commit_lead=("commit_lead", "mean"),
                mean_forced_step_confidence=("forced_step_confidence", "mean"),
                forced_applied_rate=("forced_applied", "mean"),
                replay_top1_match_rate=("replay_top1_matches", "mean"),
                target_token_change_rate=("target_token_changed", "mean"),
                non_target_token_change_rate=("non_target_token_changed", "mean"),
                mean_non_target_token_change_count=("non_target_token_change_count", "mean"),
                answer_correct_change_rate=("answer_correct_changed", "mean"),
                normalized_answer_change_rate=("normalized_answer_changed", "mean"),
            )
            .reset_index()
        )
        group_summary.to_csv(output_dir / "commit_group_summary.csv", index=False)
        summary["intervention_group_summary"] = group_summary.to_dict(orient="records")

        pair_rows: list[dict[str, Any]] = []
        for pair_id, pair in commit_df.groupby("pair_id", sort=False):
            targets = pair[pair["match_role"].eq("target")]
            controls = pair[pair["match_role"].eq("control")]
            if targets.empty or controls.empty:
                continue
            target = targets.iloc[0]
            control = controls.iloc[0]
            target_changed = set(json.loads(str(target["non_target_changed_relative_positions"])))
            control_changed = set(json.loads(str(control["non_target_changed_relative_positions"])))
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": target["example_id"],
                    "position": int(target["position"]),
                    "relative_position": int(target["relative_position"]),
                    "selected_step": int(target["selected_step"]),
                    "target_forced_step": int(target["forced_step"]),
                    "control_forced_step": int(control["forced_step"]),
                    "target_commit_lead": int(target["commit_lead"]),
                    "control_commit_lead": int(control["commit_lead"]),
                    "target_forced_step_confidence": float(target["forced_step_confidence"]),
                    "control_forced_step_confidence": float(control["forced_step_confidence"]),
                    "stable_run_fraction": float(target["stable_run_fraction"]),
                    "is_eos_eot": bool(target["is_eos_eot"]),
                    "target_non_target_changed": bool(target["non_target_token_changed"]),
                    "control_non_target_changed": bool(control["non_target_token_changed"]),
                    "delta_non_target_changed": float(target["non_target_token_changed"])
                    - float(control["non_target_token_changed"]),
                    "target_non_target_change_count": int(target["non_target_token_change_count"]),
                    "control_non_target_change_count": int(control["non_target_token_change_count"]),
                    "delta_non_target_change_count": int(target["non_target_token_change_count"])
                    - int(control["non_target_token_change_count"]),
                    "shared_non_target_change_count": int(len(target_changed & control_changed)),
                    "target_unique_non_target_change_count": int(len(target_changed - control_changed)),
                    "control_unique_non_target_change_count": int(len(control_changed - target_changed)),
                    "target_answer_changed": bool(target["answer_correct_changed"]),
                    "control_answer_changed": bool(control["answer_correct_changed"]),
                    "delta_answer_changed": float(target["answer_correct_changed"])
                    - float(control["answer_correct_changed"]),
                    "target_normalized_answer_changed": bool(target["normalized_answer_changed"]),
                    "control_normalized_answer_changed": bool(control["normalized_answer_changed"]),
                    "delta_normalized_answer_changed": float(target["normalized_answer_changed"])
                    - float(control["normalized_answer_changed"]),
                }
            )
        pair_summary = pd.DataFrame(pair_rows)
        if not pair_summary.empty:
            pair_summary.to_csv(output_dir / "commit_pair_summary.csv", index=False)
            summary["paired_complete_pairs"] = int(len(pair_summary))
            summary["paired_early_minus_late_non_target_change_rate"] = float(
                pair_summary["delta_non_target_changed"].mean()
            )
            summary["paired_early_minus_late_mean_non_target_count"] = float(
                pair_summary["delta_non_target_change_count"].mean()
            )
            summary["paired_early_minus_late_answer_change_rate"] = float(pair_summary["delta_answer_changed"].mean())
            summary["paired_early_minus_late_normalized_answer_change_rate"] = float(
                pair_summary["delta_normalized_answer_changed"].mean()
            )
        targets_only = commit_df[commit_df["match_role"].eq("target")]
        if not targets_only.empty:
            summary["early_commit_non_target_change_rate"] = float(targets_only["non_target_token_changed"].mean())
            summary["early_commit_normalized_answer_change_rate"] = float(
                targets_only["normalized_answer_changed"].mean()
            )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Settlement Fate Audit (commit mode / hypothesis A)",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2),
        "```",
    ]
    if not group_summary.empty:
        lines.extend(["", "## Commit Intervention Groups", table_text(group_summary)])
    if not pair_summary.empty:
        lines.extend(["", "## Early-vs-Late Matched Pair Summary", table_text(pair_summary)])
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Candidates are oracle-conditioned on observational stability (S=1) by "
            "design: hypothesis A measures P(commit harm | S=1), so final tokens are "
            "used to define the stable set, not to build a prospective selector. "
            "The forced commit replaces one scheduled transfer (semantic lock only); "
            "representation updates continue at every position. The late-commit "
            "control measures the schedule-perturbation floor for the same token.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_surelock_premise_summary(
    output_dir: Path,
    args: argparse.Namespace,
    drift_df: pd.DataFrame,
    interventions_df: pd.DataFrame,
    started_at: float,
) -> None:
    """Write the clean gate-specific Y0/YC/YF premise summary.

    Y0 is the baseline decode. YC is not a separate replay in this codebase:
    after LLaDA transfers a token, its identity is already fixed while the row is
    recomputed on every subsequent step. The intervention rows are YF: the same
    token identity, but the cached reference is frozen at the SureLock-style
    admission step.
    """
    is_less = args.intervention_mode == "less_premise"
    strategy = "LESS" if is_less else "SureLock-style"
    yc_description = (
        "LESS decode after token identity is committed while representation is recomputed"
        if is_less
        else "baseline decode after token identity is committed while representation is recomputed"
    )
    yf_description = (
        "LESS replay with an admitted token's reference frozen from its first post-commit row"
        if is_less
        else "replay with the same token reference frozen at the SureLock-style lock step"
    )
    tokens_with_drift = drift_df[
        drift_df.get("max_drift_mean", pd.Series(dtype=float)).notna()
        & (drift_df.get("future_steps_observed", pd.Series(dtype=float)) > 0)
    ] if not drift_df.empty else pd.DataFrame()
    summary: dict[str, Any] = {
        "intervention_mode": args.intervention_mode,
        "estimand": f"P(YF != YC | semantic_safe=1, {strategy} admitted token)",
        "y0": "LESS baseline decode" if is_less else "baseline decode",
        "yc": yc_description,
        "yf": yf_description,
        "yc_equals_y0_by_design": True,
        "examples": int(drift_df["example_id"].nunique()) if not drift_df.empty else 0,
        "committed_tokens": int(len(drift_df)),
        "tokens_with_post_commit_state": int(drift_df["base_step"].notna().sum()) if "base_step" in drift_df else 0,
        "tokens_with_drift_measurement": int(len(tokens_with_drift)),
        "semantic_safe_rate": float(drift_df["semantic_safe"].mean()) if len(drift_df) else None,
        "interventions": int(len(interventions_df)),
        "latency_s": time.perf_counter() - started_at,
        "config": vars(args),
    }
    if is_less:
        summary["less_rule"] = {
            "confidence_threshold": float(args.less_confidence_threshold),
            "jsd_threshold": float(args.less_jsd_threshold),
            "top_k": int(args.less_top_k),
            "prior_winner_history": int(args.less_history_length),
            "variant": "released LLaDA implementation semantics",
        }
        if "commit_source" in drift_df:
            summary["less_accept_token_rate"] = float(
                drift_df["commit_source"].eq("less_accept").mean()
            )
    else:
        summary["surelock_tau"] = float(args.lock_rule_tau)
    if "baseline_nfe" in drift_df and not drift_df.empty:
        baseline_nfe = drift_df.groupby("example_id", sort=False)["baseline_nfe"].first()
        summary["mean_baseline_nfe"] = float(baseline_nfe.mean())
        summary["mean_baseline_nfe_fraction"] = float(baseline_nfe.mean() / args.steps)

    premise_summary = pd.DataFrame()
    examples_df = pd.DataFrame()
    if not interventions_df.empty:
        has_full_prefix_check = (
            "pre_intervention_replay_valid" in interventions_df
            and interventions_df["pre_intervention_replay_valid"].notna().all()
        )
        replay_valid = (
            interventions_df["pre_intervention_replay_valid"].astype(bool)
            if has_full_prefix_check
            else interventions_df["selected_token_matches_replay"].astype(bool)
        )
        token_unchanged = ~interventions_df["target_token_changed"].astype(bool)
        clean_identity = replay_valid & token_unchanged
        clean_non_target = clean_identity & interventions_df["non_target_token_changed"].astype(bool)
        answer_value_changed = interventions_df.get(
            "answer_value_changed", interventions_df["normalized_answer_changed"]
        ).astype(bool)
        clean_answer = clean_identity & answer_value_changed
        baseline_correct = interventions_df["baseline_correct"].astype(bool)
        modified_correct = interventions_df["clamped_correct"].astype(bool)
        infrastructure_failures = {"sandbox_unavailable", "sandbox_error"}
        baseline_status = interventions_df.get(
            "baseline_scoring_status", pd.Series("passed", index=interventions_df.index)
        )
        modified_status = interventions_df.get(
            "clamped_scoring_status", pd.Series("passed", index=interventions_df.index)
        )
        baseline_scorable = ~baseline_status.isin(infrastructure_failures)
        modified_scorable = ~modified_status.isin(infrastructure_failures)
        paired_scorable = baseline_scorable & modified_scorable
        original_unique_example_accuracy = float(
            interventions_df[baseline_scorable]
            .assign(_baseline_correct=baseline_correct[baseline_scorable])
            .groupby("example_id", sort=False)["_baseline_correct"]
            .first()
            .mean()
        )
        baseline_scoring_failure_rate = float((~baseline_scorable).mean())
        modified_scoring_failure_rate = float((~modified_scorable).mean())
        premise_row = {
            "arm": (
                "less_admission_freeze_yf_vs_less_baseline_yc"
                if is_less
                else "surelock_freeze_yf_vs_baseline_yc"
            ),
            "candidates": int(len(interventions_df)),
            "yc_equals_y0_by_design_rate": 1.0,
            "replay_valid_rate": float(replay_valid.mean()),
            "replay_valid_definition": (
                "full pre-freeze input/top1/transfer prefix"
                if has_full_prefix_check
                else "selected token and selected step"
            ),
            "target_token_change_rate": float(interventions_df["target_token_changed"].mean()),
            "clean_identity_rate": float(clean_identity.mean()),
            "non_target_token_change_rate": float(interventions_df["non_target_token_changed"].mean()),
            "mean_non_target_token_change_count": float(
                interventions_df["non_target_token_change_count"].mean()
            ),
            "normalized_answer_change_rate": float(interventions_df["normalized_answer_changed"].mean()),
            "answer_value_change_rate": float(answer_value_changed.mean()),
            "original_accuracy": float(baseline_correct[paired_scorable].mean()),
            "original_unique_example_accuracy": original_unique_example_accuracy,
            "modified_accuracy": float(modified_correct[paired_scorable].mean()),
            "accuracy_delta": float(
                (
                    modified_correct[paired_scorable].astype(float)
                    - baseline_correct[paired_scorable].astype(float)
                ).mean()
            ),
            "answer_correct_change_rate": float(
                baseline_correct[paired_scorable].ne(modified_correct[paired_scorable]).mean()
            ),
            "correct_to_wrong_rate": float(
                (baseline_correct[paired_scorable] & ~modified_correct[paired_scorable]).mean()
            ),
            "wrong_to_correct_rate": float(
                (~baseline_correct[paired_scorable] & modified_correct[paired_scorable]).mean()
            ),
            "original_scoring_infrastructure_failure_rate": baseline_scoring_failure_rate,
            "modified_scoring_infrastructure_failure_rate": modified_scoring_failure_rate,
            "original_execution_timeout_rate": float(baseline_status.eq("timeout").mean()),
            "modified_execution_timeout_rate": float(modified_status.eq("timeout").mean()),
            "clean_b1_non_target_change_rate": float(clean_non_target.mean()),
            "clean_b1_answer_value_change_rate": float(clean_answer.mean()),
            "mean_lock_delay_steps": float(interventions_df["lock_delay_steps"].mean()),
            "median_lock_delay_steps": float(interventions_df["lock_delay_steps"].median()),
            "mean_max_drift": float(interventions_df["max_drift_mean"].mean()),
        }
        premise_summary = pd.DataFrame([premise_row])
        premise_summary.to_csv(output_dir / "surelock_premise_summary.csv", index=False)
        summary["surelock_premise_summary"] = premise_summary.to_dict(orient="records")

        changed = interventions_df[
            interventions_df["non_target_token_changed"].astype(bool)
            | answer_value_changed
        ].copy()
        if not changed.empty:
            example_columns = [
                "example_id",
                "position",
                "relative_position",
                "selected_step",
                "lock_step",
                "lock_delay_steps",
                "selected_confidence",
                "max_drift_mean",
                "non_target_token_change_count",
                "normalized_answer_changed",
                "answer_value_changed",
                "answer_correct_changed",
                "baseline_normalized_answer",
                "clamped_normalized_answer",
                "baseline_correct",
                "clamped_correct",
            ]
            if is_less:
                example_columns.extend(
                    column
                    for column in (
                        "commit_source",
                        "selected_less_jsd",
                        "selected_less_persistence",
                    )
                    if column in changed
                )
            for status_column in ("baseline_scoring_status", "clamped_scoring_status"):
                if status_column in changed:
                    example_columns.append(status_column)
            examples_df = changed[example_columns].head(20)
            examples_df.to_csv(output_dir / "surelock_premise_changed_examples.csv", index=False)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# LESS Admission-Reuse Premise Audit" if is_less else "# SureLock Premise Audit",
        "",
        "## Design",
        "",
        f"This is the clean Y0/YC/YF framing for a {strategy} admission:",
        "",
        "- `Y0`: LESS baseline decode." if is_less else "- `Y0`: baseline decode.",
        "- `YC`: token identity committed while the row/reference continues to be recomputed.",
        (
            "- `YF`: same LESS decode, but a genuinely LESS-accepted token's reference is "
            "frozen from its first post-commit row."
            if is_less
            else "- `YF`: same token, but its reference is frozen at the SureLock-style lock step."
        ),
        "",
        "In this LLaDA audit, `YC` is the baseline post-commit trajectory by design: "
        "transferred token identities are fixed, but their rows still update. The "
        "measured endpoint is therefore whether `YF` differs from baseline/`YC`.",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2),
        "```",
    ]
    if not premise_summary.empty:
        lines.extend(["", "## Premise Endpoint", table_text(premise_summary)])
    if not examples_df.empty:
        lines.extend(["", "## Changed Examples", table_text(examples_df)])
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            (
                "LESS itself changes token commitment and does not claim to freeze hidden/K/V rows. "
                "This arm deliberately asks whether a LESS admission can safely be reused for "
                "reference freezing; it is not a claim about the safety of LESS as published."
                if is_less
                else "This remains a stale-row hidden-state clamp proxy, not a real cached-K/V "
                "or row-removal runtime implementation. It tests whether a SureLock-style "
                "posterior-stability gate certifies reference-freeze safety; it does not "
                "claim wall-clock compute savings."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_polestar_cache_proxy_summary(
    output_dir: Path,
    args: argparse.Namespace,
    drift_df: pd.DataFrame,
    interventions_df: pd.DataFrame,
    event_df: pd.DataFrame,
    selected_event_count: int,
    events_truncated: bool,
    started_at: float,
) -> None:
    """Summarize the token-level attention-KL single-refresh proxy."""
    group_summary = pd.DataFrame()
    pair_summary = pd.DataFrame()
    if not interventions_df.empty:
        group_summary = (
            interventions_df.groupby("intervention_group", sort=False)
            .agg(
                interventions=("example_id", "size"),
                replay_valid_rate=("pre_intervention_replay_valid", "mean"),
                target_token_change_rate=("target_token_changed", "mean"),
                downstream_change_rate=("non_target_token_changed", "mean"),
                mean_downstream_change_count=("non_target_token_change_count", "mean"),
                answer_value_change_rate=("answer_value_changed", "mean"),
                modified_accuracy=("clamped_correct", "mean"),
                mean_attention_kl=("polestar_attention_kl", "mean"),
                mean_cache_age=("polestar_cache_age", "mean"),
                mean_refresh_drift=("polestar_refresh_drift_mean", "mean"),
            )
            .reset_index()
        )
        group_summary.to_csv(output_dir / "polestar_proxy_group_summary.csv", index=False)

        pair_rows = []
        for pair_id, pair in interventions_df.groupby("pair_id", sort=False):
            stale_rows = pair[pair["polestar_proxy_arm"].eq("stale")]
            refresh_rows = pair[pair["polestar_proxy_arm"].eq("refresh")]
            if len(stale_rows) != 1 or len(refresh_rows) != 1:
                continue
            stale = stale_rows.iloc[0]
            refresh = refresh_rows.iloc[0]
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": stale["example_id"],
                    "position": int(stale["position"]),
                    "cache_step": int(stale["polestar_cache_step"]),
                    "refresh_step": int(stale["polestar_refresh_step"]),
                    "cache_age": int(stale["polestar_cache_age"]),
                    "attention_kl": float(stale["polestar_attention_kl"]),
                    "refresh_drift_mean": float(stale["polestar_refresh_drift_mean"]),
                    "replay_valid": bool(stale["pre_intervention_replay_valid"])
                    and bool(refresh["pre_intervention_replay_valid"]),
                    "target_unchanged": not bool(stale["target_token_changed"])
                    and not bool(refresh["target_token_changed"]),
                    "stale_downstream_changed": bool(stale["non_target_token_changed"]),
                    "refresh_downstream_changed": bool(refresh["non_target_token_changed"]),
                    "stale_minus_refresh_downstream_changed": float(
                        stale["non_target_token_changed"]
                    )
                    - float(refresh["non_target_token_changed"]),
                    "stale_downstream_change_count": int(
                        stale["non_target_token_change_count"]
                    ),
                    "refresh_downstream_change_count": int(
                        refresh["non_target_token_change_count"]
                    ),
                    "stale_minus_refresh_change_count": int(
                        stale["non_target_token_change_count"]
                    )
                    - int(refresh["non_target_token_change_count"]),
                    "stale_answer_changed": bool(stale["answer_value_changed"]),
                    "refresh_answer_changed": bool(refresh["answer_value_changed"]),
                    "stale_minus_refresh_answer_changed": float(
                        stale["answer_value_changed"]
                    )
                    - float(refresh["answer_value_changed"]),
                    "baseline_correct": bool(stale["baseline_correct"]),
                    "stale_correct": bool(stale["clamped_correct"]),
                    "refresh_correct": bool(refresh["clamped_correct"]),
                }
            )
        pair_summary = pd.DataFrame(pair_rows)
        pair_summary.to_csv(output_dir / "polestar_proxy_pair_summary.csv", index=False)

    summary: dict[str, Any] = {
        "intervention_mode": "polestar_cache_proxy",
        "proxy_kind": "token_level_per_head_attention_kl_single_refresh",
        "not_full_polestar_reason": (
            "No author code is available; this run omits centroid proxy attention, quantization, "
            "sparse per-layer KV updates, suffix update packets, and Triton runtime optimizations."
        ),
        "estimand": (
            "stale-minus-refreshed hidden-row harm at online attention-KL-selected prefix-cache events"
        ),
        "examples": int(drift_df["example_id"].nunique()) if not drift_df.empty else 0,
        "committed_tokens": int(len(drift_df)),
        "selected_refresh_events": int(selected_event_count),
        "retained_refresh_events": int(len(event_df)),
        "events_truncated": bool(events_truncated),
        "event_retention_policy": (
            "deterministic_hash_reservoir" if events_truncated else "all_selected_events"
        ),
        "interventions": int(len(interventions_df)),
        "complete_pairs": int(len(pair_summary)),
        "latency_s": time.perf_counter() - started_at,
        "config": vars(args),
        "group_summary": group_summary.to_dict(orient="records"),
    }
    if not interventions_df.empty:
        summary.update(
            {
                "full_prefix_replay_valid_rate": float(
                    interventions_df["pre_intervention_replay_valid"].mean()
                ),
                "target_unchanged_rate": float(
                    (~interventions_df["target_token_changed"].astype(bool)).mean()
                ),
                "original_accuracy": float(interventions_df["baseline_correct"].mean()),
            }
        )
    if not pair_summary.empty:
        summary.update(
            {
                "paired_stale_minus_refresh_downstream_change_rate": float(
                    pair_summary["stale_minus_refresh_downstream_changed"].mean()
                ),
                "paired_stale_minus_refresh_mean_change_count": float(
                    pair_summary["stale_minus_refresh_change_count"].mean()
                ),
                "paired_stale_minus_refresh_answer_change_rate": float(
                    pair_summary["stale_minus_refresh_answer_changed"].mean()
                ),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Polestar-Cache Attention-KL Single-Refresh Proxy",
        "",
        "## Scope",
        "",
        "This is not a full Polestar implementation or a speed benchmark. It replaces "
        "Polestar's under-specified centroid proxy with token-level per-head attention KL and "
        "tests one selected refresh for a committed generated prefix token using the "
        "repository's hidden-row clamp proxy.",
        "",
        "- `YC`: full-recompute confidence baseline.",
        "- `Y_stale`: at the selected refresh step, keep serving the block-entry row.",
        "- `Y_refresh`: at the same step, refresh to the current baseline row, then hold it.",
        "",
        "A positive stale-minus-refresh effect means the attention-KL signal identified a "
        "refresh that prevented downstream divergence.",
        "",
        "```json",
        json.dumps({key: value for key, value in summary.items() if key != "config"}, indent=2),
        "```",
    ]
    if not group_summary.empty:
        lines.extend(["", "## Arms", "", table_text(group_summary)])
    if not pair_summary.empty:
        lines.extend(["", "## Paired Events", "", table_text(pair_summary.head(20))])
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            summary["not_full_polestar_reason"],
            "The run cannot establish KV-cache speedup, TPF/TPS gains, or full Polestar quality.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    drift_df: pd.DataFrame,
    interventions_df: pd.DataFrame,
    started_at: float,
) -> None:
    tokens_with_drift = drift_df[
        drift_df.get("max_drift_mean", pd.Series(dtype=float)).notna()
        & (drift_df.get("future_steps_observed", pd.Series(dtype=float)) > 0)
    ] if not drift_df.empty else pd.DataFrame()
    summary: dict[str, Any] = {
        "examples": int(drift_df["example_id"].nunique()) if not drift_df.empty else 0,
        "committed_tokens": int(len(drift_df)),
        "tokens_with_post_commit_state": int(drift_df["base_step"].notna().sum()) if "base_step" in drift_df else 0,
        "tokens_with_drift_measurement": int(len(tokens_with_drift)),
        "semantic_safe_rate": float(drift_df["semantic_safe"].mean()) if len(drift_df) else None,
        "mean_max_drift": float(drift_df["max_drift_mean"].dropna().mean()) if len(drift_df) else None,
        "median_max_drift": float(drift_df["max_drift_mean"].dropna().median()) if len(drift_df) else None,
        "interventions": int(len(interventions_df)),
        "latency_s": time.perf_counter() - started_at,
        "config": vars(args),
    }
    if not interventions_df.empty:
        group_summary = (
            interventions_df.groupby("intervention_group")
            .agg(
                candidates=("example_id", "size"),
                mean_max_drift=("max_drift_mean", "mean"),
                target_token_change_rate=("target_token_changed", "mean"),
                non_target_token_change_rate=("non_target_token_changed", "mean"),
                mean_non_target_token_change_count=("non_target_token_change_count", "mean"),
                answer_correct_change_rate=("answer_correct_changed", "mean"),
                normalized_answer_change_rate=("normalized_answer_changed", "mean"),
                replay_match_rate=("selected_token_matches_replay", "mean"),
            )
            .reset_index()
        )
        group_summary.to_csv(output_dir / "freeze_group_summary.csv", index=False)
        summary["intervention_group_summary"] = group_summary.to_dict(orient="records")
    else:
        group_summary = pd.DataFrame()
    pair_summary = pd.DataFrame()
    if not interventions_df.empty and {"pair_id", "match_role"}.issubset(interventions_df.columns):
        pair_rows: list[dict[str, Any]] = []
        for pair_id, pair in interventions_df.groupby("pair_id", sort=False):
            if not pair_id:
                continue
            targets = pair[pair["match_role"].eq("target")]
            controls = pair[pair["match_role"].eq("control")]
            if targets.empty or controls.empty:
                continue
            target = targets.iloc[0]
            control = controls.iloc[0]
            target_changed_positions = set(json.loads(str(target.get("non_target_changed_relative_positions", "[]"))))
            control_changed_positions = set(json.loads(str(control.get("non_target_changed_relative_positions", "[]"))))
            shared_changed_positions = sorted(target_changed_positions & control_changed_positions)
            target_unique_positions = sorted(target_changed_positions - control_changed_positions)
            control_unique_positions = sorted(control_changed_positions - target_changed_positions)
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": target["example_id"],
                    "target_position": int(target["position"]),
                    "control_position": int(control["position"]),
                    "target_selected_step": int(target["selected_step"]),
                    "control_selected_step": int(control["selected_step"]),
                    "target_relative_position": int(target["relative_position"]),
                    "control_relative_position": int(control["relative_position"]),
                    "target_confidence": float(target["selected_confidence"]),
                    "control_confidence": float(control["selected_confidence"]),
                    "target_future_steps_observed": int(target["future_steps_observed"]),
                    "control_future_steps_observed": int(control["future_steps_observed"]),
                    "match_distance": float(target["match_distance"]),
                    "target_max_drift": float(target["max_drift_mean"]),
                    "control_max_drift": float(control["max_drift_mean"]),
                    "target_non_target_changed": bool(target["non_target_token_changed"]),
                    "control_non_target_changed": bool(control["non_target_token_changed"]),
                    "delta_non_target_changed": float(target["non_target_token_changed"])
                    - float(control["non_target_token_changed"]),
                    "target_non_target_change_count": int(target["non_target_token_change_count"]),
                    "control_non_target_change_count": int(control["non_target_token_change_count"]),
                    "delta_non_target_change_count": int(target["non_target_token_change_count"])
                    - int(control["non_target_token_change_count"]),
                    "shared_non_target_change_count": int(len(shared_changed_positions)),
                    "target_unique_non_target_change_count": int(len(target_unique_positions)),
                    "control_unique_non_target_change_count": int(len(control_unique_positions)),
                    "shared_non_target_changed_positions": json.dumps(shared_changed_positions),
                    "target_unique_non_target_changed_positions": json.dumps(target_unique_positions),
                    "control_unique_non_target_changed_positions": json.dumps(control_unique_positions),
                    "target_answer_changed": bool(target["answer_correct_changed"]),
                    "control_answer_changed": bool(control["answer_correct_changed"]),
                    "delta_answer_changed": float(target["answer_correct_changed"])
                    - float(control["answer_correct_changed"]),
                    "target_normalized_answer_changed": bool(target["normalized_answer_changed"]),
                    "control_normalized_answer_changed": bool(control["normalized_answer_changed"]),
                    "delta_normalized_answer_changed": float(target["normalized_answer_changed"])
                    - float(control["normalized_answer_changed"]),
                }
            )
        pair_summary = pd.DataFrame(pair_rows)
        if not pair_summary.empty:
            pair_summary.to_csv(output_dir / "freeze_pair_summary.csv", index=False)
            summary["paired_complete_pairs"] = int(len(pair_summary))
            summary["paired_target_minus_control_non_target_change_rate"] = float(
                pair_summary["delta_non_target_changed"].mean()
            )
            summary["paired_target_minus_control_mean_non_target_count"] = float(
                pair_summary["delta_non_target_change_count"].mean()
            )
            summary["paired_mean_shared_non_target_change_count"] = float(
                pair_summary["shared_non_target_change_count"].mean()
            )
            summary["paired_mean_target_unique_non_target_change_count"] = float(
                pair_summary["target_unique_non_target_change_count"].mean()
            )
            summary["paired_target_minus_control_answer_change_rate"] = float(pair_summary["delta_answer_changed"].mean())
            summary["paired_target_minus_control_normalized_answer_change_rate"] = float(
                pair_summary["delta_normalized_answer_changed"].mean()
            )
    if not tokens_with_drift.empty:
        semantic_drift_summary = (
            tokens_with_drift.groupby("semantic_safe")
            .agg(
                tokens=("example_id", "size"),
                mean_selected_confidence=("selected_confidence", "mean"),
                mean_max_drift=("max_drift_mean", "mean"),
                median_max_drift=("max_drift_mean", "median"),
                p90_max_drift=("max_drift_mean", lambda values: float(values.quantile(0.9))),
            )
            .reset_index()
        )
        semantic_drift_summary.to_csv(output_dir / "semantic_drift_summary.csv", index=False)
        summary["semantic_drift_summary"] = semantic_drift_summary.to_dict(orient="records")
    else:
        semantic_drift_summary = pd.DataFrame()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Settlement Fate Audit",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2),
        "```",
        "",
        "## Drift Quantiles",
    ]
    if not drift_df.empty:
        quantiles = drift_df["max_drift_mean"].dropna().quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        lines.append("```text")
        lines.append(quantiles.to_string())
        lines.append("```")
    else:
        lines.append("No committed-token drift rows were collected.")
    if not group_summary.empty:
        lines.extend(["", "## Freeze Intervention Groups", table_text(group_summary)])
    if not pair_summary.empty:
        lines.extend(["", "## Matched Pair Summary", table_text(pair_summary)])
    if not semantic_drift_summary.empty:
        lines.extend(["", "## Semantic Safety vs Representation Drift", table_text(semantic_drift_summary)])
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This script uses a stale-row hidden-state clamp as a compute-freeze proxy. "
            "It is intended to test whether semantic commitment and representation "
            "evolution can diverge before integrating a SureLock-style cached-K/V "
            "implementation.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether committed-token identity, representation drift, and stale-row compute safety diverge."
    )
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=[
            "gsm8k",
            "math500",
            "math-500",
            "countdown",
            "humaneval",
            "human_eval",
            "aime2024",
        ],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--jsonl-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--cpv-window", type=int, default=4)
    parser.add_argument("--layers", default="16,24,32")
    parser.add_argument("--max-interventions-per-example", type=int, default=1)
    parser.add_argument(
        "--intervention-mode",
        choices=[
            "clamp",
            "commit",
            "rule_lock",
            "surelock_premise",
            "less_premise",
            "polestar_cache_proxy",
        ],
        default="clamp",
        help="clamp = stale-row reference-freeze proxy (hypotheses B/C); commit = early forced "
        "semantic lock (auxiliary timing); rule_lock = Phase 2, the same token frozen at the "
        "commit step vs at a SureLock-style gate's own lock step; surelock_premise = clean "
        "Y0/YC/YF SureLock audit, freezing only at the gate's own lock step "
        "; less_premise = run the released-code LESS sampler, then freeze only "
        "genuine LESS-accepted tokens from their first post-commit row; "
        "polestar_cache_proxy = token-level true-attention-KL single-refresh "
        "stale-vs-refreshed hidden-row proxy, not full Polestar "
        "(see docs/lock_admission_audit.md).",
    )
    parser.add_argument("--commit-min-lead", type=int, default=4)
    parser.add_argument("--commit-min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Candidate sampling seed for rule-lock and Polestar-proxy modes.",
    )
    parser.add_argument(
        "--lock-rule-tau",
        type=float,
        default=0.0,
        help="SureLock-style admission threshold on adjacent-step posterior KL, evaluated online "
        "during the baseline pass. 0.0 (default) fires only where the posterior is numerically "
        "identical between steps -- the tightest a KL gate can express.",
    )
    parser.add_argument("--less-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--less-jsd-threshold", type=float, default=0.04)
    parser.add_argument("--less-top-k", type=int, default=8)
    parser.add_argument(
        "--less-history-length",
        type=int,
        default=2,
        help="Number of prior top-1 winners required by the released LLaDA LESS code.",
    )
    parser.add_argument("--polestar-attention-layer", type=int, default=32)
    parser.add_argument("--polestar-prefix-blocks", type=int, default=2)
    parser.add_argument("--polestar-refresh-threshold", type=int, default=3)
    parser.add_argument("--polestar-refresh-fraction", type=float, default=0.5)
    parser.add_argument("--polestar-max-saved-events", type=int, default=32)
    parser.add_argument(
        "--polestar-candidate-selection",
        choices=["random", "top_kl"],
        default="random",
    )
    parser.add_argument(
        "--exclude-eos-eot",
        action="store_true",
        help="Exclude EOS/EOT tokens from intervention candidates (they drift high but are inert padding).",
    )
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument(
        "--skip-drift-time-rows",
        action="store_true",
        help="Skip per-step drift AND post-commit posterior row logging (max/final drift "
        "still tracked). Rows now stream per example, so this is rarely needed.",
    )
    parser.add_argument("--empty-cache", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.gen_length % args.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    if args.steps % (args.gen_length // args.block_length) != 0:
        raise ValueError("steps must be divisible by gen_length / block_length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    started_at = time.perf_counter()
    examples = load_examples(args.dataset, args.split, args.limit, offset=args.offset, jsonl_path=args.jsonl_path)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    layers = resolve_hook_layers(model, parse_layers(args.layers))
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        cpv_window=args.cpv_window,
        mask_id=infer_mask_token_id(tokenizer),
    )
    less_config = LessConfig(
        confidence_threshold=args.less_confidence_threshold,
        jsd_threshold=args.less_jsd_threshold,
        top_k=args.less_top_k,
        history_length=args.less_history_length,
    )
    polestar_proxy_config = (
        PolestarCacheProxyConfig(
            block_length=args.block_length,
            prefix_blocks=args.polestar_prefix_blocks,
            refresh_threshold=args.polestar_refresh_threshold,
            refresh_fraction=args.polestar_refresh_fraction,
            attention_layer=args.polestar_attention_layer,
            max_saved_events=args.polestar_max_saved_events,
            seed=args.seed,
        )
        if args.intervention_mode == "polestar_cache_proxy"
        else None
    )
    decode_policy = "less" if args.intervention_mode == "less_premise" else "confidence"

    all_drift_rows: list[dict[str, Any]] = []
    all_proposal_rows: list[dict[str, Any]] = []
    # Streamed per example rather than accumulated; stale files from a reused
    # output dir would otherwise be appended to.
    time_rows_path = output_dir / "drift_time_rows.csv"
    post_rows_path = output_dir / "post_commit_posterior.csv"
    polestar_attention_path = output_dir / "polestar_attention_kl_rows.csv"
    for path in (time_rows_path, post_rows_path, polestar_attention_path):
        path.unlink(missing_ok=True)
    baseline_by_example: dict[str, dict[str, Any]] = {}
    token_lookup: dict[tuple[str, int], CommittedToken] = {}
    all_polestar_event_rows: list[dict[str, Any]] = []
    polestar_event_lookup: dict[str, PolestarProxyRefreshEvent] = {}
    selected_polestar_event_count = 0
    polestar_events_truncated = False
    for idx, example in enumerate(examples, start=1):
        print(f"[settlement-audit] baseline {idx}/{len(examples)} {example.example_id}", flush=True)
        baseline = collect_baseline_audit(
            model,
            tokenizer,
            example,
            config,
            layers,
            collect_time_rows=not args.skip_drift_time_rows,
            lock_rule_tau=args.lock_rule_tau
            if args.intervention_mode in ("rule_lock", "surelock_premise")
            else None,
            decode_policy=decode_policy,
            less_config=less_config,
            polestar_proxy_config=polestar_proxy_config,
        )
        baseline_by_example[example.example_id] = baseline
        all_drift_rows.extend(committed_rows(baseline, layers))
        all_proposal_rows.extend(baseline["proposal_rows"])
        example_polestar_events = polestar_proxy_event_rows(baseline, layers)
        all_polestar_event_rows.extend(example_polestar_events)
        example_selected_event_count = sum(
            int(bool(row["selected_for_refresh"]))
            for row in baseline["polestar_attention_rows"]
        )
        selected_polestar_event_count += example_selected_event_count
        polestar_events_truncated |= len(example_polestar_events) < example_selected_event_count
        for event in baseline["polestar_events"]:
            polestar_event_lookup[event.event_id] = event
        for token in baseline["committed"]:
            token_lookup[(token.example_id, token.position)] = token
            # base_hidden/prev_hidden are 4096-dim per layer and are only
            # needed while the trajectory is being advanced.
            token.prev_hidden = {}
        pd.DataFrame(all_drift_rows).to_csv(output_dir / "committed_token_drift.csv", index=False)
        append_rows(time_rows_path, baseline["time_rows"])
        append_rows(post_rows_path, baseline["post_rows"])
        append_rows(polestar_attention_path, baseline["polestar_attention_rows"])
        # Drop the streamed rows; baseline_by_example is retained for the whole
        # run and these dominate its footprint at long generation lengths.
        baseline["time_rows"] = []
        baseline["post_rows"] = []
        baseline["polestar_attention_rows"] = []
        if torch.cuda.is_available() and args.empty_cache:
            torch.cuda.empty_cache()

    drift_df = pd.DataFrame(all_drift_rows)

    if args.intervention_mode == "commit":
        proposal_df = pd.DataFrame(all_proposal_rows)
        proposal_df.to_csv(output_dir / "proposal_ledger.csv", index=False)
        commit_interventions: list[dict[str, Any]] = []
        if not args.skip_interventions and not drift_df.empty:
            commit_candidates = choose_commit_candidates(
                proposal_df,
                drift_df,
                args.max_interventions_per_example,
                args.commit_min_lead,
                args.commit_min_confidence,
                exclude_eos_eot=args.exclude_eos_eot,
            )
            commit_candidates.to_csv(output_dir / "commit_candidates.csv", index=False)
            example_lookup = {example.example_id: example for example in examples}
            for idx, candidate in commit_candidates.iterrows():
                example_id = str(candidate["example_id"])
                print(
                    f"[settlement-audit] commit {idx + 1}/{len(commit_candidates)} "
                    f"{example_id} group={candidate['intervention_group']}",
                    flush=True,
                )
                result = commit_intervention_decode(
                    model,
                    tokenizer,
                    example_lookup[example_id],
                    candidate,
                    baseline_by_example[example_id],
                    config,
                )
                commit_interventions.append(result)
                pd.DataFrame(commit_interventions).to_csv(output_dir / "commit_intervention_results.csv", index=False)
                gc.collect()
                if torch.cuda.is_available() and args.empty_cache:
                    torch.cuda.empty_cache()
        else:
            pd.DataFrame().to_csv(output_dir / "commit_candidates.csv", index=False)
            pd.DataFrame().to_csv(output_dir / "commit_intervention_results.csv", index=False)
        write_commit_summary(output_dir, args, drift_df, pd.DataFrame(commit_interventions), started_at)
        return

    interventions: list[dict[str, Any]] = []
    if not args.skip_interventions and not drift_df.empty:
        if args.intervention_mode == "rule_lock":
            candidates = choose_rule_lock_candidates(
                drift_df,
                args.max_interventions_per_example,
                exclude_eos_eot=args.exclude_eos_eot,
                seed=args.seed,
            )
        elif args.intervention_mode == "surelock_premise":
            candidates = choose_surelock_premise_candidates(
                drift_df,
                args.max_interventions_per_example,
                exclude_eos_eot=args.exclude_eos_eot,
                seed=args.seed,
            )
        elif args.intervention_mode == "less_premise":
            candidates = choose_less_premise_candidates(
                drift_df,
                args.max_interventions_per_example,
                exclude_eos_eot=args.exclude_eos_eot,
                seed=args.seed,
            )
        elif args.intervention_mode == "polestar_cache_proxy":
            polestar_event_df = pd.DataFrame(all_polestar_event_rows)
            polestar_event_df.to_csv(output_dir / "polestar_refresh_events.csv", index=False)
            validate_polestar_event_retention(
                args.polestar_candidate_selection,
                selected_polestar_event_count,
                len(polestar_event_df),
            )
            candidates = choose_polestar_cache_proxy_candidates(
                polestar_event_df,
                drift_df,
                args.max_interventions_per_example,
                exclude_eos_eot=args.exclude_eos_eot,
                seed=args.seed,
                selection=args.polestar_candidate_selection,
            )
        else:
            candidates = choose_drift_candidates(
                drift_df,
                args.max_interventions_per_example,
                exclude_eos_eot=args.exclude_eos_eot,
            )
        candidates.to_csv(output_dir / "freeze_candidates.csv", index=False)
        example_lookup = {example.example_id: example for example in examples}
        for idx, candidate in candidates.iterrows():
            example_id = str(candidate["example_id"])
            position = int(candidate["position"])
            token = token_lookup[(example_id, position)]
            reference_override = None
            if args.intervention_mode == "polestar_cache_proxy":
                event = polestar_event_lookup[str(candidate["event_id"])]
                reference_override = (
                    event.cache_hidden
                    if str(candidate["proxy_arm"]) == "stale"
                    else event.refresh_hidden
                )
            print(
                f"[settlement-audit] clamp {idx + 1}/{len(candidates)} "
                f"{example_id} group={candidate['intervention_group']}",
                flush=True,
            )
            result = clamp_intervention_decode(
                model,
                tokenizer,
                example_lookup[example_id],
                candidate,
                baseline_by_example[example_id],
                token,
                config,
                layers,
                decode_policy=decode_policy,
                less_config=less_config,
                reference_override=reference_override,
            )
            interventions.append(result)
            pd.DataFrame(interventions).to_csv(output_dir / "freeze_intervention_results.csv", index=False)
            gc.collect()
            if torch.cuda.is_available() and args.empty_cache:
                torch.cuda.empty_cache()
    else:
        pd.DataFrame().to_csv(output_dir / "freeze_candidates.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "freeze_intervention_results.csv", index=False)

    interventions_df = pd.DataFrame(interventions)
    if args.intervention_mode in {"surelock_premise", "less_premise"}:
        write_surelock_premise_summary(output_dir, args, drift_df, interventions_df, started_at)
    elif args.intervention_mode == "polestar_cache_proxy":
        write_polestar_cache_proxy_summary(
            output_dir,
            args,
            drift_df,
            interventions_df,
            pd.DataFrame(all_polestar_event_rows),
            selected_polestar_event_count,
            polestar_events_truncated,
            started_at,
        )
    else:
        write_summary(output_dir, args, drift_df, interventions_df, started_at)


if __name__ == "__main__":
    main()
