#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
from regret_remasking.data import build_prompt, load_examples, normalize_answer, score_generation
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
from run_counterfactual_commit import counterfactual_token_effects
from run_logit_lens_token_fate import parse_layers, register_layer_hooks, resolve_hook_layers


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
    # First forward after selection, when the position is no longer masked.
    base_step: int | None = None
    base_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    max_drift_by_layer: dict[int, float] = field(default_factory=dict)
    final_drift_by_layer: dict[int, float] = field(default_factory=dict)
    future_steps_observed: int = 0
    final_token: int | None = None


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def tensor_value(feature_map: dict[str, torch.Tensor], name: str, batch: int, pos: int) -> float:
    return float(feature_map[name][batch, pos].detach().cpu())


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


def update_committed_drift(
    committed: dict[int, CommittedToken],
    captured: dict[int, torch.Tensor],
    layers: list[int],
    global_step: int,
    time_rows: list[dict[str, Any]],
) -> None:
    for token in committed.values():
        if global_step <= token.selected_step:
            continue
        if token.base_step is None:
            token.base_step = global_step
            for layer in layers:
                token.base_hidden[layer] = captured[layer][0, token.position].detach().float().cpu()
                token.max_drift_by_layer[layer] = 0.0
                token.final_drift_by_layer[layer] = 0.0
            continue
        token.future_steps_observed += 1
        for layer in layers:
            drift = relative_l2(captured[layer][0, token.position], token.base_hidden[layer])
            token.final_drift_by_layer[layer] = drift
            token.max_drift_by_layer[layer] = max(token.max_drift_by_layer.get(layer, 0.0), drift)
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
                }
            )


@torch.no_grad()
def collect_baseline_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    config: DecodeConfig,
    layers: list[int],
) -> dict[str, Any]:
    device = model_device(model)
    captured, handles = register_layer_hooks(model, layers)
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
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                nfe += 1
                update_committed_drift(committed, captured, layers, global_step, time_rows)
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

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                k = int(num_transfer_tokens[0, step_in_block].item())
                if k > 0:
                    _, select_index = torch.topk(score[0], k=k)
                    transfer_index[0, select_index] = True

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
                    )

                x[transfer_index] = x0[transfer_index]
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del outputs, logits, log_probs, score

        final_tokens = x[:, prompt_len:].detach().cpu()
        generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        for token in committed.values():
            token.final_token = int(final_tokens[0, token.relative_position])
        return {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "prompt_len": prompt_len,
            "final_tokens": final_tokens[0].numpy().astype(np.int64, copy=False),
            "generation": generation,
            "correct": score_generation(generation, example),
            "nfe": nfe,
            "committed": list(committed.values()),
            "time_rows": time_rows,
        }
    finally:
        for handle in handles:
            handle.remove()


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
            "future_steps_observed": token.future_steps_observed,
            "baseline_correct": bool(baseline["correct"]),
        }
        for layer in layers:
            row[f"max_drift_layer_{layer}"] = token.max_drift_by_layer.get(layer, np.nan)
            row[f"final_drift_layer_{layer}"] = token.final_drift_by_layer.get(layer, np.nan)
        row["max_drift_mean"] = float(np.nanmean(max_values)) if np.isfinite(max_values).any() else np.nan
        row["final_drift_mean"] = float(np.nanmean(final_values)) if np.isfinite(final_values).any() else np.nan
        rows.append(row)
    return rows


def choose_drift_candidates(
    drift_df: pd.DataFrame,
    per_example: int,
) -> pd.DataFrame:
    eligible = drift_df[
        drift_df["semantic_safe"].eq(1)
        & drift_df["base_step"].notna()
        & drift_df["max_drift_mean"].notna()
        & (drift_df["future_steps_observed"] > 0)
    ].copy()
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
) -> dict[str, Any]:
    device = model_device(model)
    freeze_step = int(candidate["base_step"])
    state, handles = register_row_clamp_hooks(model, layers, token.position, token.base_hidden, freeze_step)
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

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length
            block_mask_index = x[:, block_start:block_end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for step_in_block in range(steps_per_block):
                global_step = block_idx * steps_per_block + step_in_block
                state["global_step"] = global_step
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
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                k = int(num_transfer_tokens[0, step_in_block].item())
                if k > 0:
                    _, select_index = torch.topk(score[0], k=k)
                    transfer_index[0, select_index] = True
                if global_step == token.selected_step:
                    selected_token_matches = bool(
                        transfer_index[0, token.position].detach().cpu()
                        and int(x0[0, token.position].detach().cpu()) == token.selected_token
                    )
                x[transfer_index] = x0[transfer_index]
                prev_log_probs = log_probs.detach()
                prev_top1 = x0.detach()
                del logits, log_probs, score

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
        baseline_normalized_answer = normalize_answer(str(baseline["generation"]))
        clamped_normalized_answer = normalize_answer(generation)
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
            "base_step": freeze_step,
            "selected_token": token.selected_token,
            "selected_confidence": token.selected_confidence,
            "selected_entropy": token.selected_entropy,
            "selected_margin": token.selected_margin,
            "selected_kl": token.selected_kl,
            "future_steps_observed": token.future_steps_observed,
            "baseline_final_token": int(baseline["final_tokens"][token.relative_position]),
            "clamped_final_token": int(final_tokens[token.relative_position]),
            "selected_token_matches_replay": selected_token_matches,
            "baseline_correct": bool(baseline["correct"]),
            "baseline_generation": baseline["generation"],
            "clamped_generation": generation,
            "baseline_normalized_answer": baseline_normalized_answer,
            "clamped_normalized_answer": clamped_normalized_answer,
            "normalized_answer_changed": baseline_normalized_answer != clamped_normalized_answer,
            "clamped_correct": score_generation(generation, example),
            "answer_correct_changed": score_generation(generation, example) != bool(baseline["correct"]),
            "nfe": nfe,
            "max_drift_mean": float(candidate["max_drift_mean"]),
            "final_drift_mean": float(candidate["final_drift_mean"]),
            "changed_relative_positions": json.dumps(changed_positions),
            "non_target_changed_relative_positions": json.dumps(non_target_positions),
            **effects,
        }
    finally:
        for handle in handles:
            handle.remove()


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
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
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
    parser.add_argument("--skip-interventions", action="store_true")
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

    all_drift_rows: list[dict[str, Any]] = []
    all_time_rows: list[dict[str, Any]] = []
    baseline_by_example: dict[str, dict[str, Any]] = {}
    token_lookup: dict[tuple[str, int], CommittedToken] = {}
    for idx, example in enumerate(examples, start=1):
        print(f"[settlement-audit] baseline {idx}/{len(examples)} {example.example_id}", flush=True)
        baseline = collect_baseline_audit(model, tokenizer, example, config, layers)
        baseline_by_example[example.example_id] = baseline
        all_drift_rows.extend(committed_rows(baseline, layers))
        all_time_rows.extend(baseline["time_rows"])
        for token in baseline["committed"]:
            token_lookup[(token.example_id, token.position)] = token
        pd.DataFrame(all_drift_rows).to_csv(output_dir / "committed_token_drift.csv", index=False)
        pd.DataFrame(all_time_rows).to_csv(output_dir / "drift_time_rows.csv", index=False)
        if torch.cuda.is_available() and args.empty_cache:
            torch.cuda.empty_cache()

    drift_df = pd.DataFrame(all_drift_rows)
    interventions: list[dict[str, Any]] = []
    if not args.skip_interventions and not drift_df.empty:
        candidates = choose_drift_candidates(drift_df, args.max_interventions_per_example)
        candidates.to_csv(output_dir / "freeze_candidates.csv", index=False)
        example_lookup = {example.example_id: example for example in examples}
        for idx, candidate in candidates.iterrows():
            example_id = str(candidate["example_id"])
            position = int(candidate["position"])
            token = token_lookup[(example_id, position)]
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
    write_summary(output_dir, args, drift_df, interventions_df, started_at)


if __name__ == "__main__":
    main()
