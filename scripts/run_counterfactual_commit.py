#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from regret_remasking import FEATURE_NAMES
from regret_remasking.data import Example, build_prompt, load_examples, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    add_gumbel_noise,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.regret_model import RegretScorer


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


MATCH_FEATURES = ["block_t_frac", "confidence", "kl", "runlength", "relative_position", "margin"]
SELECTOR_NAMES = ("learned_trace", "learned_high_confidence", "low_confidence", "high_entropy", "low_margin")
STRICT_SAME_TRAJECTORY_CONTROL = "matched_same_trajectory_high_conf_low_risk"


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def _row_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["example_id"].astype(str)
        + ":"
        + df["global_step"].astype(str)
        + ":"
        + df["position"].astype(str)
    )


def _annotate_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candidate_row_id"] = _row_id(out)
    return out


def choose_oracle_candidates(metadata: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    df = metadata.copy()
    high_conf = df["confidence"].quantile(0.90)
    low_kl = df["kl"].quantile(0.50)
    filters = [
        (
            (df["trace_regret"] == 1)
            & (df["block_t_frac"] <= 0.5)
            & (df["confidence"] >= high_conf)
            & (df["kl"] <= low_kl),
            "stable_high_conf_early_regret",
        ),
        (
            (df["trace_regret"] == 1)
            & (df["block_t_frac"] <= 0.5)
            & (df["confidence"] >= high_conf),
            "high_conf_early_regret",
        ),
        (
            (df["trace_regret"] == 1)
            & (df["block_t_frac"] <= 0.5),
            "early_regret",
        ),
        (
            df["trace_regret"] == 1,
            "any_regret",
        ),
    ]
    candidates = pd.DataFrame()
    source = "none"
    for mask, name in filters:
        candidates = df[mask].copy()
        if not candidates.empty:
            source = name
            break
    candidates = candidates.sort_values(["confidence", "runlength"], ascending=[False, False])
    candidates = candidates.drop_duplicates("example_id", keep="first")
    candidates["candidate_source"] = source
    candidates["candidate_group"] = "oracle_regret"
    return _annotate_candidate_frame(candidates.head(max_candidates).reset_index(drop=True))


def choose_selector_candidates(
    metadata: pd.DataFrame,
    selector: str,
    max_candidates: int,
    regret_model_path: Path | None = None,
    high_confidence_threshold: float | None = None,
) -> pd.DataFrame:
    if selector not in SELECTOR_NAMES:
        raise ValueError(f"unknown candidate selector: {selector}")
    df = metadata.copy()
    required = {
        "low_confidence": ["confidence"],
        "high_entropy": ["entropy"],
        "low_margin": ["margin"],
    }
    if selector in {"learned_trace", "learned_high_confidence"}:
        if regret_model_path is None:
            raise ValueError(f"{selector} selector requires --regret-model-path")
        scorer = RegretScorer(regret_model_path)
        required[selector] = scorer.features
    missing = [name for name in required[selector] if name not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing selector features for {selector}: {missing}")
    valid = df.dropna(subset=required[selector]).copy()
    valid = valid[valid["block_t_frac"] <= 0.5].copy()
    if selector == "learned_high_confidence":
        if high_confidence_threshold is None:
            raise ValueError("learned_high_confidence requires --high-confidence-threshold from training traces")
        valid = valid[valid["confidence"] >= high_confidence_threshold].copy()
    if valid.empty:
        raise ValueError(f"no valid early candidates for selector {selector}")

    if selector in {"learned_trace", "learned_high_confidence"}:
        valid["candidate_score"] = scorer.score_matrix(valid[scorer.features].to_numpy(dtype=np.float32))
        valid["predicted_regret"] = valid["candidate_score"]
    elif selector == "low_confidence":
        valid["candidate_score"] = -valid["confidence"].astype(float)
    elif selector == "high_entropy":
        valid["candidate_score"] = valid["entropy"].astype(float)
    else:
        valid["candidate_score"] = -valid["margin"].astype(float)

    # Every selector chooses its most risky early token within each trajectory.
    # The resulting groups therefore contain the same examples when max_candidates
    # covers the evaluation slice, isolating the selector rather than the dataset.
    valid = valid.sort_values(
        ["candidate_score", "global_step", "position"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    valid = valid.drop_duplicates("example_id", keep="first")
    valid["candidate_source"] = selector
    valid["candidate_group"] = selector
    return _annotate_candidate_frame(valid.head(max_candidates).reset_index(drop=True))


def _confidence_matched_low_risk_control(
    target: pd.Series,
    scored_metadata: pd.DataFrame,
    used: set[str],
    confidence_caliper: float,
) -> pd.Series | None:
    """Find a label-free, lower-risk control in the same baseline trajectory."""
    target_score = float(target["candidate_score"])
    base = scored_metadata[
        (scored_metadata["example_id"].astype(str) == str(target["example_id"]))
        & (scored_metadata["block_t_frac"] <= 0.5)
        & (scored_metadata["candidate_score"] < target_score)
        & ~scored_metadata["candidate_row_id"].astype(str).isin(used)
    ].copy()
    if base.empty:
        return None

    base["match_confidence_abs_diff"] = (base["confidence"].astype(float) - float(target["confidence"])).abs()
    base = base[base["match_confidence_abs_diff"] <= confidence_caliper]
    if base.empty:
        return None

    # Prefer identical time in the same decode, then identical within-block time,
    # then any early step. Every fallback remains within the same trajectory.
    scopes = [
        (base[base["global_step"] == int(target["global_step"])], "same_step"),
        (base[np.isclose(base["block_t_frac"], float(target["block_t_frac"]))], "same_block_time"),
        (base, "same_trajectory"),
    ]
    for pool, scope in scopes:
        if pool.empty:
            continue
        pool = pool.copy()
        pool["match_relative_position_abs_diff"] = (
            pool["relative_position"].astype(float) - float(target["relative_position"])
        ).abs()
        # Within the confidence caliper, choose the lowest predicted-risk token.
        match = pool.sort_values(
            ["candidate_score", "match_confidence_abs_diff", "match_relative_position_abs_diff"],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0].copy()
        match["match_scope"] = scope
        match["match_global_step_abs_diff"] = abs(int(match["global_step"]) - int(target["global_step"]))
        return match
    return None


def _strict_same_trajectory_high_confidence_control(
    target: pd.Series,
    scored_metadata: pd.DataFrame,
    used: set[str],
    confidence_caliper: float,
    relative_position_caliper: int,
) -> pd.Series | None:
    """Choose a lower-risk control from the identical trace state neighborhood.

    This is deliberately stricter than the older same-trajectory matcher: the
    control must be evaluated at the same denoising step, have nearly identical
    confidence, and be close in output position. Trace-regret labels are not
    used for selection.
    """
    target_score = float(target["candidate_score"])
    base = scored_metadata[
        (scored_metadata["example_id"].astype(str) == str(target["example_id"]))
        & (scored_metadata["global_step"] == int(target["global_step"]))
        & (scored_metadata["candidate_score"] < target_score)
        & ~scored_metadata["candidate_row_id"].astype(str).isin(used)
    ].copy()
    if base.empty:
        return None
    base["match_confidence_abs_diff"] = (base["confidence"].astype(float) - float(target["confidence"])).abs()
    base["match_relative_position_abs_diff"] = (
        base["relative_position"].astype(float) - float(target["relative_position"])
    ).abs()
    base = base[
        (base["match_confidence_abs_diff"] <= confidence_caliper)
        & (base["match_relative_position_abs_diff"] <= relative_position_caliper)
    ]
    if base.empty:
        return None
    match = base.sort_values(
        ["candidate_score", "match_confidence_abs_diff", "match_relative_position_abs_diff"],
        ascending=[True, True, True],
        kind="mergesort",
    ).iloc[0].copy()
    match["match_scope"] = "same_trajectory_same_step_strict"
    match["match_global_step_abs_diff"] = 0
    return match


def _global_confidence_matched_low_risk_control(
    target: pd.Series,
    scored_metadata: pd.DataFrame,
    used: set[str],
    used_control_examples: set[str],
    target_examples: set[str],
    confidence_caliper: float,
    relative_position_caliper: int,
) -> pd.Series | None:
    """Find an independent, label-free control matched on confidence and decode time."""
    target_score = float(target["candidate_score"])
    base = scored_metadata[
        (scored_metadata["example_id"].astype(str) != str(target["example_id"]))
        & ~scored_metadata["example_id"].astype(str).isin(target_examples)
        & ~scored_metadata["example_id"].astype(str).isin(used_control_examples)
        & (scored_metadata["global_step"] == int(target["global_step"]))
        & (scored_metadata["candidate_score"] < target_score)
        & ~scored_metadata["candidate_row_id"].astype(str).isin(used)
    ].copy()
    if base.empty:
        return None
    base["match_confidence_abs_diff"] = (base["confidence"].astype(float) - float(target["confidence"])).abs()
    base["match_relative_position_abs_diff"] = (
        base["relative_position"].astype(float) - float(target["relative_position"])
    ).abs()
    base = base[
        (base["match_confidence_abs_diff"] <= confidence_caliper)
        & (base["match_relative_position_abs_diff"] <= relative_position_caliper)
    ]
    if base.empty:
        return None
    match = base.sort_values(
        ["candidate_score", "match_confidence_abs_diff", "match_relative_position_abs_diff"],
        ascending=[True, True, True],
        kind="mergesort",
    ).iloc[0].copy()
    match["match_scope"] = "global_same_step"
    match["match_global_step_abs_diff"] = 0
    return match


def _scaled_match_space(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = [name for name in MATCH_FEATURES if name in df.columns]
    values = df[features].astype(float)
    center = values.mean()
    scale = values.std().replace(0.0, 1.0).fillna(1.0)
    return (values - center) / scale, center, scale


def _nearest_control(
    target: pd.Series,
    pool: pd.DataFrame,
    center: pd.Series,
    scale: pd.Series,
    rng: np.random.Generator,
) -> pd.Series | None:
    if pool.empty:
        return None
    features = [name for name in MATCH_FEATURES if name in pool.columns]
    same_example = pool[pool["example_id"].astype(str) == str(target["example_id"])]
    search = same_example if not same_example.empty else pool
    target_values = ((target[features].astype(float) - center[features]) / scale[features]).to_numpy(dtype=np.float32)
    pool_values = ((search[features].astype(float) - center[features]) / scale[features]).to_numpy(dtype=np.float32)
    distances = np.linalg.norm(pool_values - target_values[None, :], axis=1)
    best_distance = distances.min()
    best_indices = np.flatnonzero(np.isclose(distances, best_distance))
    return search.iloc[int(rng.choice(best_indices))]


def add_matched_controls(
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    controls: list[str],
    seed: int,
    controls_for_selectors: set[str] | None = None,
    regret_model_path: Path | None = None,
    high_confidence_threshold: float | None = None,
    confidence_caliper: float = 0.05,
    relative_position_caliper: int = 4,
) -> pd.DataFrame:
    if not controls:
        return targets
    df = _annotate_candidate_frame(metadata)
    high_conf_controls = {
        "matched_high_conf_low_risk",
        "matched_global_high_conf_low_risk",
        STRICT_SAME_TRAJECTORY_CONTROL,
    }
    if high_conf_controls.intersection(controls):
        if regret_model_path is None or high_confidence_threshold is None:
            raise ValueError("matched_high_conf_low_risk requires a regret model and training-derived confidence threshold")
        scorer = RegretScorer(regret_model_path)
        missing = [name for name in scorer.features if name not in df.columns]
        if missing:
            raise ValueError(f"metadata is missing matched-control features: {missing}")
        valid = ~df[scorer.features].isna().any(axis=1)
        df["candidate_score"] = np.nan
        df.loc[valid, "candidate_score"] = scorer.score_matrix(
            df.loc[valid, scorer.features].to_numpy(dtype=np.float32)
        )
        df = df[(df["confidence"] >= high_confidence_threshold) & np.isfinite(df["candidate_score"])].copy()
    _, center, scale = _scaled_match_space(df)
    rng = np.random.default_rng(seed)
    used = set(targets["candidate_row_id"].astype(str))
    target_examples = set(targets["example_id"].astype(str))
    used_global_control_examples: set[str] = set()
    rows: list[pd.Series] = []
    high_conf_threshold = float(df["confidence"].quantile(0.75))
    for _, target in targets.iterrows():
        target = target.copy()
        target["pair_id"] = target["candidate_row_id"]
        rows.append(target)
        if controls_for_selectors is not None and str(target.get("candidate_group", "")) not in controls_for_selectors:
            continue
        for control_name in controls:
            if control_name == "matched_non_regret":
                pool = df[(df["trace_regret"] == 0) & (df["block_t_frac"] <= 0.5)].copy()
            elif control_name == "matched_random_high_conf":
                pool = df[(df["confidence"] >= high_conf_threshold) & (df["block_t_frac"] <= 0.5)].copy()
            elif control_name == "matched_oracle_regret":
                pool = df[(df["trace_regret"] == 1) & (df["block_t_frac"] <= 0.5)].copy()
            elif control_name == "matched_high_conf_low_risk":
                match = _confidence_matched_low_risk_control(target, df, used, confidence_caliper)
                if match is None:
                    continue
                match = match.copy()
                match["candidate_group"] = control_name
                match["candidate_source"] = control_name
                match["matched_to_row_id"] = target["candidate_row_id"]
                match["matched_to_example_id"] = target["example_id"]
                match["pair_id"] = target["candidate_row_id"]
                used.add(str(match["candidate_row_id"]))
                rows.append(match)
                continue
            elif control_name == STRICT_SAME_TRAJECTORY_CONTROL:
                match = _strict_same_trajectory_high_confidence_control(
                    target,
                    df,
                    used,
                    confidence_caliper,
                    relative_position_caliper,
                )
                if match is None:
                    continue
                match = match.copy()
                match["candidate_group"] = control_name
                match["candidate_source"] = control_name
                match["matched_to_row_id"] = target["candidate_row_id"]
                match["matched_to_example_id"] = target["example_id"]
                match["pair_id"] = target["candidate_row_id"]
                used.add(str(match["candidate_row_id"]))
                rows.append(match)
                continue
            elif control_name == "matched_global_high_conf_low_risk":
                match = _global_confidence_matched_low_risk_control(
                    target,
                    df,
                    used,
                    used_global_control_examples,
                    target_examples,
                    confidence_caliper,
                    relative_position_caliper,
                )
                if match is None:
                    continue
                match = match.copy()
                match["candidate_group"] = control_name
                match["candidate_source"] = control_name
                match["matched_to_row_id"] = target["candidate_row_id"]
                match["matched_to_example_id"] = target["example_id"]
                match["pair_id"] = target["candidate_row_id"]
                used.add(str(match["candidate_row_id"]))
                used_global_control_examples.add(str(match["example_id"]))
                rows.append(match)
                continue
            else:
                raise ValueError(f"unknown control: {control_name}")
            pool = pool[~pool["candidate_row_id"].astype(str).isin(used)]
            match = _nearest_control(target, pool, center, scale, rng)
            if match is None:
                continue
            match = match.copy()
            match["candidate_group"] = control_name
            match["candidate_source"] = control_name
            match["matched_to_row_id"] = target["candidate_row_id"]
            match["matched_to_example_id"] = target["example_id"]
            match["pair_id"] = target["candidate_row_id"]
            used.add(str(match["candidate_row_id"]))
            rows.append(match)
    return pd.DataFrame(rows).reset_index(drop=True)


@torch.no_grad()
def oracle_avoid_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Example,
    metadata: pd.DataFrame,
    config: DecodeConfig,
) -> dict[str, Any]:
    device = model_device(model)
    prompt = build_prompt(example.question, example.dataset)
    input_ids, attention_mask = prepare_prompts(tokenizer, [prompt], device)
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
    oracle_rows = metadata[metadata["example_id"].astype(str) == example.example_id]
    regret_lookup = {
        (int(row.global_step), int(row.position)): int(row.trace_regret)
        for row in oracle_rows.itertuples(index=False)
    }
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    nfe = 0
    avoided = 0
    fallback_steps = 0

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
            adjusted_score = score.clone()
            for pos in torch.nonzero(allowed[0], as_tuple=False).flatten().tolist():
                if regret_lookup.get((global_step, int(pos)), 0) == 1:
                    adjusted_score[0, pos] = -torch.inf

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            k = int(num_transfer_tokens[0, step_in_block].item())
            if k > 0:
                finite_count = int(torch.isfinite(adjusted_score[0]).sum().detach().cpu())
                source_score = adjusted_score if finite_count >= k else score
                if finite_count < k:
                    fallback_steps += 1
                _, select_index = torch.topk(source_score[0], k=k)
                transfer_index[0, select_index] = True
                avoided += int((score[0, select_index] != adjusted_score[0, select_index]).sum().detach().cpu())
            x[transfer_index] = x0[transfer_index]
            del logits, logits_with_noise, probs, x0_p, score, adjusted_score

    generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return {
        "example_id": example.example_id,
        "oracle_generation": generation,
        "oracle_correct": score_generation(generation, example),
        "oracle_nfe": nfe,
        "oracle_avoided_regret_positions": avoided,
        "oracle_fallback_steps": fallback_steps,
    }


def baseline_final_token_sequence(
    metadata: pd.DataFrame,
    example_id: str,
    gen_length: int,
) -> np.ndarray:
    """Recover one baseline output-token sequence from dense trace labels."""
    rows = metadata[metadata["example_id"].astype(str) == str(example_id)]
    if rows.empty:
        raise ValueError(f"no trace rows found for example {example_id}")
    if "relative_position" not in rows or "final_token" not in rows:
        raise ValueError("metadata must include relative_position and final_token")
    by_position = rows.groupby("relative_position", sort=True)["final_token"]
    inconsistent = by_position.nunique()
    if bool((inconsistent > 1).any()):
        bad_positions = inconsistent[inconsistent > 1].index.tolist()
        raise ValueError(f"inconsistent final-token labels for {example_id} at positions {bad_positions[:5]}")
    positions = inconsistent.index.to_numpy(dtype=int)
    expected = np.arange(gen_length, dtype=int)
    if not np.array_equal(positions, expected):
        missing = sorted(set(expected.tolist()) - set(positions.tolist()))
        raise ValueError(f"incomplete baseline token sequence for {example_id}; missing positions {missing[:5]}")
    return by_position.first().to_numpy(dtype=np.int64)


def counterfactual_token_effects(
    forced_final_tokens: np.ndarray,
    baseline_final_tokens: np.ndarray,
    target_relative_position: int,
) -> dict[str, Any]:
    """Separate the forced position's expected change from downstream effects."""
    forced = np.asarray(forced_final_tokens, dtype=np.int64)
    baseline = np.asarray(baseline_final_tokens, dtype=np.int64)
    if forced.shape != baseline.shape or forced.ndim != 1:
        raise ValueError("forced and baseline output-token sequences must be one-dimensional and same-sized")
    if target_relative_position < 0 or target_relative_position >= len(forced):
        raise ValueError(f"target relative position {target_relative_position} is outside output length {len(forced)}")
    changed = forced != baseline
    non_target = changed.copy()
    non_target[target_relative_position] = False
    suffix = changed.copy()
    suffix[: target_relative_position + 1] = False
    non_target_count = int(non_target.sum())
    suffix_count = int(suffix.sum())
    return {
        "target_token_changed": bool(changed[target_relative_position]),
        # Retained for compatibility with earlier result files.
        "final_token_changed": bool(changed[target_relative_position]),
        "non_target_token_changed": bool(non_target_count > 0),
        "non_target_token_change_count": non_target_count,
        "non_target_token_change_fraction": float(non_target_count / max(len(forced) - 1, 1)),
        "suffix_token_changed": bool(suffix_count > 0),
        "suffix_token_change_count": suffix_count,
    }


@torch.no_grad()
def force_commit_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Example,
    candidate: pd.Series,
    config: DecodeConfig,
    baseline_final_tokens: np.ndarray,
) -> dict[str, Any]:
    device = model_device(model)
    prompt = build_prompt(example.question, example.dataset)
    input_ids, attention_mask = prepare_prompts(tokenizer, [prompt], device)
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
    target_step = int(candidate["global_step"])
    target_pos = int(candidate["position"])
    forced_token = int(candidate["top_token"])
    forced_applied = False
    forced_allowed = False
    nfe = 0

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

            if global_step == target_step:
                forced_allowed = bool(allowed[0, target_pos].detach().cpu()) if target_pos < allowed.shape[1] else False
                if forced_allowed:
                    x0[0, target_pos] = forced_token
                    score[0, target_pos] = torch.inf
                    forced_applied = True

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for batch_idx in range(1):
                k = int(num_transfer_tokens[batch_idx, step_in_block].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(score[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True

            x[transfer_index] = x0[transfer_index]
            del logits, logits_with_noise, probs, x0_p, score

    final_tokens = x[:, prompt_len:].detach().cpu()
    rel_pos = int(candidate["relative_position"])
    final_token_ids = final_tokens[0].numpy().astype(np.int64, copy=False)
    if int(baseline_final_tokens[rel_pos]) != int(candidate["final_token"]):
        raise ValueError(
            "candidate final token does not agree with reconstructed baseline sequence "
            f"for {example.example_id} at output position {rel_pos}"
        )
    effects = counterfactual_token_effects(final_token_ids, baseline_final_tokens, rel_pos)
    final_token = int(final_token_ids[rel_pos])
    generation = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return {
        "example_id": example.example_id,
        "forced_applied": forced_applied,
        "forced_allowed": forced_allowed,
        "target_step": target_step,
        "target_position": target_pos,
        "relative_position": rel_pos,
        "forced_token": forced_token,
        "baseline_final_token": int(candidate["final_token"]),
        "forced_final_token": final_token,
        "forced_token_survived": final_token == forced_token,
        "forced_generation": generation,
        "forced_correct": score_generation(generation, example),
        "nfe": nfe,
        **effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactually force early-regret token commitments.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--candidate-mode", choices=["oracle", "prospective"], default="oracle")
    parser.add_argument("--regret-model-path", default=None)
    parser.add_argument(
        "--candidate-selectors",
        default="learned_trace",
        help="Comma-separated prospective selectors: learned_trace, learned_high_confidence, low_confidence, high_entropy, low_margin.",
    )
    parser.add_argument(
        "--controls",
        default="matched_non_regret,matched_random_high_conf",
        help=(
            "Comma-separated controls: matched_non_regret, matched_random_high_conf, "
            "matched_oracle_regret, matched_high_conf_low_risk, "
            "matched_same_trajectory_high_conf_low_risk, "
            "matched_global_high_conf_low_risk."
        ),
    )
    parser.add_argument(
        "--controls-for-selectors",
        default="learned_trace",
        help="Comma-separated target selectors that receive matched controls; empty means every selector.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=None,
        help="Fixed confidence threshold computed on training traces for high-confidence selection.",
    )
    parser.add_argument(
        "--confidence-caliper",
        type=float,
        default=0.05,
        help="Maximum absolute confidence gap for matched_high_conf_low_risk controls.",
    )
    parser.add_argument(
        "--relative-position-caliper",
        type=int,
        default=4,
        help="Maximum output-position gap for matched_global_high_conf_low_risk controls.",
    )
    parser.add_argument("--run-oracle-upper-bound", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the prospective candidate and match ledger without loading the model or running interventions.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metadata = pd.read_csv(args.metadata_csv)
    if args.candidate_mode == "prospective":
        selectors = [item.strip() for item in args.candidate_selectors.split(",") if item.strip()]
        if not selectors:
            raise ValueError("--candidate-selectors must contain at least one selector")
        learned_selectors = {"learned_trace", "learned_high_confidence"}
        if learned_selectors.intersection(selectors) and not args.regret_model_path:
            raise ValueError("learned selectors require --regret-model-path")
        candidates = pd.concat(
            [
                choose_selector_candidates(
                    metadata,
                    selector,
                    args.max_candidates,
                    Path(args.regret_model_path) if args.regret_model_path else None,
                    args.high_confidence_threshold,
                )
                for selector in selectors
            ],
            ignore_index=True,
        )
    else:
        candidates = choose_oracle_candidates(metadata, args.max_candidates)
    controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    controls_for_selectors = {item.strip() for item in args.controls_for_selectors.split(",") if item.strip()}
    candidates = add_matched_controls(
        metadata,
        candidates,
        controls,
        args.seed,
        controls_for_selectors if args.candidate_mode == "prospective" and controls_for_selectors else None,
        Path(args.regret_model_path) if args.regret_model_path else None,
        args.high_confidence_threshold,
        args.confidence_caliper,
        args.relative_position_caliper,
    )
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    if args.dry_run:
        selection_summary = {
            "candidate_mode": args.candidate_mode,
            "candidates": int(len(candidates)),
            "groups": candidates["candidate_group"].value_counts().sort_index().to_dict(),
            "match_scopes": candidates["match_scope"].dropna().value_counts().sort_index().to_dict()
            if "match_scope" in candidates
            else {},
            "mean_match_confidence_abs_diff": float(candidates["match_confidence_abs_diff"].dropna().mean())
            if "match_confidence_abs_diff" in candidates and candidates["match_confidence_abs_diff"].notna().any()
            else None,
            "mean_match_relative_position_abs_diff": float(candidates["match_relative_position_abs_diff"].dropna().mean())
            if "match_relative_position_abs_diff" in candidates and candidates["match_relative_position_abs_diff"].notna().any()
            else None,
        }
        (output_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
        print(json.dumps(selection_summary, indent=2), flush=True)
        return
    baseline = load_jsonl(Path(args.baseline_jsonl))
    baseline_final_tokens = {
        str(example_id): baseline_final_token_sequence(metadata, str(example_id), args.gen_length)
        for example_id in candidates["example_id"].astype(str).unique()
    }
    examples = {
        example.example_id: example
        for example in load_examples(args.dataset, args.split, args.limit, offset=args.offset)
    }
    if candidates.empty:
        summary = {
            "candidates": 0,
            "forced_applied_rate": 0.0,
            "forced_token_survival_rate": 0.0,
            "final_token_change_rate": 0.0,
            "non_target_token_change_rate": 0.0,
            "mean_non_target_token_change_count": 0.0,
            "suffix_token_change_rate": 0.0,
            "baseline_accuracy": 0.0,
            "forced_accuracy": 0.0,
            "answer_correct_change_rate": 0.0,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "summary.md").write_text(
            "# Counterfactual Commit Summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
    )

    rows = []
    for idx, candidate in candidates.iterrows():
        example = examples[str(candidate["example_id"])]
        print(
            f"[counterfactual] {idx + 1}/{len(candidates)} "
            f"{example.example_id} group={candidate.get('candidate_group', 'target')}",
            flush=True,
        )
        result = force_commit_decode(
            model,
            tokenizer,
            example,
            candidate,
            config,
            baseline_final_tokens[example.example_id],
        )
        base = baseline.get(example.example_id, {})
        for key in [
            "candidate_group",
            "candidate_source",
            "candidate_row_id",
            "pair_id",
            "matched_to_row_id",
            "matched_to_example_id",
            "match_scope",
            "match_confidence_abs_diff",
            "match_relative_position_abs_diff",
            "match_global_step_abs_diff",
            "trace_regret",
            "predicted_regret",
            "candidate_score",
            "confidence",
            "entropy",
            "margin",
            "kl",
            "jsd",
            "top1_flip",
            "runlength",
            "block_t_frac",
            "t_frac",
            "local_mask_ratio",
            "context_volatility",
        ]:
            if key in candidate:
                value = candidate[key]
                if pd.notna(value):
                    result[key] = value.item() if hasattr(value, "item") else value
        result["baseline_correct"] = bool(base.get("correct", False))
        result["baseline_generation"] = base.get("generation", "")
        result["answer_correct_changed"] = result["forced_correct"] != result["baseline_correct"]
        rows.append(result)
        pd.DataFrame(rows).to_csv(output_dir / "counterfactual_results.csv", index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    group_summary = (
        df.groupby("candidate_group")
        .agg(
            candidates=("example_id", "size"),
            forced_applied_rate=("forced_applied", "mean"),
            forced_token_survival_rate=("forced_token_survived", "mean"),
            final_token_change_rate=("final_token_changed", "mean"),
            non_target_token_change_rate=("non_target_token_changed", "mean"),
            mean_non_target_token_change_count=("non_target_token_change_count", "mean"),
            suffix_token_change_rate=("suffix_token_changed", "mean"),
            baseline_accuracy=("baseline_correct", "mean"),
            forced_accuracy=("forced_correct", "mean"),
            answer_correct_change_rate=("answer_correct_changed", "mean"),
            observed_trace_regret_rate=("trace_regret", "mean"),
            mean_candidate_score=("candidate_score", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_margin=("margin", "mean"),
            mean_block_t_frac=("block_t_frac", "mean"),
            mean_relative_position=("relative_position", "mean"),
        )
        .reset_index()
        if "candidate_group" in df.columns
        else pd.DataFrame()
    )
    if not group_summary.empty:
        group_summary.to_csv(output_dir / "group_summary.csv", index=False)
    summary: dict[str, Any] = {
        "candidate_mode": args.candidate_mode,
        "candidates": int(len(df)),
        "forced_applied_rate": float(df["forced_applied"].mean()) if len(df) else 0.0,
        "forced_token_survival_rate": float(df["forced_token_survived"].mean()) if len(df) else 0.0,
        "final_token_change_rate": float(df["final_token_changed"].mean()) if len(df) else 0.0,
        "non_target_token_change_rate": float(df["non_target_token_changed"].mean()) if len(df) else 0.0,
        "mean_non_target_token_change_count": float(df["non_target_token_change_count"].mean()) if len(df) else 0.0,
        "suffix_token_change_rate": float(df["suffix_token_changed"].mean()) if len(df) else 0.0,
        "baseline_accuracy": float(df["baseline_correct"].mean()) if len(df) else 0.0,
        "forced_accuracy": float(df["forced_correct"].mean()) if len(df) else 0.0,
        "answer_correct_change_rate": float(df["answer_correct_changed"].mean()) if len(df) else 0.0,
    }
    if args.run_oracle_upper_bound:
        oracle_rows = []
        for idx, example in enumerate(examples.values(), start=1):
            print(f"[oracle-upper-bound] {idx}/{len(examples)} {example.example_id}", flush=True)
            result = oracle_avoid_decode(model, tokenizer, example, metadata, config)
            base = baseline.get(example.example_id, {})
            result["baseline_correct"] = bool(base.get("correct", False))
            result["oracle_correct_changed"] = result["oracle_correct"] != result["baseline_correct"]
            oracle_rows.append(result)
            pd.DataFrame(oracle_rows).to_csv(output_dir / "oracle_upper_bound_results.csv", index=False)
        oracle_df = pd.DataFrame(oracle_rows)
        summary["oracle_upper_bound"] = {
            "examples": int(len(oracle_df)),
            "baseline_accuracy": float(oracle_df["baseline_correct"].mean()) if len(oracle_df) else 0.0,
            "oracle_accuracy": float(oracle_df["oracle_correct"].mean()) if len(oracle_df) else 0.0,
            "oracle_correct_change_rate": float(oracle_df["oracle_correct_changed"].mean()) if len(oracle_df) else 0.0,
            "mean_avoided_regret_positions": float(oracle_df["oracle_avoided_regret_positions"].mean())
            if len(oracle_df)
            else 0.0,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Counterfactual Commit Summary", "", "```json", json.dumps(summary, indent=2), "```"]
    if not group_summary.empty:
        lines.extend(["", "## Group Summary", table_text(group_summary)])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
