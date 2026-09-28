#!/usr/bin/env python3
"""Generate direct Phase-1 commitment-safety labels by intervention.

Each candidate is an unresolved token that the ordinary confidence transfer
schedule would *not* select at update t. The intervention adds that candidate
to the ordinary transfer set without displacing any baseline transfer. Hidden
representations remain live on all subsequent updates, so this labels semantic
commitment safety C rather than representation-freeze safety R.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark_phase_lock_batched_online import _answer_changed
from benchmark_phase_lock_fused_replay import _examples
from benchmark_tracelock_vs_phaselock import _install_hooks
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices
from regret_remasking.phase_lock_influence import (
    SampledAttentionInfluenceCapture,
    multilayer_max_drift,
)
from regret_remasking.phase_lock_semantic import update_token_identity_patience
from regret_remasking.tracelock_baseline import (
    TraceLockConfig,
    TraceLockController,
    pack_left_padded_controller_batch,
    soft_crop_window,
)
from run_four_arm_audit import transformer_blocks


FEATURE_COLUMNS = [
    "confidence",
    "margin",
    "entropy",
    "token_age",
    "trace_probability",
    "block_age_fraction",
    "max_hidden_drift",
    "attention_influence",
    "drift_available",
]


class CounterfactualReplayIneligible(RuntimeError):
    """A registered candidate cannot receive a distinct replay intervention."""


def ordinary_transfer_mask(
    confidence: torch.Tensor,
    allowed: torch.Tensor,
    transfer_counts: torch.Tensor,
) -> torch.Tensor:
    """Return the ordinary top-confidence transfer set for one update."""
    if confidence.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("confidence and allowed must be aligned")
    if transfer_counts.shape != (allowed.shape[0],):
        raise ValueError("transfer counts must have one entry per batch row")
    selected = torch.zeros_like(allowed)
    masked = torch.where(allowed, confidence, torch.full_like(confidence, -torch.inf))
    for batch_index in range(allowed.shape[0]):
        available = int(allowed[batch_index].sum().item())
        count = min(int(transfer_counts[batch_index].item()), available)
        if count <= 0:
            continue
        positions = torch.topk(masked[batch_index], k=count).indices
        selected[batch_index, positions] = True
    return selected


def add_counterfactual_commit(
    ordinary: torch.Tensor,
    allowed: torch.Tensor,
    target_positions: list[int | None],
) -> torch.Tensor:
    """Add one target per applicable row without removing ordinary transfers."""
    if ordinary.shape != allowed.shape or ordinary.dtype != torch.bool:
        raise ValueError("ordinary and allowed masks must align")
    if len(target_positions) != ordinary.shape[0]:
        raise ValueError("target position list must align with batch")
    selected = ordinary.clone()
    for batch_index, position in enumerate(target_positions):
        if position is None:
            continue
        if not 0 <= position < ordinary.shape[1] or not bool(allowed[batch_index, position]):
            raise CounterfactualReplayIneligible(
                "counterfactual target is not unresolved at its registered update: "
                f"batch={batch_index}, position={position}"
            )
        if bool(ordinary[batch_index, position]):
            raise CounterfactualReplayIneligible(
                "counterfactual target is already in the ordinary transfer set"
            )
        selected[batch_index, position] = True
    return selected


def _percentile_rank(values: np.ndarray, *, higher_is_safer: bool) -> np.ndarray:
    work = np.asarray(values, dtype=np.float64)
    if not higher_is_safer:
        work = -work
    order = np.argsort(work, kind="mergesort")
    rank = np.empty(len(work), dtype=np.float64)
    rank[order] = np.arange(len(work), dtype=np.float64)
    return rank / max(len(work) - 1, 1)


def select_stratified_candidates(
    rows: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Select a proxy-stratified, position-diverse set without future labels."""
    if maximum <= 0 or not rows:
        return []
    frame = pd.DataFrame(rows).reset_index(drop=True)
    oriented = {
        "confidence": True,
        "margin": True,
        "entropy": False,
        "token_age": True,
        "trace_probability": True,
        "max_hidden_drift": False,
        "attention_influence": False,
    }
    ranks = [
        _percentile_rank(frame[name].to_numpy(dtype=float), higher_is_safer=direction)
        for name, direction in oriented.items()
    ]
    frame["selection_proxy"] = np.mean(np.stack(ranks, axis=1), axis=1)
    frame = frame.sort_values(
        ["selection_proxy", "global_step", "relative_position"], kind="mergesort"
    ).reset_index(drop=True)
    desired = np.linspace(0, len(frame) - 1, num=min(maximum, len(frame))).round().astype(int)
    chosen: list[int] = []
    used_positions: set[int] = set()
    for target in desired.tolist():
        candidates = sorted(
            range(len(frame)),
            key=lambda index: (abs(index - target), index),
        )
        picked = next(
            (
                index
                for index in candidates
                if index not in chosen
                and int(frame.iloc[index]["relative_position"]) not in used_positions
            ),
            None,
        )
        if picked is None:
            picked = next((index for index in candidates if index not in chosen), None)
        if picked is None:
            break
        chosen.append(picked)
        used_positions.add(int(frame.iloc[picked]["relative_position"]))
    output = frame.iloc[chosen].copy()
    output["selection_stratum"] = np.arange(len(output), dtype=int)
    output["selection_proxy_quantile"] = (
        output["selection_proxy"].rank(method="first", pct=True).to_numpy(dtype=float)
    )
    return output.to_dict("records")


def _trace_probability(
    controller: TraceLockController,
    projection: torch.Tensor,
    captures: dict[int, torch.Tensor],
    trace_layers: list[int],
    prompt_attention: torch.Tensor,
    allowed: torch.Tensor,
    *,
    prompt_len: int,
    generation_length: int,
    window_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = torch.stack(
        [captures[layer].float().matmul(projection) for layer in trace_layers],
        dim=2,
    )
    packed, controller_attention, controller_active, prompt_lengths = (
        pack_left_padded_controller_batch(
            projected,
            prompt_attention,
            allowed,
            generation_length=generation_length,
        )
    )
    domain = torch.zeros_like(allowed)
    for batch_index in range(allowed.shape[0]):
        actual_prompt = int(prompt_lengths[batch_index].item())
        start, end = soft_crop_window(
            controller_active[batch_index],
            prompt_length=actual_prompt,
            window_width=window_width,
        )
        controller_attention[batch_index, end:] = False
        packed_window = controller_active[batch_index].clone()
        packed_window[:start] = False
        packed_window[end:] = False
        domain[batch_index, prompt_len:] = packed_window[
            actual_prompt : actual_prompt + generation_length
        ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = controller(packed, controller_attention)
    probability = torch.zeros_like(allowed, dtype=torch.float32)
    for batch_index in range(allowed.shape[0]):
        actual_prompt = int(prompt_lengths[batch_index].item())
        probability[batch_index, prompt_len:] = output["probability"][
            batch_index,
            actual_prompt : actual_prompt + generation_length,
        ].float()
    return probability, domain


@torch.inference_mode()
def collect_baseline_candidates(
    model: torch.nn.Module,
    tokenizer: Any,
    controller: TraceLockController,
    projection: torch.Tensor,
    example: Any,
    config: DecodeConfig,
    *,
    trace_layers: list[int],
    representation_layers: list[int],
    influence_layer: int,
    max_queries: int,
    window_width: int,
    maximum_candidates: int,
) -> dict[str, Any]:
    device = model_device(model)
    prompt = build_prompt(example.question, example.dataset)
    input_ids, prompt_attention = prepare_prompts(tokenizer, [prompt], device)
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + config.gen_length
    x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    model_attention = torch.cat(
        [
            prompt_attention,
            torch.ones((1, config.gen_length), dtype=prompt_attention.dtype, device=device),
        ],
        dim=-1,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    capture_layers = sorted(set(trace_layers) | set(representation_layers))
    captures, handles = _install_hooks(model, capture_layers)
    influence = SampledAttentionInfluenceCapture(
        transformer_blocks(model)[influence_layer - 1], max_queries=max_queries
    )
    influence.install()
    previous_hidden: dict[int, torch.Tensor] | None = None
    previous_tokens: torch.Tensor | None = None
    token_age = torch.zeros_like(x, dtype=torch.long)
    candidates: list[dict[str, Any]] = []
    nfe = 0
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = block_start + config.block_length
            initial_mask = x[:, block_start:block_end] == config.mask_id
            transfers = get_num_transfer_tokens(initial_mask, steps_per_block)
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                mask = x == config.mask_id
                allowed = mask.clone()
                allowed[:, :block_start] = False
                allowed[:, block_end:] = False
                unresolved = mask.clone()
                unresolved[:, :prompt_len] = False
                captures.clear()
                influence.set_query_mask(unresolved)
                logits = model(x, attention_mask=model_attention).logits
                nfe += 1
                if set(captures) != set(capture_layers):
                    raise RuntimeError("candidate collector missed a requested hidden layer")
                if influence.latest is None:
                    raise RuntimeError("candidate collector missed attention influence")
                proposals = torch.argmax(logits, dim=-1)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                top = torch.topk(log_probs, k=2, dim=-1).values.exp()
                confidence = top[:, :, 0]
                margin = top[:, :, 0] - top[:, :, 1]
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                token_age = update_token_identity_patience(
                    token_age, proposals, previous_tokens, allowed
                )
                current_hidden = {
                    layer: captures[layer].detach() for layer in representation_layers
                }
                if previous_hidden is None:
                    drift = torch.zeros_like(confidence)
                    drift_available = 0.0
                else:
                    drift, _ = multilayer_max_drift(current_hidden, previous_hidden)
                    drift_available = 1.0
                trace_probability, trace_domain = _trace_probability(
                    controller,
                    projection,
                    captures,
                    trace_layers,
                    prompt_attention,
                    allowed,
                    prompt_len=prompt_len,
                    generation_length=config.gen_length,
                    window_width=window_width,
                )
                ordinary = ordinary_transfer_mask(
                    confidence, allowed, transfers[:, step_in_block]
                )
                candidate_mask = allowed & trace_domain & ~ordinary
                for position in torch.nonzero(
                    candidate_mask[0], as_tuple=False
                ).flatten().tolist():
                    candidates.append(
                        {
                            "dataset": example.dataset,
                            "example_id": example.example_id,
                            "global_step": global_step,
                            "block": block_index,
                            "step_in_block": step_in_block,
                            "block_age": step_in_block + 1,
                            "block_age_fraction": (step_in_block + 1) / steps_per_block,
                            "position": int(position),
                            "relative_position": int(position - prompt_len),
                            "top_token": int(proposals[0, position].item()),
                            "ordinary_transfer_count": int(
                                transfers[0, step_in_block].item()
                            ),
                            "confidence": float(confidence[0, position].item()),
                            "margin": float(margin[0, position].item()),
                            "entropy": float(entropy[0, position].item()),
                            "token_age": int(token_age[0, position].item()),
                            "trace_probability": float(
                                trace_probability[0, position].item()
                            ),
                            "max_hidden_drift": float(drift[0, position].item()),
                            "attention_influence": float(
                                influence.latest[0, position].item()
                            ),
                            "drift_available": drift_available,
                        }
                    )
                x[ordinary] = proposals[ordinary]
                previous_tokens = proposals.detach()
                previous_hidden = current_hidden
                del logits, log_probs, top, confidence, margin, entropy
    finally:
        influence.remove()
        for handle in handles:
            handle.remove()

    final_tokens = x[:, prompt_len:].detach().cpu()[0]
    generation = tokenizer.decode(final_tokens, skip_special_tokens=True).strip()
    selected = select_stratified_candidates(candidates, maximum_candidates)
    return {
        "candidates": selected,
        "baseline_token_ids": final_tokens.tolist(),
        "baseline_generation": generation,
        "baseline_correct": bool(score_generation(generation, example)),
        "baseline_nfe": nfe,
        "candidate_pool_size": len(candidates),
    }


@torch.inference_mode()
def run_counterfactual_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    candidates: list[dict[str, Any]],
    config: DecodeConfig,
) -> dict[str, Any]:
    if not candidates:
        return {"token_ids": [], "generations": [], "nfe": 0}
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset)] * len(candidates)
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
    applied = [False] * batch_size
    nfe = 0
    for block_index in range(num_blocks):
        block_start = prompt_len + block_index * config.block_length
        block_end = block_start + config.block_length
        initial_mask = x[:, block_start:block_end] == config.mask_id
        transfers = get_num_transfer_tokens(initial_mask, steps_per_block)
        for step_in_block in range(steps_per_block):
            if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                break
            global_step = block_index * steps_per_block + step_in_block
            mask = x == config.mask_id
            allowed = mask.clone()
            allowed[:, :block_start] = False
            allowed[:, block_end:] = False
            logits = model(x, attention_mask=model_attention).logits
            nfe += 1
            proposals = torch.argmax(logits, dim=-1)
            maximum = logits.amax(dim=-1)
            confidence = torch.exp(
                maximum.float() - torch.logsumexp(logits.float(), dim=-1)
            )
            ordinary = ordinary_transfer_mask(
                confidence, allowed, transfers[:, step_in_block]
            )
            targets: list[int | None] = []
            for batch_index, candidate in enumerate(candidates):
                if global_step != int(candidate["global_step"]):
                    targets.append(None)
                    continue
                position = prompt_len + int(candidate["relative_position"])
                if int(proposals[batch_index, position].item()) != int(
                    candidate["top_token"]
                ):
                    raise CounterfactualReplayIneligible(
                        "counterfactual replay diverged before the registered commitment"
                    )
                targets.append(position)
                applied[batch_index] = True
            selected = add_counterfactual_commit(ordinary, allowed, targets)
            x[selected] = proposals[selected]
            del logits, proposals, maximum, confidence
    if not all(applied):
        missing = [index for index, value in enumerate(applied) if not value]
        raise RuntimeError(f"counterfactual commitments were not applied: {missing}")
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    return {
        "token_ids": final_tokens.tolist(),
        "generations": [text.strip() for text in generations],
        "nfe": nfe,
    }


def label_effect(
    candidate: dict[str, Any],
    forced_tokens: list[int],
    forced_generation: str,
    baseline: dict[str, Any],
    example: Any,
) -> dict[str, Any]:
    reference = np.asarray(baseline["baseline_token_ids"], dtype=np.int64)
    forced = np.asarray(forced_tokens, dtype=np.int64)
    if reference.shape != forced.shape:
        raise ValueError("counterfactual and baseline token sequences must align")
    changed = forced != reference
    relative_position = int(candidate["relative_position"])
    forced_correct = bool(score_generation(forced_generation, example))
    baseline_correct = bool(baseline["baseline_correct"])
    return {
        **candidate,
        "sequence_safe": int(not bool(changed.any())),
        "answer_safe": int(
            not _answer_changed(
                example.dataset,
                forced_generation,
                str(baseline["baseline_generation"]),
            )
        ),
        "quality_safe": int((not baseline_correct) or forced_correct),
        "correctness_unchanged": int(forced_correct == baseline_correct),
        "target_identity_safe": int(
            int(forced[relative_position]) == int(reference[relative_position])
        ),
        "changed_token_count": int(changed.sum()),
        "suffix_changed_token_count": int(changed[relative_position + 1 :].sum()),
        "baseline_correct": int(baseline_correct),
        "forced_correct": int(forced_correct),
        "baseline_generation": baseline["baseline_generation"],
        "forced_generation": forced_generation,
    }


def process_partition(
    model: torch.nn.Module,
    tokenizer: Any,
    controller: TraceLockController,
    projection: torch.Tensor,
    examples: list[Any],
    config: DecodeConfig,
    output_dir: Path,
    *,
    trace_layers: list[int],
    representation_layers: list[int],
    influence_layer: int,
    max_queries: int,
    window_width: int,
    maximum_candidates: int,
    intervention_batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples, start=1):
        print(
            f"[commitment-labels] partition={output_dir.name} "
            f"example={example_index}/{len(examples)} id={example.example_id}",
            flush=True,
        )
        baseline = collect_baseline_candidates(
            model,
            tokenizer,
            controller,
            projection,
            example,
            config,
            trace_layers=trace_layers,
            representation_layers=representation_layers,
            influence_layer=influence_layer,
            max_queries=max_queries,
            window_width=window_width,
            maximum_candidates=maximum_candidates,
        )
        candidates = baseline["candidates"]
        baselines.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "candidate_pool_size": baseline["candidate_pool_size"],
                "selected_candidates": len(candidates),
                "baseline_nfe": baseline["baseline_nfe"],
                "baseline_correct": int(baseline["baseline_correct"]),
                "baseline_generation": baseline["baseline_generation"],
            }
        )
        for start in range(0, len(candidates), intervention_batch_size):
            chunk = candidates[start : start + intervention_batch_size]
            try:
                result = run_counterfactual_batch(
                    model, tokenizer, example, chunk, config
                )
            except CounterfactualReplayIneligible as error:
                # A replay that naturally transfers the target has no distinct
                # treatment. Exclude it instead of assigning a false safe label
                # or aborting the remaining causal-label collection. The pilot
                # enforces intervention_batch_size == 1, so the reason maps to
                # exactly one registered candidate.
                for candidate in chunk:
                    exclusions.append(
                        {
                            **candidate,
                            "exclusion_reason": str(error),
                        }
                    )
                pd.DataFrame(exclusions).to_csv(
                    output_dir / "replay_exclusions.csv", index=False
                )
                print(
                    "[commitment-labels] excluded replay-ineligible candidate "
                    f"example={example.example_id} reason={error}",
                    flush=True,
                )
                continue
            for candidate, tokens, generation in zip(
                chunk, result["token_ids"], result["generations"]
            ):
                rows.append(
                    label_effect(
                        candidate,
                        tokens,
                        generation,
                        baseline,
                        example,
                    )
                )
            pd.DataFrame(rows).to_csv(output_dir / "labels.csv", index=False)
        pd.DataFrame(baselines).to_csv(output_dir / "baselines.csv", index=False)
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    summary = {
        "examples": len(examples),
        "labels": len(frame),
        "excluded_candidates": len(exclusions),
        "exclusion_reasons": (
            pd.Series(
                [row["exclusion_reason"] for row in exclusions], dtype="object"
            )
            .value_counts()
            .to_dict()
        ),
        "sequence_safe_rate": float(frame["sequence_safe"].mean()) if len(frame) else None,
        "answer_safe_rate": float(frame["answer_safe"].mean()) if len(frame) else None,
        "quality_safe_rate": float(frame["quality_safe"].mean()) if len(frame) else None,
        "mean_changed_tokens": float(frame["changed_token_count"].mean()) if len(frame) else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-example-indices", required=True)
    parser.add_argument("--eval-example-indices", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-layers", default="30,31,32")
    parser.add_argument("--representation-layers", default="8,16,24")
    parser.add_argument("--influence-layer", type=int, default=24)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--window-width", type=int, default=32)
    parser.add_argument("--candidates-per-example", type=int, default=6)
    parser.add_argument(
        "--intervention-batch-size",
        type=int,
        default=1,
        help=(
            "Keep at one for exact counterfactual replay. Changing the model batch "
            "shape can alter near-tied BF16 transfer rankings before intervention."
        ),
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()
    if args.candidates_per_example < 2:
        raise ValueError("use at least two candidates per example")
    if args.intervention_batch_size != 1:
        raise ValueError(
            "exact counterfactual replay currently requires intervention batch size one"
        )
    trace_layers = [int(value) for value in args.trace_layers.split(",") if value]
    representation_layers = [
        int(value) for value in args.representation_layers.split(",") if value
    ]
    if len(trace_layers) != 3 or not representation_layers:
        raise ValueError("TraceLock needs three trace layers and drift needs at least one layer")
    train_indices = parse_example_indices(args.train_example_indices)
    eval_indices = parse_example_indices(args.eval_example_indices)
    if set(train_indices) & set(eval_indices):
        raise ValueError("training and evaluation example indices must be disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {**vars(args)}
    for key in ("checkpoint", "projection", "output_dir"):
        config_payload[key] = str(config_payload[key])
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller = TraceLockController(TraceLockConfig(**checkpoint["config"]))
    controller.load_state_dict(checkpoint["model"])
    projection = torch.from_numpy(np.load(args.projection)).float()
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
    common = {
        "trace_layers": trace_layers,
        "representation_layers": representation_layers,
        "influence_layer": args.influence_layer,
        "max_queries": args.max_queries,
        "window_width": args.window_width,
        "maximum_candidates": args.candidates_per_example,
        "intervention_batch_size": args.intervention_batch_size,
    }
    process_partition(
        model,
        tokenizer,
        controller,
        projection,
        _examples(args.dataset, args.split, train_indices),
        config,
        args.output_dir / "train",
        **common,
    )
    process_partition(
        model,
        tokenizer,
        controller,
        projection,
        _examples(args.dataset, args.split, eval_indices),
        config,
        args.output_dir / "eval",
        **common,
    )


if __name__ == "__main__":
    main()
