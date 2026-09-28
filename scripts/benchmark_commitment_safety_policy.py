#!/usr/bin/env python3
"""Benchmark direct commitment-safety PhaseLock policies end to end."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_phase_lock_batched_online import _answer_changed, _write_csv
from benchmark_phase_lock_fused_replay import _examples
from benchmark_tracelock_vs_phaselock import _install_hooks
from generate_commitment_safety_data import (
    _trace_probability,
    ordinary_transfer_mask,
)
from regret_remasking.commitment_safety import CommitmentSafetyScorer
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
from regret_remasking.phase_lock import select_semantic_transfer
from regret_remasking.phase_lock_semantic import update_token_identity_patience
from regret_remasking.tracelock_baseline import TraceLockConfig, TraceLockController
from run_four_arm_audit import transformer_blocks


METHODS = (
    "dense",
    "capped_confidence",
    "confidence",
    "trace",
    "safety_cheap",
    "safety_full",
)


def select_extra_commits(
    score: torch.Tensor,
    candidate: torch.Tensor,
    *,
    threshold: float,
    maximum_per_sample: int,
) -> torch.Tensor:
    if score.shape != candidate.shape or candidate.dtype != torch.bool:
        raise ValueError("score and candidate tensors must align")
    if maximum_per_sample < 1:
        raise ValueError("maximum extra commitments must be positive")
    eligible = candidate & (score >= threshold)
    selected = torch.zeros_like(candidate)
    masked = torch.where(eligible, score, torch.full_like(score, -torch.inf))
    for batch_index in range(candidate.shape[0]):
        count = min(maximum_per_sample, int(eligible[batch_index].sum().item()))
        if count <= 0:
            continue
        positions = torch.topk(masked[batch_index], k=count).indices
        selected[batch_index, positions] = True
    return selected


@torch.inference_mode()
def decode_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    controller: TraceLockController,
    projection: torch.Tensor,
    examples: list[Any],
    config: DecodeConfig,
    *,
    method: str,
    threshold: float,
    safety_scorer: CommitmentSafetyScorer | None,
    trace_layers: list[int],
    representation_layers: list[int],
    influence_layer: int,
    max_queries: int,
    window_width: int,
    maximum_extra_per_update: int,
    capped_lock_fraction: float,
    capped_confidence_threshold: float,
    capped_optional_steps_per_block: int,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown commitment policy: {method}")
    if method.startswith("safety_") and safety_scorer is None:
        raise ValueError("direct-safety method requires a trained scorer")
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
    uses_trace = method in {"trace", "safety_cheap", "safety_full"}
    uses_full = method == "safety_full"
    uses_safety = method in {"safety_cheap", "safety_full"}
    capture_layers = sorted(
        (set(trace_layers) if uses_trace else set())
        | (set(representation_layers) if uses_full else set())
    )
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    if capture_layers:
        captures, handles = _install_hooks(model, capture_layers)
    influence: SampledAttentionInfluenceCapture | None = None
    if uses_full:
        influence = SampledAttentionInfluenceCapture(
            transformer_blocks(model)[influence_layer - 1], max_queries=max_queries
        )
        influence.install()

    previous_hidden: dict[int, torch.Tensor] | None = None
    previous_tokens: torch.Tensor | None = None
    token_age = torch.zeros_like(x, dtype=torch.long)
    semantic_plans: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(batch_size)
    ]
    extra_plans: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(batch_size)
    ]
    nfe = 0
    extra_commit_count = 0
    ordinary_commit_count = 0
    controller_latency_s = 0.0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
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
                if capture_layers:
                    captures.clear()
                if influence is not None:
                    unresolved = mask.clone()
                    unresolved[:, :prompt_len] = False
                    influence.set_query_mask(unresolved)
                logits = model(x, attention_mask=model_attention).logits
                nfe += 1
                if set(captures) != set(capture_layers):
                    raise RuntimeError("policy decoder missed a requested hidden layer")
                if influence is not None and influence.latest is None:
                    raise RuntimeError("policy decoder missed attention influence")
                proposals = torch.argmax(logits, dim=-1)
                maximum = logits.amax(dim=-1)
                confidence = torch.exp(
                    maximum.float() - torch.logsumexp(logits.float(), dim=-1)
                )
                ordinary = ordinary_transfer_mask(
                    confidence, allowed, transfers[:, step_in_block]
                )
                ordinary_commit_count += int(ordinary.sum().item())
                extra = torch.zeros_like(ordinary)
                if method == "capped_confidence":
                    torch.cuda.synchronize(device)
                    controller_started = time.perf_counter()
                    available = allowed.sum(dim=1).to(dtype=torch.long)
                    base_k = torch.minimum(
                        available, transfers[:, step_in_block]
                    )
                    total_k = base_k.clone()
                    accelerated_eligible = torch.zeros_like(allowed)
                    if step_in_block < capped_optional_steps_per_block:
                        requested = torch.ceil(
                            capped_lock_fraction * available.float()
                        ).to(dtype=torch.long)
                        total_k = torch.minimum(
                            available, torch.maximum(base_k, requested)
                        )
                        accelerated_eligible = allowed & (
                            confidence >= capped_confidence_threshold
                        )
                    selected, _scheduled, extra = select_semantic_transfer(
                        allowed,
                        confidence,
                        confidence,
                        base_k,
                        total_k,
                        accelerated_eligible=accelerated_eligible,
                    )
                    torch.cuda.synchronize(device)
                    controller_latency_s += time.perf_counter() - controller_started
                elif method != "dense":
                    torch.cuda.synchronize(device)
                    controller_started = time.perf_counter()
                    domain = allowed & ~ordinary
                    if method == "confidence":
                        score = confidence
                    else:
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
                        domain &= trace_domain
                        if method == "trace":
                            score = trace_probability
                        else:
                            if safety_scorer is None:
                                raise RuntimeError("missing direct-safety scorer")
                            log_probs = F.log_softmax(logits.float(), dim=-1)
                            top = torch.topk(log_probs, k=2, dim=-1).values.exp()
                            margin = top[:, :, 0] - top[:, :, 1]
                            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                            token_age = update_token_identity_patience(
                                token_age, proposals, previous_tokens, allowed
                            )
                            feature_map: dict[str, torch.Tensor] = {
                                "confidence": confidence,
                                "margin": margin,
                                "entropy": entropy,
                                "token_age": token_age.float(),
                                "trace_probability": trace_probability,
                                "block_age_fraction": torch.full_like(
                                    confidence,
                                    (step_in_block + 1) / steps_per_block,
                                ),
                            }
                            if uses_full:
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
                                    raise RuntimeError("missing full-safety influence")
                                feature_map.update(
                                    {
                                        "max_hidden_drift": drift,
                                        "attention_influence": influence.latest,
                                        "drift_available": drift_available,
                                    }
                                )
                                previous_hidden = current_hidden
                            score = safety_scorer.score_tensors(feature_map)
                            del log_probs, top, margin, entropy
                    extra = select_extra_commits(
                        score,
                        domain,
                        threshold=threshold,
                        maximum_per_sample=maximum_extra_per_update,
                    )
                    torch.cuda.synchronize(device)
                    controller_latency_s += time.perf_counter() - controller_started
                else:
                    selected = ordinary
                if method not in {"dense", "capped_confidence"}:
                    selected = ordinary | extra
                extra_commit_count += int(extra.sum().item())
                x[selected] = proposals[selected]
                for batch_index in range(batch_size):
                    all_events = [
                        (
                            int(position - prompt_len),
                            int(proposals[batch_index, position].item()),
                        )
                        for position in torch.nonzero(
                            selected[batch_index], as_tuple=False
                        ).flatten().tolist()
                    ]
                    extra_events = [
                        (
                            int(position - prompt_len),
                            int(proposals[batch_index, position].item()),
                        )
                        for position in torch.nonzero(
                            extra[batch_index], as_tuple=False
                        ).flatten().tolist()
                    ]
                    if all_events:
                        semantic_plans[batch_index][global_step] = all_events
                    if extra_events:
                        extra_plans[batch_index][global_step] = extra_events
                if uses_safety:
                    previous_tokens = proposals.detach()
                del logits, proposals, maximum, confidence
    finally:
        if influence is not None:
            influence.remove()
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    return {
        "method": method,
        "token_ids": final_tokens.tolist(),
        "generations": [text.strip() for text in generations],
        "latency_s": latency_s,
        "controller_latency_s": controller_latency_s,
        "nfe": nfe,
        "ordinary_commit_count": ordinary_commit_count,
        "extra_commit_count": extra_commit_count,
        "semantic_plans": semantic_plans,
        "extra_plans": extra_plans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-layers", default="30,31,32")
    parser.add_argument("--representation-layers", default="8,16,24")
    parser.add_argument("--influence-layer", type=int, default=24)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--window-width", type=int, default=32)
    parser.add_argument("--maximum-extra-per-update", type=int, default=2)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--capped-lock-fraction", type=float, default=0.06)
    parser.add_argument("--capped-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--capped-optional-steps-per-block", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=313)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timing_repeats < 1 or args.timing_warmups < 0:
        raise ValueError("batch and timing counts must be valid")
    methods = tuple(
        value.strip()
        for value in args.methods.replace(":", ",").split(",")
        if value.strip()
    )
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"unknown benchmark methods: {unknown_methods}")
    if "dense" not in methods:
        raise ValueError("--methods must include dense")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must not contain duplicates")
    trace_layers = [int(value) for value in args.trace_layers.split(",") if value]
    representation_layers = [
        int(value) for value in args.representation_layers.split(",") if value
    ]
    examples = _examples(
        args.dataset, args.split, parse_example_indices(args.example_indices)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.predictor_dir / "policy_config.json").open(encoding="utf-8") as handle:
        policy_config = json.load(handle)
    scorers = {
        "safety_cheap": CommitmentSafetyScorer(
            args.predictor_dir / "commitment_safety_cheap.joblib"
        ),
        "safety_full": CommitmentSafetyScorer(
            args.predictor_dir / "commitment_safety_full.joblib"
        ),
    }
    thresholds = {
        "dense": 1.000001,
        "capped_confidence": args.capped_confidence_threshold,
        "confidence": float(policy_config["baselines"]["confidence"]["threshold"]),
        "trace": float(policy_config["baselines"]["trace"]["threshold"]),
        "safety_cheap": scorers["safety_cheap"].threshold,
        "safety_full": scorers["safety_full"].threshold,
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller = TraceLockController(TraceLockConfig(**checkpoint["config"]))
    controller.load_state_dict(checkpoint["model"])
    projection = torch.from_numpy(np.load(args.projection)).float()
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
    rows: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], int] = {}

    def score(text: str, example: Any) -> int:
        key = (str(example.example_id), text)
        if key not in score_cache:
            score_cache[key] = int(bool(score_generation(text, example)))
        return score_cache[key]

    common = {
        "trace_layers": trace_layers,
        "representation_layers": representation_layers,
        "influence_layer": args.influence_layer,
        "max_queries": args.max_queries,
        "window_width": args.window_width,
        "maximum_extra_per_update": args.maximum_extra_per_update,
        "capped_lock_fraction": args.capped_lock_fraction,
        "capped_confidence_threshold": args.capped_confidence_threshold,
        "capped_optional_steps_per_block": args.capped_optional_steps_per_block,
    }
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(f"[commitment-policy] batch={batch_id} size={len(batch)}", flush=True)

        def run(method: str) -> dict[str, Any]:
            return decode_policy(
                model,
                tokenizer,
                controller,
                projection,
                batch,
                config,
                method=method,
                threshold=thresholds[method],
                safety_scorer=scorers.get(method),
                **common,
            )

        for warmup in range(args.timing_warmups):
            shift = warmup % len(methods)
            for method in methods[shift:] + methods[:shift]:
                run(method)
        for repeat in range(args.timing_repeats):
            shift = repeat % len(methods)
            order = methods[shift:] + methods[:shift]
            records = {method: run(method) for method in order}
            timing.append(
                {
                    "batch_id": batch_id,
                    "repeat": repeat,
                    "execution_order": "_then_".join(order),
                    **{
                        f"{method}_latency_s": float(records[method]["latency_s"])
                        for method in methods
                    },
                    **{
                        f"{method}_controller_latency_s": float(
                            records[method]["controller_latency_s"]
                        )
                        for method in methods
                    },
                }
            )
            if repeat > 0:
                continue
            for local_index, example in enumerate(batch):
                row: dict[str, Any] = {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "example_id": example.example_id,
                }
                for method in methods:
                    generation = records[method]["generations"][local_index]
                    row[f"{method}_correct"] = score(generation, example)
                    row[f"{method}_exact_dense"] = int(
                        records[method]["token_ids"][local_index]
                        == records["dense"]["token_ids"][local_index]
                    )
                    row[f"{method}_answer_changed_dense"] = int(
                        _answer_changed(
                            args.dataset,
                            generation,
                            records["dense"]["generations"][local_index],
                        )
                    )
                    row[f"{method}_nfe"] = int(records[method]["nfe"])
                    row[f"{method}_extra_commits"] = int(
                        records[method]["extra_commit_count"]
                    )
                    plans.append(
                        {
                            "dataset": args.dataset,
                            "example_id": example.example_id,
                            "method": method,
                            "semantic_plan": records[method]["semantic_plans"][local_index],
                            "extra_plan": records[method]["extra_plans"][local_index],
                        }
                    )
                rows.append(row)

    medians: list[dict[str, float]] = []
    for batch_id in sorted({int(row["batch_id"]) for row in timing}):
        group = [row for row in timing if int(row["batch_id"]) == batch_id]
        medians.append(
            {
                f"{method}_latency_s": float(
                    np.median([row[f"{method}_latency_s"] for row in group])
                )
                for method in methods
            }
        )
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "examples": len(rows),
        "thresholds": thresholds,
        "methods": {},
    }
    for method in methods:
        candidate_time = sum(row[f"{method}_latency_s"] for row in medians)
        dense_time = sum(row["dense_latency_s"] for row in medians)
        summary["methods"][method] = {
            "accuracy": float(np.mean([row[f"{method}_correct"] for row in rows])),
            "exact_dense_rate": float(
                np.mean([row[f"{method}_exact_dense"] for row in rows])
            ),
            "answer_change_rate": float(
                np.mean([row[f"{method}_answer_changed_dense"] for row in rows])
            ),
            "mean_nfe": float(np.mean([row[f"{method}_nfe"] for row in rows])),
            "mean_extra_commits": float(
                np.mean([row[f"{method}_extra_commits"] for row in rows])
            ),
            "runtime_gain_vs_dense": float(1.0 - candidate_time / dense_time),
        }
    _write_csv(args.output_dir / "rows.csv", rows)
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "plans.json").write_text(
        json.dumps(plans, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
