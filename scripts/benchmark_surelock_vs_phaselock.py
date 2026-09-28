#!/usr/bin/env python3
"""Matched LLaDA benchmark of paper-spec SureLock and frozen PhaseLock.

A dense decode is retained only as a shared measurement anchor.  The headline
comparison is SureLock versus PhaseLock on identical examples, schedules,
batches, arm ordering, and hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_phase_lock_batched_online import (
    _answer_changed,
    _global_plan,
    decode_batch as decode_phaselock,
)
from benchmark_phase_lock_fused_replay import _examples, _load_plans
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices, surelock_decision
from regret_remasking.phase_lock_sparse import (
    BucketedReferenceRuntime,
    FusedLatencyTable,
)
from run_four_arm_audit import transformer_blocks


METHOD_DENSE = "dense_anchor"
METHOD_PHASELOCK = "phaselock"


def _surelock_name(epsilon: float) -> str:
    return f"surelock_eps{epsilon:.0e}".replace("+", "")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_gain(
    batches: list[dict[str, float]],
    numerator: str,
    denominator: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, list[float]]:
    faster = np.asarray([row[numerator] for row in batches], dtype=np.float64)
    reference = np.asarray([row[denominator] for row in batches], dtype=np.float64)
    point = 1.0 - faster.sum() / reference.sum()
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(batches), size=(samples, len(batches)))
    gains = 1.0 - faster[draws].sum(axis=1) / reference[draws].sum(axis=1)
    return float(point), [float(value) for value in np.quantile(gains, [0.025, 0.975])]


@torch.inference_mode()
def decode_surelock(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: DecodeConfig,
    *,
    epsilon: float | None,
    confidence_percentile: float,
    bucket_width_multiple: int,
) -> dict[str, Any]:
    """Run the ordinary sampler or Algorithm-1 SureLock with exact row removal."""
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset) for example in examples]
    input_ids, attention_mask = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length
    x = torch.full(
        (batch_size, total_len), config.mask_id, dtype=torch.long, device=device
    )
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

    runtime: BucketedReferenceRuntime | None = None
    if epsilon is not None:
        runtime = BucketedReferenceRuntime(
            transformer_blocks(model),
            batch_size=batch_size,
            total_len=total_len,
            fusion_mode="exact",
        )
        runtime.install_dense()

    previous_log_probs: torch.Tensor | None = None
    nfe = 0
    lock_count = 0
    removed_row_token_forwards = 0
    first_lock_step: int | None = None
    decision_trace: list[dict[str, Any]] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = prompt_len + (block_index + 1) * config.block_length
            block_mask = x[:, block_start:block_end] == config.mask_id
            transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                if runtime is not None and runtime.compute_active:
                    removed_row_token_forwards += runtime.locked_count

                mask_before = x == config.mask_id
                logits = model(x, attention_mask=attention_mask).logits
                nfe += 1
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probabilities = log_probs.exp()
                proposals = torch.argmax(logits, dim=-1)
                confidence = probabilities.gather(
                    -1, proposals.unsqueeze(-1)
                ).squeeze(-1)

                allowed = mask_before.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                for batch_index in range(batch_size):
                    count = min(
                        int(allowed[batch_index].sum().item()),
                        int(transfer_schedule[batch_index, step_in_block].item()),
                    )
                    if count <= 0:
                        continue
                    scores = torch.where(
                        allowed[batch_index],
                        confidence[batch_index],
                        torch.full_like(confidence[batch_index], -torch.inf),
                    )
                    positions = torch.topk(scores, k=count).indices
                    x[batch_index, positions] = proposals[batch_index, positions]

                if runtime is not None:
                    # Algorithm 1 evaluates all active, unmasked positions after
                    # the sampler update. This includes prompt/padded positions,
                    # exactly as the paper's length-N active set specifies.
                    candidate = (x != config.mask_id) & ~runtime.locked
                    decision = surelock_decision(
                        current_log_probs=log_probs,
                        previous_log_probs=previous_log_probs,
                        candidate=candidate,
                        epsilon=float(epsilon),
                        confidence_percentile=confidence_percentile,
                    )
                    newly_locked = [
                        (int(batch_index), int(position))
                        for batch_index, position in torch.nonzero(
                            decision.selected, as_tuple=False
                        ).tolist()
                    ]
                    if newly_locked:
                        runtime.commit(newly_locked)
                        lock_count += len(newly_locked)
                        if first_lock_step is None:
                            first_lock_step = global_step
                        plans = _global_plan(
                            runtime.active_counts, total_len, bucket_width_multiple
                        )
                        if runtime.compute_active:
                            runtime.set_compute_plan(plans)
                        else:
                            runtime.activate_compute(plans)
                    decision_trace.append(
                        {
                            "step": global_step,
                            "candidate_count": int(candidate.sum().item()),
                            "new_lock_count": len(newly_locked),
                            "total_lock_count": runtime.locked_count,
                            "active_counts": list(runtime.active_counts),
                        }
                    )
                previous_log_probs = log_probs.detach()
                del logits, probabilities, confidence
    finally:
        if runtime is not None:
            runtime.remove()
    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    return {
        "token_ids": final_tokens.tolist(),
        "generations": [text.strip() for text in generations],
        "latency_s": latency_s,
        "nfe": nfe,
        "lock_count": lock_count,
        "removed_row_token_forwards": removed_row_token_forwards,
        "first_lock_step": first_lock_step,
        "prompt_len": prompt_len,
        "total_len": total_len,
        "decision_trace": decision_trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--phaselock-plan-dir", type=Path, required=True)
    parser.add_argument("--representation-label", default="mli_ultra")
    parser.add_argument("--latency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--surelock-epsilons", default="5e-4,5e-3")
    parser.add_argument("--surelock-confidence-percentile", type=float, default=20.0)
    parser.add_argument("--bucket-width-multiple", type=int, default=16)
    parser.add_argument("--max-bucket-size", type=int, default=8)
    parser.add_argument("--compute-min-locked-fraction", type=float, default=0.20)
    parser.add_argument("--compute-min-remaining-steps", type=int, default=8)
    parser.add_argument("--compute-min-projected-saved-rows", type=int, default=1500)
    parser.add_argument("--compute-min-expected-gain", type=float, default=0.03)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("matched fused benchmark requires batch size at least two")
    if args.timing_warmups < 1 or args.timing_repeats < 4:
        raise ValueError("use at least one warmup and four timing repeats")
    epsilons = [float(value) for value in args.surelock_epsilons.split(",")]
    if not epsilons:
        raise ValueError("provide at least one SureLock epsilon")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    plans_by_example = _load_plans(args.phaselock_plan_dir, args.representation_label)
    missing = [example.example_id for example in examples if example.example_id not in plans_by_example]
    if missing:
        raise RuntimeError(f"missing frozen PhaseLock plans: {missing}")
    with args.latency_table.open(encoding="utf-8") as handle:
        latency_table = FusedLatencyTable.from_rows(json.load(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    surelock_methods = {_surelock_name(epsilon): epsilon for epsilon in epsilons}
    methods = [METHOD_DENSE, *surelock_methods, METHOD_PHASELOCK]
    phaselock_common = {
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
        "bucket_width_multiple": args.bucket_width_multiple,
        "max_bucket_size": args.max_bucket_size,
        "force_semantic_plan_tokens": False,
    }

    def run_method(method: str, batch: list[Any], batch_plans: list[dict[str, Any]]) -> dict[str, Any]:
        if method == METHOD_DENSE:
            return decode_surelock(
                model,
                tokenizer,
                batch,
                config,
                epsilon=None,
                confidence_percentile=args.surelock_confidence_percentile,
                bucket_width_multiple=args.bucket_width_multiple,
            )
        if method in surelock_methods:
            return decode_surelock(
                model,
                tokenizer,
                batch,
                config,
                epsilon=surelock_methods[method],
                confidence_percentile=args.surelock_confidence_percentile,
                bucket_width_multiple=args.bucket_width_multiple,
            )
        replay = {
            "semantic_plans": [row["semantic_plan"] for row in batch_plans],
            "reference_plans": [row["reference_plan"] for row in batch_plans],
        }
        record = decode_phaselock(
            model,
            tokenizer,
            batch,
            config,
            "A3",
            latency_table,
            **phaselock_common,
            **replay,
        )
        record["lock_count"] = record["reference_lock_count"]
        record["first_lock_step"] = record["switch_step"]
        return record

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        batch_plans = [plans_by_example[example.example_id] for example in batch]
        print(f"[published-baselines] batch={batch_id} size={len(batch)}", flush=True)
        for warmup in range(args.timing_warmups):
            shift = warmup % len(methods)
            order = methods[shift:] + methods[:shift]
            for method in order:
                run_method(method, batch, batch_plans)
        for repeat in range(args.timing_repeats):
            shift = repeat % len(methods)
            order = methods[shift:] + methods[:shift]
            records = {
                method: run_method(method, batch, batch_plans) for method in order
            }
            correctness = {
                method: [
                    int(bool(score_generation(generation, example)))
                    for generation, example in zip(records[method]["generations"], batch)
                ]
                for method in methods
            }
            for local_index, example in enumerate(batch):
                dense = records[METHOD_DENSE]
                row: dict[str, Any] = {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "repeat": repeat,
                    "batch_size": len(batch),
                    "execution_order": "_then_".join(order),
                    "example_id": example.example_id,
                    "prompt_len": dense["prompt_len"],
                    "total_len": dense["total_len"],
                }
                for method in methods:
                    record = records[method]
                    prefix = method
                    row[f"{prefix}_latency_s"] = record["latency_s"]
                    row[f"{prefix}_correct"] = correctness[method][local_index]
                    row[f"{prefix}_exact_dense"] = int(
                        record["token_ids"][local_index]
                        == dense["token_ids"][local_index]
                    )
                    row[f"{prefix}_answer_changed_dense"] = int(
                        _answer_changed(
                            args.dataset,
                            record["generations"][local_index],
                            dense["generations"][local_index],
                        )
                    )
                    row[f"{prefix}_nfe"] = record["nfe"]
                    row[f"{prefix}_lock_count"] = record.get("lock_count", 0)
                    row[f"{prefix}_removed_row_token_forwards"] = record.get(
                        "removed_row_token_forwards", 0
                    )
                    row[f"{prefix}_first_lock_step"] = record.get("first_lock_step")
                rows.append(row)
                if any(
                    correctness[method][local_index] != correctness[METHOD_DENSE][local_index]
                    for method in methods[1:]
                ):
                    failures.append(
                        {
                            **row,
                            "outputs": {
                                method: records[method]["generations"][local_index]
                                for method in methods
                            },
                        }
                    )

    unique_batches: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        unique_batches.setdefault((int(row["batch_id"]), int(row["repeat"])), row)
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in unique_batches.values():
        by_batch.setdefault(int(row["batch_id"]), []).append(row)
    batch_medians = [
        {
            f"{method}_latency_s": float(
                np.median([row[f"{method}_latency_s"] for row in group])
            )
            for method in methods
        }
        for _, group in sorted(by_batch.items())
    ]
    stable_examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable_examples.setdefault(str(row["example_id"]), row)
    stable = list(stable_examples.values())
    summary: dict[str, Any] = {
        "examples": len(stable),
        "batches": len(batch_medians),
        "timed_batches": len(unique_batches),
        "methods": methods,
    }
    for method_index, method in enumerate(methods):
        summary[method] = {
            "accuracy": float(np.mean([row[f"{method}_correct"] for row in stable])),
            "exact_dense_rate": float(
                np.mean([row[f"{method}_exact_dense"] for row in stable])
            ),
            "answer_change_dense_rate": float(
                np.mean([row[f"{method}_answer_changed_dense"] for row in stable])
            ),
            "mean_nfe": float(np.mean([row[f"{method}_nfe"] for row in stable])),
            "mean_lock_count_per_batch": float(
                np.mean(
                    [
                        row[f"{method}_lock_count"]
                        for row in unique_batches.values()
                    ]
                )
            ),
            "mean_removed_row_fraction": float(
                np.mean(
                    [
                        row[f"{method}_removed_row_token_forwards"]
                        / (
                            row["batch_size"]
                            * row["total_len"]
                            * max(row[f"{method}_nfe"], 1)
                        )
                        for row in unique_batches.values()
                    ]
                )
            ),
        }
        if method_index:
            gain, interval = _bootstrap_gain(
                batch_medians,
                f"{method}_latency_s",
                f"{METHOD_DENSE}_latency_s",
                samples=args.bootstrap_samples,
                seed=args.seed + method_index,
            )
            summary[method]["runtime_gain_vs_dense"] = gain
            summary[method]["runtime_gain_vs_dense_ci95"] = interval
    for method_index, method in enumerate(methods[1:-1], start=1):
        gain, interval = _bootstrap_gain(
            batch_medians,
            f"{METHOD_PHASELOCK}_latency_s",
            f"{method}_latency_s",
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + method_index,
        )
        summary[f"phaselock_vs_{method}"] = {
            "runtime_gain": gain,
            "runtime_gain_ci95": interval,
            "accuracy_delta": (
                summary[METHOD_PHASELOCK]["accuracy"] - summary[method]["accuracy"]
            ),
        }

    run_config = {
        **vars(args),
        "phaselock_plan_dir": str(args.phaselock_plan_dir),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "baseline_status": {
            "surelock": "paper-spec reimplementation; no official code released",
            "phaselock": "frozen saved semantic and mli_ultra representation plans",
            "dense_anchor": "internal shared measurement anchor, not headline baseline",
        },
        "surelock_algorithm": {
            "posterior": "raw pre-temperature softmax",
            "kl_direction": "KL(p_t || p_t-1)",
            "confidence_percentile": args.surelock_confidence_percentile,
            "candidate": "active and unmasked after ordinary sampler update",
            "intervention": "freeze all-layer block inputs/KV and remove rows next step",
            "executor": "exact fused global active-row packing",
        },
        "timing": {
            "cuda_synchronized": True,
            "balanced_rotating_order": True,
            "online_controller_included": True,
            "task_scoring_excluded": True,
        },
        "runtime_environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(model_device(model)),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "quality_changes.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "rows.csv", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
