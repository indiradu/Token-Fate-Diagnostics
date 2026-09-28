#!/usr/bin/env python3
"""End-to-end batched online audit of the three PhaseLock phases."""

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

from regret_remasking.data import (
    build_prompt,
    load_examples,
    normalize_answer,
    score_generation,
)
from regret_remasking.llada_trace import (
    DecodeConfig,
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.phase_lock import (
    cosine_distance,
    representation_ready,
    select_representation_lock,
    select_semantic_transfer,
)
from regret_remasking.phase_lock_sparse import (
    ActiveRowBucket,
    BucketedReferenceRuntime,
    FusedLatencyTable,
    choose_fused_compute_plan,
)
from run_four_arm_audit import transformer_blocks


ARMS = ("A0", "A1", "A2", "A3")
ARM_ORDERS = (
    ("A0", "A1", "A2", "A3"),
    ("A1", "A3", "A0", "A2"),
    ("A2", "A0", "A3", "A1"),
    ("A3", "A2", "A1", "A0"),
)


def _answer_changed(dataset: str, left: str, right: str) -> bool:
    """Compare task answers without treating code as a numeric response."""
    if dataset == "humaneval":
        return left.strip() != right.strip()
    return normalize_answer(left) != normalize_answer(right)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _global_plan(
    active_counts: list[int], total_rows: int, width_multiple: int
) -> tuple[ActiveRowBucket, ...]:
    width = min(
        total_rows,
        int(math.ceil(max(active_counts) / width_multiple) * width_multiple),
    )
    return (
        ActiveRowBucket(
            sample_indices=tuple(range(len(active_counts))),
            active_counts=tuple(active_counts),
            padded_active_rows=width,
            total_rows=total_rows,
        ),
    )


@torch.inference_mode()
def decode_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: DecodeConfig,
    arm: str,
    latency_table: FusedLatencyTable,
    *,
    semantic_lock_fraction: float,
    semantic_min_confidence: float,
    semantic_optional_steps_per_block: int,
    representation_policy: str,
    representation_delay_updates: int,
    representation_layer: int,
    representation_drift_threshold: float,
    representation_patience: int,
    representation_min_age: int,
    representation_lock_fraction: float,
    compute_min_locked_fraction: float,
    compute_min_remaining_steps: int,
    compute_min_projected_saved_rows: int,
    compute_min_expected_gain: float,
    bucket_width_multiple: int,
    max_bucket_size: int,
    semantic_plans: list[dict[str, list[dict[str, Any]]]] | None = None,
    reference_plans: list[dict[str, list[dict[str, Any]]]] | None = None,
    force_semantic_plan_tokens: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    device = model_device(model)
    prompts = [build_prompt(example.question, example.dataset) for example in examples]
    input_ids, attention_mask = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    replay_mode = semantic_plans is not None or reference_plans is not None
    if replay_mode:
        if arm not in {"A2", "A3"}:
            raise ValueError("fixed plan replay supports only A2 and A3")
        if semantic_plans is None or reference_plans is None:
            raise ValueError("semantic and reference plans must be supplied together")
        if len(semantic_plans) != batch_size or len(reference_plans) != batch_size:
            raise ValueError("fixed plans must have one entry per batch example")
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
    if arm in {"A2", "A3"}:
        blocks = transformer_blocks(model)
        if representation_layer > len(blocks):
            raise ValueError(
                f"representation layer must be in 1..{len(blocks)}"
            )
        runtime = BucketedReferenceRuntime(
            blocks,
            batch_size=batch_size,
            total_len=total_len,
            fusion_mode="exact",
        )
        runtime.install_dense()
    semantic_positions = torch.zeros_like(x, dtype=torch.bool)
    semantic_age = torch.zeros_like(x, dtype=torch.long)
    below_drift_count = torch.zeros_like(x, dtype=torch.long)
    previous_representation: torch.Tensor | None = None
    nfe = 0
    optional_commits = 0
    reference_locks = 0
    removed_row_token_forwards = 0
    switch_step: int | None = None
    switch_backend: str | None = None
    switch_predicted_gain: float | None = None
    replan_count = 0
    post_switch_dense_prediction_count = 0
    decision_trace: list[dict[str, Any]] = []

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * config.block_length
            block_end = prompt_len + (block_index + 1) * config.block_length
            block_mask = x[:, block_start:block_end] == config.mask_id
            transfer_schedule = get_num_transfer_tokens(
                block_mask, steps_per_block
            )
            for step_in_block in range(steps_per_block):
                if not bool((x[:, block_start:block_end] == config.mask_id).any()):
                    break
                global_step = block_index * steps_per_block + step_in_block
                remaining_steps = config.steps - global_step
                if (
                    arm == "A3"
                    and runtime is not None
                    and not runtime.compute_active
                    and runtime.locked_count > 0
                    and runtime.locked_count / (batch_size * total_len)
                    >= compute_min_locked_fraction
                    and remaining_steps >= compute_min_remaining_steps
                    and runtime.locked_count * remaining_steps
                    >= compute_min_projected_saved_rows
                ):
                    decision = choose_fused_compute_plan(
                        runtime.active_counts,
                        total_len,
                        latency_table,
                        width_multiple=bucket_width_multiple,
                        max_bucket_size=max_bucket_size,
                        max_padding_fraction=1.0,
                        min_expected_gain=compute_min_expected_gain,
                    )
                    decision_trace.append(
                        {
                            "step": global_step,
                            "backend": decision.backend,
                            "predicted_gain": decision.predicted_gain,
                            "active_counts": list(runtime.active_counts),
                        }
                    )
                    if decision.backend != "dense":
                        runtime.activate_compute(decision.buckets)
                        switch_step = global_step
                        switch_backend = decision.backend
                        switch_predicted_gain = decision.predicted_gain
                if runtime is not None and runtime.compute_active:
                    removed_row_token_forwards += runtime.locked_count

                mask_index = x == config.mask_id
                logits = model(x, attention_mask=attention_mask).logits
                nfe += 1
                x0 = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits.float(), dim=-1)
                confidence = probabilities.gather(
                    -1, x0.unsqueeze(-1)
                ).squeeze(-1)
                allowed = mask_index.clone()
                allowed[:, :prompt_len] = False
                allowed[:, block_end:] = False
                if replay_mode:
                    if semantic_plans is None:  # pragma: no cover - validated above
                        raise RuntimeError("missing semantic replay plans")
                    transfer = torch.zeros_like(allowed)
                    selected_events = []
                    for batch_index, plan in enumerate(semantic_plans):
                        for event in plan.get(str(global_step), []):
                            position = prompt_len + int(event["relative_position"])
                            if not bool(allowed[batch_index, position]):
                                raise RuntimeError(
                                    "semantic replay event is unavailable: "
                                    f"batch={batch_index}, step={global_step}, "
                                    f"position={position}"
                                )
                            transfer[batch_index, position] = True
                            if force_semantic_plan_tokens:
                                x0[batch_index, position] = int(event["token_id"])
                            selected_events.append((batch_index, position))
                            optional_commits += int(
                                event.get("selection_source") == "accelerated"
                            )
                else:
                    available = allowed.sum(dim=1).to(dtype=torch.long)
                    base_k = torch.minimum(
                        available, transfer_schedule[:, step_in_block]
                    )
                    accelerated = arm != "A0" and (
                        semantic_optional_steps_per_block < 0
                        or step_in_block < semantic_optional_steps_per_block
                    )
                    total_k = base_k.clone()
                    eligible = torch.zeros_like(allowed)
                    if accelerated:
                        requested = torch.ceil(
                            semantic_lock_fraction * available.float()
                        ).to(dtype=torch.long)
                        total_k = torch.minimum(
                            available, torch.maximum(base_k, requested)
                        )
                        eligible = allowed & (
                            confidence >= semantic_min_confidence
                        )
                    transfer, _scheduled, optional = select_semantic_transfer(
                        allowed,
                        confidence,
                        confidence,
                        base_k,
                        total_k,
                        accelerated_eligible=eligible,
                    )
                    optional_commits += int(optional.sum().item())
                    selected_events = [
                        (int(batch_index), int(position))
                        for batch_index, position in torch.nonzero(
                            transfer, as_tuple=False
                        ).tolist()
                    ]
                x[transfer] = x0[transfer]

                if runtime is not None:
                    selected_mask = torch.zeros_like(semantic_positions)
                    if selected_events:
                        batch_indices = torch.tensor(
                            [event[0] for event in selected_events],
                            dtype=torch.long,
                            device=device,
                        )
                        positions = torch.tensor(
                            [event[1] for event in selected_events],
                            dtype=torch.long,
                            device=device,
                        )
                        selected_mask[batch_indices, positions] = True
                        semantic_positions[batch_indices, positions] = True
                        semantic_age[batch_indices, positions] = 0
                        below_drift_count[batch_indices, positions] = 0
                    old_semantic = semantic_positions & ~selected_mask
                    semantic_age[old_semantic] += 1
                    if replay_mode:
                        if reference_plans is None:  # pragma: no cover - validated above
                            raise RuntimeError("missing reference replay plans")
                        newly_eligible = []
                        for batch_index, plan in enumerate(reference_plans):
                            for event in plan.get(str(global_step), []):
                                position = prompt_len + int(event["relative_position"])
                                if not bool(semantic_positions[batch_index, position]):
                                    raise RuntimeError(
                                        "reference replay precedes semantic commitment: "
                                        f"batch={batch_index}, step={global_step}, "
                                        f"position={position}"
                                    )
                                if bool(runtime.locked[batch_index, position]):
                                    raise RuntimeError(
                                        "reference replay position is already frozen: "
                                        f"batch={batch_index}, step={global_step}, "
                                        f"position={position}"
                                    )
                                newly_eligible.append((batch_index, position))
                    else:
                        current_representation = runtime.last_outputs[
                            representation_layer - 1
                        ]
                        if previous_representation is None:
                            drift = torch.zeros_like(x, dtype=torch.float32)
                        else:
                            drift = cosine_distance(
                                current_representation, previous_representation
                            )
                        below = old_semantic & (
                            drift <= representation_drift_threshold
                        )
                        below_drift_count[below] += 1
                        below_drift_count[old_semantic & ~below] = 0
                        candidate = semantic_positions & ~runtime.locked
                        if representation_policy == "delayed_one_update":
                            ready = representation_ready(
                                "always",
                                drift,
                                confidence,
                                torch.zeros_like(drift),
                                torch.ones_like(drift),
                                semantic_age,
                                below_drift_count,
                                threshold=representation_drift_threshold,
                                patience=representation_patience,
                                min_age=representation_delay_updates,
                            )
                        else:
                            ready = representation_ready(
                                representation_policy,
                                drift,
                                confidence,
                                torch.zeros_like(drift),
                                torch.ones_like(drift),
                                semantic_age,
                                below_drift_count,
                                threshold=representation_drift_threshold,
                                patience=representation_patience,
                                min_age=representation_min_age,
                            )
                        ready &= candidate
                        selected_references = select_representation_lock(
                            candidate,
                            ready,
                            drift,
                            (
                                1.0
                                if representation_policy == "delayed_one_update"
                                else representation_lock_fraction
                            ),
                        )
                        newly_eligible = [
                            (int(batch_index), int(position))
                            for batch_index, position in torch.nonzero(
                                selected_references, as_tuple=False
                            ).tolist()
                        ]
                    if newly_eligible:
                        runtime.commit(newly_eligible)
                        reference_locks += len(newly_eligible)
                        if runtime.compute_active:
                            next_decision = choose_fused_compute_plan(
                                runtime.active_counts,
                                total_len,
                                latency_table,
                                width_multiple=bucket_width_multiple,
                                max_bucket_size=max_bucket_size,
                                max_padding_fraction=1.0,
                                min_expected_gain=compute_min_expected_gain,
                            )
                            if next_decision.backend == "dense":
                                post_switch_dense_prediction_count += 1
                                plans = _global_plan(
                                    runtime.active_counts,
                                    total_len,
                                    bucket_width_multiple,
                                )
                            else:
                                plans = next_decision.buckets
                            runtime.set_compute_plan(plans)
                            replan_count += 1
                    if not replay_mode:
                        previous_representation = current_representation.detach()
                del logits, probabilities, confidence
    finally:
        if runtime is not None:
            runtime.remove()
    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(
        final_tokens, skip_special_tokens=True
    )
    return {
        "arm": arm,
        "token_ids": final_tokens.tolist(),
        "generations": [generation.strip() for generation in generations],
        "latency_s": latency_s,
        "nfe": nfe,
        "optional_commit_count": optional_commits,
        "reference_lock_count": reference_locks,
        "removed_row_token_forwards": removed_row_token_forwards,
        "switch_step": switch_step,
        "switch_backend": switch_backend,
        "switch_predicted_gain": switch_predicted_gain,
        "replan_count": replan_count,
        "post_switch_dense_prediction_count": post_switch_dense_prediction_count,
        "decision_trace": decision_trace,
        "prompt_len": prompt_len,
        "total_len": total_len,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--dataset", choices=["gsm8k", "math500", "humaneval"], default="gsm8k"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--semantic-lock-fraction", type=float, default=0.06)
    parser.add_argument("--semantic-min-confidence", type=float, default=0.95)
    parser.add_argument("--semantic-optional-steps-per-block", type=int, default=2)
    parser.add_argument(
        "--representation-policy",
        choices=["delayed_one_update", "drift", "capped_drift"],
        default="delayed_one_update",
    )
    parser.add_argument("--representation-delay-updates", type=int, default=1)
    parser.add_argument("--representation-layer", type=int, default=24)
    parser.add_argument("--representation-drift-threshold", type=float, default=0.005)
    parser.add_argument("--representation-patience", type=int, default=4)
    parser.add_argument("--representation-min-age", type=int, default=1)
    parser.add_argument("--representation-lock-fraction", type=float, default=0.06)
    parser.add_argument("--compute-min-locked-fraction", type=float, default=0.20)
    parser.add_argument("--compute-min-remaining-steps", type=int, default=8)
    parser.add_argument(
        "--compute-min-projected-saved-rows", type=int, default=1500
    )
    parser.add_argument("--compute-min-expected-gain", type=float, default=0.03)
    parser.add_argument("--bucket-width-multiple", type=int, default=16)
    parser.add_argument("--max-bucket-size", type=int, default=8)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("this benchmark targets batched execution (batch >= 2)")
    if args.representation_delay_updates < 1:
        raise ValueError("representation delay must be at least one update")
    if args.representation_layer < 1:
        raise ValueError("representation layer must be positive")
    if args.representation_patience < 1 or args.representation_min_age < 1:
        raise ValueError("representation patience and minimum age must be positive")
    if not 0.0 <= args.representation_lock_fraction <= 1.0:
        raise ValueError("representation lock fraction must be in [0, 1]")
    if args.timing_warmups < 1 or args.timing_repeats < 4:
        raise ValueError("use at least one warmup and four timing repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.latency_table.open(encoding="utf-8") as handle:
        latency_table = FusedLatencyTable.from_rows(json.load(handle))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    examples = load_examples(args.dataset, args.split, args.limit, args.offset)
    if not examples:
        raise ValueError("no examples loaded")
    model, tokenizer = load_llada(
        args.model_name, device=args.device, dtype=args.dtype
    )
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
        temperature=0.0,
    )
    call_kwargs = {
        "semantic_lock_fraction": args.semantic_lock_fraction,
        "semantic_min_confidence": args.semantic_min_confidence,
        "semantic_optional_steps_per_block": args.semantic_optional_steps_per_block,
        "representation_policy": args.representation_policy,
        "representation_delay_updates": args.representation_delay_updates,
        "representation_layer": args.representation_layer,
        "representation_drift_threshold": args.representation_drift_threshold,
        "representation_patience": args.representation_patience,
        "representation_min_age": args.representation_min_age,
        "representation_lock_fraction": args.representation_lock_fraction,
        "compute_min_locked_fraction": args.compute_min_locked_fraction,
        "compute_min_remaining_steps": args.compute_min_remaining_steps,
        "compute_min_projected_saved_rows": args.compute_min_projected_saved_rows,
        "compute_min_expected_gain": args.compute_min_expected_gain,
        "bucket_width_multiple": args.bucket_width_multiple,
        "max_bucket_size": args.max_bucket_size,
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(
            f"[batched-online] batch={batch_id} size={len(batch)}",
            flush=True,
        )
        for warmup in range(args.timing_warmups):
            for arm in ARM_ORDERS[warmup % len(ARM_ORDERS)]:
                decode_batch(
                    model, tokenizer, batch, config, arm, latency_table, **call_kwargs
                )
        for repeat in range(args.timing_repeats):
            order = ARM_ORDERS[repeat % len(ARM_ORDERS)]
            records = {
                arm: decode_batch(
                    model, tokenizer, batch, config, arm, latency_table, **call_kwargs
                )
                for arm in order
            }
            correctness = {
                arm: [
                    bool(score_generation(generation, example))
                    for generation, example in zip(
                        records[arm]["generations"], batch
                    )
                ]
                for arm in ARMS
            }
            for local_index, example in enumerate(batch):
                a0 = records["A0"]
                row = {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "repeat": repeat,
                    "execution_order": "_then_".join(order),
                    "example_id": example.example_id,
                    "a0_latency_s": records["A0"]["latency_s"],
                    "a1_latency_s": records["A1"]["latency_s"],
                    "a2_latency_s": records["A2"]["latency_s"],
                    "a3_latency_s": records["A3"]["latency_s"],
                    "a1_exact_a0": int(
                        records["A1"]["token_ids"][local_index]
                        == a0["token_ids"][local_index]
                    ),
                    "a2_exact_a1": int(
                        records["A2"]["token_ids"][local_index]
                        == records["A1"]["token_ids"][local_index]
                    ),
                    "a3_exact_a2": int(
                        records["A3"]["token_ids"][local_index]
                        == records["A2"]["token_ids"][local_index]
                    ),
                    "a0_correct": int(correctness["A0"][local_index]),
                    "a1_correct": int(correctness["A1"][local_index]),
                    "a2_correct": int(correctness["A2"][local_index]),
                    "a3_correct": int(correctness["A3"][local_index]),
                    "a1_answer_changed_a0": int(
                        _answer_changed(
                            args.dataset,
                            records["A1"]["generations"][local_index],
                            records["A0"]["generations"][local_index],
                        )
                    ),
                    "a2_answer_changed_a1": int(
                        _answer_changed(
                            args.dataset,
                            records["A2"]["generations"][local_index],
                            records["A1"]["generations"][local_index],
                        )
                    ),
                    "a3_answer_changed_a2": int(
                        _answer_changed(
                            args.dataset,
                            records["A3"]["generations"][local_index],
                            records["A2"]["generations"][local_index],
                        )
                    ),
                    "a1_correctness_changed_a0": int(
                        correctness["A1"][local_index]
                        != correctness["A0"][local_index]
                    ),
                    "a2_correctness_changed_a1": int(
                        correctness["A2"][local_index]
                        != correctness["A1"][local_index]
                    ),
                    "a3_correctness_changed_a2": int(
                        correctness["A3"][local_index]
                        != correctness["A2"][local_index]
                    ),
                    "a0_nfe": records["A0"]["nfe"],
                    "a1_nfe": records["A1"]["nfe"],
                    "a2_nfe": records["A2"]["nfe"],
                    "a3_nfe": records["A3"]["nfe"],
                    "a3_switch_step": records["A3"]["switch_step"],
                    "a3_switch_backend": records["A3"]["switch_backend"],
                    "a3_switch_predicted_gain": records["A3"][
                        "switch_predicted_gain"
                    ],
                    "a3_reference_lock_count": records["A3"][
                        "reference_lock_count"
                    ],
                    "a3_removed_row_token_forwards": records["A3"][
                        "removed_row_token_forwards"
                    ],
                    "a3_replan_count": records["A3"]["replan_count"],
                    "a3_post_switch_dense_prediction_count": records["A3"][
                        "post_switch_dense_prediction_count"
                    ],
                    "prompt_len": records["A3"]["prompt_len"],
                    "total_len": records["A3"]["total_len"],
                }
                rows.append(row)
                if (
                    not row["a1_exact_a0"]
                    or not row["a2_exact_a1"]
                    or not row["a3_exact_a2"]
                    or row["a1_answer_changed_a0"]
                    or row["a2_answer_changed_a1"]
                    or row["a3_answer_changed_a2"]
                    or row["a1_correctness_changed_a0"]
                    or row["a2_correctness_changed_a1"]
                    or row["a3_correctness_changed_a2"]
                ):
                    failures.append(
                        {
                            **row,
                            "outputs": {
                                arm: records[arm]["generations"][local_index]
                                for arm in ARMS
                            },
                            "a3_decision_trace": records["A3"]["decision_trace"],
                        }
                    )
    # Batch latency is repeated on every example row; summarize unique batches.
    unique = [
        row
        for row in rows
        if row["example_id"]
        == next(
            candidate["example_id"]
            for candidate in rows
            if candidate["batch_id"] == row["batch_id"]
            and candidate["repeat"] == row["repeat"]
        )
    ]
    totals = {
        arm: sum(float(row[f"{arm.lower()}_latency_s"]) for row in unique)
        for arm in ARMS
    }
    summary = {
        "examples": len(examples),
        "timed_batches": len(unique),
        "a1_gain_vs_a0": 1.0 - totals["A1"] / totals["A0"],
        "a2_gain_vs_a1": 1.0 - totals["A2"] / totals["A1"],
        "a3_gain_vs_a2": 1.0 - totals["A3"] / totals["A2"],
        "a3_gain_vs_a0": 1.0 - totals["A3"] / totals["A0"],
        "a1_exact_a0_rate": float(np.mean([row["a1_exact_a0"] for row in rows])),
        "a2_exact_a1_rate": float(np.mean([row["a2_exact_a1"] for row in rows])),
        "a3_exact_a2_rate": float(np.mean([row["a3_exact_a2"] for row in rows])),
        "a1_answer_change_a0_rate": float(
            np.mean([row["a1_answer_changed_a0"] for row in rows])
        ),
        "a2_answer_change_a1_rate": float(
            np.mean([row["a2_answer_changed_a1"] for row in rows])
        ),
        "a3_answer_change_a2_rate": float(
            np.mean([row["a3_answer_changed_a2"] for row in rows])
        ),
        "a1_correctness_change_a0_rate": float(
            np.mean([row["a1_correctness_changed_a0"] for row in rows])
        ),
        "a2_correctness_change_a1_rate": float(
            np.mean([row["a2_correctness_changed_a1"] for row in rows])
        ),
        "a3_correctness_change_a2_rate": float(
            np.mean([row["a3_correctness_changed_a2"] for row in rows])
        ),
        "a0_accuracy": float(np.mean([row["a0_correct"] for row in rows])),
        "a1_accuracy": float(np.mean([row["a1_correct"] for row in rows])),
        "a2_accuracy": float(np.mean([row["a2_correct"] for row in rows])),
        "a3_accuracy": float(np.mean([row["a3_correct"] for row in rows])),
        "a1_regression_a0_rate": float(
            np.mean(
                [
                    row["a0_correct"] == 1 and row["a1_correct"] == 0
                    for row in rows
                ]
            )
        ),
        "a1_improvement_a0_rate": float(
            np.mean(
                [
                    row["a0_correct"] == 0 and row["a1_correct"] == 1
                    for row in rows
                ]
            )
        ),
        "a2_regression_a1_rate": float(
            np.mean(
                [
                    row["a1_correct"] == 1 and row["a2_correct"] == 0
                    for row in rows
                ]
            )
        ),
        "a2_improvement_a1_rate": float(
            np.mean(
                [
                    row["a1_correct"] == 0 and row["a2_correct"] == 1
                    for row in rows
                ]
            )
        ),
        "a3_regression_a2_rate": float(
            np.mean(
                [
                    row["a2_correct"] == 1 and row["a3_correct"] == 0
                    for row in rows
                ]
            )
        ),
        "a3_improvement_a2_rate": float(
            np.mean(
                [
                    row["a2_correct"] == 0 and row["a3_correct"] == 1
                    for row in rows
                ]
            )
        ),
        "a3_switch_rate": float(
            np.mean([row["a3_switch_step"] is not None for row in unique])
        ),
        "mean_a0_nfe": float(np.mean([row["a0_nfe"] for row in unique])),
        "mean_a1_nfe": float(np.mean([row["a1_nfe"] for row in unique])),
        "mean_a2_nfe": float(np.mean([row["a2_nfe"] for row in unique])),
        "mean_a3_nfe": float(np.mean([row["a3_nfe"] for row in unique])),
        "mean_a3_removed_row_fraction": float(
            np.mean(
                [
                    row["a3_removed_row_token_forwards"]
                    / (row["batch_size"] * row["total_len"] * row["a3_nfe"])
                    for row in unique
                ]
            )
        ),
    }
    device = model_device(model)
    config_row = {
        **vars(args),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "algorithm": {
            "phase_1_semantic": "capped_confidence_c095_fraction006_first2",
            "phase_2_representation": (
                "delayed_one_update_dense_reference_freeze"
                if args.representation_policy == "delayed_one_update"
                else (
                    f"{args.representation_policy}_layer{args.representation_layer}_"
                    f"tau{args.representation_drift_threshold}_"
                    f"patience{args.representation_patience}_"
                    f"age{args.representation_min_age}_"
                    f"cap{args.representation_lock_fraction}"
                )
            ),
            "phase_3_compute": (
                "latency_calibrated_exact_fusion_global_or_bucketed_with_abstention"
            ),
        },
        "timing": {
            "online_decisions_included": True,
            "phase_2_materialization_included": True,
            "phase_3_switch_and_replanning_included": True,
            "task_scoring_excluded": True,
            "cuda_synchronize": True,
            "balanced_arm_order": True,
        },
        "runtime_environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_row, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "safety_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "rows.csv", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
