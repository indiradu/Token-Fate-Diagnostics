#!/usr/bin/env python3
"""Replay fixed semantic/reference schedules through dense A2 and fused A3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_phase_lock_batched_online import (
    _answer_changed,
    _write_csv,
    decode_batch,
)
from regret_remasking.data import load_examples, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    infer_mask_token_id,
    load_llada,
)
from regret_remasking.phase_lock_sparse import FusedLatencyTable
from regret_remasking.paper_baselines import parse_example_indices


ARM_ORDERS = (("A2", "A3"), ("A3", "A2"))


def _plan_hash(plan: dict[str, list[dict[str, Any]]]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _bootstrap_gain(
    batches: list[dict[str, float]], *, samples: int, seed: int
) -> tuple[float, list[float]]:
    a2 = np.asarray([row["a2_latency_s"] for row in batches], dtype=np.float64)
    a3 = np.asarray([row["a3_latency_s"] for row in batches], dtype=np.float64)
    point = 1.0 - a3.sum() / a2.sum()
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(batches), size=(samples, len(batches)))
    gains = 1.0 - a3[draws].sum(axis=1) / a2[draws].sum(axis=1)
    return float(point), [float(value) for value in np.quantile(gains, [0.025, 0.975])]


def _load_plans(
    plan_dir: Path, representation_label: str
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with (plan_dir / "phase_plans.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["representation_policy_label"]) != representation_label:
                continue
            if str(row.get("semantic_replay_mode")) != "positions":
                raise RuntimeError("fused replay requires position-only semantic plans")
            example_id = str(row["example_id"])
            if example_id in selected:
                raise RuntimeError(f"duplicate plan for {example_id}")
            selected[example_id] = row
    if not selected:
        raise ValueError(
            f"no plans found for representation label {representation_label!r}"
        )
    return selected


def _examples(dataset: str, split: str, indices: list[int]) -> list[Any]:
    if not indices:
        return []
    positions: dict[int, Any] = {}
    ordered = sorted(set(indices))
    run_start = ordered[0]
    previous = ordered[0]
    runs: list[tuple[int, int]] = []
    for index in ordered[1:]:
        if index != previous + 1:
            runs.append((run_start, previous))
            run_start = index
        previous = index
    runs.append((run_start, previous))
    for start, end in runs:
        loaded = load_examples(dataset, split, end - start + 1, offset=start)
        if len(loaded) != end - start + 1:
            raise ValueError(f"dataset index span is unavailable: {start}..{end}")
        positions.update(
            {index: example for index, example in zip(range(start, end + 1), loaded)}
        )
    return [positions[index] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--representation-label", required=True)
    parser.add_argument("--latency-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
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
    parser.add_argument("--seed", type=int, default=109)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("fused replay requires batch size at least two")
    if args.timing_warmups < 1 or args.timing_repeats < 4:
        raise ValueError("use at least one warmup and four timing repeats")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    plans_by_example = _load_plans(args.plan_dir, args.representation_label)
    missing = [example.example_id for example in examples if example.example_id not in plans_by_example]
    if missing:
        raise RuntimeError(f"missing selected plans: {missing}")
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
    common = {
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
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        batch_plans = [plans_by_example[example.example_id] for example in batch]
        replay = {
            "semantic_plans": [row["semantic_plan"] for row in batch_plans],
            "reference_plans": [row["reference_plan"] for row in batch_plans],
        }
        print(f"[fused-replay] batch={batch_id} size={len(batch)}", flush=True)
        for warmup in range(args.timing_warmups):
            for arm in ARM_ORDERS[warmup % len(ARM_ORDERS)]:
                decode_batch(
                    model,
                    tokenizer,
                    batch,
                    config,
                    arm,
                    latency_table,
                    **common,
                    **replay,
                )
        for repeat in range(args.timing_repeats):
            order = ARM_ORDERS[repeat % len(ARM_ORDERS)]
            records = {
                arm: decode_batch(
                    model,
                    tokenizer,
                    batch,
                    config,
                    arm,
                    latency_table,
                    **common,
                    **replay,
                )
                for arm in order
            }
            correctness = {
                arm: [
                    bool(score_generation(generation, example))
                    for generation, example in zip(records[arm]["generations"], batch)
                ]
                for arm in ("A2", "A3")
            }
            for local_index, (example, source_plan) in enumerate(zip(batch, batch_plans)):
                row = {
                    "dataset": args.dataset,
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "repeat": repeat,
                    "execution_order": "_then_".join(order),
                    "example_id": example.example_id,
                    "semantic_plan_sha256": _plan_hash(source_plan["semantic_plan"]),
                    "reference_plan_sha256": _plan_hash(source_plan["reference_plan"]),
                    "a2_latency_s": records["A2"]["latency_s"],
                    "a3_latency_s": records["A3"]["latency_s"],
                    "a3_exact_a2": int(
                        records["A3"]["token_ids"][local_index]
                        == records["A2"]["token_ids"][local_index]
                    ),
                    "a2_correct": int(correctness["A2"][local_index]),
                    "a3_correct": int(correctness["A3"][local_index]),
                    "a3_answer_changed_a2": int(
                        _answer_changed(
                            args.dataset,
                            records["A3"]["generations"][local_index],
                            records["A2"]["generations"][local_index],
                        )
                    ),
                    "a3_nfe": records["A3"]["nfe"],
                    "a3_switch_step": records["A3"]["switch_step"],
                    "a3_switch_backend": records["A3"]["switch_backend"],
                    "a3_switch_predicted_gain": records["A3"]["switch_predicted_gain"],
                    "a3_reference_lock_count": records["A3"]["reference_lock_count"],
                    "a3_removed_row_token_forwards": records["A3"]["removed_row_token_forwards"],
                    "a3_replan_count": records["A3"]["replan_count"],
                    "prompt_len": records["A3"]["prompt_len"],
                    "total_len": records["A3"]["total_len"],
                }
                rows.append(row)
                if (
                    not row["a3_exact_a2"]
                    or row["a3_answer_changed_a2"]
                    or row["a2_correct"] != row["a3_correct"]
                ):
                    failures.append(
                        {
                            **row,
                            "outputs": {
                                arm: records[arm]["generations"][local_index]
                                for arm in ("A2", "A3")
                            },
                            "a3_decision_trace": records["A3"]["decision_trace"],
                        }
                    )
    unique_batches: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        unique_batches.setdefault((int(row["batch_id"]), int(row["repeat"])), row)
    timed = list(unique_batches.values())
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in timed:
        by_batch.setdefault(int(row["batch_id"]), []).append(row)
    batch_medians = [
        {
            "a2_latency_s": float(
                np.median([row["a2_latency_s"] for row in group])
            ),
            "a3_latency_s": float(
                np.median([row["a3_latency_s"] for row in group])
            ),
        }
        for _, group in sorted(by_batch.items())
    ]
    gain, gain_ci = _bootstrap_gain(
        batch_medians, samples=args.bootstrap_samples, seed=args.seed + 1000
    )
    example_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        existing = example_rows.setdefault(str(row["example_id"]), row)
        for field in ("a3_exact_a2", "a3_answer_changed_a2", "a2_correct", "a3_correct"):
            if existing[field] != row[field]:
                raise RuntimeError(f"non-deterministic {field}: {row['example_id']}")
    stable = list(example_rows.values())
    summary = {
        "examples": len(stable),
        "batches": len(batch_medians),
        "timed_batches": len(timed),
        "a3_gain_a2": gain,
        "a3_gain_a2_ci95": gain_ci,
        "a3_exact_a2_rate": float(np.mean([row["a3_exact_a2"] for row in stable])),
        "a3_answer_change_a2_count": sum(row["a3_answer_changed_a2"] for row in stable),
        "a3_regression_a2_count": sum(
            row["a2_correct"] == 1 and row["a3_correct"] == 0 for row in stable
        ),
        "a3_improvement_a2_count": sum(
            row["a2_correct"] == 0 and row["a3_correct"] == 1 for row in stable
        ),
        "a2_accuracy": float(np.mean([row["a2_correct"] for row in stable])),
        "a3_accuracy": float(np.mean([row["a3_correct"] for row in stable])),
        "a3_switch_rate": float(np.mean([row["a3_switch_step"] is not None for row in timed])),
        "mean_removed_row_fraction": float(
            np.mean(
                [
                    row["a3_removed_row_token_forwards"]
                    / (row["batch_size"] * row["total_len"] * row["a3_nfe"])
                    for row in timed
                ]
            )
        ),
    }
    summary["phase_3_quality_pass"] = summary["a3_regression_a2_count"] == 0
    summary["phase_3_speed_pass"] = summary["a3_gain_a2_ci95"][0] > 0.0
    summary["phase_3_pass"] = (
        summary["phase_3_quality_pass"] and summary["phase_3_speed_pass"]
    )
    run_config = {
        **vars(args),
        "plan_dir": str(args.plan_dir),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "semantic_replay_mode": "positions",
        "phase_2_plan_source": "dense A1 multi-layer/influence planner",
        "phase_3_backend": "latency-calibrated exact fused global/bucketed",
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
    (args.output_dir / "safety_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "rows.csv", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
