#!/usr/bin/env python3
"""Generate batch-matched PhaseLock plans, then audit A1 -> A2 -> A3.

The representation controller observes the dense A1 trajectory for the exact
batch used by the executor.  Its token identities and reference decisions are
then frozen and replayed through dense representation freezing (A2) and fused
row removal (A3).  This avoids transferring plans generated with a different
padding/batching layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_phase_lock_batched_online import _answer_changed, decode_batch
from benchmark_phase_lock_fused_replay import _examples
from benchmark_phaselock_controller_overhead import decode_planner
from benchmark_surelock_vs_phaselock import decode_surelock
from regret_remasking.data import score_generation
from regret_remasking.llada_trace import DecodeConfig, infer_mask_token_id, load_llada
from regret_remasking.paper_baselines import parse_example_indices
from regret_remasking.phase_lock_sparse import FusedLatencyTable


def semantic_replay_plan(
    plan: dict[int, list[tuple[int, int]]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert a planner signature into exact-token replay events."""
    return {
        str(step): [
            {
                "relative_position": int(position),
                "token_id": int(token_id),
                "selection_source": "planner",
            }
            for position, token_id in events
        ]
        for step, events in sorted(plan.items())
    }


def reference_replay_plan(
    plan: dict[int, list[int]],
) -> dict[str, list[dict[str, int]]]:
    """Convert a planner reference signature into replay events."""
    return {
        str(step): [
            {"relative_position": int(position)} for position in positions
        ]
        for step, positions in sorted(plan.items())
    }


def transition_counts(
    reference: list[int], candidate: list[int]
) -> dict[str, float | int]:
    """Summarize paired correctness changes for one lock transition."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("correctness vectors must be non-empty and equally sized")
    regressions = sum(left == 1 and right == 0 for left, right in zip(reference, candidate))
    improvements = sum(left == 0 and right == 1 for left, right in zip(reference, candidate))
    return {
        "regressions": int(regressions),
        "improvements": int(improvements),
        "regression_rate": float(regressions / len(reference)),
        "improvement_rate": float(improvements / len(reference)),
    }


def _bootstrap_gain(
    batches: list[dict[str, float]],
    numerator: str,
    denominator: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, list[float]]:
    candidate = np.asarray([row[numerator] for row in batches], dtype=np.float64)
    reference = np.asarray([row[denominator] for row in batches], dtype=np.float64)
    point = 1.0 - candidate.sum() / reference.sum()
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(batches), size=(samples, len(batches)))
    gains = 1.0 - candidate[draws].sum(axis=1) / reference[draws].sum(axis=1)
    return float(point), [float(value) for value in np.quantile(gains, [0.025, 0.975])]


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
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--compute-min-locked-fraction", type=float, default=0.20)
    parser.add_argument("--compute-min-remaining-steps", type=int, default=8)
    parser.add_argument("--compute-min-projected-saved-rows", type=int, default=1500)
    parser.add_argument("--compute-min-expected-gain", type=float, default=0.03)
    parser.add_argument("--bucket-width-multiple", type=int, default=16)
    parser.add_argument("--max-bucket-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.timing_warmups < 1 or args.timing_repeats < 4:
        raise ValueError("use batch >=2, at least one warmup, and at least four repeats")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
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
    planner_common = {
        "representation_layers": [8, 16, 24],
        "influence_layer": 24,
        "drift_threshold": 0.002,
        "influence_threshold": 0.5,
        "patience": 6,
        "min_age": 4,
        "lock_fraction": 0.04,
        "max_queries": 8,
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
        "bucket_width_multiple": args.bucket_width_multiple,
        "max_bucket_size": args.max_bucket_size,
        "force_semantic_plan_tokens": True,
    }
    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    timing_batches: list[dict[str, float]] = []
    score_cache: dict[tuple[str, str], int] = {}

    def score(text: str, example: Any) -> int:
        key = (str(example.example_id), text)
        if key not in score_cache:
            score_cache[key] = int(bool(score_generation(text, example)))
        return score_cache[key]

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(f"[phaselock-matched] batch={batch_id} size={len(batch)}", flush=True)
        planner = decode_planner(
            model, tokenizer, batch, config, with_phase2=True, **planner_common
        )
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
        for warmup in range(args.timing_warmups):
            orders = (("A0", "A2", "A3"), ("A2", "A3", "A0"), ("A3", "A0", "A2"))
            order = orders[warmup % len(orders)]
            for arm in order:
                if arm == "A0":
                    decode_surelock(
                        model, tokenizer, batch, config, epsilon=None,
                        confidence_percentile=20.0,
                        bucket_width_multiple=args.bucket_width_multiple,
                    )
                else:
                    decode_batch(
                        model, tokenizer, batch, config, arm, latency_table,
                        **replay_common, **replay,
                    )
        repeat_records: list[dict[str, Any]] = []
        for repeat in range(args.timing_repeats):
            orders = (("A0", "A2", "A3"), ("A2", "A3", "A0"), ("A3", "A0", "A2"))
            order = orders[repeat % len(orders)]
            records: dict[str, Any] = {}
            for arm in order:
                if arm == "A0":
                    records[arm] = decode_surelock(
                        model, tokenizer, batch, config, epsilon=None,
                        confidence_percentile=20.0,
                        bucket_width_multiple=args.bucket_width_multiple,
                    )
                else:
                    records[arm] = decode_batch(
                        model, tokenizer, batch, config, arm, latency_table,
                        **replay_common, **replay,
                    )
            repeat_records.append(records)
            timing_batches.append(
                {
                    "batch_id": float(batch_id),
                    "repeat": float(repeat),
                    "a0_latency_s": float(records["A0"]["latency_s"]),
                    "a2_latency_s": float(records["A2"]["latency_s"]),
                    "a3_latency_s": float(records["A3"]["latency_s"]),
                    "a3_reference_lock_count": float(
                        records["A3"]["reference_lock_count"]
                    ),
                    "a3_removed_row_token_forwards": float(
                        records["A3"]["removed_row_token_forwards"]
                    ),
                    "a3_switch_step": records["A3"]["switch_step"],
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
                        "example_id": example.example_id,
                        "batch_id": batch_id,
                        "repeat": repeat,
                        "a0_correct": score(a0_text, example),
                        "a1_correct": score(a1_text, example),
                        "a2_correct": score(a2_text, example),
                        "a3_correct": score(a3_text, example),
                        "a1_exact_a0": int(planner_tokens[local_index] == records["A0"]["token_ids"][local_index]),
                        "a2_exact_a1": int(records["A2"]["token_ids"][local_index] == planner_tokens[local_index]),
                        "a3_exact_a2": int(records["A3"]["token_ids"][local_index] == records["A2"]["token_ids"][local_index]),
                        "a3_exact_a0": int(records["A3"]["token_ids"][local_index] == records["A0"]["token_ids"][local_index]),
                        "a1_answer_changed_a0": int(_answer_changed(args.dataset, a1_text, a0_text)),
                        "a2_answer_changed_a1": int(_answer_changed(args.dataset, a2_text, a1_text)),
                        "a3_answer_changed_a2": int(_answer_changed(args.dataset, a3_text, a2_text)),
                        "a3_answer_changed_a0": int(_answer_changed(args.dataset, a3_text, a0_text)),
                        "a2_latency_s": records["A2"]["latency_s"],
                        "a3_latency_s": records["A3"]["latency_s"],
                        "a2_reference_lock_count": records["A2"]["reference_lock_count"],
                        "a3_reference_lock_count": records["A3"]["reference_lock_count"],
                        "a3_removed_row_token_forwards": records["A3"]["removed_row_token_forwards"],
                        "a3_switch_step": records["A3"]["switch_step"],
                        "a3_switch_backend": records["A3"]["switch_backend"],
                    }
                )
        for arm in ("A0", "A2", "A3"):
            first_tokens = repeat_records[0][arm]["token_ids"]
            if any(record[arm]["token_ids"] != first_tokens for record in repeat_records[1:]):
                raise RuntimeError(f"{arm} output changed across timing repeats in batch {batch_id}")

    stable_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable_by_id.setdefault(str(row["example_id"]), row)
    stable = list(stable_by_id.values())
    medians: list[dict[str, float]] = []
    for batch_id in sorted({int(row["batch_id"]) for row in timing_batches}):
        group = [row for row in timing_batches if int(row["batch_id"]) == batch_id]
        medians.append(
            {
                "a0_latency_s": float(np.median([row["a0_latency_s"] for row in group])),
                "a2_latency_s": float(np.median([row["a2_latency_s"] for row in group])),
                "a3_latency_s": float(np.median([row["a3_latency_s"] for row in group])),
            }
        )
    a3_vs_a2, a3_vs_a2_interval = _bootstrap_gain(
        medians,
        "a3_latency_s",
        "a2_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    a3_vs_a0, a3_vs_a0_interval = _bootstrap_gain(
        medians,
        "a3_latency_s",
        "a0_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    a2_vs_a0, a2_vs_a0_interval = _bootstrap_gain(
        medians,
        "a2_latency_s",
        "a0_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )
    correctness = {
        arm: [int(row[f"{arm}_correct"]) for row in stable]
        for arm in ("a0", "a1", "a2", "a3")
    }
    summary = {
        "examples": len(stable),
        "batches": len(medians),
        "accuracy": {
            arm: float(np.mean(values)) for arm, values in correctness.items()
        },
        "a0_to_a1": {
            "exact_rate": float(np.mean([row["a1_exact_a0"] for row in stable])),
            "answer_change_rate": float(np.mean([row["a1_answer_changed_a0"] for row in stable])),
            **transition_counts(correctness["a0"], correctness["a1"]),
        },
        "a1_to_a2": {
            "exact_rate": float(np.mean([row["a2_exact_a1"] for row in stable])),
            "answer_change_rate": float(np.mean([row["a2_answer_changed_a1"] for row in stable])),
            **transition_counts(correctness["a1"], correctness["a2"]),
            "runtime_gain_vs_a0": a2_vs_a0,
            "runtime_gain_vs_a0_ci95": a2_vs_a0_interval,
        },
        "a2_to_a3": {
            "exact_rate": float(np.mean([row["a3_exact_a2"] for row in stable])),
            "answer_change_rate": float(np.mean([row["a3_answer_changed_a2"] for row in stable])),
            **transition_counts(correctness["a2"], correctness["a3"]),
            "runtime_gain_vs_a2": a3_vs_a2,
            "runtime_gain_vs_a2_ci95": a3_vs_a2_interval,
            "runtime_gain_vs_a0": a3_vs_a0,
            "runtime_gain_vs_a0_ci95": a3_vs_a0_interval,
        },
        "mean_reference_locks_per_batch": float(
            np.mean([row["a3_reference_lock_count"] for row in timing_batches])
        ),
        "mean_removed_row_token_forwards_per_batch": float(
            np.mean([row["a3_removed_row_token_forwards"] for row in timing_batches])
        ),
        "compute_switch_rate": float(
            np.mean([row["a3_switch_step"] is not None for row in timing_batches])
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output_dir / "matched_plans.json").write_text(
        json.dumps(plans, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "latency_table": str(args.latency_table),
                "output_dir": str(args.output_dir),
                "planner": planner_common,
                "replay": replay_common,
                "plan_scope": "generated on the exact evaluation batch",
                "semantic_replay": "positions and exact A1 token identities",
                "timing_scope": "A2/A3 executor only; planner measured separately",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
