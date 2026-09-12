#!/usr/bin/env python3
"""Paired wall-clock benchmark of dense A2 versus packed-row A3 PhaseLock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from regret_remasking.data import load_examples
from regret_remasking.llada_trace import (
    DecodeConfig,
    infer_mask_token_id,
    load_llada,
    model_device,
)
from regret_remasking.phase_lock import PhaseLockPolicy
from run_phase_lock import _warmup_model, decode_one


SEMANTIC_DEFAULTS: dict[str, Any] = {
    "semantic_lock_fraction": 0.06,
    "beta_kl": 1.0,
    "beta_fate": 2.0,
    "semantic_min_confidence": 0.0,
    "semantic_min_margin": 0.0,
    "semantic_max_kl": None,
    "semantic_min_runlength": 1,
    "semantic_min_block_age": 0,
    "semantic_optional_steps_per_block": -1,
    "semantic_require_posterior_history": False,
}

REPRESENTATION_DEFAULTS: dict[str, Any] = {
    "representation_layer": 8,
    "representation_threshold": 0.0,
    "representation_patience": 1,
    "representation_min_age": 1,
    "representation_lock_fraction": 1.0,
}


def _load_specs(path: Path, key: str, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty {key} list")
    return [{**defaults, **spec} for spec in raw]


def _policy(semantic: dict[str, Any], representation: dict[str, Any], compute: str) -> PhaseLockPolicy:
    return PhaseLockPolicy(
        semantic=str(semantic["policy"]),
        representation=str(representation["policy"]),
        compute=compute,
        semantic_lock_fraction=float(semantic["semantic_lock_fraction"]),
        beta_kl=float(semantic["beta_kl"]),
        beta_fate=float(semantic["beta_fate"]),
        semantic_min_confidence=float(semantic["semantic_min_confidence"]),
        semantic_min_margin=float(semantic["semantic_min_margin"]),
        semantic_max_kl=(
            None if semantic["semantic_max_kl"] is None else float(semantic["semantic_max_kl"])
        ),
        semantic_min_runlength=int(semantic["semantic_min_runlength"]),
        semantic_min_block_age=int(semantic["semantic_min_block_age"]),
        semantic_optional_steps_per_block=int(semantic["semantic_optional_steps_per_block"]),
        semantic_require_posterior_history=bool(semantic["semantic_require_posterior_history"]),
        representation_threshold=float(representation["representation_threshold"]),
        representation_patience=int(representation["representation_patience"]),
        representation_min_age=int(representation["representation_min_age"]),
        representation_lock_fraction=float(representation["representation_lock_fraction"]),
    )


def _plan_hash(plan: dict[int, list[dict[str, Any]]]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _bootstrap_example_mean(
    rows: list[dict[str, Any]], *, seed: int, samples: int
) -> tuple[float, float]:
    by_example: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_example[str(row["example_id"])].append(float(row["runtime_gain"]))
    example_means = np.asarray(
        [np.mean(values) for _, values in sorted(by_example.items())], dtype=np.float64
    )
    if example_means.size == 1:
        value = float(example_means[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, example_means.size, size=(samples, example_means.size))
    estimates = example_means[indices].mean(axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _summaries(
    rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["semantic_label"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, semantic), group in sorted(groups.items()):
        gains = [float(row["runtime_gain"]) for row in group]
        speedups = [float(row["speedup"]) for row in group]
        a2_times = [float(row["a2_latency_s"]) for row in group]
        a3_times = [float(row["a3_latency_s"]) for row in group]
        ci_low, ci_high = _bootstrap_example_mean(
            group, seed=seed, samples=bootstrap_samples
        )
        exact_rate = float(np.mean([int(row["a3_exact_a2"]) for row in group]))
        output.append(
            {
                "dataset": dataset,
                "semantic_label": semantic,
                "examples": len({str(row["example_id"]) for row in group}),
                "timed_pairs": len(group),
                "a3_exact_a2_rate": exact_rate,
                "answer_or_correctness_change_rate": float(
                    np.mean([int(row["answer_or_correctness_changed"]) for row in group])
                ),
                "median_a2_latency_s": _percentile(a2_times, 50),
                "median_a3_latency_s": _percentile(a3_times, 50),
                "mean_runtime_gain": float(np.mean(gains)),
                "median_runtime_gain": _percentile(gains, 50),
                "p10_runtime_gain": _percentile(gains, 10),
                "p90_runtime_gain": _percentile(gains, 90),
                "example_bootstrap_gain_ci95_low": ci_low,
                "example_bootstrap_gain_ci95_high": ci_high,
                "positive_runtime_gain_rate": float(np.mean([gain > 0.0 for gain in gains])),
                "mean_speedup": float(np.mean(speedups)),
                "mean_generation_reference_opportunity": float(
                    np.mean([float(row["generation_reference_opportunity"]) for row in group])
                ),
                "mean_total_row_removal_fraction": float(
                    np.mean([float(row["total_row_removal_fraction"]) for row in group])
                ),
                "profitability_pass": int(exact_rate == 1.0 and ci_low > 0.0),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_total_memory_bytes": properties.total_memory,
    }


def _select_examples(args: argparse.Namespace) -> list[Any]:
    if not args.example_indices:
        return load_examples(args.dataset, args.split, args.limit, offset=args.offset)
    indices = [int(item.strip()) for item in args.example_indices.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("example indices must be distinct non-negative integers")
    examples = []
    for index in indices:
        loaded = load_examples(args.dataset, args.split, 1, offset=index)
        if len(loaded) != 1:
            raise ValueError(f"dataset index is unavailable: {index}")
        examples.append(loaded[0])
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=["gsm8k", "math500", "math-500", "humaneval", "human_eval", "human-eval"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--example-indices", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--semantic-policy-configs", type=Path, required=True)
    parser.add_argument("--semantic-labels", default=None)
    parser.add_argument("--representation-policy-configs", type=Path, required=True)
    parser.add_argument("--representation-label", default="robust_delayed_one_update")
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timing_warmups < 1 or args.timing_repeats < 2:
        raise ValueError("benchmark requires at least one warmup and two timed repeats")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantics = _load_specs(
        args.semantic_policy_configs, "semantic_policies", SEMANTIC_DEFAULTS
    )
    if args.semantic_labels:
        requested = {item.strip() for item in args.semantic_labels.split(",") if item.strip()}
        semantics = [spec for spec in semantics if str(spec["label"]) in requested]
        if {str(spec["label"]) for spec in semantics} != requested:
            raise ValueError("one or more requested semantic labels were not found")
    representations = _load_specs(
        args.representation_policy_configs,
        "representation_policies",
        REPRESENTATION_DEFAULTS,
    )
    matches = [
        spec for spec in representations if str(spec["label"]) == args.representation_label
    ]
    if len(matches) != 1:
        raise ValueError("representation label must resolve to exactly one policy")
    representation = matches[0]
    if str(representation["policy"]) != "always" or int(representation["representation_min_age"]) != 1:
        raise ValueError("compute benchmark is locked to the one-update-delay representation policy")

    examples = _select_examples(args)
    if not examples:
        raise ValueError("no examples loaded")
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        mask_id=infer_mask_token_id(tokenizer),
        temperature=0.0,
        cpv_window=4,
    )
    _warmup_model(model, tokenizer, examples[0], config)

    config_record = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "semantic_policy_configs": str(args.semantic_policy_configs),
        "representation_policy_configs": str(args.representation_policy_configs),
        "resolved_semantic_policies": semantics,
        "resolved_representation_policy": representation,
        "timing_protocol": {
            "plans_excluded": True,
            "task_scoring_excluded": True,
            "backend_order": "alternating_by_repeat",
            "cuda_synchronize": True,
            "bootstrap_unit": "example",
        },
        "runtime_environment": _runtime_environment(model_device(model)),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    pair_rows: list[dict[str, Any]] = []
    failures_path = args.output_dir / "safety_failures.jsonl"
    for example_index, example in enumerate(examples, start=1):
        print(f"[compute-benchmark] {example_index}/{len(examples)} {example.example_id}", flush=True)
        for semantic in semantics:
            semantic_label = str(semantic["label"])
            dense_policy = _policy(
                semantic, {**representation, "policy": "always"}, "dense"
            )
            a1, semantic_plan, _ = decode_one(
                model,
                tokenizer,
                example,
                config,
                dense_policy,
                f"P1_{semantic_label}",
                None,
                representation_layer=int(representation["representation_layer"]),
            )
            planner, _, reference_plan = decode_one(
                model,
                tokenizer,
                example,
                config,
                dense_policy,
                f"RP_{semantic_label}_{args.representation_label}_dense_a1",
                None,
                semantic_plan=semantic_plan,
                generate_reference_plan=True,
                representation_layer=int(representation["representation_layer"]),
            )
            if planner["token_ids"] != a1["token_ids"]:
                raise RuntimeError("dense reference planner changed the A1 output")

            policies = {
                "A2": _policy(semantic, representation, "kv_cache"),
                "A3": _policy(semantic, representation, "row_sparse_packed"),
            }
            arms = {
                "A2": f"P2_{semantic_label}_{args.representation_label}_kv_cache",
                "A3": f"P3_{semantic_label}_{args.representation_label}_row_sparse_packed",
            }
            for warmup in range(args.timing_warmups):
                warm_records = {}
                for backend in ("A3", "A2") if warmup % 2 == 0 else ("A2", "A3"):
                    warm_records[backend], _, _ = decode_one(
                        model,
                        tokenizer,
                        example,
                        config,
                        policies[backend],
                        arms[backend],
                        None,
                        semantic_plan=semantic_plan,
                        reference_plan=reference_plan,
                        representation_layer=int(representation["representation_layer"]),
                    )
                if warm_records["A2"]["token_ids"] != a1["token_ids"]:
                    raise RuntimeError("A2 warmup changed the fixed A1 output")

            for repeat in range(args.timing_repeats):
                order = ("A2", "A3") if repeat % 2 == 0 else ("A3", "A2")
                timed: dict[str, dict[str, Any]] = {}
                for backend in order:
                    timed[backend], _, _ = decode_one(
                        model,
                        tokenizer,
                        example,
                        config,
                        policies[backend],
                        arms[backend],
                        None,
                        semantic_plan=semantic_plan,
                        reference_plan=reference_plan,
                        representation_layer=int(representation["representation_layer"]),
                    )
                a2 = timed["A2"]
                a3 = timed["A3"]
                a2_latency = float(a2["latency_s"])
                a3_latency = float(a3["latency_s"])
                exact = a3["token_ids"] == a2["token_ids"]
                answer_or_correctness_changed = (
                    a3["generation"] != a2["generation"]
                    or bool(a3["correct"]) != bool(a2["correct"])
                )
                row = {
                    "dataset": str(example.dataset),
                    "example_id": str(example.example_id),
                    "semantic_label": semantic_label,
                    "representation_label": args.representation_label,
                    "repeat": repeat,
                    "execution_order": "_then_".join(order),
                    "semantic_plan_sha256": _plan_hash(semantic_plan),
                    "reference_plan_sha256": _plan_hash(reference_plan),
                    "a2_latency_s": a2_latency,
                    "a3_latency_s": a3_latency,
                    "runtime_gain": 1.0 - a3_latency / max(a2_latency, 1e-12),
                    "speedup": a2_latency / max(a3_latency, 1e-12),
                    "a2_exact_a1": int(a2["token_ids"] == a1["token_ids"]),
                    "a3_exact_a2": int(exact),
                    "answer_or_correctness_changed": int(answer_or_correctness_changed),
                    "generation_reference_opportunity": float(
                        a3["reference_opportunity_fraction"]
                    ),
                    "total_row_removal_fraction": float(a3["total_row_removal_fraction"]),
                    "prompt_len": int(a3["prompt_len"]),
                    "gen_length": int(a3["gen_length"]),
                    "nfe": int(a3["nfe"]),
                }
                pair_rows.append(row)
                _append_jsonl(args.output_dir / "timing_pairs.jsonl", row)
                if not exact or answer_or_correctness_changed or not row["a2_exact_a1"]:
                    _append_jsonl(
                        failures_path,
                        {
                            **row,
                            "a1_token_ids": a1["token_ids"],
                            "a2_token_ids": a2["token_ids"],
                            "a3_token_ids": a3["token_ids"],
                        },
                    )

    summaries = _summaries(
        pair_rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    _write_csv(args.output_dir / "timing_pairs.csv", pair_rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
