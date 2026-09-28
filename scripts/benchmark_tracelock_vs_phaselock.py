#!/usr/bin/env python3
"""Matched LLaDA comparison of TraceLock transfer and frozen PhaseLock."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_phase_lock_batched_online import (
    _answer_changed,
    decode_batch as decode_phaselock,
)
from benchmark_phase_lock_fused_replay import _examples, _load_plans
from benchmark_phaselock_controller_overhead import decode_planner
from benchmark_phaselock_matched_plans import (
    reference_replay_plan,
    semantic_replay_plan,
)
from benchmark_surelock_vs_phaselock import _bootstrap_gain, decode_surelock
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.llada_trace import (
    DecodeConfig,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.paper_baselines import parse_example_indices
from regret_remasking.phase_lock_sparse import FusedLatencyTable
from regret_remasking.tracelock_baseline import (
    TraceLockConfig,
    TraceLockController,
    pack_left_padded_controller_batch,
    select_tracelock_commits,
    soft_crop_window,
)
from run_four_arm_audit import transformer_blocks


METHODS = ("dense_anchor", "tracelock", "phaselock")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _install_hooks(
    model: torch.nn.Module, layers: list[int]
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    blocks = transformer_blocks(model)
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer in layers:
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"TraceLock capture layer must be in 1..{len(blocks)}")

        def make_hook(layer_number: int):
            def hook(_module: torch.nn.Module, _args: Any, output: Any) -> None:
                captures[layer_number] = output[0] if isinstance(output, tuple) else output

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    return captures, handles


@torch.inference_mode()
def decode_tracelock(
    model: torch.nn.Module,
    tokenizer: Any,
    controller: TraceLockController,
    projection: torch.Tensor,
    layers: list[int],
    examples: list[Any],
    config: DecodeConfig,
    *,
    window_width: int,
) -> dict[str, Any]:
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
            torch.ones((batch_size, config.gen_length), dtype=prompt_attention.dtype, device=device),
        ],
        dim=-1,
    )
    captures, handles = _install_hooks(model, layers)
    nfe = 0
    commit_count = 0
    controller_latency_s = 0.0
    controller_calls = 0
    per_sample_finish_step: list[int | None] = [None] * batch_size
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        for step in range(config.steps):
            active = x == config.mask_id
            active[:, :prompt_len] = False
            if not active.any():
                break
            captures.clear()
            logits = model(x, attention_mask=model_attention).logits
            nfe += 1
            if set(captures) != set(layers):
                raise RuntimeError("TraceLock hidden hooks did not capture all layers")
            proposals = torch.argmax(logits, dim=-1)
            projected = torch.stack(
                [captures[layer].float().matmul(projection) for layer in layers],
                dim=2,
            )
            # Training traces strip tokenizer left-padding. Repack each sample
            # identically for the controller so its positional embeddings do
            # not depend on the other prompts in the serving batch.
            (
                controller_projected,
                controller_attention,
                controller_active,
                packed_prompt_lengths,
            ) = pack_left_padded_controller_batch(
                projected,
                prompt_attention,
                active,
                generation_length=config.gen_length,
            )
            for batch_index in range(batch_size):
                actual_prompt = int(packed_prompt_lengths[batch_index].item())
                if not controller_active[batch_index].any():
                    continue
                _start, end = soft_crop_window(
                    controller_active[batch_index],
                    prompt_length=actual_prompt,
                    window_width=window_width,
                )
                controller_attention[batch_index, end:] = False
            torch.cuda.synchronize(device)
            controller_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = controller(controller_projected, controller_attention)
            torch.cuda.synchronize(device)
            controller_latency_s += time.perf_counter() - controller_started
            controller_calls += 1
            selected = torch.zeros_like(active)
            for batch_index in range(batch_size):
                actual_prompt = int(packed_prompt_lengths[batch_index].item())
                packed_selected = select_tracelock_commits(
                    output["probability"][batch_index].float(),
                    controller_active[batch_index],
                    prompt_length=actual_prompt,
                    window_width=window_width,
                    threshold=controller.config.deployment_probability,
                )
                relative = torch.nonzero(
                    packed_selected[
                        actual_prompt : actual_prompt + config.gen_length
                    ],
                    as_tuple=False,
                ).flatten()
                selected[batch_index, prompt_len + relative] = True
            x[selected] = proposals[selected]
            commit_count += int(selected.sum().item())
            for batch_index in range(batch_size):
                if per_sample_finish_step[batch_index] is None and not bool(
                    (x[batch_index, prompt_len:] == config.mask_id).any()
                ):
                    per_sample_finish_step[batch_index] = step + 1
            del logits, projected, controller_projected
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - started
    final_tokens = x[:, prompt_len:].detach().cpu()
    generations = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    return {
        "token_ids": final_tokens.tolist(),
        "generations": [text.strip() for text in generations],
        "latency_s": latency_s,
        "nfe": nfe,
        "mean_sample_steps": float(
            np.mean([value if value is not None else config.steps for value in per_sample_finish_step])
        ),
        "commit_count": commit_count,
        "controller_latency_s": controller_latency_s,
        "controller_calls": controller_calls,
        "prompt_len": prompt_len,
        "total_len": total_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--layers", default="30,31,32")
    parser.add_argument("--window-width", type=int, default=32)
    parser.add_argument("--phaselock-plan-dir", type=Path, required=True)
    parser.add_argument(
        "--phaselock-plan-mode",
        choices=["matched_batch", "external"],
        default="matched_batch",
    )
    parser.add_argument("--representation-label", default="mli_ultra")
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
    parser.add_argument("--seed", type=int, default=149)
    args = parser.parse_args()
    if args.batch_size < 2 or args.timing_repeats < 4 or args.timing_warmups < 1:
        raise ValueError("use batch >=2, at least one warmup, and at least four repeats")
    indices = parse_example_indices(args.example_indices)
    examples = _examples(args.dataset, args.split, indices)
    plans: dict[str, dict[str, Any]] = {}
    if args.phaselock_plan_mode == "external":
        plans = _load_plans(args.phaselock_plan_dir, args.representation_label)
        missing = [example.example_id for example in examples if example.example_id not in plans]
        if missing:
            raise RuntimeError(f"missing frozen PhaseLock plans: {missing}")
    with args.latency_table.open(encoding="utf-8") as handle:
        latency_table = FusedLatencyTable.from_rows(json.load(handle))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller_config = TraceLockConfig(**checkpoint["config"])
    controller = TraceLockController(controller_config)
    controller.load_state_dict(checkpoint["model"])
    projection = torch.from_numpy(np.load(args.projection)).float()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if len(layers) != 3:
        raise ValueError("TraceLock requires three hidden layers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        "compute_min_locked_fraction": 0.20,
        "compute_min_remaining_steps": 8,
        "compute_min_projected_saved_rows": 1500,
        "compute_min_expected_gain": 0.03,
        "bucket_width_multiple": 16,
        "max_bucket_size": 8,
        "force_semantic_plan_tokens": args.phaselock_plan_mode == "matched_batch",
    }
    phaselock_planner_common = {
        "representation_layers": [8, 16, 24],
        "influence_layer": 24,
        "drift_threshold": 0.002,
        "influence_threshold": 0.5,
        "patience": 6,
        "min_age": 4,
        "lock_fraction": 0.04,
        "max_queries": 8,
    }

    def run(method: str, batch: list[Any], batch_plans: list[dict[str, Any]]) -> dict[str, Any]:
        if method == "dense_anchor":
            record = decode_surelock(
                model,
                tokenizer,
                batch,
                config,
                epsilon=None,
                confidence_percentile=20.0,
                bucket_width_multiple=16,
            )
            record["mean_sample_steps"] = float(record["nfe"])
            return record
        if method == "tracelock":
            return decode_tracelock(
                model,
                tokenizer,
                controller,
                projection,
                layers,
                batch,
                config,
                window_width=args.window_width,
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
        record["mean_sample_steps"] = float(record["nfe"])
        return record

    rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], int] = {}

    def cached_score(generation: str, example: Any) -> int:
        key = (example.example_id, generation)
        if key not in score_cache:
            score_cache[key] = int(bool(score_generation(generation, example)))
        return score_cache[key]
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size
        print(f"[tracelock-baseline] batch={batch_id} size={len(batch)}", flush=True)
        planner_generations: list[str] | None = None
        planner_tokens: list[list[int]] | None = None
        if args.phaselock_plan_mode == "matched_batch":
            planner = decode_planner(
                model,
                tokenizer,
                batch,
                config,
                with_phase2=True,
                **phaselock_planner_common,
            )
            planner_tokens = planner["token_ids"]
            planner_generations = [
                value.strip()
                for value in tokenizer.batch_decode(
                    planner_tokens, skip_special_tokens=True
                )
            ]
            batch_plans = [
                {
                    "semantic_plan": semantic_replay_plan(semantic),
                    "reference_plan": reference_replay_plan(reference),
                }
                for semantic, reference in zip(
                    planner["semantic_plans"], planner["reference_plans"]
                )
            ]
        else:
            batch_plans = [plans[example.example_id] for example in batch]
        for warmup in range(args.timing_warmups):
            order = METHODS[warmup:] + METHODS[:warmup]
            for method in order:
                run(method, batch, batch_plans)
        for repeat in range(args.timing_repeats):
            shift = repeat % len(METHODS)
            order = METHODS[shift:] + METHODS[:shift]
            records = {method: run(method, batch, batch_plans) for method in order}
            correctness = {
                method: [
                    cached_score(generation, example)
                    for generation, example in zip(records[method]["generations"], batch)
                ]
                for method in METHODS
            }
            for local_index, example in enumerate(batch):
                dense = records["dense_anchor"]
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
                for method in METHODS:
                    record = records[method]
                    row[f"{method}_latency_s"] = record["latency_s"]
                    row[f"{method}_correct"] = correctness[method][local_index]
                    row[f"{method}_exact_dense"] = int(
                        record["token_ids"][local_index] == dense["token_ids"][local_index]
                    )
                    row[f"{method}_answer_changed_dense"] = int(
                        _answer_changed(
                            args.dataset,
                            record["generations"][local_index],
                            dense["generations"][local_index],
                        )
                    )
                    row[f"{method}_nfe"] = record["nfe"]
                    row[f"{method}_mean_sample_steps"] = record["mean_sample_steps"]
                if planner_generations is not None and planner_tokens is not None:
                    planner_text = planner_generations[local_index]
                    row["phaselock_a1_correct"] = cached_score(planner_text, example)
                    row["phaselock_a1_exact_dense"] = int(
                        planner_tokens[local_index] == dense["token_ids"][local_index]
                    )
                    row["phaselock_a1_answer_changed_dense"] = int(
                        _answer_changed(
                            args.dataset,
                            planner_text,
                            dense["generations"][local_index],
                        )
                    )
                    row["phaselock_exact_a1"] = int(
                        records["phaselock"]["token_ids"][local_index]
                        == planner_tokens[local_index]
                    )
                row["tracelock_controller_latency_s"] = records["tracelock"]["controller_latency_s"]
                row["tracelock_commit_count"] = records["tracelock"]["commit_count"]
                rows.append(row)
                if any(
                    correctness[method][local_index] != correctness["dense_anchor"][local_index]
                    for method in METHODS[1:]
                ):
                    changes.append(
                        {
                            **row,
                            "outputs": {
                                method: records[method]["generations"][local_index]
                                for method in METHODS
                            },
                        }
                    )

    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((int(row["batch_id"]), int(row["repeat"])), row)
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in unique.values():
        by_batch.setdefault(int(row["batch_id"]), []).append(row)
    batch_medians = [
        {
            f"{method}_latency_s": float(
                np.median([row[f"{method}_latency_s"] for row in group])
            )
            for method in METHODS
        }
        for _, group in sorted(by_batch.items())
    ]
    stable: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable.setdefault(str(row["example_id"]), row)
    example_rows = list(stable.values())
    summary: dict[str, Any] = {
        "examples": len(example_rows),
        "batches": len(batch_medians),
        "methods": list(METHODS),
    }
    for method_index, method in enumerate(METHODS):
        summary[method] = {
            "accuracy": float(np.mean([row[f"{method}_correct"] for row in example_rows])),
            "exact_dense_rate": float(np.mean([row[f"{method}_exact_dense"] for row in example_rows])),
            "answer_change_dense_rate": float(np.mean([row[f"{method}_answer_changed_dense"] for row in example_rows])),
            "mean_batch_nfe": float(np.mean([row[f"{method}_nfe"] for row in unique.values()])),
            "mean_sample_steps": float(np.mean([row[f"{method}_mean_sample_steps"] for row in unique.values()])),
        }
        if method_index:
            gain, interval = _bootstrap_gain(
                batch_medians,
                f"{method}_latency_s",
                "dense_anchor_latency_s",
                samples=args.bootstrap_samples,
                seed=args.seed + method_index,
            )
            summary[method]["runtime_gain_vs_dense"] = gain
            summary[method]["runtime_gain_vs_dense_ci95"] = interval
    gain, interval = _bootstrap_gain(
        batch_medians,
        "phaselock_latency_s",
        "tracelock_latency_s",
        samples=args.bootstrap_samples,
        seed=args.seed + 50,
    )
    summary["phaselock_vs_tracelock"] = {
        "runtime_gain": gain,
        "runtime_gain_ci95": interval,
        "accuracy_delta": summary["phaselock"]["accuracy"] - summary["tracelock"]["accuracy"],
    }
    summary["tracelock"]["mean_controller_fraction"] = float(
        np.mean(
            [
                row["tracelock_controller_latency_s"] / row["tracelock_latency_s"]
                for row in unique.values()
            ]
        )
    )
    if args.phaselock_plan_mode == "matched_batch":
        summary["phaselock_a1"] = {
            "accuracy": float(
                np.mean([row["phaselock_a1_correct"] for row in example_rows])
            ),
            "exact_dense_rate": float(
                np.mean([row["phaselock_a1_exact_dense"] for row in example_rows])
            ),
            "answer_change_dense_rate": float(
                np.mean(
                    [row["phaselock_a1_answer_changed_dense"] for row in example_rows]
                )
            ),
            "a3_exact_a1_rate": float(
                np.mean([row["phaselock_exact_a1"] for row in example_rows])
            ),
        }
    config_payload = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "projection": str(args.projection),
        "phaselock_plan_dir": str(args.phaselock_plan_dir),
        "latency_table": str(args.latency_table),
        "output_dir": str(args.output_dir),
        "trace_lock_status": "paper-specified controller; seeded Gaussian LLaDA projection adaptation",
        "official_release_issue": "public TraceLock commit bc4134c omits imported tracelock/models source and LLaDA path",
        "phaselock_plan_scope": (
            "generated by the frozen controller on the exact evaluation batch"
            if args.phaselock_plan_mode == "matched_batch"
            else "external saved plan"
        ),
        "phaselock_planner": phaselock_planner_common,
        "timing": {
            "tracelock_online_controller_and_projection_included": True,
            "phaselock_online_planner_included": False,
            "cuda_synchronized": True,
            "balanced_rotating_order": True,
            "task_scoring_excluded": True,
        },
        "runtime_environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output_dir / "quality_changes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "rows.csv", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
