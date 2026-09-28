#!/usr/bin/env python3
"""Generate future-stability traces for a TraceLock LLaDA reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from regret_remasking.llada_trace import (
    get_num_transfer_tokens,
    infer_mask_token_id,
    load_llada,
    model_device,
    prepare_prompts,
)
from regret_remasking.tracelock_baseline import (
    STATE_GEN,
    STATE_LOCKED,
    STATE_PROMPT,
)
from run_four_arm_audit import transformer_blocks


GSM8K_SUFFIX = (
    "\n\nSolve this math word problem step by step.\n"
    "Show your reasoning clearly.\n"
    "At the end, output the final answer on a new line in the format:\n"
    "#### <answer>"
)


def sampled_steps(total_steps: int, ratio: float, seed: int) -> list[int]:
    if total_steps <= 0 or not 0.0 < ratio <= 1.0:
        raise ValueError("total_steps and sampling ratio must be positive")
    count = min(total_steps, max(1, int(round(total_steps * ratio))))
    if count == total_steps:
        return list(range(total_steps))
    generator = random.Random(seed)
    selected = set(generator.sample(range(total_steps), count))
    selected.add(0)
    selected.add(total_steps - 1)
    while len(selected) > count:
        removable = sorted(selected - {0, total_steps - 1})
        selected.remove(removable[-1])
    return sorted(selected)


def _prompt(source: str, row: dict[str, Any]) -> str:
    if source == "gsm8k":
        return str(row.get("question", "")).strip() + GSM8K_SUFFIX
    if source == "alpaca":
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        return (
            f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
        ).strip()
    if source == "natural_questions":
        return str(row.get("question", "")).strip()
    if source == "kodcode":
        entry = str(row.get("entry_point", "")).strip()
        requirement = (
            f"- Implement the function `{entry}` exactly as specified.\n" if entry else ""
        )
        return (
            "Write Python code that passes the given tests.\n\n"
            f"Problem:\n{str(row.get('prompt', '')).rstrip()}\n\n"
            f"Unit tests:\n{str(row.get('test', '')).rstrip()}\n\n"
            "Requirements:\n"
            f"{requirement}"
            "- Return only Python code.\n"
            "- Put the final code inside one ```python``` block.\n"
            "- Do not include explanations or extra text.\n"
        )
    raise ValueError(f"unsupported source: {source}")


def _dataset(source: str):
    if source == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="train")
    if source == "alpaca":
        return load_dataset("yahma/alpaca-cleaned", split="train")
    if source == "natural_questions":
        return load_dataset(
            "BOB12311/natural-questions-slim-short-answer", split="train"
        )
    if source == "kodcode":
        return load_dataset("BOB12311/kodcode-humaneval-like", split="train")
    raise ValueError(f"unsupported source: {source}")


def _split_for(sample_id: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if value < validation_fraction else "train"


def inventory_existing_traces(
    output_dir: Path, source: str, *, expected_steps: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Inventory complete examples from an interrupted trace-generation run.

    A sample is resumable only when all expected step files are present in one
    split.  Samples interrupted while their files were being written are left
    out of the manifest and regenerated; deterministic filenames overwrite the
    partial files.
    """
    by_sample: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        for path in (output_dir / split).glob(f"{source}-*_step_*.pt"):
            sample_id, separator, _step = path.stem.rpartition("_step_")
            if not separator:
                raise ValueError(f"invalid trace filename: {path}")
            record = by_sample.setdefault(
                sample_id,
                {"split": split, "count": 0},
            )
            if record["split"] != split:
                raise ValueError(f"trace sample appears in both splits: {sample_id}")
            record["count"] += 1

    complete: list[dict[str, Any]] = []
    partial = 0
    for sample_id, record in sorted(by_sample.items()):
        count = int(record["count"])
        if count != expected_steps:
            partial += 1
            continue
        source_index = int(sample_id.rsplit("-", 1)[1])
        complete.append(
            {
                "sample_id": sample_id,
                "source": source,
                "source_index": source_index,
                "split": str(record["split"]),
                "prompt_length": None,
                "saved_steps": count,
                "recovered_from_existing_files": True,
            }
        )
    return complete, {
        "complete_examples": len(complete),
        "partial_examples": partial,
        "trace_files": sum(int(record["count"]) for record in by_sample.values()),
    }


def _projection(
    hidden_dim: int, projected_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        hidden_dim, projected_dim, generator=generator, dtype=torch.float32
    ) / math.sqrt(float(projected_dim))
    return matrix.to(device)


def _install_hooks(
    model: torch.nn.Module, layers: list[int]
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    blocks = transformer_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer in layers:
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"capture layer must be in 1..{len(blocks)}")

        def make_hook(layer_number: int):
            def hook(_module: torch.nn.Module, _args: Any, output: Any) -> None:
                captured[layer_number] = output[0] if isinstance(output, tuple) else output

            return hook

        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    return captured, handles


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--source",
        choices=["gsm8k", "natural_questions", "alpaca", "kodcode"],
        required=True,
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--layers", default="30,31,32")
    parser.add_argument("--projected-hidden-dim", type=int, default=128)
    parser.add_argument("--step-sample-ratio", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="reuse complete trace samples already present in the output directory",
    )
    parser.add_argument(
        "--target-completed",
        type=int,
        default=None,
        help="stop after this many complete examples, including resumed examples",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit <= 0 or args.batch_size <= 0:
        raise ValueError("limit and batch size must be positive")
    if args.target_completed is not None and args.target_completed <= 0:
        raise ValueError("target-completed must be positive")
    if args.gen_length % args.block_length:
        raise ValueError("generation length must divide into blocks")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks:
        raise ValueError("steps must divide evenly across blocks")
    steps_per_block = args.steps // num_blocks
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if len(layers) != 3:
        raise ValueError("TraceLock requires exactly the last three hidden snapshots")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train").mkdir(exist_ok=True)
    (args.output_dir / "val").mkdir(exist_ok=True)
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    device = model_device(model)
    mask_id = infer_mask_token_id(tokenizer)
    captured, handles = _install_hooks(model, layers)
    dataset = _dataset(args.source)
    indices = list(range(len(dataset)))
    source_seed = {
        "gsm8k": 1,
        "natural_questions": 2,
        "alpaca": 3,
        "kodcode": 4,
    }[args.source]
    random.Random(args.seed + source_seed).shuffle(indices)
    indices = indices[args.offset : args.offset + args.limit]
    expected_saved_steps = len(sampled_steps(args.steps, args.step_sample_ratio, args.seed))
    existing_manifest: list[dict[str, Any]] = []
    resume_audit = {
        "complete_examples": 0,
        "partial_examples": 0,
        "trace_files": 0,
    }
    if args.resume_existing:
        existing_manifest, resume_audit = inventory_existing_traces(
            args.output_dir,
            args.source,
            expected_steps=expected_saved_steps,
        )
    if (
        args.target_completed is not None
        and len(existing_manifest) > args.target_completed
    ):
        raise ValueError(
            "existing complete examples exceed target-completed: "
            f"{len(existing_manifest)} > {args.target_completed}"
        )
    existing_ids = {str(record["sample_id"]) for record in existing_manifest}
    projection: torch.Tensor | None = None
    saved_steps = sum(int(record["saved_steps"]) for record in existing_manifest)
    new_saved_steps = 0
    completed = len(existing_manifest)
    resumed_examples = completed
    new_completed = 0
    skipped = 0
    total_nfe = 0
    started = time.perf_counter()
    manifests: list[dict[str, Any]] = list(existing_manifest)
    try:
        for group_start in range(0, len(indices), args.batch_size):
            if args.target_completed is not None and completed >= args.target_completed:
                break
            source_indices = indices[group_start : group_start + args.batch_size]
            source_indices = [
                index
                for index in source_indices
                if f"{args.source}-{int(index):08d}" not in existing_ids
            ]
            if not source_indices:
                continue
            prompts = [_prompt(args.source, dict(dataset[int(index)])) for index in source_indices]
            nonempty = [index for index, prompt in enumerate(prompts) if prompt]
            skipped += len(prompts) - len(nonempty)
            if not nonempty:
                continue
            source_indices = [source_indices[index] for index in nonempty]
            prompts = [prompts[index] for index in nonempty]
            input_ids, prompt_attention = prepare_prompts(tokenizer, prompts, device)
            prompt_lengths = prompt_attention.sum(dim=1).to(dtype=torch.long)
            valid = torch.nonzero(
                prompt_lengths <= args.max_prompt_tokens, as_tuple=False
            ).flatten()
            skipped += len(prompts) - int(valid.numel())
            if valid.numel() == 0:
                continue
            if args.target_completed is not None:
                remaining = args.target_completed - completed
                valid = valid[:remaining]
            source_indices = [source_indices[int(index)] for index in valid.tolist()]
            input_ids = input_ids.index_select(0, valid)
            prompt_attention = prompt_attention.index_select(0, valid)
            prompt_lengths = prompt_lengths.index_select(0, valid)
            # Remove left-padding columns contributed only by filtered prompts.
            prompt_len = int(prompt_lengths.max().item())
            input_ids = input_ids[:, -prompt_len:]
            prompt_attention = prompt_attention[:, -prompt_len:]
            batch_size = len(source_indices)
            total_len = prompt_len + args.gen_length
            x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
            x[:, :prompt_len] = input_ids
            attention_mask = torch.cat(
                [
                    prompt_attention,
                    torch.ones(
                        (batch_size, args.gen_length),
                        dtype=prompt_attention.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )
            sample_ids = [f"{args.source}-{int(index):08d}" for index in source_indices]
            kept_by_sample = [
                set(sampled_steps(args.steps, args.step_sample_ratio, args.seed + int(index)))
                for index in source_indices
            ]
            pending_by_sample: list[list[dict[str, torch.Tensor | int]]] = [
                [] for _ in source_indices
            ]
            for block_index in range(num_blocks):
                block_start = prompt_len + block_index * args.block_length
                block_end = block_start + args.block_length
                schedule = get_num_transfer_tokens(
                    x[:, block_start:block_end] == mask_id, steps_per_block
                )
                for step_in_block in range(steps_per_block):
                    global_step = block_index * steps_per_block + step_in_block
                    mask_index = x == mask_id
                    captured.clear()
                    logits = model(x, attention_mask=attention_mask).logits
                    total_nfe += 1
                    if set(captured) != set(layers):
                        raise RuntimeError("hidden hooks did not capture all TraceLock layers")
                    proposals = torch.argmax(logits, dim=-1)
                    max_logits = logits.amax(dim=-1)
                    confidence = torch.exp(
                        max_logits.float() - torch.logsumexp(logits.float(), dim=-1)
                    )
                    if projection is None:
                        hidden_dim = int(captured[layers[0]].shape[-1])
                        projection = _projection(
                            hidden_dim,
                            args.projected_hidden_dim,
                            args.seed,
                            device,
                        )
                        np.save(
                            args.output_dir / "projection.npy",
                            projection.detach().cpu().numpy().astype(np.float32),
                        )
                    if any(global_step in kept for kept in kept_by_sample):
                        projected_batch = torch.stack(
                            [captured[layer].float().matmul(projection) for layer in layers],
                            dim=2,
                        ).to(dtype=torch.float16).cpu()
                        proposals_cpu = proposals.to(dtype=torch.int32).cpu()
                        masks_cpu = mask_index.cpu()
                        for batch_index, kept in enumerate(kept_by_sample):
                            if global_step not in kept:
                                continue
                            actual_prompt = int(prompt_lengths[batch_index].item())
                            prompt_start = prompt_len - actual_prompt
                            projected = torch.cat(
                                [
                                    projected_batch[batch_index, prompt_start:prompt_len],
                                    projected_batch[batch_index, prompt_len:],
                                ],
                                dim=0,
                            ).contiguous()
                            proposal = torch.cat(
                                [
                                    proposals_cpu[batch_index, prompt_start:prompt_len],
                                    proposals_cpu[batch_index, prompt_len:],
                                ]
                            ).contiguous()
                            state = torch.full(
                                (actual_prompt + args.gen_length,),
                                STATE_LOCKED,
                                dtype=torch.int8,
                            )
                            state[:actual_prompt] = STATE_PROMPT
                            state[actual_prompt:][masks_cpu[batch_index, prompt_len:]] = STATE_GEN
                            pending_by_sample[batch_index].append(
                                {
                                    "projected_hidden": projected,
                                    "proposal": proposal,
                                    "state": state,
                                    "attention_mask": torch.ones_like(state, dtype=torch.bool),
                                    "global_step": global_step,
                                }
                            )
                    allowed = mask_index.clone()
                    allowed[:, :prompt_len] = False
                    allowed[:, block_end:] = False
                    scores = torch.where(
                        allowed,
                        confidence,
                        torch.full_like(confidence, -torch.inf),
                    )
                    for batch_index in range(batch_size):
                        count = int(schedule[batch_index, step_in_block].item())
                        selected = torch.topk(scores[batch_index], k=count).indices
                        x[batch_index, selected] = proposals[batch_index, selected]
                    del logits, max_logits, confidence

            x_cpu = x.to(dtype=torch.int32).cpu()
            for batch_index, (source_index, sample_id, pending) in enumerate(
                zip(source_indices, sample_ids, pending_by_sample)
            ):
                actual_prompt = int(prompt_lengths[batch_index].item())
                prompt_start = prompt_len - actual_prompt
                final_tokens = torch.cat(
                    [
                        x_cpu[batch_index, prompt_start:prompt_len],
                        x_cpu[batch_index, prompt_len:],
                    ]
                ).contiguous()
                split = _split_for(sample_id, args.seed, args.validation_fraction)
                for capture in pending:
                    step = int(capture["global_step"])
                    proposal = capture.pop("proposal")
                    if not isinstance(proposal, torch.Tensor):
                        raise RuntimeError("invalid proposal trace")
                    payload = {
                        **capture,
                        "label": (proposal == final_tokens).to(dtype=torch.float32),
                        "prompt_length": actual_prompt,
                        "generation_length": args.gen_length,
                        "sample_id": sample_id,
                        "source": args.source,
                        "source_index": int(source_index),
                    }
                    path = args.output_dir / split / f"{sample_id}_step_{step:04d}.pt"
                    torch.save(payload, path)
                    saved_steps += 1
                    new_saved_steps += 1
                manifests.append(
                    {
                        "sample_id": sample_id,
                        "source": args.source,
                        "source_index": int(source_index),
                        "split": split,
                        "prompt_length": actual_prompt,
                        "saved_steps": len(pending),
                    }
                )
                completed += 1
                new_completed += 1
                print(
                    f"[tracelock-trace] {completed}/"
                    f"{args.target_completed or len(indices)} {sample_id} "
                    f"steps={len(pending)} new={new_completed}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    summary = {
        "source": args.source,
        "requested_examples": args.target_completed or len(indices),
        "candidate_examples": len(indices),
        "target_completed_examples": args.target_completed,
        "completed_examples": completed,
        "resumed_examples": resumed_examples,
        "new_completed_examples": new_completed,
        "skipped_examples": skipped,
        "saved_step_samples": saved_steps,
        "new_saved_step_samples": new_saved_steps,
        "nfe": total_nfe,
        "elapsed_s": time.perf_counter() - started,
        "resume_audit": resume_audit,
        "projection": "seeded Gaussian adapter shared by three hidden layers",
        "paper_deviation": "LLaDA projection checkpoint was not released",
    }
    (args.output_dir / f"manifest_{args.source}_{args.offset:06d}.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )
    (args.output_dir / f"summary_{args.source}_{args.offset:06d}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    config = {**vars(args), "output_dir": str(args.output_dir), "layers": layers}
    (args.output_dir / f"trace_config_{args.source}_{args.offset:06d}.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.target_completed is not None and completed != args.target_completed:
        raise RuntimeError(
            f"candidate range exhausted with {completed}/{args.target_completed} "
            "complete examples"
        )


if __name__ == "__main__":
    main()
