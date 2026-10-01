#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from regret_remasking.data import (
    build_prompt,
    generated_answers_equivalent,
    load_examples,
    normalize_answer,
    score_generation_with_status,
)
from regret_remasking.llada_trace import ensure_transformers_tied_weight_compat


MASK_ID = 126336


@dataclass(frozen=True)
class ArmSpec:
    generator: str
    steps: str
    thresholded: bool


ARM_SPECS = {
    "fixed_full": ArmSpec("full", "full", False),
    "prefix_fixed": ArmSpec("prefix", "full", False),
    "parallel_full": ArmSpec("full", "one_per_block", True),
    "prefix_parallel": ArmSpec("prefix", "one_per_block", True),
    # The official GSM8K launcher uses a full step budget for dual-cache
    # decoding. This also guarantees enough iterations to remove every mask.
    "dual_parallel": ArmSpec("dual", "full", True),
}
PAIR_SPECS = (
    ("fixed_full", "prefix_fixed"),
    ("fixed_full", "parallel_full"),
    ("fixed_full", "prefix_parallel"),
    ("fixed_full", "dual_parallel"),
    ("prefix_fixed", "prefix_parallel"),
    ("parallel_full", "prefix_parallel"),
    ("parallel_full", "dual_parallel"),
    ("prefix_parallel", "dual_parallel"),
)


class CountingModel(torch.nn.Module):
    """Transparent model wrapper for method-level forward-work counters."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.reset_counts()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_counts(self) -> None:
        self.forward_calls = 0
        self.input_token_positions = 0
        self.cached_forward_calls = 0

    def forward(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        self.forward_calls += 1
        self.input_token_positions += int(input_ids.numel())
        if kwargs.get("past_key_values") is not None:
            self.cached_forward_calls += 1
        return self.model(input_ids, *args, **kwargs)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def upstream_commit(upstream_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True
    ).strip()


def resolve_steps(spec: ArmSpec, gen_length: int, block_length: int) -> int:
    if spec.steps == "full":
        return gen_length
    if spec.steps == "one_per_block":
        return gen_length // block_length
    raise ValueError(f"unsupported step schedule: {spec.steps}")


def validate_schedule(gen_length: int, block_length: int) -> None:
    if gen_length <= 0 or block_length <= 0:
        raise ValueError("generation and block lengths must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")


def load_official_fast_dllm(
    upstream_dir: Path,
    model_name: str,
    device: str,
    dtype: str,
) -> tuple[CountingModel, Any, dict[str, Callable[..., tuple[torch.Tensor, int]]]]:
    llada_dir = upstream_dir / "v1" / "llada"
    if not llada_dir.is_dir():
        raise FileNotFoundError(f"missing official Fast-dLLM LLaDA source: {llada_dir}")
    sys.path.insert(0, str(llada_dir))

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from generate import generate, generate_with_dual_cache, generate_with_prefix_cache
    from huggingface_hub import snapshot_download
    from model.modeling_llada import LLaDAModelLM
    from transformers import AutoConfig, AutoTokenizer

    ensure_transformers_tied_weight_compat()
    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype.lower()]
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.flash_attention = True
    if not hasattr(config, "use_cache"):
        config.use_cache = False
    print(f"[fast-dllm] loading model {model_name}", flush=True)
    checkpoint_dir = snapshot_download(model_name, local_files_only=True)
    with init_empty_weights():
        base_model = LLaDAModelLM(config)
    base_model.tie_weights()
    base_model = load_checkpoint_and_dispatch(
        base_model,
        checkpoint=checkpoint_dir,
        device_map={"": device},
        dtype=torch_dtype,
    )
    meta_parameters = [
        name for name, parameter in base_model.named_parameters() if parameter.is_meta
    ]
    if meta_parameters:
        raise RuntimeError(f"checkpoint dispatch left meta parameters: {meta_parameters[:5]}")
    base_model.eval()
    model = CountingModel(base_model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("[fast-dllm] model loaded", flush=True)
    return model, tokenizer, {
        "full": generate,
        "prefix": generate_with_prefix_cache,
        "dual": generate_with_dual_cache,
    }


def run_arm(
    model: CountingModel,
    prompt: torch.Tensor,
    generator: Callable[..., tuple[torch.Tensor, int]],
    spec: ArmSpec,
    *,
    gen_length: int,
    block_length: int,
    threshold: float,
) -> dict[str, Any]:
    model.reset_counts()
    sync_cuda()
    start = time.perf_counter()
    tokens, nfe = generator(
        model,
        prompt,
        steps=resolve_steps(spec, gen_length, block_length),
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=MASK_ID,
        threshold=threshold if spec.thresholded else None,
    )
    sync_cuda()
    latency = time.perf_counter() - start
    response = tokens[:, prompt.shape[1] :].detach().cpu()
    final_mask_count = int(response.eq(MASK_ID).sum())
    if final_mask_count:
        raise RuntimeError(
            f"{spec.generator} arm stopped with {final_mask_count} unresolved masks"
        )
    return {
        "response": response,
        "nfe": int(nfe),
        "forward_calls": model.forward_calls,
        "input_token_positions": model.input_token_positions,
        "cached_forward_calls": model.cached_forward_calls,
        "latency_s": latency,
        "final_mask_count": final_mask_count,
    }


@torch.no_grad()
def warm_up_model(
    model: CountingModel, prompt: torch.Tensor, *, gen_length: int
) -> None:
    warmup = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        MASK_ID,
        dtype=torch.long,
        device=prompt.device,
    )
    warmup[:, : prompt.shape[1]] = prompt
    model.reset_counts()
    model(warmup)
    sync_cuda()
    model.reset_counts()


def comparison_metrics(rows: pd.DataFrame, reference: str, candidate: str) -> dict[str, Any]:
    if rows.empty:
        return {"examples": 0}
    reference_correct = rows[f"{reference}_correct"].astype(bool)
    candidate_correct = rows[f"{candidate}_correct"].astype(bool)
    return {
        "examples": int(len(rows)),
        "output_change_rate": float(rows[f"{candidate}_output_changed_vs_{reference}"].mean()),
        "answer_value_change_rate": float(
            rows[f"{candidate}_answer_value_changed_vs_{reference}"].mean()
        ),
        "reference_accuracy": float(reference_correct.mean()),
        "candidate_accuracy": float(candidate_correct.mean()),
        "accuracy_delta": float((candidate_correct.astype(float) - reference_correct).mean()),
        "correctness_change_rate": float(reference_correct.ne(candidate_correct).mean()),
        "correct_to_wrong_rate": float((reference_correct & ~candidate_correct).mean()),
        "wrong_to_correct_rate": float((~reference_correct & candidate_correct).mean()),
        "mean_reference_nfe": float(rows[f"{reference}_nfe"].mean()),
        "mean_candidate_nfe": float(rows[f"{candidate}_nfe"].mean()),
        "relative_input_token_positions": float(
            rows[f"{candidate}_input_token_positions"].sum()
            / rows[f"{reference}_input_token_positions"].sum()
        ),
        "wall_clock_speedup": float(
            rows[f"{reference}_latency_s"].sum() / rows[f"{candidate}_latency_s"].sum()
        ),
        "total_final_masks": int(rows[f"{candidate}_final_mask_count"].sum()),
    }


def add_pair_columns(
    row: dict[str, Any],
    arm_outputs: dict[str, dict[str, Any]],
    arm_generations: dict[str, str],
    dataset: str,
    reference: str,
    candidate: str,
) -> None:
    reference_response = arm_outputs[reference]["response"]
    candidate_response = arm_outputs[candidate]["response"]
    prefix = f"{candidate}"
    row[f"{prefix}_output_changed_vs_{reference}"] = not torch.equal(
        reference_response, candidate_response
    )
    row[f"{prefix}_changed_token_count_vs_{reference}"] = int(
        reference_response.ne(candidate_response).sum()
    )
    row[f"{prefix}_answer_value_changed_vs_{reference}"] = not generated_answers_equivalent(
        arm_generations[reference], arm_generations[candidate], dataset
    )


def summarize(rows: pd.DataFrame, arms: list[str]) -> dict[str, Any]:
    if rows.empty:
        return {"examples": 0, "arms": {}, "comparisons": {}}
    arm_metrics = {}
    for arm in arms:
        arm_metrics[arm] = {
            "accuracy": float(rows[f"{arm}_correct"].mean()),
            "mean_nfe": float(rows[f"{arm}_nfe"].mean()),
            "mean_input_token_positions": float(rows[f"{arm}_input_token_positions"].mean()),
            "mean_latency_s": float(rows[f"{arm}_latency_s"].mean()),
            "total_final_masks": int(rows[f"{arm}_final_mask_count"].sum()),
        }
    comparisons = {}
    for reference, candidate in PAIR_SPECS:
        if (
            reference in arms
            and candidate in arms
            and f"{candidate}_output_changed_vs_{reference}" in rows
        ):
            comparisons[f"{reference}->{candidate}"] = comparison_metrics(
                rows, reference, candidate
            )
    return {"examples": int(len(rows)), "arms": arm_metrics, "comparisons": comparisons}


def write_summary(
    output_dir: Path, config: dict[str, Any], rows: pd.DataFrame, arms: list[str]
) -> None:
    summary = {
        "method": "official Fast-dLLM v1 LLaDA generators in the local matched harness",
        "upstream_commit": config["upstream_commit"],
        "baseline": "official full-recompute fixed-schedule generator",
        "input_token_positions_note": (
            "Sum of input_ids positions passed to model forwards; this is a transparent work "
            "proxy, not FLOPs or total attention/KV-cache cost."
        ),
        "metrics": summarize(rows, arms),
        "config": config,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# Official Fast-dLLM v1 Evaluation\n\n```json\n"
        + json.dumps({key: value for key, value in summary.items() if key != "config"}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an official Fast-dLLM v1 LLaDA component evaluation."
    )
    parser.add_argument("--upstream-dir", default=".omx/external/fast-dllm-reference")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--jsonl-path", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARM_SPECS), default=list(ARM_SPECS)
    )
    args = parser.parse_args()

    validate_schedule(args.gen_length, args.block_length)
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms must not contain duplicates")
    if "fixed_full" not in args.arms:
        raise ValueError("fixed_full is required as the reference arm")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completed.json").unlink(missing_ok=True)
    commit = upstream_commit(Path(args.upstream_dir))
    arm_definitions = {
        arm: {
            "generator": ARM_SPECS[arm].generator,
            "steps": resolve_steps(ARM_SPECS[arm], args.gen_length, args.block_length),
            "threshold": args.threshold if ARM_SPECS[arm].thresholded else None,
        }
        for arm in args.arms
    }
    config = {
        **vars(args),
        "upstream_commit": commit,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arm_definitions": arm_definitions,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    examples = load_examples(
        args.dataset,
        args.split,
        args.limit,
        offset=args.offset,
        jsonl_path=args.jsonl_path,
    )
    model, tokenizer, generators = load_official_fast_dllm(
        Path(args.upstream_dir), args.model_name, args.device, args.dtype
    )
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        print(f"[fast-dllm] {index + 1}/{len(examples)} {example.example_id}", flush=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(example.question, example.dataset)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = torch.tensor(tokenizer(rendered)["input_ids"], device=args.device).unsqueeze(0)
        if index == 0:
            print("[fast-dllm] warming up full-length model forward", flush=True)
            warm_up_model(model, prompt, gen_length=args.gen_length)

        # Rotate method order across examples to reduce systematic timing-order bias.
        arm_order = args.arms[index % len(args.arms) :] + args.arms[: index % len(args.arms)]
        arm_outputs: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            spec = ARM_SPECS[arm]
            print(f"[fast-dllm]   arm={arm}", flush=True)
            arm_outputs[arm] = run_arm(
                model,
                prompt,
                generators[spec.generator],
                spec,
                gen_length=args.gen_length,
                block_length=args.block_length,
                threshold=args.threshold,
            )

        row: dict[str, Any] = {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "gold_answer": example.gold_answer,
        }
        arm_generations: dict[str, str] = {}
        for arm in args.arms:
            output = arm_outputs[arm]
            generation = tokenizer.batch_decode(
                output["response"], skip_special_tokens=True
            )[0].strip()
            correct, status = score_generation_with_status(generation, example)
            arm_generations[arm] = generation
            row.update(
                {
                    f"{arm}_normalized_answer": normalize_answer(generation, example.dataset),
                    f"{arm}_correct": correct,
                    f"{arm}_scoring_status": status,
                    f"{arm}_nfe": output["nfe"],
                    f"{arm}_forward_calls": output["forward_calls"],
                    f"{arm}_input_token_positions": output["input_token_positions"],
                    f"{arm}_cached_forward_calls": output["cached_forward_calls"],
                    f"{arm}_latency_s": output["latency_s"],
                    f"{arm}_final_mask_count": output["final_mask_count"],
                    f"{arm}_generation": generation,
                    f"{arm}_token_ids": json.dumps(output["response"][0].tolist()),
                }
            )

        for reference, candidate in PAIR_SPECS:
            if reference in args.arms and candidate in args.arms:
                add_pair_columns(
                    row,
                    arm_outputs,
                    arm_generations,
                    example.dataset,
                    reference,
                    candidate,
                )
        rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "example_results.csv", index=False)
        write_summary(output_dir, config, frame, args.arms)

    final_metrics = summarize(pd.DataFrame(rows), args.arms)
    (output_dir / "completed.json").write_text(
        json.dumps(
            {
                "complete": True,
                "examples": len(rows),
                "expected_examples": len(examples),
                "upstream_commit": commit,
                "runner_sha256": config["runner_sha256"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
