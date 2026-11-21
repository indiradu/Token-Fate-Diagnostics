#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
EOS_ID = 126081


def elastic_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    return list(model.model.transformer.blocks)


def reset_elastic_state(model: torch.nn.Module) -> None:
    for block in elastic_blocks(model):
        block.x_cache = None
        block.q_cache = None
        block.k_cache = None
        block.v_cache = None
        block.track_token = None


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate_full_recompute(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    gen_length: int,
    window_length: int,
    threshold: float,
    gamma: float,
    track_num: int,
    mask_id: int = MASK_ID,
    eos_id: int = EOS_ID,
) -> tuple[torch.Tensor, int, float]:
    """Matched confidence/sliding-window decode with every layer recomputed."""
    reset_elastic_state(model)
    total_length = prompt.shape[1] + gen_length
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    full_positions = torch.arange(total_length, device=prompt.device)
    masked_positions = full_positions[prompt.shape[1] :].clone()
    decoded_eos = False
    nfe = 0

    while masked_positions.numel() > 0:
        query_masked_positions = masked_positions[:window_length]
        empty = full_positions[:0]
        positions = [full_positions, empty, query_masked_positions, masked_positions]
        # start_reset=-1 forces every transformer layer through the cache-update
        # branch while x_query remains the complete sequence.
        lengths = [total_length, -1, gamma, track_num]
        outputs = model(
            x,
            use_cache=True,
            lengths=lengths,
            positions=positions,
        )
        logits = outputs.logits[:, query_masked_positions, :]
        probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
        confidence, predictions = torch.max(probabilities, dim=-1)
        keep = confidence >= min(threshold, float(confidence.max()))
        decoded_positions = query_masked_positions[keep[0]]
        decoded_tokens = predictions[:, keep[0]]
        x[:, decoded_positions] = decoded_tokens
        masked_positions = masked_positions[~torch.isin(masked_positions, decoded_positions)]
        nfe += 1

        if not decoded_eos:
            eos_positions = decoded_positions[decoded_tokens.eq(eos_id)[0]]
            if eos_positions.numel() > 0:
                decoded_eos = True
                masked_positions = masked_positions[masked_positions <= int(eos_positions.min())]

    x[x.eq(mask_id)] = eos_id
    return x, nfe, 1.0


def load_official_elastic(
    upstream_dir: Path,
    model_name: str,
    device: str,
    dtype: str,
) -> tuple[Any, Any, Any]:
    llada_dir = upstream_dir / "llada"
    if not llada_dir.is_dir():
        raise FileNotFoundError(f"missing official Elastic-Cache LLaDA source: {llada_dir}")
    sys.path.insert(0, str(llada_dir))
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from generate import generate_with_elastic_cache
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
    print(f"[elastic-official] loading model {model_name}", flush=True)
    checkpoint_dir = snapshot_download(model_name, local_files_only=True)
    with init_empty_weights():
        model = LLaDAModelLM(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_dir,
        device_map={"": device},
        dtype=torch_dtype,
    )
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"checkpoint dispatch left meta parameters: {meta_parameters[:5]}")
    model.eval()
    print("[elastic-official] model loaded", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer, generate_with_elastic_cache


def upstream_commit(upstream_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"examples": 0}
    cache_frequency_column = (
        "elastic_upstream_layer_recompute_fraction"
        if "elastic_upstream_layer_recompute_fraction" in rows
        else "elastic_cache_update_frequency"
        if "elastic_cache_update_frequency" in rows
        else "elastic_layer_compute_fraction"
    )
    return {
        "examples": int(len(rows)),
        "output_change_rate": float(rows["output_changed"].mean()),
        "answer_value_change_rate": float(rows["answer_value_changed"].mean()),
        "baseline_accuracy": float(rows["baseline_correct"].mean()),
        "elastic_accuracy": float(rows["elastic_correct"].mean()),
        "accuracy_delta": float(
            (rows["elastic_correct"].astype(float) - rows["baseline_correct"].astype(float)).mean()
        ),
        "correct_to_wrong_rate": float(
            (rows["baseline_correct"].astype(bool) & ~rows["elastic_correct"].astype(bool)).mean()
        ),
        "wrong_to_correct_rate": float(
            (~rows["baseline_correct"].astype(bool) & rows["elastic_correct"].astype(bool)).mean()
        ),
        "mean_baseline_nfe": float(rows["baseline_nfe"].mean()),
        "mean_elastic_nfe": float(rows["elastic_nfe"].mean()),
        "mean_elastic_upstream_layer_recompute_fraction": float(
            rows[cache_frequency_column].mean()
        ),
        "mean_baseline_latency_s": float(rows["baseline_latency_s"].mean()),
        "mean_elastic_latency_s": float(rows["elastic_latency_s"].mean()),
    }


def write_run_summary(
    output_dir: Path,
    config: dict[str, Any],
    rows: pd.DataFrame,
) -> None:
    summary = {
        "method": "official Elastic-Cache",
        "metric_schema_version": 2,
        "upstream_commit": config["upstream_commit"],
        "comparison": "matched full-recompute decoder using the official model implementation",
        "metrics": summarize(rows),
        "config": config,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# Official Elastic-Cache Evaluation\n\n```json\n"
        + json.dumps({key: value for key, value in summary.items() if key != "config"}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official Elastic-Cache with a matched full-recompute decoder."
    )
    parser.add_argument("--upstream-dir", default="/tmp/elastic-cache-reference")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--jsonl-path", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=80)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--track-num", type=int, default=1)
    parser.add_argument("--block-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.gen_length <= 0 or args.window_length <= 0:
        raise ValueError("generation and window lengths must be positive")
    if args.window_length > args.gen_length:
        raise ValueError("window_length cannot exceed gen_length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = upstream_commit(Path(args.upstream_dir))
    config = {**vars(args), "upstream_commit": commit}
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
    model, tokenizer, elastic_generate = load_official_elastic(
        Path(args.upstream_dir), args.model_name, args.device, args.dtype
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        print(f"[elastic-official] {index}/{len(examples)} {example.example_id}", flush=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(example.question, example.dataset)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = torch.tensor(tokenizer(rendered)["input_ids"], device=args.device).unsqueeze(0)

        sync_cuda()
        baseline_start = time.perf_counter()
        baseline_tokens, baseline_nfe, _ = generate_full_recompute(
            model,
            prompt,
            gen_length=args.gen_length,
            window_length=args.window_length,
            threshold=args.threshold,
            gamma=args.gamma,
            track_num=args.track_num,
        )
        sync_cuda()
        baseline_latency = time.perf_counter() - baseline_start

        reset_elastic_state(model)
        sync_cuda()
        elastic_start = time.perf_counter()
        elastic_tokens, elastic_nfe, layer_recompute_fraction = elastic_generate(
            model,
            prompt,
            gen_length=args.gen_length,
            window_length=args.window_length,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
            threshold=args.threshold,
            gamma=args.gamma,
            track_num=args.track_num,
            block_caching=args.block_caching,
        )
        sync_cuda()
        elastic_latency = time.perf_counter() - elastic_start

        baseline_response = baseline_tokens[:, prompt.shape[1] :].detach().cpu()
        elastic_response = elastic_tokens[:, prompt.shape[1] :].detach().cpu()
        baseline_generation = tokenizer.batch_decode(
            baseline_response, skip_special_tokens=True
        )[0].strip()
        elastic_generation = tokenizer.batch_decode(
            elastic_response, skip_special_tokens=True
        )[0].strip()
        baseline_correct, baseline_status = score_generation_with_status(
            baseline_generation, example
        )
        elastic_correct, elastic_status = score_generation_with_status(
            elastic_generation, example
        )
        rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "gold_answer": example.gold_answer,
                "output_changed": not torch.equal(baseline_response, elastic_response),
                "changed_token_count": int((baseline_response != elastic_response).sum()),
                "answer_value_changed": not generated_answers_equivalent(
                    baseline_generation, elastic_generation, example.dataset
                ),
                "baseline_normalized_answer": normalize_answer(
                    baseline_generation, example.dataset
                ),
                "elastic_normalized_answer": normalize_answer(
                    elastic_generation, example.dataset
                ),
                "baseline_correct": baseline_correct,
                "elastic_correct": elastic_correct,
                "baseline_scoring_status": baseline_status,
                "elastic_scoring_status": elastic_status,
                "baseline_nfe": baseline_nfe,
                "elastic_nfe": elastic_nfe,
                "elastic_upstream_layer_recompute_fraction": layer_recompute_fraction,
                "baseline_latency_s": baseline_latency,
                "elastic_latency_s": elastic_latency,
                "baseline_generation": baseline_generation,
                "elastic_generation": elastic_generation,
                "baseline_token_ids": json.dumps(baseline_response[0].tolist()),
                "elastic_token_ids": json.dumps(elastic_response[0].tolist()),
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / "example_results.csv", index=False)

    frame = pd.DataFrame(rows)
    write_run_summary(output_dir, config, frame)
    print(json.dumps(summarize(frame), indent=2), flush=True)


if __name__ == "__main__":
    main()
