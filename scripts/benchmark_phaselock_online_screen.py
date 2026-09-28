#!/usr/bin/env python3
"""Matched LIVE online baselines and H2 controller ablations; no plan replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_phase_lock_fused_replay import _examples
from benchmark_phaselock_sure_hybrid import decode_hybrid
from benchmark_surelock_vs_phaselock import decode_surelock
from benchmark_tracelock_vs_phaselock import decode_tracelock
from regret_remasking.data import build_prompt, score_generation
from regret_remasking.llada_trace import DecodeConfig, infer_mask_token_id, load_llada
from regret_remasking.paper_baselines import parse_example_indices
from regret_remasking.tracelock_baseline import TraceLockConfig, TraceLockController


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval"], required=True)
    parser.add_argument("--example-indices", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271)
    args = parser.parse_args()
    if min(args.batch_size, args.gen_length, args.steps, args.block_length, args.timing_repeats) < 1 or args.timing_warmups < 0:
        raise ValueError("invalid schedule or timing parameters")
    # One mandatory fallback per update guarantees completion for the H2 pool.
    if args.steps != args.gen_length or args.gen_length % args.block_length:
        raise ValueError("this screen requires steps == gen_length and divisible blocks")
    policies = json.loads(args.policy_config.read_text())["methods"]
    names = args.methods.split(",") if args.methods else list(policies)
    if len(set(names)) != len(names) or not set(names) <= set(policies):
        raise ValueError("duplicate or unknown methods")
    if not {"dense", "surelock", "tracelock", "h2"} <= set(names):
        raise ValueError("every screen must include both baselines, dense and H2")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 24 * 1024 ** 3:
        raise RuntimeError(f"Allocated GPU has only {free_bytes / 1024 ** 3:.1f} GiB free; refusing a contaminated/OOM-prone benchmark")
    examples = _examples(args.dataset, "test", parse_example_indices(args.example_indices))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    # Fail closed rather than silently replace a run or mingle partial output.
    with config_path.open("x") as handle:
        json.dump({**vars(args), "policies": {k: policies[k] for k in names},
                   "status": "initializing"}, handle, indent=2, default=str)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller = TraceLockController(TraceLockConfig(**checkpoint["config"]))
    controller.load_state_dict(checkpoint["model"])
    model, tokenizer = load_llada(args.model_name, device="cuda", dtype="bf16")
    controller = controller.to("cuda").eval()
    projection = torch.from_numpy(np.load(args.projection)).float().to("cuda")
    config = DecodeConfig(steps=args.steps, gen_length=args.gen_length,
                          block_length=args.block_length, mask_id=infer_mask_token_id(tokenizer),
                          temperature=0.0)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), args.policy_config]
    sources += [root / "scripts" / name for name in (
        "benchmark_phaselock_sure_hybrid.py", "benchmark_surelock_vs_phaselock.py",
        "benchmark_tracelock_vs_phaselock.py")]
    sources += sorted((root / "src/regret_remasking").glob("*.py"))
    manifest = {
        **vars(args), "policies": {k: policies[k] for k in names},
        "stage": "development_only", "time_metric": "synchronized caller wall time including tokenization, live controller, execution, cleanup and decoding; excludes scoring/model load",
        "baseline_labels": {"surelock": "paper-spec transfer reimplementation", "tracelock": "LLaDA transfer adaptation; not official checkpoint"},
        "schedule_difference": "TraceLock uses sliding soft-crop width; others use fixed blocks of the same width. This is an intrinsic method difference, not an identical token-transfer trajectory.",
        "scoring": "legacy repository scorer (numeric extraction for GSM8K/MATH500; HumanEval single-sample tests); MATH500 scores are screening proxies, not validated symbolic accuracy",
        "source_sha256": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256(p) for p in sources},
        "checkpoint_sha256": sha256(args.checkpoint), "projection_sha256": sha256(args.projection),
        "examples": [{"id": e.example_id, "prompt_sha256": hashlib.sha256(build_prompt(e.question, e.dataset).encode()).hexdigest()} for e in examples],
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "initial_gpu_free_bytes": free_bytes, "gpu_total_bytes": total_bytes,
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(), "platform": platform.platform(),
        "model_config_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_config": tokenizer.init_kwargs,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }
    config_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    common = dict(epsilon=0.005, confidence_percentile=20.0, bucket_width_multiple=16)
    hybrid = dict(**common, representation_layers=[8, 16, 24], influence_layer=24,
                  drift_threshold=0.01, influence_threshold=0.5,
                  representation_patience=1, representation_min_age=1,
                  representation_lock_fraction=0.04, max_queries=8)
    raw_path = args.output_dir / "records.jsonl"
    wall_start = time.time()
    first_tokens = {}
    score_cache = {}
    with raw_path.open("x") as output:
        for batch_start in range(0, len(examples), args.batch_size):
            batch = examples[batch_start:batch_start + args.batch_size]
            batch_id = batch_start // args.batch_size
            for repeat in range(-args.timing_warmups, args.timing_repeats):
                shift = (batch_id + repeat + args.timing_warmups) % len(names)
                order = names[shift:] + names[:shift]
                for name in order:
                    policy = dict(policies[name])
                    kind = policy.pop("kind")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    if kind in {"dense", "surelock"}:
                        kwargs = {**common, "epsilon": None if kind == "dense" else common["epsilon"]}
                        record = decode_surelock(model, tokenizer, batch, config, **kwargs)
                    elif kind == "tracelock":
                        record = decode_tracelock(model, tokenizer, controller, projection,
                                                 [30, 31, 32], batch, config, window_width=args.block_length)
                    elif kind == "hybrid":
                        record = decode_hybrid(model, tokenizer, batch, config, mode="h2", **{**hybrid, **policy})
                    else:
                        raise ValueError(f"unknown policy kind {kind}")
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    incomplete = any(config.mask_id in tokens for tokens in record["token_ids"])
                    if incomplete:
                        raise RuntimeError(f"{name} left masked generation tokens")
                    signature_key = (batch_id, name)
                    deterministic = (signature_key not in first_tokens or first_tokens[signature_key] == record["token_ids"])
                    first_tokens.setdefault(signature_key, record["token_ids"])
                    # Missing determinism is saved, never silently scored as extra examples.
                    record.update(method=name, dataset=args.dataset, batch_id=batch_id, repeat=repeat,
                                  order=order, online_latency_s=elapsed,
                                  deterministic_vs_first=deterministic,
                                  peak_memory_bytes=torch.cuda.max_memory_allocated(),
                                  example_ids=[e.example_id for e in batch])
                    scores = []
                    for text, e in zip(record["generations"], batch):
                        key = (e.example_id, text)
                        if key not in score_cache:
                            score_cache[key] = int(bool(score_generation(text, e)))
                        scores.append(score_cache[key])
                    record["correct"] = scores
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps({"batch": batch_id, "repeat": repeat, "method": name,
                                      "online_s": elapsed, "nfe": record["nfe"], "correct": scores,
                                      "deterministic": deterministic}), flush=True)
                    if kind == "hybrid" and policy.get("speed_reference_policy") in {"delay", "drift"}:
                        if any(r["same_update_reference_locks"] for r in record["decision_trace"]):
                            raise RuntimeError("reference gate froze a same-update identity")
    (args.output_dir / "completion.json").write_text(json.dumps({
        "status": "complete", "wall_seconds_excluding_model_load": time.time() - wall_start,
        "records_sha256": sha256(raw_path), "training_gpu_seconds": None,
        "training_cost_note": "TraceLock pretraining reused; unknown cost is not zero. H2 pool has no fitted controller yet; screening cost is development cost.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
