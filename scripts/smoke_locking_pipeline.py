"""Bounded real-model engineering smoke; these are not research evaluation scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-examples", type=int, default=2)
    parser.add_argument("--gen-length", type=int, default=16)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("use a fresh smoke directory to preserve previous evidence")
    if args.trace_examples < 2:
        raise ValueError("at least two examples per trace split are required")
    output.mkdir(parents=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(output / "module_cache"))

    from regret_remasking.locking_analysis import analyze_run
    from regret_remasking.locking_experiment import load_local_llada, make_policies, run_experiment
    from regret_remasking.locking_manifest import load_manifest, write_frozen_jsonl
    from regret_remasking.locking_validation import validate_backends
    from regret_remasking.trace_commitment import train_trace_commitment

    manifest_dir = Path(args.manifest_dir)
    subset_dir = output / "manifests"
    subset_dir.mkdir()
    for split, count in (("fit", args.trace_examples), ("calibration", args.trace_examples),
                         ("smoke_math", 1), ("smoke_code", 1)):
        rows = load_manifest(manifest_dir / f"{split}.jsonl", expected_split=split)
        if len(rows) < count:
            raise ValueError(f"not enough {split} examples")
        write_frozen_jsonl(subset_dir / f"{split}.jsonl", rows[:count])
    print("[smoke] loading cached LLaDA", flush=True)
    model, tokenizer = load_local_llada(args.checkpoint, args.device, args.dtype)
    print("[smoke] validating dense/sparse reference semantics", flush=True)
    backend = validate_backends(model, tokenizer, "Solve: 2 + 2 =", gen_length=8)
    (output / "backend_validation.json").write_text(json.dumps(backend, indent=2, allow_nan=False)+"\n")
    if not backend["passed"]:
        raise RuntimeError("backend validation failed; policy smoke was not started")
    shared = dict(checkpoint=args.checkpoint, gen_length=args.gen_length, block_length=args.block_length,
                  device=args.device, dtype=args.dtype, model=model, tokenizer=tokenizer,
                  repeats=1, purpose="smoke", warmup=True)
    for split in ("fit", "calibration"):
        run_experiment([subset_dir / f"{split}.jsonl"], output / split,
                       make_policies(["B0"], [0.0], [0.95]), collect_trace=True, **shared)
    print("[smoke] fitting separate trace and confidence predictors", flush=True)
    artifacts = train_trace_commitment(
        output / "fit/traces.csv", output / "calibration/traces.csv", output / "models",
        fit_manifest=subset_dir / "fit.jsonl", calibration_manifest=subset_dir / "calibration.jsonl",
    )
    policies = make_policies(
        ["B0", "R-dense", "R-sparse", "T-fixed", "C-adaptive", "T-adaptive"],
        [0.0, 0.01], [0.5, 0.95], str(artifacts.trace_model_path), str(artifacts.confidence_model_path),
    )
    run_experiment([subset_dir / "smoke_math.jsonl", subset_dir / "smoke_code.jsonl"],
                   output / "policies", policies, collect_events=True, **shared)
    analyze_run(output / "policies", output / "analysis", bootstrap_reps=100, seed=23)
    summary = {"purpose": "engineering smoke only; tiny traces and short generations, no scientific quality claim",
               "passed": True, "backend_passed": backend["passed"], "trace_examples_per_split": args.trace_examples,
               "policy_variants": len(policies), "evaluation_examples": 2,
               "gen_length": args.gen_length, "artifacts": str(output)}
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
