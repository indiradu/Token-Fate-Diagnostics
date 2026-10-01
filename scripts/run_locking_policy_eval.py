"""Run free-continuation reference-freeze or trace-commitment experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="existing local model snapshot; never downloads")
    parser.add_argument("--arms", nargs="+", default=["B0", "R-dense", "R-sparse"])
    parser.add_argument("--epsilon", nargs="+", type=float, default=[0.0])
    parser.add_argument("--threshold", nargs="+", type=float, default=[0.95])
    parser.add_argument("--trace-checkpoint")
    parser.add_argument("--confidence-checkpoint")
    parser.add_argument("--policy-config", help="JSON policy list from development selection")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--collect-trace", action="store_true")
    parser.add_argument("--collect-events", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--purpose", choices=["smoke", "trace_collection", "selection", "evaluation"], default="evaluation")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(args.output_dir).resolve() / "module_cache"))
    from regret_remasking.locking_experiment import make_policies, run_experiment
    policies = (json.loads(Path(args.policy_config).read_text()) if args.policy_config else
                make_policies(args.arms, args.epsilon, args.threshold, args.trace_checkpoint, args.confidence_checkpoint))
    result = run_experiment(
        args.manifest, args.output_dir, policies, checkpoint=args.checkpoint,
        gen_length=args.gen_length, block_length=args.block_length, repeats=args.repeats,
        device=args.device, dtype=args.dtype, seed=args.seed, collect_trace=args.collect_trace,
        collect_events=args.collect_events, limit=args.limit, warmup=not args.no_warmup, resume=args.resume,
        purpose=args.purpose,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
