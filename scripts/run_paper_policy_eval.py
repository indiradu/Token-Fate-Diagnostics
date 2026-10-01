"""Run bounded paper-policy evaluation from frozen manifests."""
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
    parser.add_argument("--policy-config", help="JSON policy list from development selection")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--epsilon", nargs="+", type=float, default=[0.0, 0.0005, 0.005])
    parser.add_argument("--threshold", nargs="+", type=float, default=[0.95])
    parser.add_argument("--controller-checkpoint")
    parser.add_argument("--projector-checkpoint")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--freeze-scope", choices=["generated", "all_unmasked"], default="generated")
    parser.add_argument("--confidence-percentile", type=float, default=100.0)
    parser.add_argument("--kl-space", choices=["token", "representation"], default="token")
    parser.add_argument("--representation-temperature", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--collect-events", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--purpose", choices=["smoke", "selection", "evaluation"], default="evaluation")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(args.output_dir).resolve() / "module_cache"))

    from regret_remasking.paper_experiment import DEFAULT_ARMS, make_paper_policies, run_paper_experiment

    policies = (
        json.loads(Path(args.policy_config).read_text(encoding="utf-8"))
        if args.policy_config
        else make_paper_policies(
            args.arms or DEFAULT_ARMS,
            args.epsilon,
            args.threshold,
            gen_length=args.gen_length,
            block_length=args.block_length,
            window_length=args.window_length,
            freeze_scope=args.freeze_scope,
            confidence_percentile=args.confidence_percentile,
            kl_space=args.kl_space,
            representation_temperature=args.representation_temperature,
            controller_checkpoint=args.controller_checkpoint,
            projector_checkpoint=args.projector_checkpoint,
        )
    )
    result = run_paper_experiment(
        args.manifest,
        args.output_dir,
        policies,
        checkpoint=args.checkpoint,
        gen_length=args.gen_length,
        repeats=args.repeats,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        limit=args.limit,
        collect_events=args.collect_events,
        warmup=not args.no_warmup,
        resume=args.resume,
        purpose=args.purpose,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
