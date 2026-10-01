#!/usr/bin/env python3
"""Validate dense/sparse reference-locking backends on an inspectable prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="existing local LLaDA checkpoint; never downloads")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Solve: 2 + 2 =")
    parser.add_argument("--gen-length", type=int, default=8)
    parser.add_argument("--lock-offset", type=int, default=0)
    parser.add_argument("--lock-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tolerance-multiplier", type=float, default=16.0)
    parser.add_argument("--min-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--min-rel-tolerance", type=float, default=1e-4)
    parser.add_argument("--strict-logit-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--strict-kv-abs-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(output_dir / "module_cache"))

    from regret_remasking.locking_experiment import load_local_llada
    from regret_remasking.locking_validation import validate_backends

    model, tokenizer = load_local_llada(args.checkpoint, args.device, args.dtype)
    result = validate_backends(
        model,
        tokenizer,
        args.prompt,
        gen_length=args.gen_length,
        lock_offset=args.lock_offset,
        lock_count=args.lock_count,
        tolerance_multiplier=args.tolerance_multiplier,
        min_abs_tolerance=args.min_abs_tolerance,
        min_rel_tolerance=args.min_rel_tolerance,
        strict_logit_abs_tolerance=args.strict_logit_abs_tolerance,
        strict_kv_abs_tolerance=args.strict_kv_abs_tolerance,
    )
    result.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    output_path = output_dir / "backend_validation.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "passed": result["passed"]}, indent=2, sort_keys=True))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
