#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.locking_analysis import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a paired locking policy eval run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    analyze_run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

