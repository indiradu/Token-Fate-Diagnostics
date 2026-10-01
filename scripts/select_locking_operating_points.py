#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.locking_selection import DEFAULT_NFE_TARGETS, select_operating_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze development-only locking operating points from selection sweeps."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nfe-targets", nargs="+", type=float, default=list(DEFAULT_NFE_TARGETS))
    args = parser.parse_args()

    summary = select_operating_points(args.run_dir, args.output_dir, args.nfe_targets)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

