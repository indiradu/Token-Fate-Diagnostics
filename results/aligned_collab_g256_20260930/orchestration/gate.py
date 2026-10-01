"""Smoke gate between the smoke and full stages (our environment, no scoring).

Hard checks: the same-job B0 reproduces the aligned campaign's smoke B0 token-for-token, every requested
collaborator method produced complete outputs, SureLock locked rows, H2 committed in parallel.
Reported only: her dense vs our B0 token parity, capped-confidence extra commits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALIGNED = Path(__file__).resolve().parents[2] / "aligned_standard_g256_20260929"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--collab-dir", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ours = {(r["dataset"], r["example_id"]): r["token_ids"] for r in rows(args.b0_dir / "records.jsonl") if r["policy_id"] == "B0"}
    ref = {(r["dataset"], r["example_id"]): r["token_ids"] for r in rows(ALIGNED / "smoke_timed/records.jsonl") if r["policy_id"] == "B0"}
    if set(ours) != set(ref) or any(ours[k] != ref[k] for k in ref):
        raise SystemExit("same-job B0 does not reproduce the aligned campaign smoke B0")
    collab = rows(args.collab_dir / "records.jsonl")
    report = {"same_job_B0_parity": f"{len(ref)}/{len(ref)}", "methods": {}}
    for method in args.methods.split(","):
        got = [r for r in collab if r["policy_id"] == method]
        if {(r["dataset"], r["example_id"]) for r in got} != set(ref):
            raise SystemExit(f"{method} did not complete both smoke cases")
        if "SureLock" in method and not all(r["lock_count"] for r in got):
            raise SystemExit(f"{method} never locked a row")
        if "H2" in method and not all(r["nfe"] < 256 for r in got):
            raise SystemExit(f"{method} never committed in parallel")
        report["methods"][method] = [{"example": r["example_id"], "nfe": r["nfe"], "latency_s": round(r["latency_s"], 3),
                                      "lock_count": r.get("lock_count"), "extra_commit_count": r.get("extra_commit_count"),
                                      "same_tokens_as_our_B0": r["token_ids"] == ours[(r["dataset"], r["example_id"])]} for r in got]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
