#!/usr/bin/env python3
"""Apply the predeclared safety-first rule to a semantic-lock sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline_c095_s2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    baseline_matches = [row for row in rows if row["semantic_policy"] == args.baseline]
    if len(baseline_matches) != 1:
        raise ValueError(f"expected exactly one baseline row for {args.baseline!r}")
    baseline = baseline_matches[0]
    baseline_precision = baseline["pooled_lock_precision_vs_a0"]
    baseline_nfe = float(baseline["example_weighted_nfe_reduction"])
    baseline_locks = float(baseline["mean_accelerated_locks_per_example"])

    ranked: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if float(row["max_dataset_answer_harm_rate"]) > 0.0:
            reasons.append("nonzero_per_dataset_answer_harm")
        if float(row["example_weighted_exact_sequence_match_rate"]) < 1.0:
            reasons.append("not_exactly_a0_equivalent")
        precision = row["pooled_lock_precision_vs_a0"]
        if baseline_precision is not None and (
            precision is None or float(precision) < float(baseline_precision)
        ):
            reasons.append("lock_precision_below_baseline")
        ranked.append(
            {
                **row,
                "safety_eligible": not reasons,
                "rejection_reasons": ";".join(reasons),
                "nfe_gain_delta_vs_baseline": (
                    float(row["example_weighted_nfe_reduction"]) - baseline_nfe
                ),
                "locks_per_example_delta_vs_baseline": (
                    float(row["mean_accelerated_locks_per_example"]) - baseline_locks
                ),
            }
        )

    ranked.sort(
        key=lambda row: (
            bool(row["safety_eligible"]),
            float(row["example_weighted_nfe_reduction"]),
            float(row["mean_accelerated_locks_per_example"]),
            float(row["example_weighted_runtime_gain"]),
        ),
        reverse=True,
    )
    eligible_improvements = [
        row
        for row in ranked
        if row["safety_eligible"] and float(row["nfe_gain_delta_vs_baseline"]) > 0.0
    ]
    selected = eligible_improvements[0] if eligible_improvements else baseline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "semantic_candidate_ranking.csv", ranked)
    (args.output_dir / "semantic_candidate_ranking.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )
    decision = {
        "baseline": args.baseline,
        "selected_for_frozen_confirmation": selected["semantic_policy"],
        "eligible_improvements": len(eligible_improvements),
        "selection_is_provisional_until_matched_timing_and_confirmation": True,
    }
    (args.output_dir / "semantic_candidate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# Semantic-lock candidate ranking",
        "",
        f"Baseline: `{args.baseline}`",
        f"Provisional candidate: `{selected['semantic_policy']}`",
        "",
        "| policy | eligible | rejection | locks/example | lock precision | exact A0 | answer harm | NFE gain | runtime gain |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['semantic_policy']} | {int(row['safety_eligible'])} | "
            f"{row['rejection_reasons'] or '-'} | "
            f"{float(row['mean_accelerated_locks_per_example']):.3f} | "
            f"{row['pooled_lock_precision_vs_a0']} | "
            f"{float(row['example_weighted_exact_sequence_match_rate']):.3f} | "
            f"{float(row['example_weighted_answer_harm_rate']):.3f} | "
            f"{float(row['example_weighted_nfe_reduction']):.3f} | "
            f"{float(row['example_weighted_runtime_gain']):.3f} |"
        )
    (args.output_dir / "semantic_candidate_ranking.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
