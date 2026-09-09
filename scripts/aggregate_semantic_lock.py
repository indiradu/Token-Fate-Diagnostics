#!/usr/bin/env python3
"""Aggregate semantic-lock selector summaries across dataset runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be DATASET=RUN_DIR")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(
    successes: int, count: int, z: float = 1.96
) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-reference-policy", default="confidence")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    for dataset, run_dir in args.run:
        summary_path = run_dir / "semantic_analysis" / "semantic_selector_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            for row in json.load(handle):
                dataset_rows.append({"dataset": dataset, **row})
    if not dataset_rows:
        raise ValueError("no semantic selector rows loaded")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        grouped[str(row["semantic_policy"])].append(row)

    combined: list[dict[str, Any]] = []
    for policy, rows in sorted(grouped.items()):
        examples = sum(int(row["examples"]) for row in rows)
        events = sum(int(row["accelerated_lock_events"]) for row in rows)
        matched_events = round(
            sum(
                float(row["retrospective_lock_precision_vs_a0"])
                * int(row["accelerated_lock_events"])
                for row in rows
            )
        )
        harmful_examples = round(
            sum(float(row["answer_harm_rate"]) * int(row["examples"]) for row in rows)
        )
        precision_low, precision_high = _wilson_interval(matched_events, events)
        harm_low, harm_high = _wilson_interval(harmful_examples, examples)

        def example_weighted(field: str) -> float:
            return sum(float(row[field]) * int(row["examples"]) for row in rows) / examples

        reference_runtime_by_dataset = {
            str(row["dataset"]): float(row["mean_runtime_gain"])
            for row in dataset_rows
            if row["semantic_policy"] == args.runtime_reference_policy
        }
        if len(reference_runtime_by_dataset) != len({str(row["dataset"]) for row in rows}):
            raise ValueError(
                f"runtime reference policy {args.runtime_reference_policy!r} is missing from one or more datasets"
            )
        answer_change_rows = [row for row in rows if row.get("normalized_answer_changed_rate") is not None]
        answer_change_examples = sum(int(row["examples"]) for row in answer_change_rows)
        combined.append(
            {
                "semantic_policy": policy,
                "datasets": len(rows),
                "examples": examples,
                "accelerated_lock_events": events,
                "pooled_lock_precision_vs_a0": matched_events / events if events else None,
                "pooled_lock_precision_wilson95_low": precision_low,
                "pooled_lock_precision_wilson95_high": precision_high,
                "example_weighted_exact_sequence_match_rate": example_weighted("exact_sequence_match_rate"),
                "example_weighted_token_agreement": example_weighted("mean_token_agreement"),
                "math_answer_changed_rate": (
                    sum(
                        float(row["normalized_answer_changed_rate"]) * int(row["examples"])
                        for row in answer_change_rows
                    ) / answer_change_examples
                    if answer_change_examples
                    else None
                ),
                "example_weighted_answer_harm_rate": example_weighted("answer_harm_rate"),
                "answer_harm_wilson95_low": harm_low,
                "answer_harm_wilson95_high": harm_high,
                "max_dataset_answer_harm_rate": max(float(row["answer_harm_rate"]) for row in rows),
                "example_weighted_accuracy_delta": example_weighted("accuracy_delta"),
                "example_weighted_nfe_reduction": example_weighted("mean_nfe_reduction"),
                "example_weighted_runtime_gain": example_weighted("mean_runtime_gain"),
                "runtime_gain_delta_vs_confidence": sum(
                    (
                        float(row["mean_runtime_gain"])
                        - reference_runtime_by_dataset[str(row["dataset"])]
                    )
                    * int(row["examples"])
                    for row in rows
                ) / examples,
            }
        )

    _write_csv(args.output_dir / "dataset_selector_rows.csv", dataset_rows)
    _write_csv(args.output_dir / "combined_selector_summary.csv", combined)
    (args.output_dir / "combined_selector_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    lines = [
        "# Cross-dataset semantic-lock screen",
        "",
        "All selectors use the same normal scheduled transfers and the same accelerated-lock budget.",
        "",
        "| selector | lock precision | exact sequence | answer harm | NFE gain | runtime gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        lines.append(
            f"| {row['semantic_policy']} | {_fmt(row['pooled_lock_precision_vs_a0'])} | "
            f"{_fmt(row['example_weighted_exact_sequence_match_rate'])} | "
            f"{_fmt(row['example_weighted_answer_harm_rate'])} | "
            f"{_fmt(row['example_weighted_nfe_reduction'])} | "
            f"{_fmt(row['example_weighted_runtime_gain'])} |"
        )
    (args.output_dir / "combined_selector_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
