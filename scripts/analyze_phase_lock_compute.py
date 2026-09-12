#!/usr/bin/env python3
"""Validate and aggregate paired PhaseLock A2/A3 timing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_phase_lock_compute import _summaries


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no timing rows found")
    keys = [
        (
            str(row["dataset"]),
            str(row["example_id"]),
            str(row["semantic_label"]),
            int(row["repeat"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset/example/semantic/repeat timing key")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["example_id"]), str(row["semantic_label"]))].append(row)
    expected_repeats = {int(config["timing_repeats"]) for config in configs}
    if len(expected_repeats) != 1:
        raise ValueError("all runs must use one timing repeat count")
    expected_repeat_count = expected_repeats.pop()
    for key, group in grouped.items():
        if len(group) != expected_repeat_count:
            raise ValueError(f"{key} has {len(group)} repeats, expected {expected_repeat_count}")
        if len({row["semantic_plan_sha256"] for row in group}) != 1:
            raise ValueError(f"semantic plan changed across repeats for {key}")
        if len({row["reference_plan_sha256"] for row in group}) != 1:
            raise ValueError(f"reference plan changed across repeats for {key}")

    hardware = [config["runtime_environment"] for config in configs]
    if len({json.dumps(item, sort_keys=True) for item in hardware}) != 1:
        raise ValueError("runtime hardware/software metadata differs across datasets")

    return {
        "timing_rows": len(rows),
        "examples": len({(row["dataset"], row["example_id"]) for row in rows}),
        "example_semantic_groups": len(grouped),
        "timing_repeats_per_group": expected_repeat_count,
        "all_a2_exact_a1": all(int(row["a2_exact_a1"]) == 1 for row in rows),
        "all_a3_exact_a2": all(int(row["a3_exact_a2"]) == 1 for row in rows),
        "answer_or_correctness_changes": sum(
            int(row["answer_or_correctness_changed"]) for row in rows
        ),
        "fixed_plan_hashes_within_group": True,
        "runtime_environment": hardware[0],
    }


def _order_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["dataset"]), str(row["semantic_label"]), str(row["execution_order"]))
        ].append(float(row["runtime_gain"]))
    return [
        {
            "dataset": dataset,
            "semantic_label": semantic,
            "execution_order": order,
            "timed_pairs": len(values),
            "mean_runtime_gain": float(np.mean(values)),
        }
        for (dataset, semantic, order), values in sorted(grouped.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    dataset_dirs = sorted(path.parent for path in args.input_root.glob("*/timing_pairs.csv"))
    if not dataset_dirs:
        raise ValueError("input root contains no dataset timing artifacts")
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for directory in dataset_dirs:
        rows.extend(_read_csv(directory / "timing_pairs.csv"))
        configs.append(json.loads((directory / "run_config.json").read_text(encoding="utf-8")))
        if (directory / "safety_failures.jsonl").exists():
            raise ValueError(f"safety failure artifact exists: {directory}")

    validation = _validate(rows, configs)
    by_dataset = _summaries(
        rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    pooled_rows = [
        {**row, "dataset": "pooled", "example_id": f'{row["dataset"]}::{row["example_id"]}'}
        for row in rows
    ]
    pooled = _summaries(
        pooled_rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    summary_rows = by_dataset + pooled
    payload = {
        "validation": validation,
        "by_dataset_and_semantic_mode": by_dataset,
        "pooled_by_semantic_mode": pooled,
        "execution_order_check": _order_summary(rows),
    }
    (args.input_root / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csv(args.input_root / "aggregate_summary.csv", summary_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
