#!/usr/bin/env python3
"""Aggregate the matched 60-example capped-confidence/direct-safety run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_phaselock_matched_plans import _bootstrap_gain


EXPECTED_METHODS = ("dense", "capped_confidence", "safety_full")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=313)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    timing_medians: list[dict[str, float]] = []
    datasets: dict[str, Any] = {}
    for run in args.run:
        summary = _load(run / "summary.json")
        run_rows = _load(run / "rows.json")
        timing = _load(run / "timing.json")
        dataset = str(summary["dataset"])
        if tuple(summary["methods"]) != EXPECTED_METHODS:
            raise ValueError(f"unexpected method order in {run}: {list(summary['methods'])}")
        if len(run_rows) != 20:
            raise ValueError(f"expected 20 examples in {run}, found {len(run_rows)}")
        datasets[dataset] = summary
        rows.extend(run_rows)
        for batch_id in sorted({int(item["batch_id"]) for item in timing}):
            group = [item for item in timing if int(item["batch_id"]) == batch_id]
            timing_medians.append(
                {
                    "dataset": dataset,
                    "batch_id": float(batch_id),
                    **{
                        f"{method}_latency_s": float(
                            np.median(
                                [item[f"{method}_latency_s"] for item in group]
                            )
                        )
                        for method in EXPECTED_METHODS
                    },
                    **{
                        f"{method}_controller_latency_s": float(
                            np.median(
                                [
                                    item[f"{method}_controller_latency_s"]
                                    for item in group
                                ]
                            )
                        )
                        for method in EXPECTED_METHODS
                    },
                }
            )

    if len(rows) != 60 or len(timing_medians) != 15:
        raise ValueError(
            f"expected 60 examples and 15 batches, found {len(rows)} and "
            f"{len(timing_medians)}"
        )
    if len({(str(row["dataset"]), str(row["example_id"])) for row in rows}) != 60:
        raise ValueError("matched evaluation examples are not unique")

    methods: dict[str, Any] = {}
    dense_correct = [int(row["dense_correct"]) for row in rows]
    for offset, method in enumerate(EXPECTED_METHODS):
        correctness = [int(row[f"{method}_correct"]) for row in rows]
        regressions = sum(
            left == 1 and right == 0
            for left, right in zip(dense_correct, correctness)
        )
        improvements = sum(
            left == 0 and right == 1
            for left, right in zip(dense_correct, correctness)
        )
        gain, interval = _bootstrap_gain(
            timing_medians,
            f"{method}_latency_s",
            "dense_latency_s",
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        methods[method] = {
            "accuracy": float(np.mean(correctness)),
            "exact_dense_rate": float(
                np.mean([row[f"{method}_exact_dense"] for row in rows])
            ),
            "answer_change_rate": float(
                np.mean([row[f"{method}_answer_changed_dense"] for row in rows])
            ),
            "regressions_vs_dense": int(regressions),
            "improvements_vs_dense": int(improvements),
            "mean_nfe": float(np.mean([row[f"{method}_nfe"] for row in rows])),
            "runtime_gain_vs_dense": gain,
            "runtime_gain_vs_dense_ci95": interval,
            "mean_controller_latency_s_per_batch": float(
                np.mean(
                    [
                        item[f"{method}_controller_latency_s"]
                        for item in timing_medians
                    ]
                )
            ),
        }

    result = {
        "examples": len(rows),
        "batches": len(timing_medians),
        "datasets": datasets,
        "methods": methods,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
