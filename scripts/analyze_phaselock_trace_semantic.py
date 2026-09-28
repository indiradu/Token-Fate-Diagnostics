#!/usr/bin/env python3
"""Aggregate and select a preregistered Trace-assisted PhaseLock policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POLICY_ORDER = {
    "capped_confidence": 0,
    "direct_safety_full": 1,
    "guarded": 2,
    "balanced": 3,
    "open": 4,
    "open_confidence": 5,
    "open_joint": 6,
    "sure_persist2_conf80": 7,
    "sure_conf90": 8,
    "trace_sure_union": 9,
    "sure_raw": 10,
    "trace_surelite_union_d02": 11,
    "trace_surelite_union_d05": 12,
    "surelite_p2_d02": 13,
    "surelite_p2_d05": 14,
    "trace_surelite_union_d02_c90_p2": 15,
    "trace_surelite_union_d02_c80_p3": 16,
    "trace_surelite_union_d02_c80_p2": 17,
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one quality row per example and reject inconsistent replays."""
    stable: dict[str, dict[str, Any]] = {}
    quality_keys = (
        "a0_correct",
        "a1_correct",
        "a2_correct",
        "a3_correct",
        "a1_exact_a0",
        "a2_exact_a1",
        "a3_exact_a2",
        "a1_answer_changed_a0",
        "a2_answer_changed_a1",
        "a3_answer_changed_a2",
        "a0_nfe",
        "a1_nfe",
        "a2_nfe",
        "a3_nfe",
    )
    for row in rows:
        example_id = str(row["example_id"])
        previous = stable.get(example_id)
        if previous is None:
            stable[example_id] = row
            continue
        if any(previous[key] != row[key] for key in quality_keys):
            raise ValueError(f"non-deterministic replay for {example_id}")
    return list(stable.values())


def _transition(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    exact_key = f"{right}_exact_{left}"
    changed_key = f"{right}_answer_changed_{left}"
    regressions = sum(
        int(row[f"{left}_correct"] == 1 and row[f"{right}_correct"] == 0)
        for row in rows
    )
    improvements = sum(
        int(row[f"{left}_correct"] == 0 and row[f"{right}_correct"] == 1)
        for row in rows
    )
    return {
        "exact_rate": float(np.mean([row[exact_key] for row in rows])),
        "answer_change_rate": float(np.mean([row[changed_key] for row in rows])),
        "regressions": regressions,
        "improvements": improvements,
    }


def _paired_gain(
    timing: list[dict[str, float]], candidate_key: str, *, samples: int, seed: int
) -> dict[str, Any]:
    baseline = np.asarray([row["a0_latency_s"] for row in timing], dtype=np.float64)
    candidate = np.asarray([row[candidate_key] for row in timing], dtype=np.float64)
    gains = 1.0 - candidate / baseline
    rng = np.random.default_rng(seed)
    if len(gains) == 1:
        ci = [float(gains[0]), float(gains[0])]
    else:
        boot = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            choice = rng.integers(0, len(gains), size=len(gains))
            boot[index] = float(gains[choice].mean())
        ci = [float(value) for value in np.quantile(boot, [0.025, 0.975])]
    return {
        "mean": float(gains.mean()),
        "ci95": ci,
        "pairs": int(len(gains)),
    }


def _median_timing_rows(
    run_name: str, timing: list[dict[str, Any]]
) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in timing:
        grouped[int(row["batch_id"])].append(row)
    output: list[dict[str, float]] = []
    keys = (
        "a0_latency_s",
        "a2_latency_s",
        "a3_latency_s",
        "estimated_online_a3_latency_s",
        "controller_overhead_s",
        "removed_row_token_forwards",
        "compute_switched",
    )
    for batch_id, batch_rows in sorted(grouped.items()):
        aggregate = {
            key: float(np.median([float(row[key]) for row in batch_rows]))
            for key in keys
        }
        aggregate["run_name"] = run_name
        aggregate["batch_id"] = float(batch_id)
        output.append(aggregate)
    return output


def aggregate_policy(
    policy: str,
    runs: list[tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = [
        row
        for _path, run_rows, _timing in runs
        for row in stable_rows(run_rows)
    ]
    timing = [
        row
        for path, _rows, run_timing in runs
        for row in _median_timing_rows(path.name, run_timing)
    ]
    if not rows or not timing:
        raise ValueError(f"policy {policy} has no rows or timing pairs")
    datasets = sorted({str(row["dataset"]) for row in rows})
    a0_a1 = _transition(rows, "a0", "a1")
    a1_a2 = _transition(rows, "a1", "a2")
    a2_a3 = _transition(rows, "a2", "a3")
    eligible = bool(
        a0_a1["regressions"] == 0
        and a1_a2["regressions"] == 0
        and np.isclose(a1_a2["exact_rate"], 1.0)
        and a2_a3["regressions"] == 0
        and np.isclose(a2_a3["exact_rate"], 1.0)
    )
    return {
        "policy": policy,
        "datasets": datasets,
        "examples": len(rows),
        "timing_pairs": len(timing),
        "eligible": eligible,
        "accuracy": {
            arm: float(np.mean([row[f"{arm}_correct"] for row in rows]))
            for arm in ("a0", "a1", "a2", "a3")
        },
        "a0_to_a1": a0_a1,
        "a1_to_a2": a1_a2,
        "a2_to_a3": a2_a3,
        "mean_nfe": {
            arm: float(np.mean([row[f"{arm}_nfe"] for row in rows]))
            for arm in ("a0", "a1", "a2", "a3")
        },
        "executor_gain_a3_vs_a0": _paired_gain(
            timing, "a3_latency_s", samples=bootstrap_samples, seed=seed
        ),
        "estimated_online_gain_a3_vs_a0": _paired_gain(
            timing,
            "estimated_online_a3_latency_s",
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "compute_gain_a3_vs_a2": {
            **_paired_gain(timing, "a3_latency_s", samples=bootstrap_samples, seed=seed + 2),
            "note": "For this field only, gain is recomputed below against A2.",
        },
        "mean_controller_overhead_s": float(
            np.mean([row["controller_overhead_s"] for row in timing])
        ),
        "mean_removed_row_token_forwards": float(
            np.mean([row["removed_row_token_forwards"] for row in timing])
        ),
        "compute_switch_rate": float(
            np.mean([row["compute_switched"] for row in timing])
        ),
    }


def _replace_compute_gain(
    aggregate: dict[str, Any],
    runs: list[tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    timing = [
        row
        for path, _rows, run_timing in runs
        for row in _median_timing_rows(path.name, run_timing)
    ]
    transformed = [
        {"a0_latency_s": row["a2_latency_s"], "candidate": row["a3_latency_s"]}
        for row in timing
    ]
    aggregate["compute_gain_a3_vs_a2"] = _paired_gain(
        transformed, "candidate", samples=bootstrap_samples, seed=seed
    )


def select_policy(aggregates: list[dict[str, Any]]) -> str | None:
    eligible = [row for row in aggregates if row["eligible"]]
    if not eligible:
        return None
    best_gain = max(row["estimated_online_gain_a3_vs_a0"]["mean"] for row in eligible)
    within_tie = [
        row
        for row in eligible
        if row["estimated_online_gain_a3_vs_a0"]["mean"] >= best_gain - 0.01
    ]
    return min(within_tie, key=lambda row: POLICY_ORDER[row["policy"]])["policy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=251)
    args = parser.parse_args()

    by_policy: dict[str, list[tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]]] = defaultdict(list)
    for path in args.run:
        summary = _load(path / "summary.json")
        policy = str(summary["policy"])
        if policy not in POLICY_ORDER:
            raise ValueError(f"unknown policy {policy!r} in {path}")
        by_policy[policy].append((path, _load(path / "rows.json"), _load(path / "timing_batches.json")))

    aggregates = []
    for offset, policy in enumerate(sorted(by_policy, key=POLICY_ORDER.get)):
        runs = by_policy[policy]
        aggregate = aggregate_policy(
            policy,
            runs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10 * offset,
        )
        _replace_compute_gain(
            aggregate,
            runs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10 * offset + 3,
        )
        aggregates.append(aggregate)

    selected = select_policy(aggregates)
    result = {
        "selection_rule": (
            "zero A0->A1 correctness regressions; exact, zero-regression A1->A2 and "
            "A2->A3; maximize pooled estimated online gain; within 1pp choose "
            "the earlier policy in the preregistered conservative order"
        ),
        "selected_policy": selected,
        "policies": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
