#!/usr/bin/env python3
"""Analyze dense-A1 low-drift plans against count/time-matched high-drift controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in _jsonl(path)}


def _events(plan: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [event for step in sorted(plan, key=int) for event in plan[step]]


def _counts(plan: dict[str, list[dict[str, Any]]]) -> dict[int, int]:
    return {int(step): len(events) for step, events in plan.items() if events}


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _analyze_dir(input_dir: Path, matches: dict[str, str]) -> list[dict[str, Any]]:
    plans = _jsonl(input_dir / "phase_plans.jsonl")
    by_key = {
        (
            str(row["example_id"]),
            str(row["semantic_policy_label"]),
            str(row["representation_policy_label"]),
        ): row
        for row in plans
    }
    record_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def arm_records(name: str) -> dict[str, dict[str, Any]]:
        if name not in record_cache:
            record_cache[name] = _records(input_dir / f"{name}.jsonl")
        return record_cache[name]

    rows: list[dict[str, Any]] = []
    for (example_id, semantic_label, high_label), high_row in sorted(by_key.items()):
        if high_label not in matches:
            continue
        low_label = matches[high_label]
        low_row = by_key[(example_id, semantic_label, low_label)]
        low_plan = low_row["reference_plan"]
        high_plan = high_row["reference_plan"]
        low_events = _events(low_plan)
        high_events = _events(high_plan)
        low_positions = {int(event["position"]) for event in low_events}
        high_positions = {int(event["position"]) for event in high_events}
        low_event_keys = {
            (int(event["global_step"]), int(event["position"])) for event in low_events
        }
        high_event_keys = {
            (int(event["global_step"]), int(event["position"])) for event in high_events
        }
        low_drifts = [float(event["representation_drift"]) for event in low_events]
        high_drifts = [float(event["representation_drift"]) for event in high_events]

        a1_name = f"P1_{semantic_label}"
        low_name = f"P2_{semantic_label}_{low_label}_kv_cache"
        high_name = f"P2_{semantic_label}_{high_label}_kv_cache"
        a1 = arm_records(a1_name)[example_id]
        low = arm_records(low_name)[example_id]
        high = arm_records(high_name)[example_id]
        rows.append(
            {
                "input_dir": str(input_dir),
                "dataset": str(a1["dataset"]),
                "example_id": example_id,
                "semantic_label": semantic_label,
                "low_label": low_label,
                "high_label": high_label,
                "exact_step_count_match": int(_counts(low_plan) == _counts(high_plan)),
                "low_lock_count": len(low_events),
                "high_lock_count": len(high_events),
                "low_mean_source_drift": _safe_mean(low_drifts),
                "high_mean_source_drift": _safe_mean(high_drifts),
                "mean_source_drift_gap": _safe_mean(high_drifts) - _safe_mean(low_drifts),
                "selected_position_jaccard": len(low_positions & high_positions)
                / max(1, len(low_positions | high_positions)),
                "selected_event_jaccard": len(low_event_keys & high_event_keys)
                / max(1, len(low_event_keys | high_event_keys)),
                "low_exact_a1": int(low["token_ids"] == a1["token_ids"]),
                "high_exact_a1": int(high["token_ids"] == a1["token_ids"]),
                "low_correctness_changed": int(bool(low["correct"]) != bool(a1["correct"])),
                "high_correctness_changed": int(bool(high["correct"]) != bool(a1["correct"])),
                "low_harm": int(bool(a1["correct"]) and not bool(low["correct"])),
                "high_harm": int(bool(a1["correct"]) and not bool(high["correct"])),
                "low_reference_opportunity": float(low["reference_opportunity_fraction"]),
                "high_reference_opportunity": float(high["reference_opportunity_fraction"]),
            }
        )
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["semantic_label"]),
            str(row["low_label"]),
            str(row["high_label"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        n = len(group)
        output.append(
            {
                "dataset": key[0],
                "semantic_label": key[1],
                "low_label": key[2],
                "high_label": key[3],
                "examples": n,
                "exact_step_count_match_rate": mean(row["exact_step_count_match"] for row in group),
                "mean_source_drift_gap": mean(row["mean_source_drift_gap"] for row in group),
                "mean_selected_position_jaccard": mean(row["selected_position_jaccard"] for row in group),
                "mean_selected_event_jaccard": mean(row["selected_event_jaccard"] for row in group),
                "low_sequence_change_rate": 1.0 - mean(row["low_exact_a1"] for row in group),
                "high_sequence_change_rate": 1.0 - mean(row["high_exact_a1"] for row in group),
                "low_harm_rate": mean(row["low_harm"] for row in group),
                "high_harm_rate": mean(row["high_harm"] for row in group),
                "low_reference_opportunity": mean(row["low_reference_opportunity"] for row in group),
                "high_reference_opportunity": mean(row["high_reference_opportunity"] for row in group),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = {
        str(spec["label"]): str(spec["match_reference_counts_from"])
        for spec in manifest["representation_policies"]
        if "match_reference_counts_from" in spec
    }
    if not matches:
        raise ValueError("manifest contains no matched high-drift controls")
    rows = [row for input_dir in args.input_dirs for row in _analyze_dir(input_dir, matches)]
    if not rows:
        raise ValueError("no matched representation rows found")
    summaries = _summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "matched_rows.csv", rows)
    _write_csv(args.output_dir / "matched_summary.csv", summaries)
    (args.output_dir / "matched_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    if any(not row["exact_step_count_match"] for row in rows):
        raise RuntimeError("a high-drift control did not match its low-drift plan's step counts")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
