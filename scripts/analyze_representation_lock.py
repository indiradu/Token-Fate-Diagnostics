#!/usr/bin/env python3
"""Rank matched A1->A2 representation-lock arms across datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from regret_remasking.data import normalize_answer


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["example_id"])] = row
    return rows


def _answer(record: dict[str, Any]) -> str:
    dataset = str(record.get("dataset", "")).lower()
    generation = str(record.get("generation", ""))
    if dataset in {"gsm8k", "math500", "math-500"}:
        return normalize_answer(generation)
    return generation


def _paired_rows(input_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        arm_files = {
            path.stem: path
            for path in input_dir.glob("*.jsonl")
            if path.stem.startswith(("P1_", "P2_"))
        }
        p1_arms = {name: _load(path) for name, path in arm_files.items() if name.startswith("P1_")}
        p2_arms = {name: _load(path) for name, path in arm_files.items() if name.startswith("P2_")}
        for p1_name, p1_records in sorted(p1_arms.items()):
            semantic_label = p1_name.removeprefix("P1_")
            prefix = f"P2_{semantic_label}_"
            for p2_name, p2_records in sorted(p2_arms.items()):
                if not p2_name.startswith(prefix) or not p2_name.endswith("_kv_cache"):
                    continue
                representation_label = p2_name[len(prefix) : -len("_kv_cache")]
                for example_id in sorted(set(p1_records) & set(p2_records)):
                    a1 = p1_records[example_id]
                    a2 = p2_records[example_id]
                    a1_tokens = list(a1.get("token_ids", []))
                    a2_tokens = list(a2.get("token_ids", []))
                    length = max(len(a1_tokens), len(a2_tokens), 1)
                    agreement = sum(
                        left == right for left, right in zip(a1_tokens, a2_tokens)
                    ) / length
                    paired.append(
                        {
                            "input_dir": str(input_dir),
                            "dataset": str(a2.get("dataset", input_dir.name)),
                            "example_id": example_id,
                            "semantic_label": semantic_label,
                            "representation_label": representation_label,
                            "representation_policy": str(a2.get("representation_policy", "")),
                            "representation_layer": int(a2.get("representation_layer", 0)),
                            "representation_threshold": float(a2.get("representation_threshold", 0.0)),
                            "representation_patience": int(a2.get("representation_patience", 0)),
                            "representation_min_age": int(a2.get("representation_min_age", 0)),
                            "representation_lock_fraction": float(a2.get("representation_lock_fraction", 0.0)),
                            "exact_sequence_match": int(a1_tokens == a2_tokens),
                            "token_agreement": agreement,
                            "answer_changed": int(_answer(a1) != _answer(a2)),
                            "correctness_changed": int(bool(a1.get("correct")) != bool(a2.get("correct"))),
                            "a1_correct_a2_incorrect": int(bool(a1.get("correct")) and not bool(a2.get("correct"))),
                            "a1_incorrect_a2_correct": int(not bool(a1.get("correct")) and bool(a2.get("correct"))),
                            "reference_lock_count": int(a2.get("reference_lock_count", 0)),
                            "reference_token_forwards": int(a2.get("reference_token_forwards", 0)),
                            "reference_opportunity_fraction": float(a2.get("reference_opportunity_fraction", 0.0)),
                            "runtime_gain": 1.0 - float(a2.get("latency_s", 0.0)) / max(float(a1.get("latency_s", 0.0)), 1e-9),
                        }
                    )
    if not paired:
        raise ValueError("no matched P1->P2 records found")
    return paired


def _summaries(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    config_fields = (
        "representation_policy",
        "representation_layer",
        "representation_threshold",
        "representation_patience",
        "representation_min_age",
        "representation_lock_fraction",
    )
    for key, group in groups.items():
        summary = {name: value for name, value in zip(keys, key)}
        for field in config_fields:
            values = {row[field] for row in group}
            summary[field] = next(iter(values)) if len(values) == 1 else "mixed"
        count = len(group)
        summary.update(
            {
                "examples": count,
                "exact_sequence_match_rate": sum(row["exact_sequence_match"] for row in group) / count,
                "mean_token_agreement": sum(row["token_agreement"] for row in group) / count,
                "answer_changed_rate": sum(row["answer_changed"] for row in group) / count,
                "correctness_changed_rate": sum(row["correctness_changed"] for row in group) / count,
                "a1_correct_a2_incorrect_rate": sum(row["a1_correct_a2_incorrect"] for row in group) / count,
                "a1_incorrect_a2_correct_rate": sum(row["a1_incorrect_a2_correct"] for row in group) / count,
                "mean_reference_lock_count": sum(row["reference_lock_count"] for row in group) / count,
                "mean_reference_token_forwards": sum(row["reference_token_forwards"] for row in group) / count,
                "mean_reference_opportunity_fraction": sum(row["reference_opportunity_fraction"] for row in group) / count,
                "mean_runtime_gain": sum(row["runtime_gain"] for row in group) / count,
            }
        )
        summary["strict_safety_pass"] = int(
            summary["exact_sequence_match_rate"] == 1.0
            and summary["answer_changed_rate"] == 0.0
            and summary["a1_correct_a2_incorrect_rate"] == 0.0
        )
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            *(str(row[key]) for key in keys[:-1]),
            -int(row["strict_safety_pass"]),
            -float(row["mean_reference_opportunity_fraction"]),
            str(row[keys[-1]]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _paired_rows(args.input_dirs)
    by_dataset = _summaries(pairs, ("dataset", "semantic_label", "representation_label"))
    pooled = _summaries(pairs, ("semantic_label", "representation_label"))
    _write_csv(args.output_dir / "pair_rows.csv", pairs)
    _write_csv(args.output_dir / "dataset_summary.csv", by_dataset)
    _write_csv(args.output_dir / "pooled_summary.csv", pooled)
    (args.output_dir / "pooled_summary.json").write_text(
        json.dumps(pooled, indent=2), encoding="utf-8"
    )

    report = [
        "# Representation-lock development ranking",
        "",
        "Every comparison is matched A1->A2 within the same semantic mode.",
        "",
    ]
    for semantic in sorted({str(row["semantic_label"]) for row in pooled}):
        report.extend([f"## {semantic}", ""])
        for row in [item for item in pooled if item["semantic_label"] == semantic]:
            report.append(
                f"- `{row['representation_label']}`: strict_pass={row['strict_safety_pass']}, "
                f"exact={row['exact_sequence_match_rate']:.3f}, "
                f"answer_changed={row['answer_changed_rate']:.3f}, "
                f"harm={row['a1_correct_a2_incorrect_rate']:.3f}, "
                f"reference_opportunity={row['mean_reference_opportunity_fraction']:.3f}."
            )
        report.append("")
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(pooled, indent=2), flush=True)


if __name__ == "__main__":
    main()
