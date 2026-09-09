#!/usr/bin/env python3
"""Analyze nested PhaseLock arm outputs and write paired metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from regret_remasking.data import normalize_answer


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _answer_value(record: dict[str, Any]) -> str | None:
    dataset = str(record.get("dataset", "")).lower()
    if dataset in {"gsm8k", "math500", "math-500"}:
        return normalize_answer(str(record.get("generation", "")))
    return None


def _pair(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> dict[str, Any]:
    ref_tokens = list(reference.get("token_ids", []))
    cand_tokens = list(candidate.get("token_ids", []))
    length = min(len(ref_tokens), len(cand_tokens))
    agreement = sum(ref_tokens[idx] == cand_tokens[idx] for idx in range(length)) / max(1, max(len(ref_tokens), len(cand_tokens)))
    reference_answer = _answer_value(reference)
    candidate_answer = _answer_value(candidate)
    if reference_answer is None or candidate_answer is None:
        answer_changed = str(reference.get("generation", "")) != str(candidate.get("generation", ""))
    else:
        answer_changed = reference_answer != candidate_answer
    return {
        "example_id": str(reference["example_id"]),
        "pair": label,
        "reference_arm": reference["arm"],
        "candidate_arm": candidate["arm"],
        "exact_sequence_match": int(ref_tokens == cand_tokens),
        "token_agreement": agreement,
        "answer_changed": int(answer_changed),
        "correctness_changed": int(bool(reference["correct"]) != bool(candidate["correct"])),
        "reference_answer_value": reference_answer,
        "candidate_answer_value": candidate_answer,
        "reference_correct": int(bool(reference["correct"])),
        "candidate_correct": int(bool(candidate["correct"])),
        "reference_correct_candidate_incorrect": int(bool(reference["correct"]) and not bool(candidate["correct"])),
        "reference_incorrect_candidate_correct": int(not bool(reference["correct"]) and bool(candidate["correct"])),
        "reference_latency_s": float(reference["latency_s"]),
        "candidate_latency_s": float(candidate["latency_s"]),
        "runtime_gain": 1.0 - float(candidate["latency_s"]) / max(float(reference["latency_s"]), 1e-9),
        "reference_nfe": int(reference["nfe"]),
        "candidate_nfe": int(candidate["nfe"]),
    }


def _write_drift_audit(
    output: Path,
    arms: dict[str, dict[str, dict[str, Any]]],
    ledger: list[dict[str, Any]],
    threshold: float,
) -> None:
    """Relate P1 representation drift to per-position A0/P1 divergence.

    This is an observational diagnostic, not a causal estimate.  The causal
    representation test remains the nested P1->P2 intervention above.  Here
    we ask whether positions with larger post-commit drift are more likely to
    finish with a different token from A0, which is the operational check for
    premise C at the token level.
    """
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, int], list[float]] = {}
    for event in ledger:
        if event.get("phase") != "representation_observation":
            continue
        if event.get("compute_policy") != "dense":
            continue
        # P1 keeps representations live and is tagged with representation
        # policy ``none``.  A0 also has dense observations, but it is the
        # baseline and must not be mixed into the P1 drift audit.
        if event.get("representation_policy") != "none":
            continue
        semantic = str(event.get("semantic_policy", ""))
        example_id = str(event.get("example_id", ""))
        position = int(event.get("relative_position", -1))
        drift = float(event.get("representation_drift", 0.0))
        if position >= 0:
            by_key.setdefault((semantic, example_id, position), []).append(drift)

    for semantic in sorted({key[0] for key in by_key}):
        p1_rows = arms.get(f"P1_{semantic}", {})
        a0_rows = arms.get("A0", {})
        for example_id, p1 in p1_rows.items():
            a0 = a0_rows.get(example_id)
            if a0 is None:
                continue
            p1_tokens = list(p1.get("token_ids", []))
            a0_tokens = list(a0.get("token_ids", []))
            for (policy_name, row_example, position), values in by_key.items():
                if policy_name != semantic or row_example != example_id:
                    continue
                if position >= len(p1_tokens) or position >= len(a0_tokens):
                    continue
                rows.append(
                    {
                        "semantic_policy": semantic,
                        "example_id": example_id,
                        "relative_position": position,
                        "observations": len(values),
                        "max_representation_drift": max(values),
                        "median_representation_drift": statistics.median(values),
                        "token_changed_from_a0": int(p1_tokens[position] != a0_tokens[position]),
                        "a0_correct": int(bool(a0.get("correct"))),
                        "p1_correct": int(bool(p1.get("correct"))),
                    }
                )

    if not rows:
        return
    fields = list(rows[0])
    with (output / "drift_token_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["semantic_policy"]), []).append(row)
    for semantic, policy_rows in sorted(grouped.items()):
        changed = [r for r in policy_rows if r["token_changed_from_a0"]]
        unchanged = [r for r in policy_rows if not r["token_changed_from_a0"]]
        high = [r for r in policy_rows if float(r["max_representation_drift"]) > threshold]
        low = [r for r in policy_rows if float(r["max_representation_drift"]) <= threshold]
        summaries.append(
            {
                "semantic_policy": semantic,
                "positions": len(policy_rows),
                "changed_positions": len(changed),
                "changed_position_rate": len(changed) / len(policy_rows),
                "median_max_drift_changed": statistics.median(
                    [float(r["max_representation_drift"]) for r in changed]
                ) if changed else None,
                "median_max_drift_unchanged": statistics.median(
                    [float(r["max_representation_drift"]) for r in unchanged]
                ) if unchanged else None,
                "high_drift_positions": len(high),
                "high_drift_changed_rate": (
                    sum(int(r["token_changed_from_a0"]) for r in high) / len(high) if high else None
                ),
                "low_drift_positions": len(low),
                "low_drift_changed_rate": (
                    sum(int(r["token_changed_from_a0"]) for r in low) / len(low) if low else None
                ),
                "drift_threshold": threshold,
            }
        )
    with (output / "drift_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    arms = {path.stem: _load(path) for path in args.input_dir.glob("*.jsonl") if path.stem != "phase_ledger"}
    if "A0" not in arms:
        raise ValueError("A0.jsonl is required")
    pairs: list[dict[str, Any]] = []
    for arm_name, rows in sorted(arms.items()):
        if not arm_name.startswith("P1_"):
            continue
        semantic = arm_name.removeprefix("P1_")
        for example_id, candidate in rows.items():
            if example_id in arms["A0"]:
                pairs.append(_pair(arms["A0"][example_id], candidate, f"A0_vs_{arm_name}"))
        for ref_name, ref_rows in sorted(arms.items()):
            prefix = f"P2_{semantic}_"
            if not ref_name.startswith(prefix) or not ref_name.endswith("_kv_cache"):
                continue
            for example_id, candidate in ref_rows.items():
                if example_id in rows:
                    pairs.append(_pair(rows[example_id], candidate, f"{arm_name}_vs_{ref_name}"))
            representation_suffix = ref_name[len(prefix):-len("_kv_cache")]
            for compute_name in ("row_sparse", "row_sparse_packed"):
                row_sparse = f"P3_{semantic}_" + representation_suffix + f"_{compute_name}"
                if row_sparse not in arms:
                    continue
                for example_id, candidate in arms[row_sparse].items():
                    if example_id in ref_rows:
                        pairs.append(_pair(ref_rows[example_id], candidate, f"{ref_name}_vs_{row_sparse}"))
    if not pairs:
        raise ValueError("no paired arms found")
    fieldnames = list(pairs[0])
    with (output / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    summaries: list[dict[str, Any]] = []
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_pair.setdefault(str(row["pair"]), []).append(row)
    for pair, rows in sorted(by_pair.items()):
        summaries.append(
            {
                "pair": pair,
                "examples": len(rows),
                "exact_sequence_match_rate": sum(int(r["exact_sequence_match"]) for r in rows) / len(rows),
                "mean_token_agreement": sum(float(r["token_agreement"]) for r in rows) / len(rows),
                "answer_changed_rate": sum(int(r["answer_changed"]) for r in rows) / len(rows),
                "correctness_changed_rate": sum(int(r["correctness_changed"]) for r in rows) / len(rows),
                "reference_correct_candidate_incorrect_rate": sum(int(r["reference_correct_candidate_incorrect"]) for r in rows) / len(rows),
                "reference_incorrect_candidate_correct_rate": sum(int(r["reference_incorrect_candidate_correct"]) for r in rows) / len(rows),
                "mean_runtime_gain": sum(float(r["runtime_gain"]) for r in rows) / len(rows),
                "positive_runtime_gain_rate": sum(float(r["runtime_gain"]) > 0 for r in rows) / len(rows),
                "mean_reference_nfe": sum(int(r["reference_nfe"]) for r in rows) / len(rows),
                "mean_candidate_nfe": sum(int(r["candidate_nfe"]) for r in rows) / len(rows),
            }
        )
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report: list[str] = ["# PhaseLock paired audit", "", "Pairs are nested: A0→P1 tests semantic commitment; P1→P2 tests reference freezing; P2→P3 tests row compute removal.", ""]
    for row in summaries:
        report.append(
            f"- `{row['pair']}`: exact={row['exact_sequence_match_rate']:.3f}, "
            f"answer_changed={row['answer_changed_rate']:.3f}, "
            f"reference→candidate harm={row['reference_correct_candidate_incorrect_rate']:.3f}, "
            f"runtime_gain={row['mean_runtime_gain']:.3f}."
        )
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    ledger = _load_ledger(args.input_dir / "phase_ledger.jsonl")
    if ledger:
        config_path = args.input_dir / "run_config.json"
        threshold = 0.02
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                threshold = float(json.load(handle).get("representation_threshold", threshold))
        _write_drift_audit(output, arms, ledger, threshold)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
