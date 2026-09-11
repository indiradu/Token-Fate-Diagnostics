#!/usr/bin/env python3
"""Audit semantic-lock selectors under a matched acceleration budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from regret_remasking.data import normalize_answer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_example(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in _read_jsonl(path)}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _answer_value(record: dict[str, Any]) -> str | None:
    dataset = str(record.get("dataset", "")).lower()
    if dataset in {"gsm8k", "math500", "math-500"}:
        return normalize_answer(str(record.get("generation", "")))
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = _by_example(input_dir / "A0.jsonl")
    if not baseline:
        raise ValueError(f"missing or empty A0.jsonl in {input_dir}")
    ledger = _read_jsonl(input_dir / "phase_ledger.jsonl")
    event_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("P1_*.jsonl")):
        arm = path.stem
        policy = arm.removeprefix("P1_")
        candidate = _by_example(path)
        common = sorted(set(baseline) & set(candidate))
        policy_events = [
            event
            for event in ledger
            if event.get("arm") == arm
            and event.get("phase") == "semantic_commit"
            and event.get("selection_source") == "accelerated"
        ]

        for event in policy_events:
            example_id = str(event["example_id"])
            relative_position = int(event["relative_position"])
            a0 = baseline.get(example_id)
            p1 = candidate.get(example_id)
            if a0 is None or p1 is None or relative_position >= len(a0.get("token_ids", [])):
                continue
            baseline_token = int(a0["token_ids"][relative_position])
            committed_token = int(event["token_id"])
            event_rows.append(
                {
                    "semantic_policy": policy,
                    "example_id": example_id,
                    "relative_position": relative_position,
                    "global_step": int(event["global_step"]),
                    "committed_token_id": committed_token,
                    "a0_final_token_id": baseline_token,
                    "matches_a0_final_token": int(committed_token == baseline_token),
                    "a0_correct": int(bool(a0["correct"])),
                    "p1_correct": int(bool(p1["correct"])),
                    "confidence": float(event["confidence_online"]),
                    "margin": float(event["margin_online"]),
                    "entropy": float(event["entropy_online"]),
                    "posterior_kl": float(event["kl_online"]),
                    "runlength": float(event["runlength_online"]),
                    "block_runlength": float(
                        event.get("block_runlength_online", event["runlength_online"])
                    ),
                    "top1_flip": float(event["top1_flip_online"]),
                    "context_volatility": float(event["context_volatility_online"]),
                    "predicted_regret": event.get("predicted_regret"),
                    "selector_fallback": event.get("selector_fallback"),
                }
            )

        exact: list[float] = []
        agreement: list[float] = []
        harm: list[float] = []
        gain: list[float] = []
        answer_changed: list[float] = []
        correctness_changed: list[float] = []
        nfe_reduction: list[float] = []
        masked_reduction: list[float] = []
        runtime_gain: list[float] = []
        accuracy_delta: list[float] = []
        accelerated_counts: list[float] = []
        for example_id in common:
            a0 = baseline[example_id]
            p1 = candidate[example_id]
            a0_tokens = list(a0.get("token_ids", []))
            p1_tokens = list(p1.get("token_ids", []))
            compared = min(len(a0_tokens), len(p1_tokens))
            exact.append(float(a0_tokens == p1_tokens))
            agreement.append(
                sum(a0_tokens[index] == p1_tokens[index] for index in range(compared))
                / max(1, max(len(a0_tokens), len(p1_tokens)))
            )
            a0_correct = bool(a0["correct"])
            p1_correct = bool(p1["correct"])
            harm.append(float(a0_correct and not p1_correct))
            gain.append(float(not a0_correct and p1_correct))
            a0_answer = _answer_value(a0)
            p1_answer = _answer_value(p1)
            if a0_answer is not None and p1_answer is not None:
                answer_changed.append(float(a0_answer != p1_answer))
            correctness_changed.append(float(a0_correct != p1_correct))
            accuracy_delta.append(float(p1_correct) - float(a0_correct))
            nfe_reduction.append(1.0 - float(p1["nfe"]) / max(float(a0["nfe"]), 1.0))
            masked_reduction.append(
                1.0
                - float(p1["masked_token_forwards"])
                / max(float(a0["masked_token_forwards"]), 1.0)
            )
            runtime_gain.append(
                1.0 - float(p1["latency_s"]) / max(float(a0["latency_s"]), 1e-9)
            )
            accelerated_counts.append(float(p1.get("accelerated_commit_count", 0)))

        selected = [row for row in event_rows if row["semantic_policy"] == policy]
        matches = [float(row["matches_a0_final_token"]) for row in selected]
        high_confidence = [row for row in selected if float(row["confidence"]) >= 0.9]
        lock_precision_low, lock_precision_high = _wilson_interval(int(sum(matches)), len(matches))
        harm_low, harm_high = _wilson_interval(int(sum(harm)), len(harm))
        summaries.append(
            {
                "semantic_policy": policy,
                "examples": len(common),
                "accelerated_lock_events": len(selected),
                "mean_accelerated_locks_per_example": _mean(accelerated_counts),
                "retrospective_lock_precision_vs_a0": _mean(matches),
                "retrospective_lock_precision_wilson95_low": lock_precision_low,
                "retrospective_lock_precision_wilson95_high": lock_precision_high,
                "accelerated_lock_error_rate_vs_a0": None if not matches else 1.0 - float(_mean(matches)),
                "high_confidence_accelerated_locks": len(high_confidence),
                "high_confidence_lock_error_rate_vs_a0": (
                    _mean([1.0 - float(row["matches_a0_final_token"]) for row in high_confidence])
                ),
                "exact_sequence_match_rate": _mean(exact),
                "mean_token_agreement": _mean(agreement),
                "normalized_answer_changed_rate": _mean(answer_changed),
                "correctness_changed_rate": _mean(correctness_changed),
                "answer_harm_rate": _mean(harm),
                "answer_harm_wilson95_low": harm_low,
                "answer_harm_wilson95_high": harm_high,
                "answer_gain_rate": _mean(gain),
                "accuracy_delta": _mean(accuracy_delta),
                "mean_nfe_reduction": _mean(nfe_reduction),
                "mean_masked_token_forward_reduction": _mean(masked_reduction),
                "mean_runtime_gain": _mean(runtime_gain),
                "posterior_history_fallback_events": sum(
                    row["selector_fallback"] == "confidence_no_posterior_history" for row in selected
                ),
            }
        )
    return event_rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, summaries = audit(args.input_dir)
    if not summaries:
        raise ValueError("no P1 semantic-lock arms found")
    _write_csv(args.output_dir / "accelerated_lock_events.csv", events)
    _write_csv(args.output_dir / "semantic_selector_summary.csv", summaries)
    (args.output_dir / "semantic_selector_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# Semantic-lock selector audit",
        "",
        "Only selector-controlled accelerated locks are used for lock precision; normal scheduled transfers are excluded.",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- `{row['semantic_policy']}`: lock precision={_fmt(row['retrospective_lock_precision_vs_a0'])}, "
            f"exact sequence={_fmt(row['exact_sequence_match_rate'])}, answer harm={_fmt(row['answer_harm_rate'])}, "
            f"NFE reduction={_fmt(row['mean_nfe_reduction'])}, runtime gain={_fmt(row['mean_runtime_gain'])}."
        )
    (args.output_dir / "semantic_selector_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
