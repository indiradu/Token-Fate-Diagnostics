#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.data import generated_answers_equivalent


Metric = Callable[[pd.DataFrame], float]
PAIR_SPECS = (
    ("fixed_full", "prefix_fixed"),
    ("fixed_full", "parallel_full"),
    ("fixed_full", "prefix_parallel"),
    ("fixed_full", "dual_parallel"),
    ("prefix_fixed", "prefix_parallel"),
    ("parallel_full", "prefix_parallel"),
    ("parallel_full", "dual_parallel"),
    ("prefix_parallel", "dual_parallel"),
)


def bool_series(rows: pd.DataFrame, column: str) -> pd.Series:
    values = rows[column]
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"column {column} contains non-boolean values")
    return normalized.eq("true")


def metric_functions(reference: str, candidate: str) -> dict[str, Metric]:
    output_column = f"{candidate}_output_changed_vs_{reference}"
    answer_column = f"{candidate}_answer_value_changed_vs_{reference}"
    return {
        "output_change_rate": lambda d: float(bool_series(d, output_column).mean()),
        "answer_value_change_rate": lambda d: float(bool_series(d, answer_column).mean()),
        "reference_accuracy": lambda d: float(bool_series(d, f"{reference}_correct").mean()),
        "candidate_accuracy": lambda d: float(bool_series(d, f"{candidate}_correct").mean()),
        "accuracy_delta": lambda d: float(
            (
                bool_series(d, f"{candidate}_correct").astype(float)
                - bool_series(d, f"{reference}_correct").astype(float)
            ).mean()
        ),
        "correctness_change_rate": lambda d: float(
            bool_series(d, f"{reference}_correct")
            .ne(bool_series(d, f"{candidate}_correct"))
            .mean()
        ),
        "correct_to_wrong_rate": lambda d: float(
            (
                bool_series(d, f"{reference}_correct")
                & ~bool_series(d, f"{candidate}_correct")
            ).mean()
        ),
        "wrong_to_correct_rate": lambda d: float(
            (
                ~bool_series(d, f"{reference}_correct")
                & bool_series(d, f"{candidate}_correct")
            ).mean()
        ),
        "mean_reference_nfe": lambda d: float(d[f"{reference}_nfe"].mean()),
        "mean_candidate_nfe": lambda d: float(d[f"{candidate}_nfe"].mean()),
        "relative_input_token_positions": lambda d: float(
            d[f"{candidate}_input_token_positions"].sum()
            / d[f"{reference}_input_token_positions"].sum()
        ),
        "wall_clock_speedup": lambda d: float(
            d[f"{reference}_latency_s"].sum() / d[f"{candidate}_latency_s"].sum()
        ),
    }


def add_derived_pair_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for reference, candidate in PAIR_SPECS:
        required = {
            "dataset",
            f"{reference}_token_ids",
            f"{candidate}_token_ids",
            f"{reference}_generation",
            f"{candidate}_generation",
        }
        if not required.issubset(rows.columns):
            continue
        output_column = f"{candidate}_output_changed_vs_{reference}"
        answer_column = f"{candidate}_answer_value_changed_vs_{reference}"
        if output_column not in rows:
            reference_tokens = rows[f"{reference}_token_ids"].map(json.loads)
            candidate_tokens = rows[f"{candidate}_token_ids"].map(json.loads)
            rows[output_column] = [
                reference_value != candidate_value
                for reference_value, candidate_value in zip(
                    reference_tokens, candidate_tokens, strict=True
                )
            ]
        if answer_column not in rows:
            rows[answer_column] = [
                not generated_answers_equivalent(
                    reference_generation, candidate_generation, dataset
                )
                for reference_generation, candidate_generation, dataset in zip(
                    rows[f"{reference}_generation"],
                    rows[f"{candidate}_generation"],
                    rows["dataset"],
                    strict=True,
                )
            ]
    return rows


def available_pairs(rows: pd.DataFrame) -> list[tuple[str, str]]:
    columns = set(rows.columns)
    return [
        (reference, candidate)
        for reference, candidate in PAIR_SPECS
        if {
            f"{reference}_correct",
            f"{candidate}_correct",
            f"{candidate}_output_changed_vs_{reference}",
            f"{candidate}_answer_value_changed_vs_{reference}",
        }.issubset(columns)
    ]


def bootstrap_metrics(rows: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rows), size=(reps, len(rows)))
    output: list[dict[str, float | str]] = []
    for reference, candidate in available_pairs(rows):
        metrics = metric_functions(reference, candidate)
        bootstrap_values = {name: [] for name in metrics}
        for indices in sampled:
            sample = rows.iloc[indices]
            for name, metric in metrics.items():
                bootstrap_values[name].append(metric(sample))
        for name, metric in metrics.items():
            low, high = np.percentile(bootstrap_values[name], [2.5, 97.5])
            output.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": name,
                    "point": metric(rows),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return pd.DataFrame(output)


def pair_counts(rows: pd.DataFrame, reference: str, candidate: str) -> dict[str, int | float]:
    reference_correct = bool_series(rows, f"{reference}_correct")
    candidate_correct = bool_series(rows, f"{candidate}_correct")
    correct_to_wrong = int((reference_correct & ~candidate_correct).sum())
    wrong_to_correct = int((~reference_correct & candidate_correct).sum())
    discordant = correct_to_wrong + wrong_to_correct
    sign_p = (
        float(binomtest(min(correct_to_wrong, wrong_to_correct), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "examples": int(len(rows)),
        "output_changed": int(
            bool_series(rows, f"{candidate}_output_changed_vs_{reference}").sum()
        ),
        "answer_value_changed": int(
            bool_series(rows, f"{candidate}_answer_value_changed_vs_{reference}").sum()
        ),
        "reference_correct": int(reference_correct.sum()),
        "candidate_correct": int(candidate_correct.sum()),
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "correctness_exact_sign_p": sign_p,
    }


def validate_run_artifact(rows: pd.DataFrame, config: dict) -> dict[str, int | bool]:
    expected_examples = int(config["limit"])
    if len(rows) != expected_examples:
        raise ValueError(f"partial run: expected {expected_examples} rows, found {len(rows)}")
    if rows["example_id"].nunique() != len(rows):
        raise ValueError("run contains duplicate example IDs")

    arms = list(config["arms"])
    generation_length = int(config["gen_length"])
    required_columns = ["example_id", "dataset"]
    for arm in arms:
        required_columns.extend(
            [
                f"{arm}_token_ids",
                f"{arm}_generation",
                f"{arm}_correct",
                f"{arm}_nfe",
                f"{arm}_forward_calls",
                f"{arm}_final_mask_count",
            ]
        )
    missing_columns = sorted(set(required_columns) - set(rows.columns))
    if missing_columns:
        raise ValueError(f"run is missing required columns: {missing_columns}")
    if rows[required_columns].isna().any().any():
        raise ValueError("run contains missing required values")

    total_masks = 0
    for arm in arms:
        total_masks += int(rows[f"{arm}_final_mask_count"].sum())
        if not rows[f"{arm}_nfe"].eq(rows[f"{arm}_forward_calls"]).all():
            raise ValueError(f"{arm} NFE does not match observed forward calls")
        lengths = rows[f"{arm}_token_ids"].map(lambda value: len(json.loads(value)))
        if not lengths.eq(generation_length).all():
            raise ValueError(f"{arm} has a response with the wrong token length")
    if total_masks:
        raise ValueError(f"run contains {total_masks} unresolved masks")
    return {
        "validated": True,
        "examples": len(rows),
        "unique_examples": int(rows["example_id"].nunique()),
        "arms": len(arms),
        "total_outputs": len(rows) * len(arms),
        "total_unresolved_masks": total_masks,
    }


def table_text(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap paired official Fast-dLLM component comparisons."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    raw_rows = pd.read_csv(run_dir / "example_results.csv")
    validation = validate_run_artifact(raw_rows, config)
    rows = add_derived_pair_columns(raw_rows)
    metrics = bootstrap_metrics(rows, args.bootstrap_reps, args.seed)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    counts = {
        f"{reference}->{candidate}": pair_counts(rows, reference, candidate)
        for reference, candidate in available_pairs(rows)
    }
    summary = {
        "source_run": str(run_dir),
        "examples": int(len(rows)),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "example",
        "validation": validation,
        "runner_sha256": config.get("runner_sha256"),
        "pair_counts": counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# Official Fast-dLLM v1 Reanalysis\n\n"
        f"Examples: {len(rows)}. Bootstrap: {args.bootstrap_reps} example resamples.\n\n"
        + table_text(metrics)
        + "\n\n```json\n"
        + json.dumps(counts, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(table_text(metrics))


if __name__ == "__main__":
    main()
