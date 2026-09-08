#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


Metric = Callable[[pd.DataFrame], float]


def upstream_layer_recompute_fraction(rows: pd.DataFrame) -> pd.Series:
    column = (
        "elastic_upstream_layer_recompute_fraction"
        if "elastic_upstream_layer_recompute_fraction" in rows
        else "elastic_cache_update_frequency"
        if "elastic_cache_update_frequency" in rows
        else "elastic_layer_compute_fraction"
    )
    return rows[column]


def metric_functions() -> dict[str, Metric]:
    return {
        "output_change_rate": lambda d: float(d["output_changed"].mean()),
        "mean_changed_token_count": lambda d: float(d["changed_token_count"].mean()),
        "answer_value_change_rate": lambda d: float(d["answer_value_changed"].mean()),
        "baseline_accuracy": lambda d: float(d["baseline_correct"].mean()),
        "elastic_accuracy": lambda d: float(d["elastic_correct"].mean()),
        "accuracy_delta": lambda d: float(
            (d["elastic_correct"].astype(float) - d["baseline_correct"].astype(float)).mean()
        ),
        "correctness_change_rate": lambda d: float(
            d["baseline_correct"].ne(d["elastic_correct"]).mean()
        ),
        "correct_to_wrong_rate": lambda d: float(
            (d["baseline_correct"] & ~d["elastic_correct"]).mean()
        ),
        "wrong_to_correct_rate": lambda d: float(
            (~d["baseline_correct"] & d["elastic_correct"]).mean()
        ),
        "mean_baseline_nfe": lambda d: float(d["baseline_nfe"].mean()),
        "mean_elastic_nfe": lambda d: float(d["elastic_nfe"].mean()),
        "mean_nfe_delta": lambda d: float((d["elastic_nfe"] - d["baseline_nfe"]).mean()),
        "relative_layer_recompute_count": lambda d: float(
            (d["elastic_nfe"] * upstream_layer_recompute_fraction(d)).sum()
            / d["baseline_nfe"].sum()
        ),
        "wall_clock_speedup": lambda d: float(
            d["baseline_latency_s"].sum() / d["elastic_latency_s"].sum()
        ),
    }


def bootstrap_metrics(rows: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    metrics = metric_functions()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rows), size=(reps, len(rows)))
    bootstrap_values = {name: [] for name in metrics}
    for indices in sampled:
        sample = rows.iloc[indices]
        for name, metric in metrics.items():
            bootstrap_values[name].append(metric(sample))

    output = []
    for name, metric in metrics.items():
        low, high = np.percentile(bootstrap_values[name], [2.5, 97.5])
        output.append(
            {
                "metric": name,
                "point": metric(rows),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(output)


def table_text(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an official Elastic-Cache comparison.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(run_dir / "example_results.csv")
    metrics = bootstrap_metrics(rows, args.bootstrap_reps, args.seed)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    baseline_correct = rows["baseline_correct"].astype(bool)
    elastic_correct = rows["elastic_correct"].astype(bool)
    correct_to_wrong = int((baseline_correct & ~elastic_correct).sum())
    wrong_to_correct = int((~baseline_correct & elastic_correct).sum())
    discordant = correct_to_wrong + wrong_to_correct
    correctness_sign_p = (
        float(binomtest(min(correct_to_wrong, wrong_to_correct), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    summary = {
        "source_run": str(run_dir),
        "examples": int(len(rows)),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "example",
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "correctness_exact_sign_p": correctness_sign_p,
        "metrics": {row.metric: row.point for row in metrics.itertuples(index=False)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# Official Elastic-Cache Reanalysis\n\n"
        f"Examples: {len(rows)}. Bootstrap: {args.bootstrap_reps} example resamples.\n\n"
        + table_text(metrics)
        + "\n\n"
        + f"Correct -> wrong: {correct_to_wrong}. Wrong -> correct: {wrong_to_correct}. "
        + f"Exact paired sign-test p-value: {correctness_sign_p:.6g}.\n",
        encoding="utf-8",
    )
    print(table_text(metrics))


if __name__ == "__main__":
    main()
