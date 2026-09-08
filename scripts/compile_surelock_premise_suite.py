#!/usr/bin/env python3
"""Compile multiple SureLock premise reanalyses into one comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTED_METRICS = [
    "non_target_token_change_rate",
    "mean_non_target_token_change_count",
    "answer_value_change_rate",
    "original_accuracy",
    "modified_accuracy",
    "accuracy_delta",
    "correctness_change_rate",
    "correct_to_wrong_rate",
    "wrong_to_correct_rate",
    "original_scoring_infrastructure_failure_rate",
    "modified_scoring_infrastructure_failure_rate",
]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def load_analysis(spec: str) -> tuple[dict[str, object], pd.DataFrame]:
    label, separator, directory = spec.partition("=")
    if not separator:
        raise ValueError(f"analysis must be LABEL=DIR, got {spec}")
    path = Path(directory)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(path / "surelock_premise_metrics.csv")
    metrics.insert(0, "label", label)
    metrics.insert(1, "examples", int(summary["examples"]))
    metrics.insert(2, "interventions", int(summary["interventions"]))
    return summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile SureLock premise analysis directories.")
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        help="LABEL=DIR containing summary.json and surelock_premise_metrics.csv; repeatable.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    frames = []
    for spec in args.analysis:
        summary, metrics = load_analysis(spec)
        summaries.append({"label": metrics["label"].iloc[0], **summary})
        frames.append(metrics)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(output_dir / "suite_metrics_long.csv", index=False)

    selected = long[long["metric"].isin(SELECTED_METRICS)].copy()
    selected["estimate_95ci"] = selected.apply(
        lambda row: f"{row['point']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
        axis=1,
    )
    wide = selected.pivot(
        index=["label", "examples", "interventions"],
        columns="metric",
        values="estimate_95ci",
    ).reset_index()
    wide.columns.name = None
    wide.to_csv(output_dir / "suite_metrics_wide.csv", index=False)

    suite_summary = {
        "analyses": len(summaries),
        "labels": [item["label"] for item in summaries],
        "interval": "95% percentile bootstrap",
        "bootstrap_unit": "example_id clusters",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(suite_summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# SureLock Premise Suite",
        "",
        "Each cell reports point estimate and example-clustered 95% bootstrap interval.",
        "",
        table_text(wide),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(wide))


if __name__ == "__main__":
    main()
