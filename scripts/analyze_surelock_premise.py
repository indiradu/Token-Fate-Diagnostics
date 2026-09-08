#!/usr/bin/env python3
"""Regrade and summarize a SureLock or LESS YF-vs-YC intervention artifact.

The baseline decode is YC in this audit: committed token identities remain
fixed while their hidden rows are recomputed. YF freezes the same token's row
at the strategy-specific reference step. Bootstrap intervals resample example
ids so multiple candidate interventions from one decode stay together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from regret_remasking.data import (
    answers_equivalent,
    load_examples,
    normalize_answer,
    score_generation_with_status,
)


MetricFn = Callable[[pd.DataFrame], float]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def load_yf_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "freeze_intervention_results.csv"
    rows = pd.read_csv(path)
    accepted_arms = {"freeze_at_rule_lock", "surelock_freeze_yf", "less_freeze_yf"}
    rows = rows[rows["intervention_group"].isin(accepted_arms)].copy()
    if rows.empty:
        raise ValueError(f"no supported YF rows found in {path}")
    return rows


def validate_less_replay_prefix(rows: pd.DataFrame) -> None:
    column = "pre_intervention_replay_valid"
    if column not in rows:
        raise ValueError("LESS causal analysis requires the full pre-intervention replay check")

    def parse_flag(value: object) -> bool | None:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return None

    replay_valid = rows[column].map(parse_flag)
    malformed = int(replay_valid.isna().sum())
    if malformed:
        raise ValueError(f"LESS causal analysis has {malformed} malformed replay-validity values")
    invalid = int((~replay_valid.astype(bool)).sum())
    if invalid:
        raise ValueError(f"LESS causal analysis has {invalid} invalid replay prefixes")


def report_copy(intervention_mode: str) -> tuple[str, str, str]:
    if intervention_mode == "less_premise":
        return (
            "# LESS Admission-Reuse Reanalysis",
            "`YC` is the original LESS decode with committed-token representations recomputed. "
            "`YF` freezes a genuinely LESS-admitted token from its first post-commit row.",
            "## Changed Answers Or Programs",
        )
    return (
        "# SureLock Premise Reanalysis",
        "`YC` is the original decode with committed-token representations recomputed. "
        "`YF` freezes the same token's row at its SureLock-style lock step.",
        "## Mathematically Changed Answers",
    )


def regrade_rows(rows: pd.DataFrame, examples_by_id: dict[str, object]) -> pd.DataFrame:
    rows = rows.copy()
    missing = sorted(set(rows["example_id"].astype(str)) - set(examples_by_id))
    if missing:
        raise ValueError(f"missing gold examples for ids: {missing[:5]}")

    baseline_answers = []
    modified_answers = []
    baseline_correct = []
    modified_correct = []
    baseline_status = []
    modified_status = []
    value_changed = []
    for row in rows.itertuples(index=False):
        example = examples_by_id[str(row.example_id)]
        baseline_generation = str(row.baseline_generation)
        modified_generation = str(row.clamped_generation)
        baseline_answers.append(normalize_answer(baseline_generation, example.dataset))
        modified_answers.append(normalize_answer(modified_generation, example.dataset))
        baseline_passed, baseline_score_status = score_generation_with_status(
            baseline_generation, example
        )
        modified_passed, modified_score_status = score_generation_with_status(
            modified_generation, example
        )
        baseline_correct.append(baseline_passed)
        modified_correct.append(modified_passed)
        baseline_status.append(baseline_score_status)
        modified_status.append(modified_score_status)
        if str(example.dataset).lower() in {"humaneval", "human_eval"}:
            value_changed.append(baseline_answers[-1] != modified_answers[-1])
        else:
            value_changed.append(not answers_equivalent(baseline_generation, modified_generation))

    rows["baseline_extracted_answer_regraded"] = baseline_answers
    rows["modified_extracted_answer_regraded"] = modified_answers
    rows["baseline_correct_regraded"] = baseline_correct
    rows["modified_correct_regraded"] = modified_correct
    rows["baseline_scoring_status_regraded"] = baseline_status
    rows["modified_scoring_status_regraded"] = modified_status
    rows["extracted_answer_changed_regraded"] = rows[
        "baseline_extracted_answer_regraded"
    ].ne(rows["modified_extracted_answer_regraded"])
    rows["answer_value_changed_regraded"] = value_changed
    rows["correctness_changed_regraded"] = rows["baseline_correct_regraded"].ne(
        rows["modified_correct_regraded"]
    )
    rows["correct_to_wrong_regraded"] = rows["baseline_correct_regraded"] & ~rows[
        "modified_correct_regraded"
    ]
    rows["wrong_to_correct_regraded"] = ~rows["baseline_correct_regraded"] & rows[
        "modified_correct_regraded"
    ]
    return rows


def metric_functions() -> dict[str, MetricFn]:
    infrastructure_failures = {"sandbox_unavailable", "sandbox_error"}

    def paired_scorable(d: pd.DataFrame) -> pd.Series:
        return ~d["baseline_scoring_status_regraded"].isin(infrastructure_failures) & ~d[
            "modified_scoring_status_regraded"
        ].isin(infrastructure_failures)

    return {
        "non_target_token_change_rate": lambda d: float(d["non_target_token_changed"].mean()),
        "mean_non_target_token_change_count": lambda d: float(
            d["non_target_token_change_count"].mean()
        ),
        "extracted_answer_change_rate": lambda d: float(
            d["extracted_answer_changed_regraded"].mean()
        ),
        "answer_value_change_rate": lambda d: float(d["answer_value_changed_regraded"].mean()),
        "original_accuracy": lambda d: float(
            d.loc[paired_scorable(d), "baseline_correct_regraded"].mean()
        ),
        "original_unique_example_accuracy": lambda d: float(
            d[~d["baseline_scoring_status_regraded"].isin(infrastructure_failures)]
            .groupby("example_id", sort=False)["baseline_correct_regraded"]
            .first()
            .mean()
        ),
        "modified_accuracy": lambda d: float(
            d.loc[paired_scorable(d), "modified_correct_regraded"].mean()
        ),
        "accuracy_delta": lambda d: float(
            (
                d.loc[paired_scorable(d), "modified_correct_regraded"].astype(float)
                - d.loc[paired_scorable(d), "baseline_correct_regraded"].astype(float)
            ).mean()
        ),
        "correctness_change_rate": lambda d: float(
            d.loc[paired_scorable(d), "correctness_changed_regraded"].mean()
        ),
        "correct_to_wrong_rate": lambda d: float(
            d.loc[paired_scorable(d), "correct_to_wrong_regraded"].mean()
        ),
        "wrong_to_correct_rate": lambda d: float(
            d.loc[paired_scorable(d), "wrong_to_correct_regraded"].mean()
        ),
        "original_scoring_infrastructure_failure_rate": lambda d: float(
            d["baseline_scoring_status_regraded"].isin(infrastructure_failures).mean()
        ),
        "modified_scoring_infrastructure_failure_rate": lambda d: float(
            d["modified_scoring_status_regraded"].isin(infrastructure_failures).mean()
        ),
        "original_execution_timeout_rate": lambda d: float(
            d["baseline_scoring_status_regraded"].eq("timeout").mean()
        ),
        "modified_execution_timeout_rate": lambda d: float(
            d["modified_scoring_status_regraded"].eq("timeout").mean()
        ),
    }


def cluster_bootstrap_metrics(
    rows: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    metrics = metric_functions()
    bootstrap_rows = rows.copy()
    infrastructure_failures = {"sandbox_unavailable", "sandbox_error"}
    baseline_scorable = ~bootstrap_rows["baseline_scoring_status_regraded"].isin(
        infrastructure_failures
    )
    modified_scorable = ~bootstrap_rows["modified_scoring_status_regraded"].isin(
        infrastructure_failures
    )
    paired_scorable = baseline_scorable & modified_scorable
    bootstrap_rows["_original_accuracy"] = bootstrap_rows[
        "baseline_correct_regraded"
    ].astype(float).where(paired_scorable)
    bootstrap_rows["_original_unique_example_accuracy"] = bootstrap_rows[
        "baseline_correct_regraded"
    ].astype(float).where(baseline_scorable)
    bootstrap_rows["_modified_accuracy"] = bootstrap_rows[
        "modified_correct_regraded"
    ].astype(float).where(paired_scorable)
    bootstrap_rows["_accuracy_delta"] = (
        bootstrap_rows["modified_correct_regraded"].astype(float)
        - bootstrap_rows["baseline_correct_regraded"].astype(float)
    ).where(paired_scorable)
    bootstrap_rows["_correctness_change"] = bootstrap_rows[
        "correctness_changed_regraded"
    ].astype(float).where(paired_scorable)
    bootstrap_rows["_correct_to_wrong"] = bootstrap_rows["correct_to_wrong_regraded"].astype(
        float
    ).where(paired_scorable)
    bootstrap_rows["_wrong_to_correct"] = bootstrap_rows["wrong_to_correct_regraded"].astype(
        float
    ).where(paired_scorable)
    specs = {
        "non_target_token_change_rate": ("non_target_token_changed", False),
        "mean_non_target_token_change_count": ("non_target_token_change_count", False),
        "extracted_answer_change_rate": ("extracted_answer_changed_regraded", False),
        "answer_value_change_rate": ("answer_value_changed_regraded", False),
        "original_accuracy": ("_original_accuracy", False),
        "original_unique_example_accuracy": ("_original_unique_example_accuracy", True),
        "modified_accuracy": ("_modified_accuracy", False),
        "accuracy_delta": ("_accuracy_delta", False),
        "correctness_change_rate": ("_correctness_change", False),
        "correct_to_wrong_rate": ("_correct_to_wrong", False),
        "wrong_to_correct_rate": ("_wrong_to_correct", False),
        "original_scoring_infrastructure_failure_rate": (
            "_baseline_scoring_infrastructure_failure",
            False,
        ),
        "modified_scoring_infrastructure_failure_rate": (
            "_modified_scoring_infrastructure_failure",
            False,
        ),
        "original_execution_timeout_rate": ("_baseline_execution_timeout", False),
        "modified_execution_timeout_rate": ("_modified_execution_timeout", False),
    }
    bootstrap_rows["_baseline_scoring_infrastructure_failure"] = bootstrap_rows[
        "baseline_scoring_status_regraded"
    ].isin(infrastructure_failures)
    bootstrap_rows["_modified_scoring_infrastructure_failure"] = bootstrap_rows[
        "modified_scoring_status_regraded"
    ].isin(infrastructure_failures)
    bootstrap_rows["_baseline_execution_timeout"] = bootstrap_rows[
        "baseline_scoring_status_regraded"
    ].eq("timeout")
    bootstrap_rows["_modified_execution_timeout"] = bootstrap_rows[
        "modified_scoring_status_regraded"
    ].eq("timeout")
    grouped = list(bootstrap_rows.groupby("example_id", sort=False))
    cluster_count = len(grouped)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, cluster_count, size=(reps, cluster_count))

    output = []
    for name, function in metrics.items():
        column, unique_per_example = specs[name]
        if unique_per_example:
            values = [group[column].dropna() for _, group in grouped]
            numerators = np.asarray(
                [float(value.iloc[0]) if not value.empty else 0.0 for value in values],
                dtype=float,
            )
            denominators = np.asarray([not value.empty for value in values], dtype=float)
        else:
            numerators = np.asarray(
                [float(group[column].astype(float).sum()) for _, group in grouped], dtype=float
            )
            denominators = np.asarray(
                [float(group[column].notna().sum()) for _, group in grouped], dtype=float
            )
        sampled_denominators = denominators[sampled].sum(axis=1)
        sampled_values = np.divide(
            numerators[sampled].sum(axis=1),
            sampled_denominators,
            out=np.full(reps, np.nan, dtype=float),
            where=sampled_denominators > 0,
        )
        low, high = np.nanpercentile(sampled_values, [2.5, 97.5])
        output.append(
            {
                "metric": name,
                "point": function(rows),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(output)


def changed_answer_examples(rows: pd.DataFrame) -> pd.DataFrame:
    changed = rows[rows["answer_value_changed_regraded"]].copy()
    columns = [
        "example_id",
        "position",
        "freeze_step",
        "baseline_extracted_answer_regraded",
        "modified_extracted_answer_regraded",
        "baseline_correct_regraded",
        "modified_correct_regraded",
        "baseline_scoring_status_regraded",
        "modified_scoring_status_regraded",
        "non_target_token_change_count",
    ]
    return changed[columns].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regrade and bootstrap a YF-vs-YC premise run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    examples = load_examples(
        config["dataset"],
        config["split"],
        config["limit"],
        offset=config["offset"],
        jsonl_path=config.get("jsonl_path"),
    )
    examples_by_id = {str(example.example_id): example for example in examples}
    raw_rows = load_yf_rows(run_dir)
    intervention_mode = str(config.get("intervention_mode", ""))
    if intervention_mode == "less_premise":
        validate_less_replay_prefix(raw_rows)
    rows = regrade_rows(raw_rows, examples_by_id)
    metrics = cluster_bootstrap_metrics(rows, args.bootstrap_reps, args.seed)
    changed = changed_answer_examples(rows)

    rows.to_csv(output_dir / "surelock_premise_regraded.csv", index=False)
    metrics.to_csv(output_dir / "surelock_premise_metrics.csv", index=False)
    changed.to_csv(output_dir / "surelock_premise_changed_answers.csv", index=False)
    summary = {
        "source_run": str(run_dir),
        "estimand": (
            "YF versus YC for genuine LESS admissions reused for reference freezing"
            if intervention_mode == "less_premise"
            else "YF versus YC for tokens admitted by the SureLock-style gate"
        ),
        "examples": int(rows["example_id"].nunique()),
        "interventions": int(len(rows)),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "example_id clusters",
        "scoring": "safe symbolic equivalence for math; task scorer otherwise",
        "metrics": {row.metric: row.point for row in metrics.itertuples(index=False)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    title, description, changed_heading = report_copy(intervention_mode)
    lines = [
        title,
        "",
        description,
        "",
        f"Examples: {summary['examples']}. Interventions: {summary['interventions']}.",
        f"Bootstrap: {args.bootstrap_reps} resamples of example-id clusters.",
        "",
        "## Metrics",
        "",
        table_text(metrics),
        "",
        changed_heading,
        "",
        table_text(changed),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(metrics))


if __name__ == "__main__":
    main()
