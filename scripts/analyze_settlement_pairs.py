#!/usr/bin/env python3
"""Paired analysis for settlement-fate audits (freeze/clamp and commit modes).

Reads one or more run directories containing freeze_pair_summary.csv (clamp
mode, hypotheses B/C) or commit_pair_summary.csv (commit mode, hypothesis A),
then reports per-run and pooled paired deltas with example-clustered bootstrap
confidence intervals and a paired sign test. Bootstrap resamples example ids,
never individual pairs, matching the repo's bootstrap discipline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EOS_ID = 126081
EOT_ID = 126348

DELTA_COLUMNS = [
    "delta_non_target_changed",
    "delta_non_target_change_count",
    "delta_normalized_answer_changed",
    "delta_answer_changed",
]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def load_pairs(run_dir: Path, label: str) -> pd.DataFrame:
    for name, mode in (("freeze_pair_summary.csv", "clamp"), ("commit_pair_summary.csv", "commit")):
        path = run_dir / name
        if path.exists():
            pairs = pd.read_csv(path)
            pairs["run"] = label
            pairs["mode"] = mode
            return pairs
    raise FileNotFoundError(f"no pair summary found in {run_dir}")


def annotate_eos(pairs: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Mark pairs whose target or control token is EOS/EOT.

    Commit-mode pair files already carry is_eos_eot; clamp-mode files need a
    join against committed_token_drift.csv for the selected tokens.
    """
    pairs = pairs.copy()
    if "is_eos_eot" in pairs.columns:
        pairs["pair_has_eos_eot"] = pairs["is_eos_eot"].astype(bool)
        return pairs
    drift_path = run_dir / "committed_token_drift.csv"
    if not drift_path.exists():
        pairs["pair_has_eos_eot"] = False
        return pairs
    drift = pd.read_csv(drift_path)[["example_id", "position", "selected_token"]]
    token_lookup = {
        (str(row.example_id), int(row.position)): int(row.selected_token)
        for row in drift.itertuples(index=False)
    }

    def has_eos(row: pd.Series) -> bool:
        tokens = [
            token_lookup.get((str(row["example_id"]), int(row["target_position"]))),
            token_lookup.get((str(row["example_id"]), int(row["control_position"]))),
        ]
        return any(token in (EOS_ID, EOT_ID) for token in tokens if token is not None)

    pairs["pair_has_eos_eot"] = pairs.apply(has_eos, axis=1)
    return pairs


def cluster_bootstrap_ci(
    pairs: pd.DataFrame,
    column: str,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    values_by_example = {
        example_id: group[column].to_numpy(dtype=float)
        for example_id, group in pairs.groupby("example_id", sort=False)
    }
    example_ids = list(values_by_example)
    point = float(pairs[column].mean())
    if len(example_ids) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for rep in range(reps):
        sampled = rng.choice(len(example_ids), size=len(example_ids), replace=True)
        chunks = [values_by_example[example_ids[idx]] for idx in sampled]
        means[rep] = float(np.concatenate(chunks).mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return point, float(low), float(high)


def sign_test(pairs: pd.DataFrame, column: str) -> dict[str, Any]:
    deltas = pairs[column].to_numpy(dtype=float)
    positive = int((deltas > 0).sum())
    negative = int((deltas < 0).sum())
    discordant = positive + negative
    if discordant == 0:
        return {"positive": 0, "negative": 0, "p_value": None}
    try:
        from scipy.stats import binomtest

        p_value = float(binomtest(positive, discordant, 0.5).pvalue)
    except ImportError:
        p_value = None
    return {"positive": positive, "negative": negative, "p_value": p_value}


def analyze(pairs: pd.DataFrame, label: str, reps: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for column in DELTA_COLUMNS:
        if column not in pairs.columns:
            continue
        point, low, high = cluster_bootstrap_ci(pairs, column, reps, seed)
        sign = sign_test(pairs, column)
        rows.append(
            {
                "slice": label,
                "pairs": int(len(pairs)),
                "examples": int(pairs["example_id"].nunique()),
                "metric": column,
                "mean_delta": point,
                "ci_low": low,
                "ci_high": high,
                "sign_positive": sign["positive"],
                "sign_negative": sign["negative"],
                "sign_p_value": sign["p_value"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap paired settlement-fate deltas across runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=RUN_DIR containing freeze_pair_summary.csv or commit_pair_summary.csv; repeatable.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for spec in args.run:
        label, _, run_dir = spec.partition("=")
        if not run_dir:
            raise ValueError(f"--run must be LABEL=RUN_DIR, got: {spec}")
        pairs = load_pairs(Path(run_dir), label)
        pairs = annotate_eos(pairs, Path(run_dir))
        all_pairs.append(pairs)
    pooled = pd.concat(all_pairs, ignore_index=True)
    modes = pooled["mode"].unique().tolist()
    if len(modes) > 1:
        raise ValueError(f"refusing to pool clamp and commit runs together: {modes}")

    rows: list[dict[str, Any]] = []
    for run_label, run_pairs in pooled.groupby("run", sort=False):
        rows.extend(analyze(run_pairs, f"{run_label}", args.bootstrap_reps, args.seed))
        no_eos = run_pairs[~run_pairs["pair_has_eos_eot"]]
        if len(no_eos) < len(run_pairs) and not no_eos.empty:
            rows.extend(analyze(no_eos, f"{run_label} (no EOS/EOT)", args.bootstrap_reps, args.seed))
    if pooled["run"].nunique() > 1:
        rows.extend(analyze(pooled, "pooled", args.bootstrap_reps, args.seed))
        no_eos = pooled[~pooled["pair_has_eos_eot"]]
        if not no_eos.empty:
            rows.extend(analyze(no_eos, "pooled (no EOS/EOT)", args.bootstrap_reps, args.seed))

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    summary = {
        "mode": modes[0],
        "runs": pooled["run"].nunique(),
        "total_pairs": int(len(pooled)),
        "pairs_with_eos_eot": int(pooled["pair_has_eos_eot"].sum()),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "example_id clusters",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Settlement Fate Paired Bootstrap",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "Bootstrap resamples example ids (cluster level), 95% percentile intervals.",
        "Sign test is an exact binomial test on discordant pairs.",
        "",
        table_text(result),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(result))


if __name__ == "__main__":
    main()
