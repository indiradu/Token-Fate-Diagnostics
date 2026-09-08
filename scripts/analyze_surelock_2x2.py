#!/usr/bin/env python3
"""Example-level bootstrap contrasts for the difficulty x generation 2x2."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = {
    "non_target_token_change_rate": "non_target_token_changed",
    "mean_non_target_token_change_count": "non_target_token_change_count",
    "answer_value_change_rate": "answer_value_changed_regraded",
    "original_accuracy": "baseline_correct_regraded",
    "modified_accuracy": "modified_correct_regraded",
    "accuracy_delta": "_accuracy_delta",
}


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def example_metrics(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path / "surelock_premise_regraded.csv")
    rows["_accuracy_delta"] = (
        rows["modified_correct_regraded"].astype(float)
        - rows["baseline_correct_regraded"].astype(float)
    )
    return rows.groupby("example_id", sort=False)[list(METRIC_COLUMNS.values())].mean()


def bootstrap_effects(
    lt5_g64: pd.DataFrame,
    lt5_g256: pd.DataFrame,
    l5_g64: pd.DataFrame,
    l5_g256: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    if set(lt5_g64.index) != set(lt5_g256.index):
        raise ValueError("Level-1-4 cells must contain the same example ids")
    if set(l5_g64.index) != set(l5_g256.index):
        raise ValueError("Level-5 cells must contain the same example ids")
    lt5_g256 = lt5_g256.loc[lt5_g64.index]
    l5_g256 = l5_g256.loc[l5_g64.index]

    rng = np.random.default_rng(seed)
    lt5_samples = rng.integers(0, len(lt5_g64), size=(reps, len(lt5_g64)))
    l5_samples = rng.integers(0, len(l5_g64), size=(reps, len(l5_g64)))
    rows = []
    for metric, column in METRIC_COLUMNS.items():
        lt5_64 = lt5_g64[column].to_numpy(dtype=float)
        lt5_256 = lt5_g256[column].to_numpy(dtype=float)
        l5_64 = l5_g64[column].to_numpy(dtype=float)
        l5_256 = l5_g256[column].to_numpy(dtype=float)

        point_effects = {
            "horizon_effect_levels1to4": float(lt5_256.mean() - lt5_64.mean()),
            "horizon_effect_level5": float(l5_256.mean() - l5_64.mean()),
            "difficulty_effect_gen64": float(l5_64.mean() - lt5_64.mean()),
            "difficulty_effect_gen256": float(l5_256.mean() - lt5_256.mean()),
        }
        sampled_effects = {
            "horizon_effect_levels1to4": (
                lt5_256[lt5_samples].mean(axis=1) - lt5_64[lt5_samples].mean(axis=1)
            ),
            "horizon_effect_level5": (
                l5_256[l5_samples].mean(axis=1) - l5_64[l5_samples].mean(axis=1)
            ),
            "difficulty_effect_gen64": (
                l5_64[l5_samples].mean(axis=1) - lt5_64[lt5_samples].mean(axis=1)
            ),
            "difficulty_effect_gen256": (
                l5_256[l5_samples].mean(axis=1) - lt5_256[lt5_samples].mean(axis=1)
            ),
        }
        point_effects["difficulty_by_horizon_interaction"] = (
            point_effects["horizon_effect_level5"]
            - point_effects["horizon_effect_levels1to4"]
        )
        sampled_effects["difficulty_by_horizon_interaction"] = (
            sampled_effects["horizon_effect_level5"]
            - sampled_effects["horizon_effect_levels1to4"]
        )

        for contrast, point in point_effects.items():
            low, high = np.percentile(sampled_effects[contrast], [2.5, 97.5])
            rows.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "point": point,
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the SureLock 2x2 contrasts.")
    parser.add_argument("--lt5-g64", required=True)
    parser.add_argument("--lt5-g256", required=True)
    parser.add_argument("--l5-g64", required=True)
    parser.add_argument("--l5-g256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = bootstrap_effects(
        example_metrics(Path(args.lt5_g64)),
        example_metrics(Path(args.lt5_g256)),
        example_metrics(Path(args.l5_g64)),
        example_metrics(Path(args.l5_g256)),
        args.bootstrap_reps,
        args.seed,
    )
    result.to_csv(output_dir / "factorial_contrasts.csv", index=False)
    lines = [
        "# SureLock Difficulty x Generation-Length Contrasts",
        "",
        "Effects are differences in rates or means. Intervals are example-level paired "
        "bootstrap intervals within each difficulty group.",
        "",
        table_text(result),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(result))


if __name__ == "__main__":
    main()
