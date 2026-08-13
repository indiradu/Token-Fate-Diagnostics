from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FateThresholds:
    high_confidence: float
    low_kl: float


def make_thresholds(df: pd.DataFrame) -> FateThresholds:
    """Choose data-dependent thresholds for descriptive fate buckets."""
    return FateThresholds(
        high_confidence=float(df["confidence"].quantile(0.75)),
        low_kl=float(df["kl"].quantile(0.50)),
    )


def add_row_labels(df: pd.DataFrame, thresholds: FateThresholds | None = None) -> pd.DataFrame:
    out = df.copy()
    if thresholds is None:
        thresholds = make_thresholds(out)
    out["early_in_block"] = out["block_t_frac"] <= 0.5
    out["high_confidence"] = out["confidence"] >= thresholds.high_confidence
    out["low_kl"] = out["kl"] <= thresholds.low_kl
    out["early_regret"] = out["early_in_block"] & (out["trace_regret"] == 1)
    out["stable_regret"] = out["high_confidence"] & out["low_kl"] & (out["trace_regret"] == 1)
    return out


def token_fate_table(df: pd.DataFrame, thresholds: FateThresholds | None = None) -> pd.DataFrame:
    """Assign one descriptive fate label per generated token position."""
    labeled = add_row_labels(df, thresholds)
    rows: list[dict[str, object]] = []
    group_cols = ["example_id", "position", "relative_position"]
    for key, group in labeled.sort_values("global_step").groupby(group_cols, sort=False):
        example_id, position, relative_position = key
        regrets = group["trace_regret"].to_numpy(dtype=np.int64)
        flip_count = int(group["top1_flip"].sum())
        unique_top_tokens = int(group["top_token"].nunique())
        early_wrong = bool(group["early_regret"].any())
        stable_wrong = bool(group["stable_regret"].any())
        ever_correct = bool((group["trace_regret"] == 0).any())
        first_correct_idx = np.flatnonzero(regrets == 0)
        first_correct_step = (
            int(group.iloc[int(first_correct_idx[0])]["global_step"])
            if len(first_correct_idx)
            else None
        )

        if flip_count >= 2 or unique_top_tokens >= 3:
            fate = "oscillating"
        elif stable_wrong:
            fate = "stable_wrong"
        elif early_wrong and ever_correct:
            fate = "late_corrected"
        elif bool((group["trace_regret"] == 0).all()):
            fate = "early_correct"
        elif early_wrong:
            fate = "early_wrong"
        else:
            fate = "other"

        rows.append(
            {
                "example_id": example_id,
                "position": int(position),
                "relative_position": int(relative_position),
                "fate": fate,
                "steps_observed": int(len(group)),
                "flip_count": flip_count,
                "unique_top_tokens": unique_top_tokens,
                "mean_confidence": float(group["confidence"].mean()),
                "max_confidence": float(group["confidence"].max()),
                "mean_kl": float(group["kl"].mean()),
                "selected_any": bool(group["selected"].any()),
                "first_correct_step": first_correct_step,
            }
        )
    return pd.DataFrame(rows)


def fate_summary(df: pd.DataFrame, thresholds: FateThresholds | None = None) -> pd.DataFrame:
    table = token_fate_table(df, thresholds)
    if table.empty:
        return pd.DataFrame(columns=["fate", "count", "fraction"])
    counts = table["fate"].value_counts().rename_axis("fate").reset_index(name="count")
    counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts.sort_values(["count", "fate"], ascending=[False, True]).reset_index(drop=True)
