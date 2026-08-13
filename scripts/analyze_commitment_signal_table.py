#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("runs must be LABEL=PATH")
    label, path = text.split("=", 1)
    return label, Path(path)


def load_metadata(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = run_dir / "train" / "metadata.csv"
    eval_path = run_dir / "eval" / "metadata.csv"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"expected train/eval metadata under {run_dir}")
    return pd.read_csv(train_path), pd.read_csv(eval_path)


def target_mask(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "all_trace_regret":
        return pd.Series(True, index=df.index)
    if target == "early_trace_regret":
        return df["block_t_frac"] <= 0.5
    if target == "stable_high_conf_trace_regret":
        high_conf = df["confidence"].quantile(0.75)
        low_kl = df["kl"].quantile(0.50)
        return (df["confidence"] >= high_conf) & (df["kl"] <= low_kl)
    raise ValueError(f"unknown target: {target}")


def add_boundary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "relative_position" not in out.columns:
        return out
    if "block" in out.columns and "block_t_frac" in out.columns:
        block_length = int(out.groupby("block")["relative_position"].nunique().median())
        block_length = max(block_length, 1)
        pos_in_block = out["relative_position"].astype(int) - out["block"].astype(int) * block_length
    else:
        block_length = max(int(out["relative_position"].max()) + 1, 1)
        pos_in_block = out["relative_position"].astype(int)
    right_distance = (block_length - 1 - pos_in_block).clip(lower=0)
    edge_distance = np.minimum(pos_in_block.clip(lower=0), right_distance)
    out["right_boundary_nearness"] = 1.0 / (1.0 + right_distance.astype(float))
    out["edge_boundary_nearness"] = 1.0 / (1.0 + edge_distance.astype(float))
    return out


def signal_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    df = add_boundary_columns(df)
    signals: dict[str, np.ndarray] = {}
    for name in ["confidence", "entropy", "margin", "kl", "top1_flip", "runlength", "jsd"]:
        if name in df.columns:
            signals[name] = df[name].astype(float).to_numpy()
    if "right_boundary_nearness" in df.columns:
        signals["right_boundary_nearness"] = df["right_boundary_nearness"].astype(float).to_numpy()
    if "edge_boundary_nearness" in df.columns:
        signals["edge_boundary_nearness"] = df["edge_boundary_nearness"].astype(float).to_numpy()
    return signals


def orient_score(y_train: np.ndarray, train_score: np.ndarray, eval_score: np.ndarray) -> tuple[np.ndarray, str]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_train)) < 2:
        return eval_score, "raw"
    clean = np.isfinite(train_score)
    if clean.sum() == 0:
        return eval_score, "raw"
    auc = roc_auc_score(y_train[clean], train_score[clean])
    if auc < 0.5:
        return -eval_score, "inverted"
    return eval_score, "raw"


def fit_trace_predictor(
    train: pd.DataFrame,
    y_train: np.ndarray,
    eval_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = [name for name in FEATURE_NAMES if name in train.columns and name in eval_df.columns]
    if "jsd" in train.columns and "jsd" in eval_df.columns:
        features.append("jsd")
    clean_train = train[features + ["trace_regret"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean_train["trace_regret"].nunique() < 2:
        return np.full(len(eval_df), np.nan, dtype=float), features
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=23),
    )
    clf.fit(clean_train[features].to_numpy(dtype=np.float32), clean_train["trace_regret"].to_numpy(dtype=np.int64))
    clean_eval = eval_df[features].replace([np.inf, -np.inf], np.nan)
    scores = np.full(len(eval_df), np.nan, dtype=float)
    mask = ~clean_eval.isna().any(axis=1)
    if mask.any():
        scores[mask.to_numpy()] = clf.predict_proba(clean_eval.loc[mask, features].to_numpy(dtype=np.float32))[:, 1]
    return scores, features


def score_rows(
    label: str,
    probe: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_score: np.ndarray,
    eval_score: np.ndarray,
    target: str,
    top_k: int,
    orientation: str,
) -> dict[str, object] | None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    train_mask = target_mask(train_df, target).to_numpy()
    eval_mask = target_mask(eval_df, target).to_numpy()
    y_train = train_df.loc[train_mask, "trace_regret"].to_numpy(dtype=np.int64)
    y_eval = eval_df.loc[eval_mask, "trace_regret"].to_numpy(dtype=np.int64)
    raw_train = train_score[train_mask]
    raw_eval = eval_score[eval_mask]
    if orientation == "auto":
        oriented_eval, direction = orient_score(y_train, raw_train, raw_eval)
    elif orientation == "invert":
        oriented_eval, direction = -raw_eval, "inverted"
    else:
        oriented_eval, direction = raw_eval, "raw"
    clean = np.isfinite(oriented_eval)
    if clean.sum() == 0 or len(np.unique(y_eval[clean])) < 2:
        return None
    auc = roc_auc_score(y_eval[clean], oriented_eval[clean])
    ap = average_precision_score(y_eval[clean], oriented_eval[clean])
    eval_subset = eval_df.loc[eval_mask].copy()
    eval_subset["risk_score"] = oriented_eval
    eval_subset["label"] = y_eval
    eval_subset = eval_subset[np.isfinite(eval_subset["risk_score"])]
    eval_subset = eval_subset.sort_values("risk_score", ascending=False)
    if "example_id" in eval_subset.columns:
        eval_subset = eval_subset.drop_duplicates("example_id", keep="first")
    top = eval_subset.head(top_k)
    base_rate = float(y_eval[clean].mean())
    topk_rate = float(top["label"].mean()) if len(top) else np.nan
    return {
        "run": label,
        "target": target,
        "signal": probe,
        "orientation": direction,
        "eval_rows": int(clean.sum()),
        "base_regret_rate": base_rate,
        "auroc": float(auc),
        "average_precision": float(ap),
        "top_k": int(len(top)),
        "topk_realized_regret": topk_rate,
        "topk_lift": float(topk_rate / base_rate) if base_rate > 0 and len(top) else np.nan,
    }


def analyze_run(label: str, run_dir: Path, targets: list[str], top_k: int) -> pd.DataFrame:
    train, eval_df = load_metadata(run_dir)
    train = add_boundary_columns(train)
    eval_df = add_boundary_columns(eval_df)
    rows: list[dict[str, object]] = []
    train_signals = signal_scores(train)
    eval_signals = signal_scores(eval_df)
    for target in targets:
        train_mask = target_mask(train, target).to_numpy()
        y_train = train.loc[train_mask, "trace_regret"].to_numpy(dtype=np.int64)
        trace_scores, trace_features = fit_trace_predictor(train.loc[train_mask], y_train, eval_df)
        row = score_rows(
            label,
            "learned_trace_predictor",
            train,
            eval_df,
            np.full(len(train), np.nan, dtype=float),
            trace_scores,
            target,
            top_k,
            "raw",
        )
        if row is not None:
            row["features"] = ",".join(trace_features)
            rows.append(row)
        for signal, eval_score in eval_signals.items():
            train_score = train_signals[signal]
            row = score_rows(label, signal, train, eval_df, train_score, eval_score, target, top_k, "auto")
            if row is not None:
                row["features"] = signal
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_causal(causal_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if causal_dir is None:
        return pd.DataFrame(), pd.DataFrame()
    group_path = causal_dir / "group_summary.csv"
    oracle_path = causal_dir / "oracle_upper_bound_results.csv"
    group_summary = pd.read_csv(group_path) if group_path.exists() else pd.DataFrame()
    oracle_summary = pd.DataFrame()
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        if len(oracle):
            oracle_summary = pd.DataFrame(
                [
                    {
                        "examples": len(oracle),
                        "baseline_accuracy": oracle["baseline_correct"].mean(),
                        "oracle_accuracy": oracle["oracle_correct"].mean(),
                        "accuracy_delta": oracle["oracle_correct"].mean() - oracle["baseline_correct"].mean(),
                        "correctness_change_rate": oracle["oracle_correct_changed"].mean(),
                    }
                ]
            )
    return group_summary, oracle_summary


def compact_cross_task(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df
    view = signal_df[signal_df["target"].eq("all_trace_regret")].copy()
    keep = []
    for run, group in view.groupby("run"):
        learned = group[group["signal"].eq("learned_trace_predictor")]
        best_baseline = group[~group["signal"].eq("learned_trace_predictor")].sort_values("auroc", ascending=False).head(1)
        keep.extend(learned.to_dict("records"))
        keep.extend(best_baseline.to_dict("records"))
    cols = ["run", "signal", "base_regret_rate", "auroc", "average_precision", "topk_realized_regret", "topk_lift"]
    return pd.DataFrame(keep)[cols] if keep else pd.DataFrame(columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build commitment-risk signal and causal-enrichment tables.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=token_fate_run_dir")
    parser.add_argument("--causal-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--targets",
        default="all_trace_regret,early_trace_regret",
        help="Comma-separated target slices.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    signal_tables = [analyze_run(label, run_dir, targets, args.top_k) for label, run_dir in args.run]
    signal_df = pd.concat(signal_tables, ignore_index=True) if signal_tables else pd.DataFrame()
    signal_df.to_csv(args.output_dir / "commitment_signal_leaderboard.csv", index=False)

    compact = compact_cross_task(signal_df)
    compact.to_csv(args.output_dir / "cross_task_commitment_summary.csv", index=False)

    group_summary, oracle_summary = summarize_causal(args.causal_dir)
    if not group_summary.empty:
        group_summary.to_csv(args.output_dir / "causal_group_summary.csv", index=False)
    if not oracle_summary.empty:
        oracle_summary.to_csv(args.output_dir / "oracle_upper_bound_summary.csv", index=False)

    lines = ["# Commitment Signal Tables", ""]
    lines.extend(["## Cross-task summary", table_text(compact), ""])
    if not signal_df.empty:
        selected = signal_df[
            signal_df["signal"].isin(
                [
                    "learned_trace_predictor",
                    "confidence",
                    "entropy",
                    "margin",
                    "kl",
                    "runlength",
                    "top1_flip",
                    "jsd",
                    "right_boundary_nearness",
                ]
            )
        ]
        cols = [
            "run",
            "target",
            "signal",
            "orientation",
            "auroc",
            "average_precision",
            "topk_realized_regret",
            "topk_lift",
        ]
        lines.extend(["## Signal leaderboard", table_text(selected[cols]), ""])
    if not group_summary.empty:
        lines.extend(["## Matched causal controls", table_text(group_summary), ""])
    if not oracle_summary.empty:
        lines.extend(["## Oracle upper bound", table_text(oracle_summary), ""])
    (args.output_dir / "commitment_signal_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
