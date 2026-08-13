#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


ATTENTION_FEATURES = ["attn_context_volatility", "attn_mask_ratio", "attn_local_mask_ratio"]
LENS_FEATURES = [
    "lens_top1_matches_surface",
    "final_rank",
    "surface_rank",
    "final_in_top10",
    "surface_in_top10",
    "final_logit_minus_surface_logit",
    "lens_margin",
]


def classifier():
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=80,
            tol=1e-3,
            class_weight="balanced",
            random_state=23,
        ),
    )


def fit_score(name: str, train: pd.DataFrame, eval_df: pd.DataFrame, features: list[str]) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score

    missing = [feature for feature in features if feature not in train.columns or feature not in eval_df.columns]
    if missing:
        return {"probe": name, "skipped": "missing features: " + ",".join(missing)}
    train_local = train.dropna(subset=features + ["trace_regret"])
    eval_local = eval_df.dropna(subset=features + ["trace_regret"])
    y_train = train_local["trace_regret"].to_numpy(dtype=np.int64)
    y_eval = eval_local["trace_regret"].to_numpy(dtype=np.int64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
        return {
            "probe": name,
            "train_rows": int(len(y_train)),
            "eval_rows": int(len(y_eval)),
            "positive_rate": float(y_eval.mean()) if len(y_eval) else 0.0,
            "skipped": "single-class labels",
        }
    clf = classifier()
    x_train = train_local[features].to_numpy(dtype=np.float32)
    x_eval = eval_local[features].to_numpy(dtype=np.float32)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_eval)[:, 1]
    y_pred = y_prob >= 0.5
    return {
        "probe": name,
        "features": ",".join(features),
        "train_rows": int(len(y_train)),
        "eval_rows": int(len(y_eval)),
        "positive_rate": float(y_eval.mean()),
        "auc": float(roc_auc_score(y_eval, y_prob)),
        "average_precision": float(average_precision_score(y_eval, y_prob)),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "brier": float(brier_score_loss(y_eval, y_prob)),
    }


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trace probes with attention-derived dependency features.")
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--eval-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--early-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    train = pd.read_csv(args.train_rows)
    eval_df = pd.read_csv(args.eval_rows)
    train = train[train["layer"] == args.layer].copy()
    eval_df = eval_df[eval_df["layer"] == args.layer].copy()
    if args.early_only:
        train = train[train["block_t_frac"] <= 0.5].copy()
        eval_df = eval_df[eval_df["block_t_frac"] <= 0.5].copy()

    probes = [
        ("trace_features", FEATURE_NAMES),
        ("trace_plus_attention", FEATURE_NAMES + ATTENTION_FEATURES),
        ("trace_plus_lens", FEATURE_NAMES + LENS_FEATURES),
        ("trace_plus_attention_plus_lens", FEATURE_NAMES + ATTENTION_FEATURES + LENS_FEATURES),
        ("attention_only", ATTENTION_FEATURES),
    ]
    results = [fit_score(name, train, eval_df, features) for name, features in probes]
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "attention_probe_results.csv", index=False)
    lines = ["# Attention Dependency Probe Results", "", table_text(results_df)]
    (output_dir / "attention_probe_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
