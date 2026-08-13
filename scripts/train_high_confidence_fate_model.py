#!/usr/bin/env python3
"""Train a token-fate predictor conditional on high current confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


def evaluate(model: object, df: pd.DataFrame, features: list[str]) -> dict[str, float | int]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    clean = df.dropna(subset=features + ["trace_regret"])
    y = clean["trace_regret"].to_numpy(dtype=np.int64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"rows": int(len(y)), "positive_rate": float(y.mean()) if len(y) else 0.0}
    score = model.predict_proba(clean[features].to_numpy(dtype=np.float32))[:, 1]
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a confidence-conditioned high-confidence token-fate model.")
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-quantile", type=float, default=0.75)
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Retain residual variation in confidence; use caliper-matched controls for the causal comparison.",
    )
    parser.add_argument("--early-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_all = pd.read_csv(args.train_metadata)
    threshold = float(train_all["confidence"].quantile(args.confidence_quantile))
    features = [
        name
        for name in FEATURE_NAMES
        if name in train_all.columns and (args.include_confidence or name != "confidence")
    ]
    train = train_all[(train_all["confidence"] >= threshold) & (train_all["block_t_frac"] <= args.early_max)].copy()
    train = train.dropna(subset=features + ["trace_regret"])
    if train["trace_regret"].nunique() < 2:
        raise RuntimeError("high-confidence training slice has only one trace-regret class")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
    )
    model.fit(train[features].to_numpy(dtype=np.float32), train["trace_regret"].to_numpy(dtype=np.int64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": model,
        "features": features,
        "high_confidence_threshold": threshold,
        "confidence_quantile": args.confidence_quantile,
        "include_confidence": args.include_confidence,
        "early_max": args.early_max,
    }
    dump(bundle, args.output_dir / "high_confidence_fate_model.joblib")
    report: dict[str, object] = {
        "threshold": threshold,
        "features": features,
        "train": evaluate(model, train, features),
    }
    if args.eval_metadata:
        eval_all = pd.read_csv(args.eval_metadata)
        eval_df = eval_all[(eval_all["confidence"] >= threshold) & (eval_all["block_t_frac"] <= args.early_max)].copy()
        report["eval"] = evaluate(model, eval_df, features)
    (args.output_dir / "high_confidence_fate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
