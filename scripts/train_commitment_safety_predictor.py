#!/usr/bin/env python3
"""Train calibrated direct commitment-safety predictors on causal labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regret_remasking.commitment_safety import (
    CHEAP_COMMITMENT_FEATURES,
    FULL_COMMITMENT_FEATURES,
    select_safety_threshold,
    wilson_upper_bound,
)


def _load_partitions(input_dirs: list[Path], partition: str) -> pd.DataFrame:
    frames = []
    for root in input_dirs:
        path = root / partition / "labels.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_dir"] = str(root)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output["group_id"] = (
        output["dataset"].astype(str) + ":" + output["example_id"].astype(str)
    )
    return output


def _clean(frame: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    required = list(dict.fromkeys(features + [target, "group_id", "dataset", "example_id"]))
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"commitment label table is missing columns: {missing}")
    output = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    output[target] = output[target].astype(int)
    if output[target].nunique() < 2:
        raise ValueError(f"{target} has fewer than two classes")
    return output


def _grouped_fit_calibration_split(
    frame: pd.DataFrame, target: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame["group_id"].astype(str).to_numpy()
    labels = frame[target].to_numpy(dtype=int)
    for offset in range(100):
        rng = np.random.default_rng(seed + offset)
        calibration_groups: set[str] = set()
        for dataset, subset in frame.groupby("dataset", sort=True):
            del dataset
            available = np.asarray(sorted(subset["group_id"].astype(str).unique()))
            count = max(1, int(round(0.25 * len(available))))
            calibration_groups.update(
                rng.choice(available, size=min(count, len(available)), replace=False).tolist()
            )
        calibration_mask = np.isin(groups, list(calibration_groups))
        calibration = np.flatnonzero(calibration_mask)
        fit = np.flatnonzero(~calibration_mask)
        if np.unique(labels[fit]).size == 2 and np.unique(labels[calibration]).size == 2:
            return fit, calibration
    raise RuntimeError("could not create a group-disjoint two-class calibration split")


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    probability = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def _selection_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    selected = np.asarray(scores) >= threshold
    count = int(selected.sum())
    unsafe = int((labels[selected] == 0).sum()) if count else 0
    return {
        "selected": count,
        "unsafe": unsafe,
        "coverage": float(count / len(labels)) if len(labels) else 0.0,
        "observed_unsafe_rate": float(unsafe / count) if count else 0.0,
        "unsafe_rate_upper95": wilson_upper_bound(unsafe, count),
    }


def _fit_model(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    seed: int,
    max_observed_unsafe_rate: float,
    minimum_selected: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fit_indices, calibration_indices = _grouped_fit_calibration_split(
        train, target, seed
    )
    x = train[features].to_numpy(dtype=np.float32)
    y = train[target].to_numpy(dtype=np.int64)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )
    pipeline.fit(x[fit_indices], y[fit_indices])
    raw_calibration = pipeline.decision_function(x[calibration_indices]).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=seed + 100)
    calibrator.fit(raw_calibration, y[calibration_indices])
    calibration_probability = calibrator.predict_proba(raw_calibration)[:, 1]
    selected = select_safety_threshold(
        calibration_probability,
        y[calibration_indices],
        max_observed_unsafe_rate=max_observed_unsafe_rate,
        minimum_selected=minimum_selected,
    )
    evaluation_probability = calibrator.predict_proba(
        pipeline.decision_function(
            evaluation[features].to_numpy(dtype=np.float32)
        ).reshape(-1, 1)
    )[:, 1]
    report = {
        "features": features,
        "target": target,
        "fit_rows": int(len(fit_indices)),
        "calibration_rows": int(len(calibration_indices)),
        "fit_examples": int(train.iloc[fit_indices]["group_id"].nunique()),
        "calibration_examples": int(
            train.iloc[calibration_indices]["group_id"].nunique()
        ),
        "train_safe_rate": float(y.mean()),
        "calibration_metrics": _classification_metrics(
            y[calibration_indices], calibration_probability
        ),
        "calibration_threshold": asdict(selected),
        "evaluation_metrics": _classification_metrics(
            evaluation[target].to_numpy(dtype=int), evaluation_probability
        ),
        "evaluation_selection": _selection_metrics(
            evaluation[target].to_numpy(dtype=int),
            evaluation_probability,
            selected.threshold,
        ),
    }
    bundle = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "features": features,
        "threshold": selected.threshold,
        "target": target,
        "calibration_threshold": asdict(selected),
    }
    return bundle, report, evaluation_probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=["sequence_safe", "answer_safe", "quality_safe"],
        default="sequence_safe",
    )
    parser.add_argument("--max-observed-unsafe-rate", type=float, default=0.0)
    parser.add_argument("--minimum-selected", type=int, default=8)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_all = _load_partitions(args.input_dir, "train")
    evaluation_all = _load_partitions(args.input_dir, "eval")
    overlap = set(train_all["group_id"]) & set(evaluation_all["group_id"])
    if overlap:
        raise ValueError(f"training/evaluation examples overlap: {sorted(overlap)[:5]}")

    reports: dict[str, Any] = {
        "target": args.target,
        "train_rows": int(len(train_all)),
        "evaluation_rows": int(len(evaluation_all)),
        "train_examples": int(train_all["group_id"].nunique()),
        "evaluation_examples": int(evaluation_all["group_id"].nunique()),
        "models": {},
        "baselines": {},
    }
    scored = evaluation_all[
        ["dataset", "example_id", "global_step", "relative_position", args.target]
    ].copy()
    policy_config: dict[str, Any] = {
        "target": args.target,
        "max_observed_unsafe_rate": args.max_observed_unsafe_rate,
        "minimum_selected": args.minimum_selected,
        "models": {},
        "baselines": {},
    }
    from joblib import dump

    for name, features in (
        (("cheap", CHEAP_COMMITMENT_FEATURES), ("full", FULL_COMMITMENT_FEATURES))
    ):
        train = _clean(train_all, features, args.target)
        evaluation = _clean(evaluation_all, features, args.target)
        bundle, report, probability = _fit_model(
            train,
            evaluation,
            features=features,
            target=args.target,
            seed=args.seed,
            max_observed_unsafe_rate=args.max_observed_unsafe_rate,
            minimum_selected=args.minimum_selected,
        )
        path = args.output_dir / f"commitment_safety_{name}.joblib"
        dump(bundle, path)
        reports["models"][name] = report
        policy_config["models"][name] = {
            "path": str(path),
            "threshold": float(bundle["threshold"]),
            "features": features,
        }
        if len(evaluation) != len(scored):
            raise RuntimeError("feature cleaning changed the shared evaluation subset")
        scored[f"{name}_probability"] = probability

    # Calibrate observational-stability baselines with the identical causal
    # labels and risk rule. These thresholds are not tuned on evaluation data.
    reference_train = _clean(
        train_all, FULL_COMMITMENT_FEATURES, args.target
    )
    reference_eval = _clean(
        evaluation_all, FULL_COMMITMENT_FEATURES, args.target
    )
    _fit_indices, calibration_indices = _grouped_fit_calibration_split(
        reference_train, args.target, args.seed
    )
    calibration_labels = reference_train.iloc[calibration_indices][args.target].to_numpy(
        dtype=int
    )
    for name, column in (
        ("confidence", "confidence"),
        ("trace", "trace_probability"),
    ):
        calibration_scores = reference_train.iloc[calibration_indices][column].to_numpy(
            dtype=float
        )
        threshold = select_safety_threshold(
            calibration_scores,
            calibration_labels,
            max_observed_unsafe_rate=args.max_observed_unsafe_rate,
            minimum_selected=args.minimum_selected,
        )
        evaluation_scores = reference_eval[column].to_numpy(dtype=float)
        reports["baselines"][name] = {
            "calibration_threshold": asdict(threshold),
            "evaluation_metrics": _classification_metrics(
                reference_eval[args.target].to_numpy(dtype=int), evaluation_scores
            ),
            "evaluation_selection": _selection_metrics(
                reference_eval[args.target].to_numpy(dtype=int),
                evaluation_scores,
                threshold.threshold,
            ),
        }
        policy_config["baselines"][name] = {
            "feature": column,
            "threshold": threshold.threshold,
        }
        scored[f"{name}_score"] = evaluation_scores

    scored.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "policy_config.json").write_text(
        json.dumps(policy_config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
