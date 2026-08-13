#!/usr/bin/env python3
"""Evaluate whether token-fate features add value beyond surface confidence.

All thresholds and fitted models use training traces only. Evaluation traces are
split by example already, so this reports generalization to unseen trajectories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_metadata(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = run_dir / "train" / "metadata.csv"
    eval_path = run_dir / "eval" / "metadata.csv"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"expected train/eval metadata under {run_dir}")
    return pd.read_csv(train_path), pd.read_csv(eval_path)


def feature_sets(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, list[str]]:
    full = [name for name in FEATURE_NAMES if name in train and name in eval_df]
    if "jsd" in train and "jsd" in eval_df:
        full.append("jsd")
    lite_names = [
        "confidence",
        "entropy",
        "margin",
        "top1_flip",
        "runlength",
        "t_frac",
        "local_mask_ratio",
    ]
    lite = [name for name in lite_names if name in train and name in eval_df]
    return {"confidence_only": ["confidence"], "fate_lite": lite, "full_trace": full}


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def fit_predictors(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit confidence-only and token-fate predictors on training traces."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = train["trace_regret"].to_numpy(dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("training labels contain only one class")
    scores: dict[str, np.ndarray] = {}
    clean_conf = train[["confidence", "trace_regret"]].replace([np.inf, -np.inf], np.nan).dropna()
    confidence_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=23),
    )
    confidence_model.fit(
        clean_conf[["confidence"]].to_numpy(dtype=np.float32),
        clean_conf["trace_regret"].to_numpy(dtype=np.int64),
    )
    eval_conf = eval_df[["confidence"]].replace([np.inf, -np.inf], np.nan)
    valid_conf = ~eval_conf.isna().any(axis=1)
    confidence_scores = np.full(len(eval_df), np.nan, dtype=float)
    confidence_scores[valid_conf.to_numpy()] = confidence_model.predict_proba(
        eval_conf.loc[valid_conf, ["confidence"]].to_numpy(dtype=np.float32)
    )[:, 1]
    scores["confidence_only"] = confidence_scores

    for name, features in feature_sets(train, eval_df).items():
        if name == "confidence_only" or not features:
            continue
        clean_train = train[features + ["trace_regret"]].replace([np.inf, -np.inf], np.nan).dropna()
        model = make_pipeline(
            StandardScaler(),
            # This analysis compares probability quality with a confidence-only
            # calibrator, so preserve the natural training prevalence here.
            LogisticRegression(max_iter=1000, random_state=23),
        )
        model.fit(clean_train[features].to_numpy(dtype=np.float32), clean_train["trace_regret"].to_numpy(dtype=np.int64))
        clean_eval = eval_df[features].replace([np.inf, -np.inf], np.nan)
        valid = ~clean_eval.isna().any(axis=1)
        out = np.full(len(eval_df), np.nan, dtype=float)
        out[valid.to_numpy()] = model.predict_proba(clean_eval.loc[valid, features].to_numpy(dtype=np.float32))[:, 1]
        scores[name] = out
    return scores


def metrics_for_slice(
    label: str,
    slice_name: str,
    eval_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    mask: np.ndarray,
    top_k: int,
) -> list[dict[str, object]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    rows: list[dict[str, object]] = []
    y_all = eval_df["trace_regret"].to_numpy(dtype=np.int64)
    for method, all_scores in scores.items():
        valid = mask & np.isfinite(all_scores)
        y = y_all[valid]
        score = all_scores[valid]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        score = np.clip(score, 1e-6, 1 - 1e-6)
        subset = eval_df.loc[valid, ["example_id", "trace_regret"]].copy()
        subset["score"] = score
        subset = subset.sort_values("score", ascending=False, kind="mergesort")
        subset = subset.drop_duplicates("example_id", keep="first")
        selected = subset.head(min(top_k, len(subset)))
        base_rate = float(y.mean())
        top_rate = float(selected["trace_regret"].mean()) if len(selected) else np.nan
        rows.append(
            {
                "run": label,
                "slice": slice_name,
                "method": method,
                "eval_rows": int(len(y)),
                "examples": int(eval_df.loc[valid, "example_id"].nunique()),
                "base_regret_rate": base_rate,
                "auroc": float(roc_auc_score(y, score)),
                "average_precision": float(average_precision_score(y, score)),
                "log_loss": float(log_loss(y, score, labels=[0, 1])),
                "brier": float(brier_score_loss(y, score)),
                "top_k": int(len(selected)),
                "topk_realized_regret": top_rate,
                "topk_lift": float(top_rate / base_rate) if base_rate else np.nan,
            }
        )
    return rows


def slice_masks(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {"all": np.ones(len(eval_df), dtype=bool)}
    for start, end in zip([0.0, 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.000001]):
        masks[f"activation_{start:.2f}_{min(end, 1.0):.2f}"] = (
            (eval_df["block_t_frac"].to_numpy(dtype=float) >= start)
            & (eval_df["block_t_frac"].to_numpy(dtype=float) < end)
        )
    quantiles = train["confidence"].quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)
    confidence = eval_df["confidence"].to_numpy(dtype=float)
    edges = np.concatenate(([-np.inf], quantiles, [np.inf]))
    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        masks[f"confidence_quartile_{idx}"] = (confidence >= start) & (confidence < end if idx < 4 else confidence <= end)
    masks["high_confidence_q75"] = confidence >= quantiles[-1]
    masks["early_high_confidence_q75"] = masks["high_confidence_q75"] & (
        eval_df["block_t_frac"].to_numpy(dtype=float) <= 0.5
    )
    return masks


def example_bootstrap_predictive_deltas(
    label: str,
    eval_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    bootstrap_samples: int,
    seed: int,
    candidate_method: str = "fate_lite",
    baseline_method: str = "confidence_only",
    slice_names: list[str] | None = None,
) -> pd.DataFrame:
    """Bootstrap fate-vs-confidence metric differences by held-out example."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if bootstrap_samples <= 0 or baseline_method not in scores or candidate_method not in scores:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    labels = eval_df["trace_regret"].to_numpy(dtype=np.int64)
    for slice_name in slice_names or ["high_confidence_q75", "early_high_confidence_q75"]:
        mask = masks.get(slice_name)
        if mask is None:
            continue
        fate_scores = scores[candidate_method]
        confidence_scores = scores[baseline_method]
        valid = mask & np.isfinite(fate_scores) & np.isfinite(confidence_scores)
        if valid.sum() < 2:
            continue
        frame = pd.DataFrame(
            {
                "example_id": eval_df.loc[valid, "example_id"].astype(str).to_numpy(),
                "label": labels[valid],
                "candidate": fate_scores[valid],
                "baseline": confidence_scores[valid],
            }
        )
        unique_examples = frame["example_id"].drop_duplicates().to_numpy()
        if len(unique_examples) < 2 or frame["label"].nunique() < 2:
            continue
        row_indices = {
            example_id: group.index.to_numpy(dtype=int)
            for example_id, group in frame.groupby("example_id", sort=False)
        }
        draws: dict[str, list[float]] = {"auroc": [], "average_precision": []}
        for _ in range(bootstrap_samples):
            sampled_ids = rng.choice(unique_examples, size=len(unique_examples), replace=True)
            indices = np.concatenate([row_indices[example_id] for example_id in sampled_ids])
            sampled = frame.loc[indices]
            y = sampled["label"].to_numpy(dtype=np.int64)
            if len(np.unique(y)) < 2:
                continue
            fate = sampled["candidate"].to_numpy(dtype=float)
            confidence = sampled["baseline"].to_numpy(dtype=float)
            draws["auroc"].append(float(roc_auc_score(y, fate) - roc_auc_score(y, confidence)))
            draws["average_precision"].append(
                float(average_precision_score(y, fate) - average_precision_score(y, confidence))
            )
        for metric, values in draws.items():
            if not values:
                continue
            y = frame["label"].to_numpy(dtype=np.int64)
            fate = frame["candidate"].to_numpy(dtype=float)
            confidence = frame["baseline"].to_numpy(dtype=float)
            point = (
                float(roc_auc_score(y, fate) - roc_auc_score(y, confidence))
                if metric == "auroc"
                else float(average_precision_score(y, fate) - average_precision_score(y, confidence))
            )
            rows.append(
                {
                    "run": label,
                    "slice": slice_name,
                    "comparison": f"{candidate_method}_minus_{baseline_method}",
                    "metric": metric,
                    "eval_rows": int(len(frame)),
                    "bootstrap_examples": int(len(unique_examples)),
                    "point_delta": point,
                    "bootstrap_ci_low": float(np.quantile(values, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_samples": int(len(values)),
                    "bootstrap_unit": "example",
                }
            )
    return pd.DataFrame(rows)


def analyze_run(
    label: str,
    run_dir: Path,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, eval_df = load_metadata(run_dir)
    scores = fit_predictors(train, eval_df)
    masks = slice_masks(train, eval_df)
    rows: list[dict[str, object]] = []
    for slice_name, mask in masks.items():
        rows.extend(metrics_for_slice(label, slice_name, eval_df, scores, mask, top_k))
    result = pd.DataFrame(rows)
    if result.empty:
        return result, pd.DataFrame()
    confidence = result[result["method"].eq("confidence_only")][["run", "slice", "log_loss", "brier"]].rename(
        columns={"log_loss": "confidence_log_loss", "brier": "confidence_brier"}
    )
    result = result.merge(confidence, on=["run", "slice"], how="left")
    result["delta_log_loss_vs_confidence"] = result["log_loss"] - result["confidence_log_loss"]
    result["delta_brier_vs_confidence"] = result["brier"] - result["confidence_brier"]
    bootstrap = example_bootstrap_predictive_deltas(label, eval_df, scores, masks, bootstrap_samples, seed)
    return result, bootstrap


def high_confidence_selector_bootstrap(
    train_metadata: Path,
    eval_metadata: Path,
    selector_model: Path,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate a pre-fit high-confidence selector against confidence only.

    The selector bundle carries the training-derived threshold and feature set;
    evaluation labels are only used after scores have been computed.
    """
    from joblib import load
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = pd.read_csv(train_metadata)
    eval_df = pd.read_csv(eval_metadata)
    bundle = load(selector_model)
    threshold = float(bundle["high_confidence_threshold"])
    early_max = float(bundle.get("early_max", 0.5))
    features = list(bundle["features"])
    train_slice = train[
        (train["confidence"] >= threshold) & (train["block_t_frac"] <= early_max)
    ].copy()
    clean_train = train_slice[["confidence", "trace_regret"]].replace([np.inf, -np.inf], np.nan).dropna()
    confidence_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    confidence_model.fit(
        clean_train[["confidence"]].to_numpy(dtype=np.float32),
        clean_train["trace_regret"].to_numpy(dtype=np.int64),
    )
    clean_eval = eval_df[features].replace([np.inf, -np.inf], np.nan)
    valid = ~clean_eval.isna().any(axis=1)
    candidate_scores = np.full(len(eval_df), np.nan, dtype=float)
    candidate_scores[valid.to_numpy()] = bundle["pipeline"].predict_proba(
        clean_eval.loc[valid, features].to_numpy(dtype=np.float32)
    )[:, 1]
    confidence_frame = eval_df[["confidence"]].replace([np.inf, -np.inf], np.nan)
    valid_confidence = ~confidence_frame.isna().any(axis=1)
    confidence_scores = np.full(len(eval_df), np.nan, dtype=float)
    confidence_scores[valid_confidence.to_numpy()] = confidence_model.predict_proba(
        confidence_frame.loc[valid_confidence, ["confidence"]].to_numpy(dtype=np.float32)
    )[:, 1]
    masks = {
        "selector_high_confidence_early": (
            (eval_df["confidence"].to_numpy(dtype=float) >= threshold)
            & (eval_df["block_t_frac"].to_numpy(dtype=float) <= early_max)
        )
    }
    return example_bootstrap_predictive_deltas(
        "high_confidence_selector",
        eval_df,
        {"selector": candidate_scores, "confidence_only": confidence_scores},
        masks,
        bootstrap_samples,
        seed,
        candidate_method="selector",
        baseline_method="confidence_only",
        slice_names=["selector_high_confidence_early"],
    )


def fixed_selector_bootstrap(
    train_metadata: Path,
    eval_metadata: Path,
    selector_model: Path,
    confidence_threshold: float,
    early_max: float,
    label: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate a frozen trace selector against a train-fit confidence baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from regret_remasking.regret_model import RegretScorer

    train = pd.read_csv(train_metadata)
    eval_df = pd.read_csv(eval_metadata)
    scorer = RegretScorer(selector_model)
    features = scorer.features
    train_slice = train[
        (train["confidence"] >= confidence_threshold) & (train["block_t_frac"] <= early_max)
    ].copy()
    clean_train = train_slice[["confidence", "trace_regret"]].replace([np.inf, -np.inf], np.nan).dropna()
    confidence_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    confidence_model.fit(
        clean_train[["confidence"]].to_numpy(dtype=np.float32),
        clean_train["trace_regret"].to_numpy(dtype=np.int64),
    )
    selector_frame = eval_df[features].replace([np.inf, -np.inf], np.nan)
    valid_selector = ~selector_frame.isna().any(axis=1)
    selector_scores = np.full(len(eval_df), np.nan, dtype=float)
    selector_scores[valid_selector.to_numpy()] = scorer.score_matrix(
        selector_frame.loc[valid_selector, features].to_numpy(dtype=np.float32)
    )
    confidence_frame = eval_df[["confidence"]].replace([np.inf, -np.inf], np.nan)
    valid_confidence = ~confidence_frame.isna().any(axis=1)
    confidence_scores = np.full(len(eval_df), np.nan, dtype=float)
    confidence_scores[valid_confidence.to_numpy()] = confidence_model.predict_proba(
        confidence_frame.loc[valid_confidence, ["confidence"]].to_numpy(dtype=np.float32)
    )[:, 1]
    masks = {
        "fixed_selector_high_confidence_early": (
            (eval_df["confidence"].to_numpy(dtype=float) >= confidence_threshold)
            & (eval_df["block_t_frac"].to_numpy(dtype=float) <= early_max)
        )
    }
    return example_bootstrap_predictive_deltas(
        label,
        eval_df,
        {"fixed_selector": selector_scores, "confidence_only": confidence_scores},
        masks,
        bootstrap_samples,
        seed,
        candidate_method="fixed_selector",
        baseline_method="confidence_only",
        slice_names=["fixed_selector_high_confidence_early"],
    )


def causal_comparisons(
    causal_results: Path,
    bootstrap_samples: int,
    seed: int,
    reference_group: str,
    pairing: str,
) -> pd.DataFrame:
    df = pd.read_csv(causal_results)
    required = {"example_id", "candidate_group", "final_token_changed", "answer_correct_changed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"causal results missing columns: {sorted(missing)}")
    reference = df[df["candidate_group"].eq(reference_group)].copy()
    if reference.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group, baseline in df.groupby("candidate_group"):
        if group == reference_group:
            continue
        metrics = [
            metric
            for metric in [
                "non_target_token_changed",
                "non_target_token_change_count",
                "non_target_token_change_fraction",
                "suffix_token_changed",
                "suffix_token_change_count",
                "final_token_changed",
                "answer_correct_changed",
            ]
            if metric in reference and metric in baseline
        ]
        for metric in metrics:
            if pairing == "paired":
                pair_key = "pair_id" if "pair_id" in baseline and baseline["pair_id"].notna().any() else "example_id"
                reference_key = "candidate_row_id" if pair_key == "pair_id" else "example_id"
                comparison = reference.set_index(reference_key).join(
                    baseline.set_index(pair_key),
                    how="inner",
                    lsuffix="_learned",
                    rsuffix="_baseline",
                )
                if comparison.empty:
                    continue
                reference_values = comparison[f"{metric}_learned"].to_numpy(dtype=float)
                comparison_values = comparison[f"{metric}_baseline"].to_numpy(dtype=float)
                draws = np.array(
                    [
                        rng.choice(reference_values - comparison_values, size=len(comparison), replace=True).mean()
                        for _ in range(bootstrap_samples)
                    ]
                )
                same_trajectory = bool(
                    (comparison["example_id_learned"].astype(str) == comparison["example_id_baseline"].astype(str)).all()
                )
                comparison_design = "same_trajectory_paired" if same_trajectory else "matched_pair_bootstrap"
            else:
                reference_values = reference[metric].to_numpy(dtype=float)
                comparison_values = baseline[metric].to_numpy(dtype=float)
                if len(reference_values) == 0 or len(comparison_values) == 0:
                    continue
                draws = np.array(
                    [
                        rng.choice(reference_values, size=len(reference_values), replace=True).mean()
                        - rng.choice(comparison_values, size=len(comparison_values), replace=True).mean()
                        for _ in range(bootstrap_samples)
                    ]
                )
                comparison_design = "unpaired_example_bootstrap"
            rows.append(
                {
                    "reference": reference_group,
                    "comparison": group,
                    "metric": metric,
                    "comparison_design": comparison_design,
                    "reference_examples": int(len(reference_values)),
                    "comparison_examples": int(len(comparison_values)),
                    "learned_rate": float(reference_values.mean()),
                    "comparison_rate": float(comparison_values.mean()),
                    "delta_learned_minus_comparison": float(reference_values.mean() - comparison_values.mean()),
                    "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_samples": int(len(draws)),
                    "bootstrap_unit": "matched example pair" if pairing == "paired" else "example",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure token-fate value beyond confidence.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--causal-results", type=Path)
    parser.add_argument("--causal-reference", default="learned_trace")
    parser.add_argument("--causal-pairing", choices=["paired", "unpaired"], default="paired")
    parser.add_argument("--high-confidence-train-metadata", type=Path)
    parser.add_argument("--high-confidence-eval-metadata", type=Path)
    parser.add_argument("--high-confidence-selector-model", type=Path)
    parser.add_argument("--fixed-selector-model", type=Path)
    parser.add_argument("--fixed-selector-train-metadata", type=Path)
    parser.add_argument("--fixed-selector-eval-metadata", type=Path)
    parser.add_argument("--fixed-selector-confidence-threshold", type=float)
    parser.add_argument("--fixed-selector-early-max", type=float, default=0.5)
    parser.add_argument("--fixed-selector-label", default="fixed_trace_selector")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyses = [analyze_run(label, run_dir, args.top_k, args.bootstrap_samples, args.seed) for label, run_dir in args.run]
    observational_tables = [result for result, _ in analyses]
    bootstrap_tables = [result for _, result in analyses]
    observational = pd.concat(observational_tables, ignore_index=True) if observational_tables else pd.DataFrame()
    predictive_bootstrap = pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame()
    observational.to_csv(args.output_dir / "confidence_conditioned_superiority.csv", index=False)
    predictive_bootstrap.to_csv(args.output_dir / "predictive_metric_bootstrap.csv", index=False)

    high_confidence_bootstrap = pd.DataFrame()
    high_confidence_args = [
        args.high_confidence_train_metadata,
        args.high_confidence_eval_metadata,
        args.high_confidence_selector_model,
    ]
    if any(item is not None for item in high_confidence_args):
        if not all(item is not None for item in high_confidence_args):
            raise ValueError(
                "--high-confidence-train-metadata, --high-confidence-eval-metadata, and "
                "--high-confidence-selector-model must be supplied together"
            )
        high_confidence_bootstrap = high_confidence_selector_bootstrap(
            args.high_confidence_train_metadata,
            args.high_confidence_eval_metadata,
            args.high_confidence_selector_model,
            args.bootstrap_samples,
            args.seed,
        )
    high_confidence_bootstrap.to_csv(args.output_dir / "high_confidence_selector_bootstrap.csv", index=False)

    fixed_selector_bootstrap_results = pd.DataFrame()
    fixed_selector_args = [
        args.fixed_selector_model,
        args.fixed_selector_train_metadata,
        args.fixed_selector_eval_metadata,
        args.fixed_selector_confidence_threshold,
    ]
    if any(item is not None for item in fixed_selector_args):
        if not all(item is not None for item in fixed_selector_args):
            raise ValueError(
                "all --fixed-selector-* metadata/model/threshold arguments must be supplied together"
            )
        fixed_selector_bootstrap_results = fixed_selector_bootstrap(
            args.fixed_selector_train_metadata,
            args.fixed_selector_eval_metadata,
            args.fixed_selector_model,
            args.fixed_selector_confidence_threshold,
            args.fixed_selector_early_max,
            args.fixed_selector_label,
            args.bootstrap_samples,
            args.seed,
        )
    fixed_selector_bootstrap_results.to_csv(args.output_dir / "fixed_selector_bootstrap.csv", index=False)

    causal = pd.DataFrame()
    if args.causal_results and args.causal_results.exists():
        causal = causal_comparisons(
            args.causal_results,
            args.bootstrap_samples,
            args.seed,
            args.causal_reference,
            args.causal_pairing,
        )
        causal.to_csv(args.output_dir / "causal_selector_comparisons.csv", index=False)

    lines = ["# Token-Fate Superiority Analysis", ""]
    if not observational.empty:
        report_slices = ["all", "activation_0.00_0.25", "high_confidence_q75", "early_high_confidence_q75"]
        report = observational[observational["slice"].isin(report_slices)].copy()
        cols = [
            "run",
            "slice",
            "method",
            "auroc",
            "average_precision",
            "delta_log_loss_vs_confidence",
            "topk_realized_regret",
            "topk_lift",
        ]
        lines.extend(["## Held-out conditional results", table_text(report[cols]), ""])
    if not predictive_bootstrap.empty:
        lines.extend(["## Example-bootstrap prediction deltas", table_text(predictive_bootstrap), ""])
    if not high_confidence_bootstrap.empty:
        lines.extend(["## High-confidence selector bootstrap", table_text(high_confidence_bootstrap), ""])
    if not fixed_selector_bootstrap_results.empty:
        lines.extend(["## Frozen selector bootstrap", table_text(fixed_selector_bootstrap_results), ""])
    if not causal.empty:
        lines.extend(["## Causal group comparisons", table_text(causal), ""])
    (args.output_dir / "superiority_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
