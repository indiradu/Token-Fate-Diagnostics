#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES
from regret_remasking.token_fate import add_row_labels, fate_summary, make_thresholds, token_fate_table


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def load_layer_files(run_dir: Path) -> dict[str, Path]:
    return {path.stem.replace("hidden_layer_", "layer_"): path for path in sorted(run_dir.glob("hidden_layer_*.npy"))}


def stratified_sample(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) < 2:
        return rng.choice(len(y), size=max_rows, replace=False)
    per_class = max(1, max_rows // len(classes))
    indices = []
    for value in classes:
        class_idx = np.flatnonzero(y == value)
        take = min(len(class_idx), per_class)
        indices.extend(rng.choice(class_idx, size=take, replace=False).tolist())
    remaining = max_rows - len(indices)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(y)), np.asarray(indices), assume_unique=False)
        if len(pool) > 0:
            indices.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    rng.shuffle(indices)
    return np.asarray(indices, dtype=np.int64)


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
            max_iter=50,
            tol=1e-3,
            class_weight="balanced",
            random_state=19,
        ),
    )


def score_probe(name: str, clf, x_eval: np.ndarray, y_eval: np.ndarray, n_train: int) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score

    y_prob = clf.predict_proba(x_eval)[:, 1]
    y_pred = y_prob >= 0.5
    row: dict[str, object] = {
        "probe": name,
        "train_rows": int(n_train),
        "eval_rows": int(len(y_eval)),
        "positive_rate": float(y_eval.mean()),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "brier": float(brier_score_loss(y_eval, y_prob)),
    }
    if len(np.unique(y_eval)) >= 2:
        row["auc"] = float(roc_auc_score(y_eval, y_prob))
        row["average_precision"] = float(average_precision_score(y_eval, y_prob))
    else:
        row["auc"] = None
        row["average_precision"] = None
    return row


def fit_and_score(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    max_train_rows: int,
    seed: int,
):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
        return None, {
            "probe": name,
            "train_rows": int(len(y_train)),
            "eval_rows": int(len(y_eval)),
            "positive_rate": float(y_eval.mean()) if len(y_eval) else 0.0,
            "accuracy": None,
            "brier": None,
            "auc": None,
            "average_precision": None,
            "skipped": "single-class labels",
        }
    keep = stratified_sample(y_train, max_train_rows, seed)
    clf = classifier()
    clf.fit(x_train[keep].astype(np.float32), y_train[keep].astype(np.int64))
    return clf, score_probe(name, clf, x_eval.astype(np.float32), y_eval.astype(np.int64), len(keep))


def subset_masks(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    high_conf = float(train["confidence"].quantile(0.75))
    low_kl = float(train["kl"].quantile(0.50))
    return {
        "all_trace_regret": (
            np.ones(len(train), dtype=bool),
            np.ones(len(eval_df), dtype=bool),
            "trace_regret",
        ),
        "early_trace_regret": (
            (train["block_t_frac"] <= 0.5).to_numpy(),
            (eval_df["block_t_frac"] <= 0.5).to_numpy(),
            "trace_regret",
        ),
        "stable_high_conf_trace_regret": (
            ((train["confidence"] >= high_conf) & (train["kl"] <= low_kl)).to_numpy(),
            ((eval_df["confidence"] >= high_conf) & (eval_df["kl"] <= low_kl)).to_numpy(),
            "trace_regret",
        ),
    }


def hidden_dynamics_features(df: pd.DataFrame, hidden: np.ndarray) -> np.ndarray:
    hidden_arr = np.asarray(hidden, dtype=np.float32)
    norms = np.linalg.norm(hidden_arr, axis=1)
    delta_norms = np.zeros(len(df), dtype=np.float32)
    prev_cosines = np.zeros(len(df), dtype=np.float32)
    grouped = df.reset_index().sort_values(["example_id", "position", "global_step"]).groupby(["example_id", "position"])
    for _, group in grouped:
        indices = group["index"].to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        current = hidden_arr[indices[1:]]
        previous = hidden_arr[indices[:-1]]
        delta_norms[indices[1:]] = np.linalg.norm(current - previous, axis=1)
        numerator = (current * previous).sum(axis=1)
        denominator = (norms[indices[1:]] * norms[indices[:-1]]) + 1e-6
        prev_cosines[indices[1:]] = numerator / denominator
    return np.stack([norms, delta_norms, prev_cosines], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logits-only and hidden-state token-fate probes.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-rows", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    train = pd.read_csv(train_dir / "metadata.csv")
    eval_df = pd.read_csv(eval_dir / "metadata.csv")
    thresholds = make_thresholds(train)
    train = add_row_labels(train, thresholds)
    eval_df = add_row_labels(eval_df, thresholds)

    token_fate_table(train, thresholds).to_csv(output_dir / "token_fates_train.csv", index=False)
    token_fate_table(eval_df, thresholds).to_csv(output_dir / "token_fates_eval.csv", index=False)
    fate_summary(train, thresholds).to_csv(output_dir / "fate_summary_train.csv", index=False)
    fate_summary(eval_df, thresholds).to_csv(output_dir / "fate_summary_eval.csv", index=False)

    train_layers = load_layer_files(train_dir)
    eval_layers = load_layer_files(eval_dir)
    common_layers = sorted(set(train_layers) & set(eval_layers))
    train_logits = train[FEATURE_NAMES].to_numpy(dtype=np.float32)
    eval_logits = eval_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    layer_cache_train: dict[str, np.ndarray] = {}
    layer_cache_eval: dict[str, np.ndarray] = {}
    dynamics_cache_train: dict[str, np.ndarray] = {}
    dynamics_cache_eval: dict[str, np.ndarray] = {}

    results: list[dict[str, object]] = []
    best_model = None
    best_auc = -1.0
    best_name = ""
    for target_name, (train_mask, eval_mask, label_name) in subset_masks(train, eval_df).items():
        y_train = train.loc[train_mask, label_name].to_numpy(dtype=np.int64)
        y_eval = eval_df.loc[eval_mask, label_name].to_numpy(dtype=np.int64)
        if len(y_train) == 0 or len(y_eval) == 0:
            continue

        clf, row = fit_and_score(
            f"{target_name}/logits",
            train_logits[train_mask],
            y_train,
            eval_logits[eval_mask],
            y_eval,
            args.max_train_rows,
            args.seed,
        )
        results.append({"target": target_name, **row})
        if clf is not None and row.get("auc") is not None and float(row["auc"]) > best_auc:
            best_model, best_auc, best_name = clf, float(row["auc"]), f"{target_name}/logits"

        for layer_name in common_layers:
            if layer_name not in layer_cache_train:
                layer_cache_train[layer_name] = np.load(train_layers[layer_name], mmap_mode="r")
                layer_cache_eval[layer_name] = np.load(eval_layers[layer_name], mmap_mode="r")
                dynamics_cache_train[layer_name] = hidden_dynamics_features(train, layer_cache_train[layer_name])
                dynamics_cache_eval[layer_name] = hidden_dynamics_features(eval_df, layer_cache_eval[layer_name])
            clf, row = fit_and_score(
                f"{target_name}/{layer_name}_hidden",
                np.asarray(layer_cache_train[layer_name][train_mask]),
                y_train,
                np.asarray(layer_cache_eval[layer_name][eval_mask]),
                y_eval,
                args.max_train_rows,
                args.seed,
            )
            results.append({"target": target_name, **row})
            if clf is not None and row.get("auc") is not None and float(row["auc"]) > best_auc:
                best_model, best_auc, best_name = clf, float(row["auc"]), f"{target_name}/{layer_name}_hidden"

            clf, row = fit_and_score(
                f"{target_name}/{layer_name}_hidden_dynamics",
                dynamics_cache_train[layer_name][train_mask],
                y_train,
                dynamics_cache_eval[layer_name][eval_mask],
                y_eval,
                args.max_train_rows,
                args.seed,
            )
            results.append({"target": target_name, **row})
            if clf is not None and row.get("auc") is not None and float(row["auc"]) > best_auc:
                best_model, best_auc, best_name = clf, float(row["auc"]), f"{target_name}/{layer_name}_hidden_dynamics"

            clf, row = fit_and_score(
                f"{target_name}/{layer_name}_hidden_dynamics_plus_logits",
                np.concatenate([train_logits[train_mask], dynamics_cache_train[layer_name][train_mask]], axis=1),
                y_train,
                np.concatenate([eval_logits[eval_mask], dynamics_cache_eval[layer_name][eval_mask]], axis=1),
                y_eval,
                args.max_train_rows,
                args.seed,
            )
            results.append({"target": target_name, **row})
            if clf is not None and row.get("auc") is not None and float(row["auc"]) > best_auc:
                best_model, best_auc, best_name = (
                    clf,
                    float(row["auc"]),
                    f"{target_name}/{layer_name}_hidden_dynamics_plus_logits",
                )

            clf, row = fit_and_score(
                f"{target_name}/{layer_name}_hidden_plus_logits",
                np.concatenate([train_logits[train_mask], np.asarray(layer_cache_train[layer_name][train_mask])], axis=1),
                y_train,
                np.concatenate([eval_logits[eval_mask], np.asarray(layer_cache_eval[layer_name][eval_mask])], axis=1),
                y_eval,
                args.max_train_rows,
                args.seed,
            )
            results.append({"target": target_name, **row})
            if clf is not None and row.get("auc") is not None and float(row["auc"]) > best_auc:
                best_model, best_auc, best_name = clf, float(row["auc"]), f"{target_name}/{layer_name}_hidden_plus_logits"

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "probe_results.csv", index=False)
    if best_model is not None:
        joblib.dump({"model": best_model, "name": best_name, "auc": best_auc}, output_dir / "best_probe.joblib")

    lines = ["# Token-Fate Probe Results", ""]
    lines.append(f"Best probe: `{best_name}` AUC `{best_auc:.4f}`" if best_model is not None else "No probe fit.")
    lines.append("")
    if not results_df.empty:
        view = results_df[["probe", "eval_rows", "positive_rate", "auc", "average_precision", "accuracy", "brier"]]
        lines.append(table_text(view))
    lines.append("")
    lines.append("## Fate Summary, Train")
    lines.append(table_text(fate_summary(train, thresholds)))
    lines.append("")
    lines.append("## Fate Summary, Eval")
    lines.append(table_text(fate_summary(eval_df, thresholds)))
    (output_dir / "probe_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
