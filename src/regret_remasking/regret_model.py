from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "predicted": 0.0, "observed": 0.0})
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "predicted": float(y_prob[mask].mean()),
                "observed": float(y_true[mask].mean()),
            }
        )
    return rows


def train_regret_predictors(
    trace_csv: str | Path,
    output_dir: str | Path,
    max_rows: int = 250_000,
    seed: int = 13,
) -> dict[str, object]:
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(trace_csv)
    df = df.dropna(subset=FEATURE_NAMES + ["trace_regret"])
    df = df[df["final_token"] != df.get("mask_token_id", -1)]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    if df["trace_regret"].nunique() < 2:
        raise RuntimeError("trace labels have only one class; cannot train regret predictor")

    reports: dict[str, object] = {}
    for name, features in {
        "regret_model": FEATURE_NAMES,
        "regret_model_no_cpv": [f for f in FEATURE_NAMES if f != "context_volatility"],
    }.items():
        x = df[features].to_numpy(dtype=np.float32)
        y = df["trace_regret"].to_numpy(dtype=np.int64)
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        clf.fit(x_train, y_train)
        y_prob = clf.predict_proba(x_val)[:, 1]
        report = {
            "features": features,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "positive_rate": float(y.mean()),
            "auc": float(roc_auc_score(y_val, y_prob)),
            "brier": float(brier_score_loss(y_val, y_prob)),
            "log_loss": float(log_loss(y_val, y_prob)),
            "calibration": _calibration_table(y_val, y_prob),
        }
        dump({"pipeline": clf, "features": features}, output / f"{name}.joblib")
        pd.DataFrame(report["calibration"]).to_csv(output / f"{name}_calibration.csv", index=False)
        reports[name] = report

    with (output / "regret_training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    return reports


class RegretScorer:
    def __init__(self, path: str | Path):
        from joblib import load

        bundle = load(path)
        self.pipeline = bundle["pipeline"]
        self.features: list[str] = list(bundle["features"])
        final_estimator = getattr(self.pipeline, "steps", [[None, self.pipeline]])[-1][1]
        if final_estimator.__class__.__name__ == "LogisticRegression" and not hasattr(final_estimator, "multi_class"):
            final_estimator.multi_class = "auto"

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(matrix.astype(np.float32))[:, 1]
