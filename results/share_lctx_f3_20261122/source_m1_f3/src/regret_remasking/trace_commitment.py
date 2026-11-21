from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regret_remasking import FEATURE_NAMES


REQUIRED_TRACE_COLUMNS = ["example_id", "top_token", "final_token"]
EVENT_KEY_COLUMNS = ["example_id", "global_step", "position"]
PROVENANCE_COLUMNS = [
    "source_sha256",
    "gen_length",
    "block_length",
    "collection_source",
    "source",
    "dataset",
    "model_name",
    "model_revision",
    "tokenizer_revision",
    "feature_schema",
    "feature_schema_hash",
    "feature_names",
    "trace_feature_schema",
]
MODEL_FILENAMES = {
    "trace": "trace_model.joblib",
    "confidence": "confidence_model.joblib",
}


@dataclass(frozen=True)
class TraceCommitmentArtifacts:
    output_dir: Path
    trace_model_path: Path
    confidence_model_path: Path
    metadata_path: Path
    metrics_path: Path
    provenance_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_manifest(path: str | Path | None, split: str) -> pd.DataFrame | None:
    if path is None:
        return None
    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(manifest_path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(manifest_path, lines=True)
    elif suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "examples" in payload:
                payload = payload["examples"]
            else:
                payload = [payload]
        frame = pd.DataFrame(payload)
    else:
        raise ValueError(f"{split} manifest must be .csv, .json, or .jsonl: {manifest_path}")

    if "example_id" not in frame.columns:
        raise ValueError(f"{split} manifest is missing example_id")
    if frame["example_id"].isna().any():
        raise ValueError(f"{split} manifest contains missing example_id")
    if frame["example_id"].duplicated().any():
        dupes = sorted(frame.loc[frame["example_id"].duplicated(), "example_id"].astype(str).unique())
        raise ValueError(f"{split} manifest contains duplicate example_id values: {dupes[:5]}")
    return frame


def _hash_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in ("content_hash", "prompt_hash", "normalized_prompt_hash", "normalized_hash")
        if name in frame.columns
    ]


def _validate_manifest(
    traces: pd.DataFrame,
    manifest: pd.DataFrame | None,
    split: str,
) -> None:
    if manifest is None:
        return
    manifest_ids = set(manifest["example_id"].astype(str))
    trace_ids = set(traces["example_id"].astype(str))
    missing = sorted(trace_ids - manifest_ids)
    if missing:
        raise ValueError(f"{split} traces contain examples absent from manifest: {missing[:5]}")
    extra = sorted(manifest_ids - trace_ids)
    if extra:
        raise ValueError(f"{split} manifest contains examples absent from traces: {extra[:5]}")

    shared_hash_columns = [name for name in _hash_columns(manifest) if name in traces.columns]
    if not shared_hash_columns:
        return
    manifest_index = manifest.set_index(manifest["example_id"].astype(str))
    for column in shared_hash_columns:
        expected = manifest_index[column].astype(str)
        observed = traces[["example_id", column]].drop_duplicates()
        for row in observed.itertuples(index=False):
            example_id = str(row.example_id)
            if example_id in expected.index and str(getattr(row, column)) != expected.loc[example_id]:
                raise ValueError(
                    f"{split} trace {column} mismatch for example_id={example_id}: "
                    f"{getattr(row, column)!r} != {expected.loc[example_id]!r}"
                )


def _validate_cross_csv_hashes(fit: pd.DataFrame, calibration: pd.DataFrame) -> None:
    for column in set(_hash_columns(fit)) & set(_hash_columns(calibration)):
        fit_hashes = set(fit[column].dropna().astype(str))
        calibration_hashes = set(calibration[column].dropna().astype(str))
        overlap = sorted(fit_hashes & calibration_hashes)
        if overlap:
            raise ValueError(f"fit and calibration CSVs overlap by {column}: {overlap[:5]}")


def _validate_provenance_compatibility(fit: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, str]:
    shared = [name for name in PROVENANCE_COLUMNS if name in fit.columns and name in calibration.columns]
    compatible: dict[str, str] = {}
    for column in shared:
        fit_values = set(fit[column].dropna().astype(str))
        calibration_values = set(calibration[column].dropna().astype(str))
        if len(fit_values) > 1:
            raise ValueError(f"fit CSV has multiple {column} provenance values")
        if len(calibration_values) > 1:
            raise ValueError(f"calibration CSV has multiple {column} provenance values")
        if fit_values and calibration_values and fit_values != calibration_values:
            raise ValueError(f"fit and calibration CSVs have incompatible {column} provenance")
        if fit_values or calibration_values:
            compatible[column] = next(iter(fit_values or calibration_values))
    return compatible


def _validate_trace_split_column(frame: pd.DataFrame, expected: set[str], split: str) -> None:
    if "split" not in frame.columns:
        return
    values = set(frame["split"].dropna().astype(str))
    if len(values) != 1:
        raise ValueError(f"{split} CSV must contain exactly one split value")
    actual = next(iter(values))
    if actual not in expected:
        expected_text = ", ".join(sorted(expected))
        raise ValueError(f"{split} CSV split value must be one of {{{expected_text}}}, got {actual!r}")


def _check_disjoint_manifests(
    fit_manifest: pd.DataFrame | None,
    calibration_manifest: pd.DataFrame | None,
) -> None:
    if fit_manifest is None or calibration_manifest is None:
        return
    fit_ids = set(fit_manifest["example_id"].astype(str))
    calibration_ids = set(calibration_manifest["example_id"].astype(str))
    overlap = sorted(fit_ids & calibration_ids)
    if overlap:
        raise ValueError(f"fit and calibration manifests overlap by example_id: {overlap[:5]}")
    for column in set(_hash_columns(fit_manifest)) & set(_hash_columns(calibration_manifest)):
        fit_hashes = set(fit_manifest[column].dropna().astype(str))
        calibration_hashes = set(calibration_manifest[column].dropna().astype(str))
        hash_overlap = sorted(fit_hashes & calibration_hashes)
        if hash_overlap:
            raise ValueError(f"fit and calibration manifests overlap by {column}: {hash_overlap[:5]}")


def _load_trace_csv(path: str | Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_TRACE_COLUMNS + FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"{split} CSV is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{split} CSV is empty")
    if frame.duplicated().any():
        raise ValueError(f"{split} CSV contains duplicate trace rows")
    if frame["example_id"].isna().any():
        raise ValueError(f"{split} CSV contains missing example_id")
    if all(column in frame.columns for column in EVENT_KEY_COLUMNS):
        if frame[EVENT_KEY_COLUMNS].isna().any().any():
            raise ValueError(f"{split} CSV contains missing event key values")
        if frame.duplicated(subset=EVENT_KEY_COLUMNS).any():
            raise ValueError(f"{split} CSV contains duplicate event keys")

    features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    feature_values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"{split} CSV contains nonfinite or nonnumeric feature values")
    frame = frame.copy()
    frame[FEATURE_NAMES] = features

    token_values = frame[["top_token", "final_token"]].apply(pd.to_numeric, errors="coerce")
    token_array = token_values.to_numpy(dtype=np.float64)
    if not np.isfinite(token_array).all():
        raise ValueError(f"{split} CSV contains missing or nonnumeric top_token/final_token values")
    if not np.equal(token_array, np.floor(token_array)).all():
        raise ValueError(f"{split} CSV contains non-integer top_token/final_token values")
    frame["top_token"] = token_values["top_token"].to_numpy(dtype=np.int64)
    frame["final_token"] = token_values["final_token"].to_numpy(dtype=np.int64)

    direct_stable = (frame["top_token"] == frame["final_token"]).astype(np.int64)
    if "stable" in frame.columns:
        explicit = pd.to_numeric(frame["stable"], errors="coerce")
        if explicit.isna().any() or not explicit.isin([0, 1]).all():
            raise ValueError(f"{split} CSV contains invalid stable labels")
        if not np.array_equal(explicit.to_numpy(dtype=np.int64), direct_stable.to_numpy(dtype=np.int64)):
            raise ValueError(f"{split} stable labels disagree with top_token == final_token")
    if "trace_regret" in frame.columns:
        regret = pd.to_numeric(frame["trace_regret"], errors="coerce")
        if regret.isna().any() or not regret.isin([0, 1]).all():
            raise ValueError(f"{split} CSV contains invalid trace_regret labels")
        expected_regret = 1 - direct_stable.to_numpy(dtype=np.int64)
        if not np.array_equal(regret.to_numpy(dtype=np.int64), expected_regret):
            raise ValueError(f"{split} trace_regret labels disagree with 1 - stable")
    frame["stable_label"] = direct_stable
    frame["example_id"] = frame["example_id"].astype(str)
    return frame


def _example_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("example_id")["example_id"].transform("size").to_numpy(dtype=np.float64)
    weights = 1.0 / counts
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("failed to compute positive finite example weights")
    return weights


def _ensure_two_classes(frame: pd.DataFrame, split: str) -> None:
    labels = set(frame["stable_label"].astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"{split} labels must contain both stable classes 0 and 1")


def _positive_index(classes: np.ndarray, positive_label: int = 1) -> int:
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(f"positive class {positive_label} missing from estimator classes {classes!r}")
    return int(matches[0])


def _fit_base_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x, y, scaler__sample_weight=weights, logistic__sample_weight=weights)
    return model


def _check_output_paths(output: Path) -> None:
    output_files = [
        MODEL_FILENAMES["trace"],
        MODEL_FILENAMES["confidence"],
        "metadata.json",
        "metrics.json",
        "provenance.json",
        "trace_fit_uncalibrated_reliability.csv",
        "trace_fit_calibrated_reliability.csv",
        "trace_calibration_uncalibrated_reliability.csv",
        "trace_calibration_calibrated_reliability.csv",
        "confidence_fit_uncalibrated_reliability.csv",
        "confidence_fit_calibrated_reliability.csv",
        "confidence_calibration_uncalibrated_reliability.csv",
        "confidence_calibration_calibrated_reliability.csv",
    ]
    existing = [name for name in output_files if (output / name).exists()]
    if existing:
        raise FileExistsError(f"output model artifacts already exist: {existing[:5]}")


def _fit_sigmoid_calibrator(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    seed: int,
):
    from sklearn.linear_model import LogisticRegression

    calibrator = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    calibrator.fit(decision_scores.reshape(-1, 1), labels, sample_weight=weights)
    return calibrator


def _predict_uncalibrated(model: Any, x: np.ndarray) -> np.ndarray:
    logistic = model.named_steps["logistic"]
    pos_index = _positive_index(logistic.classes_)
    return model.predict_proba(x)[:, pos_index]


def _predict_calibrated(model: Any, calibrator: Any, x: np.ndarray) -> np.ndarray:
    scores = model.decision_function(x).reshape(-1, 1)
    pos_index = _positive_index(calibrator.classes_)
    return calibrator.predict_proba(scores)[:, pos_index]


def _metric_dict(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("model produced invalid probability values")
    return {
        "auroc": float(roc_auc_score(y, p, sample_weight=weights)),
        "average_precision": float(average_precision_score(y, p, sample_weight=weights)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
    }


def _reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "rows": 0,
                    "weight": 0.0,
                    "predicted": 0.0,
                    "observed": 0.0,
                }
            )
            continue
        bin_weight = float(weights[mask].sum())
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "rows": int(mask.sum()),
                "weight": bin_weight,
                "predicted": float(np.average(p[mask], weights=weights[mask])),
                "observed": float(np.average(y[mask], weights=weights[mask])),
            }
        )
    return rows


def _train_one(
    name: str,
    features: list[str],
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_fit = fit[features].to_numpy(dtype=np.float64)
    y_fit = fit["stable_label"].to_numpy(dtype=np.int64)
    w_fit = _example_weights(fit)
    x_cal = calibration[features].to_numpy(dtype=np.float64)
    y_cal = calibration["stable_label"].to_numpy(dtype=np.int64)
    w_cal = _example_weights(calibration)

    model = _fit_base_model(x_fit, y_fit, w_fit, seed)
    calibrator = _fit_sigmoid_calibrator(model.decision_function(x_cal), y_cal, w_cal, seed)

    logistic = model.named_steps["logistic"]
    calibration_coef = float(calibrator.coef_.reshape(-1)[0])
    bundle = {
        "model_kind": name,
        "features": list(features),
        "positive_class": 1,
        "base_positive_index": _positive_index(logistic.classes_),
        "calibration_positive_index": _positive_index(calibrator.classes_),
        "score_polarity": "increasing" if calibration_coef >= 0.0 else "decreasing",
        "pipeline": model,
        "calibrator": calibrator,
    }

    split_metrics: dict[str, Any] = {}
    for split_name, x, y, weights in (
        ("fit", x_fit, y_fit, w_fit),
        ("calibration", x_cal, y_cal, w_cal),
    ):
        uncalibrated = _predict_uncalibrated(model, x)
        calibrated = _predict_calibrated(model, calibrator, x)
        split_metrics[split_name] = {
            "rows": int(len(y)),
            "examples": int(fit["example_id"].nunique() if split_name == "fit" else calibration["example_id"].nunique()),
            "positive_rate_row": float(y.mean()),
            "positive_rate_example_weighted": float(np.average(y, weights=weights)),
            "example_weight_total": float(weights.sum()),
            "uncalibrated": _metric_dict(y, uncalibrated, weights),
            "calibrated": _metric_dict(y, calibrated, weights),
            "reliability_uncalibrated": _reliability_table(y, uncalibrated, weights),
            "reliability_calibrated": _reliability_table(y, calibrated, weights),
        }
    return bundle, split_metrics


def train_trace_commitment(
    fit_csv: str | Path,
    calibration_csv: str | Path,
    output_dir: str | Path,
    fit_manifest: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    seed: int = 23,
) -> TraceCommitmentArtifacts:
    from joblib import dump

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _check_output_paths(output)

    fit = _load_trace_csv(fit_csv, "fit")
    calibration = _load_trace_csv(calibration_csv, "calibration")
    _ensure_two_classes(fit, "fit")
    _ensure_two_classes(calibration, "calibration")
    _validate_trace_split_column(fit, {"fit", "train"}, "fit")
    _validate_trace_split_column(calibration, {"calibration", "cal", "calib"}, "calibration")
    _validate_cross_csv_hashes(fit, calibration)
    provenance_compatibility = _validate_provenance_compatibility(fit, calibration)

    fit_ids = set(fit["example_id"])
    calibration_ids = set(calibration["example_id"])
    overlap = sorted(fit_ids & calibration_ids)
    if overlap:
        raise ValueError(f"fit and calibration CSVs overlap by example_id: {overlap[:5]}")

    fit_manifest_frame = _read_manifest(fit_manifest, "fit")
    calibration_manifest_frame = _read_manifest(calibration_manifest, "calibration")
    _check_disjoint_manifests(fit_manifest_frame, calibration_manifest_frame)
    _validate_manifest(fit, fit_manifest_frame, "fit")
    _validate_manifest(calibration, calibration_manifest_frame, "calibration")

    models = {
        "trace": FEATURE_NAMES,
        "confidence": ["confidence"],
    }
    metrics: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    for name, features in models.items():
        bundle, model_metrics = _train_one(name, list(features), fit, calibration, seed)
        model_path = output / MODEL_FILENAMES[name]
        dump(bundle, model_path)
        model_paths[name] = model_path
        metrics[name] = model_metrics
        for split_name, split_metrics in model_metrics.items():
            for calibration_state in ("uncalibrated", "calibrated"):
                reliability = split_metrics[f"reliability_{calibration_state}"]
                pd.DataFrame(reliability).to_csv(
                    output / f"{name}_{split_name}_{calibration_state}_reliability.csv",
                    index=False,
                )

    metadata = {
        "target": "stable",
        "target_definition": "int(top_token == final_token)",
        "positive_class": 1,
        "fit_csv": str(fit_csv),
        "calibration_csv": str(calibration_csv),
        "fit_manifest": str(fit_manifest) if fit_manifest is not None else None,
        "calibration_manifest": str(calibration_manifest) if calibration_manifest is not None else None,
        "seed": int(seed),
        "feature_schema": list(FEATURE_NAMES),
        "models": {
            "trace": {
                "path": MODEL_FILENAMES["trace"],
                "features": list(FEATURE_NAMES),
                "estimator": "StandardScaler + LogisticRegression + LogisticRegression sigmoid calibration",
            },
            "confidence": {
                "path": MODEL_FILENAMES["confidence"],
                "features": ["confidence"],
                "estimator": "StandardScaler + LogisticRegression + LogisticRegression sigmoid calibration",
            },
        },
        "fit_rows": int(len(fit)),
        "calibration_rows": int(len(calibration)),
        "fit_examples": int(fit["example_id"].nunique()),
        "calibration_examples": int(calibration["example_id"].nunique()),
        "fit_example_weight_total": float(_example_weights(fit).sum()),
        "calibration_example_weight_total": float(_example_weights(calibration).sum()),
        "provenance_compatibility": provenance_compatibility,
    }
    provenance = {
        "inputs": {
            "fit_csv": {"path": str(fit_csv), "sha256": _sha256_file(Path(fit_csv))},
            "calibration_csv": {"path": str(calibration_csv), "sha256": _sha256_file(Path(calibration_csv))},
            "fit_manifest": (
                {"path": str(fit_manifest), "sha256": _sha256_file(Path(fit_manifest))}
                if fit_manifest is not None
                else None
            ),
            "calibration_manifest": (
                {"path": str(calibration_manifest), "sha256": _sha256_file(Path(calibration_manifest))}
                if calibration_manifest is not None
                else None
            ),
        },
        "outputs": {
            "trace_model": MODEL_FILENAMES["trace"],
            "confidence_model": MODEL_FILENAMES["confidence"],
            "metadata": "metadata.json",
            "metrics": "metrics.json",
        },
    }

    metadata_path = output / "metadata.json"
    metrics_path = output / "metrics.json"
    provenance_path = output / "provenance.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(_jsonable(provenance), indent=2) + "\n", encoding="utf-8")

    return TraceCommitmentArtifacts(
        output_dir=output,
        trace_model_path=model_paths["trace"],
        confidence_model_path=model_paths["confidence"],
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        provenance_path=provenance_path,
    )


class TraceCommitmentScorer:
    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.calibrator = bundle["calibrator"]
        self.features: list[str] = list(bundle["features"])
        self.positive_class = int(bundle.get("positive_class", 1))
        self.score_polarity = str(bundle.get("score_polarity", "increasing"))

    @classmethod
    def load(cls, path: str | Path) -> "TraceCommitmentScorer":
        from joblib import load

        bundle = load(path)
        if not isinstance(bundle, dict):
            raise ValueError("trace commitment model bundle must be a dictionary")
        for key in ("pipeline", "calibrator", "features"):
            if key not in bundle:
                raise ValueError(f"trace commitment model bundle missing {key}")
        return cls(bundle)

    def predict_stability(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("matrix must be a 2D batch array")
        if values.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features, got {values.shape[1]}")
        if not np.isfinite(values).all():
            raise ValueError("matrix contains nonfinite feature values")
        probabilities = _predict_calibrated(self.pipeline, self.calibrator, values)
        if probabilities.ndim != 1 or len(probabilities) != values.shape[0]:
            raise ValueError("model returned an invalid probability shape")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("model returned invalid stability probabilities")
        return probabilities.astype(np.float64, copy=False)
