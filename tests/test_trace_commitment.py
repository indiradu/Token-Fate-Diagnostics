from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking import FEATURE_NAMES
from regret_remasking.trace_commitment import TraceCommitmentScorer, train_trace_commitment


def _row(example_id: str, index: int, stable: int, confidence: float) -> dict[str, object]:
    row: dict[str, object] = {
        "example_id": example_id,
        "global_step": index,
        "position": index % 4,
        "top_token": 100 + index if stable else 900 + index,
        "final_token": 100 + index,
        "stable": stable,
        "trace_regret": 1 - stable,
        "confidence": confidence,
    }
    row.update(
        {
            "entropy": 1.0 - 0.35 * stable + 0.01 * index,
            "margin": confidence - 0.15 + 0.01 * stable,
            "kl": 0.2 - 0.08 * stable + 0.005 * index,
            "top1_flip": float(1 - stable),
            "runlength": float(index % 3 + stable),
            "t_frac": 0.05 * (index + 1),
            "local_mask_ratio": 0.6 - 0.1 * stable + 0.01 * index,
            "context_volatility": 0.4 - 0.12 * stable + 0.005 * index,
        }
    )
    return row


def _write_trace_csv(path: pathlib.Path, prefix: str, examples: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for example_idx in range(examples):
        example_id = f"{prefix}-{example_idx}"
        for row_idx in range(4):
            stable = int((example_idx + row_idx) % 2 == 0)
            confidence = 0.7 + 0.08 * stable - 0.01 * row_idx + 0.005 * example_idx
            rows.append(_row(example_id, row_idx + example_idx * 4, stable, confidence))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _write_valid_pair(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pd.DataFrame, pd.DataFrame]:
    fit_csv = tmp_path / "fit.csv"
    calibration_csv = tmp_path / "calibration.csv"
    fit = _write_trace_csv(fit_csv, "fit")
    calibration = _write_trace_csv(calibration_csv, "cal")
    return fit_csv, calibration_csv, fit, calibration


def _write_manifest(path: pathlib.Path, frame: pd.DataFrame) -> None:
    manifest = (
        frame[["example_id"]]
        .drop_duplicates()
        .assign(content_hash=lambda data: data["example_id"].map(lambda value: f"hash-{value}"))
    )
    manifest.to_csv(path, index=False)


def test_train_save_load_probability_consistency_and_schema(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit_manifest = tmp_path / "fit_manifest.csv"
    calibration_manifest = tmp_path / "calibration_manifest.csv"
    _write_manifest(fit_manifest, fit)
    _write_manifest(calibration_manifest, calibration)

    artifacts = train_trace_commitment(
        fit_csv,
        calibration_csv,
        tmp_path / "out",
        fit_manifest=fit_manifest,
        calibration_manifest=calibration_manifest,
        seed=7,
    )

    scorer = TraceCommitmentScorer.load(artifacts.trace_model_path)
    assert scorer.features == FEATURE_NAMES
    matrix = calibration[FEATURE_NAMES].to_numpy(dtype=np.float64)
    first = scorer.predict_stability(matrix)
    second = TraceCommitmentScorer.load(artifacts.trace_model_path).predict_stability(matrix)
    assert first.shape == (len(calibration),)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert np.allclose(first, second)

    confidence_scorer = TraceCommitmentScorer.load(artifacts.confidence_model_path)
    assert confidence_scorer.features == ["confidence"]
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["target_definition"] == "int(top_token == final_token)"
    assert metadata["fit_example_weight_total"] == pytest.approx(fit["example_id"].nunique())


def test_class_ordering_and_polarity_are_explicit(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, _, calibration = _write_valid_pair(tmp_path)

    artifacts = train_trace_commitment(fit_csv, calibration_csv, tmp_path / "out", seed=11)
    scorer = TraceCommitmentScorer.load(artifacts.trace_model_path)
    assert scorer.positive_class == 1
    assert scorer.score_polarity in {"increasing", "decreasing"}

    probabilities = scorer.predict_stability(calibration[FEATURE_NAMES].to_numpy(dtype=np.float64))
    stable_mask = calibration["stable"].to_numpy(dtype=bool)
    assert probabilities[stable_mask].mean() > probabilities[~stable_mask].mean()


def test_rejects_same_example_leakage(tmp_path: pathlib.Path) -> None:
    fit_csv = tmp_path / "fit.csv"
    calibration_csv = tmp_path / "calibration.csv"
    fit = _write_trace_csv(fit_csv, "shared")
    fit.to_csv(calibration_csv, index=False)

    with pytest.raises(ValueError, match="overlap by example_id"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "out")


def test_fit_only_scaler_and_separate_calibration(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    calibration.loc[:, "confidence"] = calibration["confidence"] + 10.0
    calibration.to_csv(calibration_csv, index=False)

    artifacts = train_trace_commitment(fit_csv, calibration_csv, tmp_path / "out", seed=5)
    scorer = TraceCommitmentScorer.load(artifacts.trace_model_path)
    scaler = scorer.pipeline.named_steps["scaler"]
    per_example_weight = fit.groupby("example_id")["example_id"].transform("size")
    expected_weighted_mean = np.average(fit["confidence"], weights=1.0 / per_example_weight)
    assert scaler.mean_[FEATURE_NAMES.index("confidence")] == pytest.approx(float(expected_weighted_mean))

    calibrated = scorer.predict_stability(calibration[FEATURE_NAMES].to_numpy(dtype=np.float64))
    uncalibrated = scorer.pipeline.predict_proba(calibration[FEATURE_NAMES].to_numpy(dtype=np.float64))[:, 1]
    assert not np.allclose(calibrated, uncalibrated)


def test_online_feature_schema_validation(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, _, _ = _write_valid_pair(tmp_path)
    artifacts = train_trace_commitment(fit_csv, calibration_csv, tmp_path / "out")
    scorer = TraceCommitmentScorer.load(artifacts.trace_model_path)

    with pytest.raises(ValueError, match="2D batch"):
        scorer.predict_stability(np.ones(len(FEATURE_NAMES)))
    with pytest.raises(ValueError, match="expected 9 features"):
        scorer.predict_stability(np.ones((2, len(FEATURE_NAMES) - 1)))
    bad = np.ones((2, len(FEATURE_NAMES)))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scorer.predict_stability(bad)


def test_rejects_one_class_invalid_labels_nonfinite_and_duplicates(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, _ = _write_valid_pair(tmp_path)

    one_class = fit.copy()
    one_class["top_token"] = one_class["final_token"]
    one_class["stable"] = 1
    one_class["trace_regret"] = 0
    one_class.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="both stable classes"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "one_class")

    invalid = fit.copy()
    invalid.loc[0, "stable"] = 2
    invalid.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="invalid stable"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "invalid")

    nonfinite = fit.copy()
    nonfinite.loc[0, "confidence"] = np.inf
    nonfinite.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "nonfinite")

    duplicate = pd.concat([fit, fit.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="duplicate trace rows"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "duplicate")


def test_rejects_missing_or_noninteger_tokens_and_trace_regret_mismatch(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, _ = _write_valid_pair(tmp_path)

    missing_token = fit.copy()
    missing_token.loc[0, "top_token"] = np.nan
    missing_token.loc[0, "stable"] = 0
    missing_token.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="missing or nonnumeric top_token/final_token"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "missing_token")

    fractional_token = fit.copy()
    fractional_token["top_token"] = fractional_token["top_token"].astype(float)
    fractional_token.loc[0, "top_token"] = 1.5
    fractional_token.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="non-integer top_token/final_token"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "fractional_token")

    bad_regret = fit.copy()
    bad_regret.loc[0, "trace_regret"] = int(bad_regret.loc[0, "stable"])
    bad_regret.to_csv(fit_csv, index=False)
    with pytest.raises(ValueError, match="trace_regret labels disagree"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "bad_regret")


def test_rejects_duplicate_event_keys_even_when_rows_differ(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, _ = _write_valid_pair(tmp_path)
    duplicate_key = fit.copy()
    duplicate_key.loc[1, ["example_id", "global_step", "position"]] = duplicate_key.loc[
        0, ["example_id", "global_step", "position"]
    ]
    duplicate_key.loc[1, "confidence"] = duplicate_key.loc[1, "confidence"] + 0.123
    duplicate_key.to_csv(fit_csv, index=False)

    with pytest.raises(ValueError, match="duplicate event keys"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "duplicate_key")


def test_rejects_csv_content_hash_overlap_without_manifests(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit["content_hash"] = [f"fit-hash-{idx}" for idx in range(len(fit))]
    calibration["content_hash"] = [f"cal-hash-{idx}" for idx in range(len(calibration))]
    calibration.loc[0, "content_hash"] = fit.loc[0, "content_hash"]
    fit.to_csv(fit_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)

    with pytest.raises(ValueError, match="overlap by content_hash"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "hash_overlap")


def test_manifest_requires_exact_id_coverage(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit_manifest = tmp_path / "fit_manifest.csv"
    calibration_manifest = tmp_path / "calibration_manifest.csv"
    _write_manifest(fit_manifest, fit)
    _write_manifest(calibration_manifest, calibration)

    too_small = pd.read_csv(fit_manifest).iloc[:-1]
    too_small.to_csv(fit_manifest, index=False)
    with pytest.raises(ValueError, match="absent from manifest"):
        train_trace_commitment(
            fit_csv,
            calibration_csv,
            tmp_path / "manifest_missing",
            fit_manifest=fit_manifest,
            calibration_manifest=calibration_manifest,
        )

    _write_manifest(fit_manifest, fit)
    too_large = pd.read_csv(fit_manifest)
    too_large.loc[len(too_large)] = {"example_id": "extra-fit", "content_hash": "hash-extra-fit"}
    too_large.to_csv(fit_manifest, index=False)
    with pytest.raises(ValueError, match="manifest contains examples absent from traces"):
        train_trace_commitment(
            fit_csv,
            calibration_csv,
            tmp_path / "manifest_extra",
            fit_manifest=fit_manifest,
            calibration_manifest=calibration_manifest,
        )


def test_rejects_incompatible_provenance_when_columns_present(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit["collection_source"] = "baseline-v1"
    calibration["collection_source"] = "baseline-v2"
    fit["feature_schema_hash"] = "schema-a"
    calibration["feature_schema_hash"] = "schema-a"
    fit.to_csv(fit_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)

    with pytest.raises(ValueError, match="incompatible collection_source"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "bad_source")


def test_valid_provenance_and_output_artifacts_are_not_overwritten(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit["collection_source"] = "baseline-v1"
    calibration["collection_source"] = "baseline-v1"
    fit["feature_schema_hash"] = "schema-a"
    calibration["feature_schema_hash"] = "schema-a"
    fit.to_csv(fit_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)

    output_dir = tmp_path / "out"
    artifacts = train_trace_commitment(fit_csv, calibration_csv, output_dir)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["provenance_compatibility"] == {
        "collection_source": "baseline-v1",
        "feature_schema_hash": "schema-a",
    }

    with pytest.raises(FileExistsError, match="already exist"):
        train_trace_commitment(fit_csv, calibration_csv, output_dir)


def test_runner_shaped_split_and_config_hashes_train_end_to_end(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit["split"] = "fit"
    calibration["split"] = "calibration"
    fit["content_hash"] = [f"fit-content-{idx}" for idx in range(len(fit))]
    calibration["content_hash"] = [f"cal-content-{idx}" for idx in range(len(calibration))]
    fit["source_sha256"] = "same-source-sha"
    calibration["source_sha256"] = "same-source-sha"
    fit["config_sha256"] = "fit-config-sha"
    calibration["config_sha256"] = "calibration-config-sha"
    fit["collection_source"] = "llada-baseline-runner"
    calibration["collection_source"] = "llada-baseline-runner"
    fit["feature_schema_hash"] = "trace-feature-schema-v1"
    calibration["feature_schema_hash"] = "trace-feature-schema-v1"
    fit.to_csv(fit_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)

    artifacts = train_trace_commitment(fit_csv, calibration_csv, tmp_path / "runner_out")
    scorer = TraceCommitmentScorer.load(artifacts.trace_model_path)
    probabilities = scorer.predict_stability(calibration[FEATURE_NAMES].to_numpy(dtype=np.float64))
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert probabilities.shape == (len(calibration),)
    assert metadata["provenance_compatibility"]["collection_source"] == "llada-baseline-runner"
    assert metadata["provenance_compatibility"]["feature_schema_hash"] == "trace-feature-schema-v1"
    assert "split" not in metadata["provenance_compatibility"]
    assert "config_sha256" not in metadata["provenance_compatibility"]


def test_rejects_unexpected_trace_split_values(tmp_path: pathlib.Path) -> None:
    fit_csv, calibration_csv, fit, calibration = _write_valid_pair(tmp_path)
    fit["split"] = "calibration"
    calibration["split"] = "calibration"
    fit.to_csv(fit_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)

    with pytest.raises(ValueError, match="fit CSV split value"):
        train_trace_commitment(fit_csv, calibration_csv, tmp_path / "bad_split")
