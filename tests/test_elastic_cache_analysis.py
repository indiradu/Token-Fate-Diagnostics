from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_elastic_cache_official import bootstrap_metrics, metric_functions


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "output_changed": True,
                "changed_token_count": 4,
                "answer_value_changed": True,
                "baseline_correct": True,
                "elastic_correct": False,
                "baseline_nfe": 10,
                "elastic_nfe": 8,
                "elastic_upstream_layer_recompute_fraction": 0.5,
                "baseline_latency_s": 2.0,
                "elastic_latency_s": 1.0,
            },
            {
                "output_changed": False,
                "changed_token_count": 0,
                "answer_value_changed": False,
                "baseline_correct": False,
                "elastic_correct": True,
                "baseline_nfe": 10,
                "elastic_nfe": 10,
                "elastic_upstream_layer_recompute_fraction": 0.5,
                "baseline_latency_s": 2.0,
                "elastic_latency_s": 1.0,
            },
        ]
    )


def test_metrics_include_quality_compute_and_latency() -> None:
    metrics = {name: function(_rows()) for name, function in metric_functions().items()}
    assert metrics["output_change_rate"] == 0.5
    assert metrics["answer_value_change_rate"] == 0.5
    assert metrics["accuracy_delta"] == 0.0
    assert metrics["correct_to_wrong_rate"] == 0.5
    assert metrics["wrong_to_correct_rate"] == 0.5
    assert metrics["mean_nfe_delta"] == -1.0
    assert metrics["relative_layer_recompute_count"] == 0.45
    assert metrics["wall_clock_speedup"] == 2.0


def test_bootstrap_is_seeded() -> None:
    first = bootstrap_metrics(_rows(), reps=100, seed=7)
    second = bootstrap_metrics(_rows(), reps=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
