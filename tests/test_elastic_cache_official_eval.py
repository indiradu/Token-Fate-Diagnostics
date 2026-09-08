from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_elastic_cache_official_eval import summarize


def test_summarize_separates_output_answer_and_correctness_changes() -> None:
    rows = pd.DataFrame(
        [
            {
                "output_changed": True,
                "answer_value_changed": False,
                "baseline_correct": True,
                "elastic_correct": True,
                "baseline_nfe": 10,
                "elastic_nfe": 8,
                "elastic_upstream_layer_recompute_fraction": 0.5,
                "baseline_latency_s": 2.0,
                "elastic_latency_s": 1.0,
            },
            {
                "output_changed": True,
                "answer_value_changed": True,
                "baseline_correct": True,
                "elastic_correct": False,
                "baseline_nfe": 12,
                "elastic_nfe": 9,
                "elastic_upstream_layer_recompute_fraction": 0.75,
                "baseline_latency_s": 2.2,
                "elastic_latency_s": 1.2,
            },
        ]
    )
    metrics = summarize(rows)
    assert metrics["output_change_rate"] == 1.0
    assert metrics["answer_value_change_rate"] == 0.5
    assert metrics["baseline_accuracy"] == 1.0
    assert metrics["elastic_accuracy"] == 0.5
    assert metrics["accuracy_delta"] == -0.5
    assert metrics["correct_to_wrong_rate"] == 0.5
    assert metrics["wrong_to_correct_rate"] == 0.0
    assert metrics["mean_elastic_upstream_layer_recompute_fraction"] == 0.625
