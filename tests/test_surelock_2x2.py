from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_surelock_2x2 import bootstrap_effects


def _cell(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "non_target_token_changed": values,
            "non_target_token_change_count": values,
            "answer_value_changed_regraded": values,
            "baseline_correct_regraded": [0.0] * len(values),
            "modified_correct_regraded": [0.0] * len(values),
            "_accuracy_delta": [0.0] * len(values),
        },
        index=[f"e{index}" for index in range(len(values))],
    )


def test_factorial_contrasts_recover_known_interaction():
    lt5_g64 = _cell([0.0, 0.0])
    lt5_g256 = _cell([0.0, 0.0])
    l5_g64 = _cell([0.0, 0.0])
    l5_g256 = _cell([1.0, 1.0])
    result = bootstrap_effects(lt5_g64, lt5_g256, l5_g64, l5_g256, reps=100, seed=7)
    row = result[
        (result["metric"] == "non_target_token_change_rate")
        & (result["contrast"] == "difficulty_by_horizon_interaction")
    ].iloc[0]
    assert row["point"] == 1.0
    assert row["ci_low"] == 1.0
    assert row["ci_high"] == 1.0
