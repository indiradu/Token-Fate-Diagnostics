from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_settlement_fate_audit import write_polestar_cache_proxy_summary


def test_polestar_proxy_summary_pairs_result_arm_column(tmp_path) -> None:
    shared = {
        "example_id": "ex1",
        "position": 10,
        "pair_id": "event-1",
        "pre_intervention_replay_valid": True,
        "target_token_changed": False,
        "answer_value_changed": False,
        "baseline_correct": True,
        "polestar_cache_step": 8,
        "polestar_refresh_step": 12,
        "polestar_cache_age": 4,
        "polestar_attention_kl": 0.2,
        "polestar_refresh_drift_mean": 0.3,
    }
    interventions = pd.DataFrame(
        [
            {
                **shared,
                "intervention_group": "polestar_proxy_stale",
                "polestar_proxy_arm": "stale",
                "non_target_token_changed": True,
                "non_target_token_change_count": 3,
                "clamped_correct": False,
            },
            {
                **shared,
                "intervention_group": "polestar_proxy_refresh",
                "polestar_proxy_arm": "refresh",
                "non_target_token_changed": False,
                "non_target_token_change_count": 0,
                "clamped_correct": True,
            },
        ]
    )
    write_polestar_cache_proxy_summary(
        tmp_path,
        argparse.Namespace(intervention_mode="polestar_cache_proxy"),
        pd.DataFrame({"example_id": ["ex1"]}),
        interventions,
        pd.DataFrame({"event_id": ["event-1"]}),
        selected_event_count=1,
        events_truncated=False,
        started_at=time.perf_counter(),
    )
    pairs = pd.read_csv(tmp_path / "polestar_proxy_pair_summary.csv")
    assert len(pairs) == 1
    assert pairs.loc[0, "stale_minus_refresh_downstream_changed"] == 1.0
    assert pairs.loc[0, "stale_minus_refresh_change_count"] == 3
