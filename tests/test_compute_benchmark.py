from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_phase_lock_compute import _plan_hash, _runtime_environment, _summaries
from analyze_phase_lock_compute import _validate


def _row(example: str, repeat: int, gain: float, exact: int = 1) -> dict[str, object]:
    a2 = 1.0
    a3 = 1.0 - gain
    return {
        "dataset": "gsm8k",
        "semantic_label": "safe",
        "example_id": example,
        "repeat": repeat,
        "runtime_gain": gain,
        "speedup": a2 / a3,
        "a2_latency_s": a2,
        "a3_latency_s": a3,
        "a3_exact_a2": exact,
        "answer_or_correctness_changed": 0,
        "generation_reference_opportunity": 0.5,
        "total_row_removal_fraction": 0.25,
    }


def test_compute_summary_bootstraps_examples_and_requires_exact_output() -> None:
    rows = [
        _row(example, repeat, gain)
        for example, gain in (("a", 0.10), ("b", 0.20))
        for repeat in range(3)
    ]
    summary = _summaries(rows, seed=17, bootstrap_samples=1000)[0]

    assert summary["examples"] == 2
    assert summary["timed_pairs"] == 6
    assert summary["mean_runtime_gain"] == 0.15
    assert summary["example_bootstrap_gain_ci95_low"] > 0.0
    assert summary["mean_total_row_removal_fraction"] == 0.25
    assert summary["profitability_pass"] == 1

    rows[0]["a3_exact_a2"] = 0
    assert _summaries(rows, seed=17, bootstrap_samples=1000)[0]["profitability_pass"] == 0


def test_plan_hash_is_order_stable_but_content_sensitive() -> None:
    left = {1: [{"position": 3, "token_id": 7}], 2: []}
    reordered = {2: [], 1: [{"token_id": 7, "position": 3}]}
    changed = {1: [{"position": 3, "token_id": 8}], 2: []}

    assert _plan_hash(left) == _plan_hash(reordered)
    assert _plan_hash(left) != _plan_hash(changed)


def test_runtime_environment_resolves_device_metadata(monkeypatch) -> None:
    import benchmark_phase_lock_compute as benchmark

    class Properties:
        total_memory = 32_000_000_000

    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda device: Properties())
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda device: "test-gpu")
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda device: (8, 9))

    record = _runtime_environment(benchmark.torch.device("cuda:0"))

    assert record["device_name"] == "test-gpu"
    assert record["device_capability"] == [8, 9]
    assert record["device_total_memory_bytes"] == 32_000_000_000


def test_aggregate_validation_requires_fixed_plans_and_exact_repeats() -> None:
    rows = []
    for repeat in range(2):
        rows.append(
            {
                **_row("a", repeat, 0.01),
                "semantic_plan_sha256": "semantic",
                "reference_plan_sha256": "reference",
                "a2_exact_a1": 1,
            }
        )
    config = {
        "timing_repeats": 2,
        "runtime_environment": {"device_name": "test-gpu"},
    }

    validation = _validate(rows, [config])

    assert validation["timing_rows"] == 2
    assert validation["all_a2_exact_a1"]
    assert validation["all_a3_exact_a2"]
    assert validation["fixed_plan_hashes_within_group"]
