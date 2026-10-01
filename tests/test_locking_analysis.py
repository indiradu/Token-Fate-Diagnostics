from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from regret_remasking.locking_analysis import analyze_run


def _write_run(
    run_dir: Path,
    *,
    examples: list[dict[str, str]] | None = None,
    policies: list[dict[str, object]] | None = None,
    repeats: int = 2,
    records: list[dict[str, object]] | None = None,
) -> None:
    examples = examples or [{"example_id": "a", "dataset": "math"}]
    policies = policies or [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    config = {
        "examples": examples,
        "policies": policies,
        "repeats": repeats,
        "gen_length": 5,
    }
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = records if records is not None else _records(examples, policies, repeats)
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _records(
    examples: list[dict[str, str]],
    policies: list[dict[str, object]],
    repeats: int,
) -> list[dict[str, object]]:
    return [
        _record(example, policy, repeat)
        for example in examples
        for policy in policies
        for repeat in range(1, repeats + 1)
    ]


def _record(
    example: dict[str, str],
    policy: dict[str, object],
    repeat: int,
    **overrides: object,
) -> dict[str, object]:
    policy_id = str(policy["policy_id"])
    base = {
        "example_id": example["example_id"],
        "dataset": example["dataset"],
        "policy_id": policy_id,
        "arm": policy["arm"],
        "epsilon": policy.get("epsilon"),
        "threshold": policy.get("threshold"),
        "repeat": repeat,
        "nfe": 4 if policy_id == "B0" else 2,
        "forward_calls": 4 if policy_id == "B0" else 2,
        "latency_s": 10.0 if policy_id == "B0" else 5.0,
        "token_ids": [1, 2, 3, 4, 5] if policy_id == "B0" else [1, 2, 9, 4, 5],
        "generation": "42" if policy_id == "B0" else "43",
        "correct": policy_id == "B0",
        "scoring_status": "passed",
        "final_mask_count": 0,
        "immutable_token_violations": 0,
        "prompt_length": 7,
        "gen_length": 5,
        "total_query_rows": 20,
        "generated_row_updates_avoided": 0 if policy_id == "B0" else 8,
        "frozen_count": 0 if policy_id == "B0" else 3,
        "gate_time_s": 0.01,
        "scorer_time_s": 0.02,
        "status": "ok",
    }
    base.update(overrides)
    return base


def _metric(output_dir: Path, dataset: str, candidate: str, metric: str) -> float:
    with (output_dir / "metrics.csv").open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            if (
                row["dataset"] == dataset
                and row["candidate_policy_id"] == candidate
                and row["metric"] == metric
            ):
                return float(row["point"])
    raise AssertionError(f"missing metric {dataset}/{candidate}/{metric}")


def test_analyze_run_uses_repeat_one_for_quality_and_repeated_medians_for_latency(tmp_path):
    examples = [{"example_id": "same", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    records = _records(examples, policies, repeats=2)
    for row in records:
        if row["policy_id"] == "B0" and row["repeat"] == 1:
            row["latency_s"] = 10.0
            row["generation"] = "2 + 2"
        if row["policy_id"] == "B0" and row["repeat"] == 2:
            row["latency_s"] = 14.0
            row["generation"] = "changed repeat"
        if row["policy_id"] == "P1" and row["repeat"] == 1:
            row["latency_s"] = 5.0
            row["generation"] = "4"
            row["correct"] = False
        if row["policy_id"] == "P1" and row["repeat"] == 2:
            row["latency_s"] = 7.0
            row["correct"] = True
    _write_run(tmp_path / "run", examples=examples, policies=policies, records=records)

    summary = analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=25, seed=3)

    assert summary["bootstrap_unit"].startswith("dataset/example_id")
    assert _metric(tmp_path / "out", "math", "P1", "baseline_accuracy") == 1.0
    assert _metric(tmp_path / "out", "math", "P1", "candidate_accuracy") == 0.0
    assert _metric(tmp_path / "out", "math", "P1", "wall_clock_speedup") == 2.0
    assert _metric(tmp_path / "out", "math", "P1", "wall_clock_speedup_percent") == 100.0
    assert _metric(tmp_path / "out", "math", "P1", "answer_value_change_rate") == 0.0
    assert _metric(tmp_path / "out", "math", "P1", "nondeterministic_example_rate") == 1.0


def test_dataset_is_part_of_the_pair_key_when_example_ids_are_shared(tmp_path):
    examples = [
        {"example_id": "shared", "dataset": "math"},
        {"example_id": "shared", "dataset": "humaneval"},
    ]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        if row["dataset"] == "humaneval" and row["policy_id"] == "P1":
            row["generation"] = "def f():\n    return 1"
        if row["dataset"] == "humaneval" and row["policy_id"] == "B0":
            row["generation"] = "def f():\n    return 0"
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=10, seed=3)

    assert _metric(tmp_path / "out", "math", "P1", "quality_denominator") == 1.0
    assert _metric(tmp_path / "out", "humaneval", "P1", "quality_denominator") == 1.0


def test_infrastructure_failures_are_excluded_but_timeouts_count_incorrect(tmp_path):
    examples = [{"example_id": "infra", "dataset": "math"}, {"example_id": "timeout", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        if row["example_id"] == "infra" and row["policy_id"] == "B0":
            row["scoring_status"] = "sandbox_error"
        if row["example_id"] == "infra" and row["policy_id"] == "P1":
            row["correct"] = True
        if row["example_id"] == "timeout" and row["policy_id"] == "B0":
            row["correct"] = True
        if row["example_id"] == "timeout" and row["policy_id"] == "P1":
            row["scoring_status"] = "timeout"
            row["correct"] = False
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=10, seed=3)

    assert _metric(tmp_path / "out", "math", "P1", "quality_denominator") == 1.0
    assert _metric(tmp_path / "out", "math", "P1", "excluded_infrastructure_pairs") == 1.0
    assert _metric(tmp_path / "out", "math", "P1", "candidate_accuracy") == 0.0


def test_generated_coverage_uses_fixed_generated_budget_not_query_rows(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}, {"example_id": "b", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "R-dense", "arm": "R-dense", "epsilon": 0.01, "threshold": None},
        {"policy_id": "R-sparse", "arm": "R-sparse", "epsilon": 0.01, "threshold": None},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        if row["policy_id"] == "B0":
            row["forward_calls"] = 5
            row["nfe"] = 5
            row["total_query_rows"] = 75 if row["example_id"] == "a" else 175
            row["prompt_length"] = 10 if row["example_id"] == "a" else 30
        elif row["policy_id"] == "R-dense":
            row["generated_row_updates_avoided"] = 10
            row["total_query_rows"] = 50 if row["example_id"] == "a" else 150
            row["frozen_count"] = 2
            row["prompt_length"] = 10 if row["example_id"] == "a" else 30
        elif row["policy_id"] == "R-sparse":
            row["generated_row_updates_avoided"] = 10
            row["total_query_rows"] = 25 if row["example_id"] == "a" else 75
            row["frozen_count"] = 2
            row["prompt_length"] = 10 if row["example_id"] == "a" else 30
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)

    dense_generated = _metric(
        tmp_path / "out", "math", "R-dense", "generated_row_updates_avoided_fraction"
    )
    sparse_generated = _metric(
        tmp_path / "out", "math", "R-sparse", "generated_row_updates_avoided_fraction"
    )
    dense_query = _metric(
        tmp_path / "out", "math", "R-dense", "query_row_updates_avoided_fraction"
    )
    sparse_query = _metric(
        tmp_path / "out", "math", "R-sparse", "query_row_updates_avoided_fraction"
    )
    assert dense_generated == sparse_generated == 20 / 50
    assert dense_query == 50 / 250
    assert sparse_query == 150 / 250


def test_full_attention_sparse_reports_no_query_savings_but_projection_and_ffn_savings(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}, {"example_id": "b", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "R-dense", "arm": "R-dense", "epsilon": 0.01, "threshold": None},
        {"policy_id": "R-sparse", "arm": "R-sparse", "epsilon": 0.01, "threshold": None},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        row["nfe"] = 5
        row["forward_calls"] = 5
        row["total_query_rows"] = 250
        row["total_projection_rows"] = 250
        row["total_ffn_rows"] = 250
        if row["policy_id"] in {"R-dense", "R-sparse"}:
            row["generated_row_updates_avoided"] = 10
            row["frozen_count"] = 2
        if row["policy_id"] == "R-sparse":
            row["total_projection_rows"] = 150
            row["total_ffn_rows"] = 125
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)

    assert _metric(tmp_path / "out", "math", "R-dense", "query_row_updates_avoided_fraction") == 0.0
    assert _metric(tmp_path / "out", "math", "R-sparse", "query_row_updates_avoided_fraction") == 0.0
    assert _metric(tmp_path / "out", "math", "R-dense", "projection_row_updates_avoided_fraction") == 0.0
    assert _metric(tmp_path / "out", "math", "R-dense", "ffn_row_updates_avoided_fraction") == 0.0
    assert _metric(tmp_path / "out", "math", "R-sparse", "projection_row_updates_avoided_fraction") == 200 / 500
    assert _metric(tmp_path / "out", "math", "R-sparse", "ffn_row_updates_avoided_fraction") == 250 / 500
    assert _metric(tmp_path / "out", "math", "R-sparse", "generated_row_updates_avoided_fraction") == 20 / 50


def test_query_savings_use_baseline_denominator_for_fewer_forward_candidates(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "T-adaptive", "arm": "T-adaptive", "epsilon": None, "threshold": 0.95},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        if row["policy_id"] == "B0":
            row["nfe"] = 5
            row["forward_calls"] = 5
            row["total_query_rows"] = 250
            row["total_projection_rows"] = 250
            row["total_ffn_rows"] = 250
        else:
            row["nfe"] = 2
            row["forward_calls"] = 2
            row["total_query_rows"] = 100
            row["total_projection_rows"] = 100
            row["total_ffn_rows"] = 100
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)

    assert _metric(tmp_path / "out", "math", "T-adaptive", "query_row_updates_avoided_fraction") == 150 / 250
    assert _metric(tmp_path / "out", "math", "T-adaptive", "projection_row_updates_avoided_fraction") == 150 / 250
    assert _metric(tmp_path / "out", "math", "T-adaptive", "ffn_row_updates_avoided_fraction") == 150 / 250


def test_historical_records_fall_back_to_query_rows_for_projection_and_ffn_semantics(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "legacy", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    records = _records(examples, policies, repeats=1)
    for row in records:
        row.pop("total_projection_rows", None)
        row.pop("total_ffn_rows", None)
        if row["policy_id"] == "B0":
            row["total_query_rows"] = 250
        else:
            row["total_query_rows"] = 100
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1, records=records)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)

    assert _metric(tmp_path / "out", "math", "legacy", "query_row_updates_avoided_fraction") == 150 / 250
    assert _metric(tmp_path / "out", "math", "legacy", "projection_row_updates_avoided_fraction") == 150 / 250
    assert _metric(tmp_path / "out", "math", "legacy", "ffn_row_updates_avoided_fraction") == 150 / 250


def test_missing_repeat_or_duplicate_record_rejects_incomplete_runs(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    missing = _records(examples, policies, repeats=2)[:-1]
    _write_run(tmp_path / "missing", examples=examples, policies=policies, records=missing)
    with pytest.raises(ValueError, match="missing record"):
        analyze_run(tmp_path / "missing", tmp_path / "missing_out", bootstrap_reps=1)

    duplicate = _records(examples, policies, repeats=1)
    duplicate.append(dict(duplicate[-1]))
    _write_run(
        tmp_path / "duplicate",
        examples=examples,
        policies=policies,
        repeats=1,
        records=duplicate,
    )
    with pytest.raises(ValueError, match="duplicate record"):
        analyze_run(tmp_path / "duplicate", tmp_path / "duplicate_out", bootstrap_reps=1)


def test_decode_errors_and_invalid_success_rows_block_completeness(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "P1", "arm": "candidate", "epsilon": 0.01, "threshold": 0.5},
    ]
    failed = _records(examples, policies, repeats=1)
    failed[-1] = {
        "example_id": "a",
        "dataset": "math",
        "policy_id": "P1",
        "repeat": 1,
        "status": "error",
        "error": "decode failed",
    }
    _write_run(tmp_path / "failed", examples=examples, policies=policies, repeats=1, records=failed)
    with pytest.raises(ValueError, match="not a complete decode"):
        analyze_run(tmp_path / "failed", tmp_path / "failed_out", bootstrap_reps=1)

    masked = _records(examples, policies, repeats=1)
    masked[-1]["final_mask_count"] = 1
    _write_run(tmp_path / "masked", examples=examples, policies=policies, repeats=1, records=masked)
    with pytest.raises(ValueError, match="final masked tokens"):
        analyze_run(tmp_path / "masked", tmp_path / "masked_out", bootstrap_reps=1)


def test_policy_family_pairs_are_added_when_metadata_matches(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {"policy_id": "R-dense-e1", "arm": "R-dense", "epsilon": 0.01, "threshold": None},
        {"policy_id": "R-sparse-e1", "arm": "R-sparse", "epsilon": 0.01, "threshold": None},
        {"policy_id": "T-adaptive-5", "arm": "T-adaptive", "threshold": 0.5, "epsilon": None},
        {"policy_id": "C-adaptive-5", "arm": "C-adaptive", "threshold": 0.5, "epsilon": None},
    ]
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    kinds = {comparison["comparison_kind"] for comparison in summary["comparisons"]}

    assert "R_dense_vs_R_sparse" in kinds
    assert "T_adaptive_vs_C_adaptive" in kinds


def test_trace_pairs_match_by_operating_point_despite_distinct_checkpoints(tmp_path):
    examples = [{"example_id": "a", "dataset": "math"}]
    policies = [
        {"policy_id": "B0", "arm": "baseline", "epsilon": None, "threshold": None},
        {
            "policy_id": "T-adaptive-op1",
            "arm": "T-adaptive",
            "threshold": 0.9,
            "epsilon": None,
            "checkpoint": "/tmp/trace.ckpt",
            "checkpoint_sha256": "trace-hash",
            "operating_point": "trace-op-1",
        },
        {
            "policy_id": "C-adaptive-op1",
            "arm": "C-adaptive",
            "threshold": 0.8,
            "epsilon": None,
            "checkpoint": "/tmp/conf.ckpt",
            "checkpoint_sha256": "confidence-hash",
            "operating_point": "trace-op-1",
        },
    ]
    _write_run(tmp_path / "run", examples=examples, policies=policies, repeats=1)

    analyze_run(tmp_path / "run", tmp_path / "out", bootstrap_reps=5, seed=3)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    pair_ids = {
        (comparison["baseline_policy_id"], comparison["candidate_policy_id"])
        for comparison in summary["comparisons"]
        if comparison["comparison_kind"] == "T_adaptive_vs_C_adaptive"
    }

    assert ("T-adaptive-op1", "C-adaptive-op1") in pair_ids
