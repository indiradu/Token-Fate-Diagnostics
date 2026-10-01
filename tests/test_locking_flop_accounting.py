from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from locking_flop_accounting import (  # noqa: E402
    ControllerConfig,
    ModelConfig,
    ProjectorConfig,
    analyze_records,
    backbone_flops,
    controller_flops,
    load_model_config,
)


def _record(**overrides):
    row = {
        "dataset": "toy",
        "example_id": "ex1",
        "policy_id": "B0",
        "repeat": 1,
        "status": "ok",
        "correct": True,
        "latency_s": 10.0,
        "prompt_length": 3,
        "gen_length": 2,
        "row_count_basis": "per_layer_equivalent; output_head_count_is_actual",
        "total_query_rows": 7,
        "total_projection_rows": 11,
        "total_ffn_rows": 13,
        "total_output_head_rows": 17,
    }
    row.update(overrides)
    return row


def _assert_raises_contains(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_backbone_flops_counts_attention_gqa_and_head():
    cfg = ModelConfig(
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=8,
        vocab_size=10,
    )
    work = backbone_flops(_record(), cfg)

    assert cfg.kv_hidden_size == 2
    assert work["sequence_length"] == 5
    assert work["attention_flops"] == 4 * 2 * 4 * 5 * 7
    assert work["projection_flops"] == 2 * (4 * 4 * 4 + 4 * 4 * 2) * 11
    assert work["ffn_flops"] == 6 * 2 * 4 * 8 * 13
    assert work["lm_head_flops"] == 2 * 4 * 10 * 17
    assert work["backbone_flops"] == (
        work["attention_flops"] + work["projection_flops"] + work["ffn_flops"]
    )
    assert work["total_with_lm_head_flops"] == work["backbone_flops"] + work["lm_head_flops"]


def test_backbone_flops_rejects_failed_or_unsupported_rows():
    cfg = ModelConfig(4, 1, 2, 2, 8, 10)
    try:
        backbone_flops(_record(status="failed"), cfg)
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("failed rows must be rejected")

    try:
        backbone_flops(_record(row_count_basis="actual"), cfg)
    except ValueError as exc:
        assert "row_count_basis" in str(exc)
    else:
        raise AssertionError("unsupported row basis must be rejected")


def test_controller_flops_includes_threshold_head_and_quadratic_attention():
    projector = ProjectorConfig(hidden_size=5, state_hidden=7, state_dim=3, delta_hidden=11, delta_dim=2)
    controller = ControllerConfig(model_dim=13, num_layers=2, ffn_dim=17, moving_window=4, input_feature_dim=19)

    work = controller_flops([2, 4], projector, controller)

    per_prefix_projector = 3 * (2 * 5 * 7 + 2 * 7 * 3) + 2 * (2 * 5 * 11 + 2 * 11 * 2)
    expected_projector = per_prefix_projector * 6
    expected_input = 2 * 2 * 19 * 13 + 2 * 4 * 19 * 13
    expected_transformer = 0
    expected_heads = 0
    for m in [2, 4]:
        s = m + 1
        expected_transformer += 2 * (8 * s * 13 * 13 + 4 * s * s * 13 + 4 * s * 13 * 17)
        expected_heads += 2 * 13 * (m + 1)

    assert work["controller_calls"] == 2
    assert work["controller_prefix_tokens"] == 6
    assert work["projector_flops"] == expected_projector
    assert work["controller_input_flops"] == expected_input
    assert work["controller_transformer_flops"] == expected_transformer
    assert work["controller_token_threshold_head_flops"] == expected_heads
    assert work["controller_total_flops"] == expected_projector + expected_input + expected_transformer + expected_heads


def test_controller_flops_rejects_fractional_prefix_lengths():
    projector = ProjectorConfig(hidden_size=5, state_hidden=7, state_dim=3, delta_hidden=11, delta_dim=2)
    controller = ControllerConfig(model_dim=13, num_layers=2, ffn_dim=17)
    try:
        controller_flops([2.5], projector, controller)  # type: ignore[list-item]
    except ValueError as exc:
        assert "prefix length" in str(exc)
    else:
        raise AssertionError("fractional prefix lengths must be rejected")


def test_load_model_config_accepts_llada_names(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "d_model": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 32,
                "mlp_hidden_size": 12288,
                "vocab_size": 126464,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_model_config(path)
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288


def test_analyze_records_uses_repeat1_flops_and_median_latency(tmp_path):
    records = [
        _record(policy_id="B0", repeat=1, latency_s=10.0, correct=True),
        _record(policy_id="B0", repeat=2, latency_s=12.0, correct=True),
        _record(policy_id="B0", repeat=3, latency_s=14.0, correct=True),
        _record(policy_id="P", repeat=1, latency_s=5.0, correct=False, total_query_rows=3),
        _record(policy_id="P", repeat=2, latency_s=6.0, correct=False, total_query_rows=3),
        _record(policy_id="P", repeat=3, latency_s=7.0, correct=False, total_query_rows=3),
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    summary = analyze_records(
        records_path,
        ModelConfig(4, 2, 4, 2, 8, 10),
        tmp_path / "out",
        policy_ids=["B0", "P"],
        expected_cases=1,
    )

    assert summary["controller_profile_status"] == "not_provided"
    with (tmp_path / "out" / "aggregate_flops.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overall_p = next(row for row in rows if row["dataset"] == "overall" and row["policy_id"] == "P")
    assert float(overall_p["speedup_pct"]) == 100.0
    assert float(overall_p["accuracy_pct"]) == 0.0
    assert float(overall_p["backbone_reduction_pct"]) > 0.0
    with (tmp_path / "out" / "percase_flops.csv").open(newline="", encoding="utf-8") as f:
        percase = list(csv.DictReader(f))
    assert next(row for row in percase if row["policy_id"] == "P")["controller_total_tflops"] == "0.0"


def test_analyze_records_marks_profiles_without_shapes_unknown(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(
        json.dumps({"dataset": "toy", "example_id": "ex1", "policy_id": "B0", "controller_prefix_lengths": [3]})
        + "\n",
        encoding="utf-8",
    )
    summary = analyze_records(
        records_path,
        ModelConfig(4, 2, 4, 2, 8, 10),
        tmp_path / "out",
        profiles_path=profiles,
        expected_cases=1,
    )
    assert summary["controller_profile_status"] == "profiles_provided_without_shapes"


def test_analyze_records_rejects_duplicate_repeats(tmp_path):
    records_path = tmp_path / "records.jsonl"
    rows = [_record(repeat=1, latency_s=1.0), _record(repeat=1, latency_s=2.0)]
    records_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    _assert_raises_contains(
        lambda: analyze_records(records_path, ModelConfig(4, 2, 4, 2, 8, 10), tmp_path / "out"),
        "duplicate repeat",
    )


def test_analyze_records_rejects_repeat_work_drift(tmp_path):
    records_path = tmp_path / "records.jsonl"
    rows = [_record(repeat=1), _record(repeat=2, total_query_rows=8)]
    records_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    _assert_raises_contains(
        lambda: analyze_records(records_path, ModelConfig(4, 2, 4, 2, 8, 10), tmp_path / "out"),
        "work counters vary",
    )


def test_analyze_records_rejects_non_ok_and_missing_correctness(tmp_path):
    bad_status = tmp_path / "bad_status.jsonl"
    bad_status.write_text(json.dumps(_record(status="failed")) + "\n", encoding="utf-8")
    _assert_raises_contains(
        lambda: analyze_records(bad_status, ModelConfig(4, 2, 4, 2, 8, 10), tmp_path / "out1"),
        "non-ok status",
    )

    missing_correct = tmp_path / "missing_correct.jsonl"
    row = _record()
    row["correct"] = None
    missing_correct.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _assert_raises_contains(
        lambda: analyze_records(missing_correct, ModelConfig(4, 2, 4, 2, 8, 10), tmp_path / "out2"),
        "correctness must be boolean",
    )


def test_analyze_records_requires_baseline_and_complete_pairs(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_record(policy_id="P")) + "\n", encoding="utf-8")
    _assert_raises_contains(
        lambda: analyze_records(
            records_path,
            ModelConfig(4, 2, 4, 2, 8, 10),
            tmp_path / "out1",
            baseline_id="B0",
            policy_ids=["P"],
        ),
        "baseline_id",
    )

    incomplete_path = tmp_path / "incomplete.jsonl"
    incomplete_rows = [_record(policy_id="B0"), _record(example_id="ex2", policy_id="P")]
    incomplete_path.write_text("\n".join(json.dumps(row) for row in incomplete_rows) + "\n", encoding="utf-8")
    _assert_raises_contains(
        lambda: analyze_records(
            incomplete_path,
            ModelConfig(4, 2, 4, 2, 8, 10),
            tmp_path / "out2",
            baseline_id="B0",
            policy_ids=["B0", "P"],
        ),
        "no complete paired cases",
    )


def test_analyze_records_rejects_duplicate_profiles(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_record(controller_calls=1)) + "\n", encoding="utf-8")
    profiles = tmp_path / "profiles.jsonl"
    profile = {"dataset": "toy", "example_id": "ex1", "policy_id": "B0", "controller_prefix_lengths": [3]}
    profiles.write_text(json.dumps(profile) + "\n" + json.dumps(profile) + "\n", encoding="utf-8")

    _assert_raises_contains(
        lambda: analyze_records(
            records_path,
            ModelConfig(4, 2, 4, 2, 8, 10),
            tmp_path / "out",
            profiles_path=profiles,
            projector_config=ProjectorConfig(4, 3, 2, 3, 2),
            controller_config=ControllerConfig(5, 1, 7),
        ),
        "duplicate controller profile",
    )


def test_controller_profiles_must_cover_all_learned_cases_for_aggregate_total(tmp_path):
    records = [
        _record(policy_id="B0", controller_calls=0),
        _record(policy_id="Learned", controller_calls=2, correct=False),
        _record(dataset="toy", example_id="ex2", policy_id="B0", controller_calls=0),
        _record(dataset="toy", example_id="ex2", policy_id="Learned", controller_calls=2, correct=True),
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(
        json.dumps(
            {
                "dataset": "toy",
                "example_id": "ex1",
                "policy_id": "Learned",
                "controller_prefix_lengths": [3, 4],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    analyze_records(
        records_path,
        ModelConfig(4, 2, 4, 2, 8, 10),
        tmp_path / "out",
        policy_ids=["B0", "Learned"],
        profiles_path=profiles,
        projector_config=ProjectorConfig(4, 3, 2, 3, 2),
        controller_config=ControllerConfig(5, 1, 7),
    )

    with (tmp_path / "out" / "aggregate_flops.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    learned = next(row for row in rows if row["dataset"] == "overall" and row["policy_id"] == "Learned")
    assert learned["controller_profile_expected_cases"] == "2"
    assert learned["controller_profile_covered_cases"] == "1"
    assert learned["controller_profile_pending_cases"] == "1"
    assert learned["controller_total_tflops_per_example"] == ""

    baseline = next(row for row in rows if row["dataset"] == "overall" and row["policy_id"] == "B0")
    assert baseline["controller_profile_expected_cases"] == "0"
    assert baseline["controller_total_tflops_per_example"] == "0.0"


def test_controller_profile_length_mismatch_is_pending_not_partial_average(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_record(controller_calls=2)) + "\n", encoding="utf-8")
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(
        json.dumps({"dataset": "toy", "example_id": "ex1", "policy_id": "B0", "controller_prefix_lengths": [3]})
        + "\n",
        encoding="utf-8",
    )

    analyze_records(
        records_path,
        ModelConfig(4, 2, 4, 2, 8, 10),
        tmp_path / "out",
        profiles_path=profiles,
        projector_config=ProjectorConfig(4, 3, 2, 3, 2),
        controller_config=ControllerConfig(5, 1, 7),
    )
    with (tmp_path / "out" / "aggregate_flops.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["controller_profile_pending_cases"] == "1"
    assert row["controller_total_tflops_per_example"] == ""


if __name__ == "__main__":
    test_backbone_flops_counts_attention_gqa_and_head()
    test_backbone_flops_rejects_failed_or_unsupported_rows()
    test_controller_flops_includes_threshold_head_and_quadratic_attention()
    print("ok")
