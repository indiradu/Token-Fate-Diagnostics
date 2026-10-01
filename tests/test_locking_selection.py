from __future__ import annotations

import json
from pathlib import Path

import pytest

from regret_remasking.locking_analysis import analyze_run
from regret_remasking.locking_manifest import write_frozen_jsonl
from regret_remasking.locking_selection import select_operating_points


def _manifest_record(example_id: str, split: str = "selection") -> dict[str, object]:
    return {
        "record_id": f"{split}:math:{example_id}",
        "source_id": f"math:{example_id}",
        "dataset": "math",
        "example_id": example_id,
        "question": f"{example_id}?",
        "gold_answer": "1",
        "metadata": {},
        "split": split,
        "seed": 23,
        "source_path": "/tmp/source.jsonl",
        "source_revision": "source-rev",
        "content_hash": _content_hash(f"{example_id}?", "math"),
    }


def _content_hash(question: str, dataset: str) -> str:
    from regret_remasking.locking_manifest import prompt_content_hash

    return prompt_content_hash(question, dataset)


def _policy(policy_id: str, arm: str, *, epsilon=None, threshold=None) -> dict[str, object]:
    policy = {
        "policy_id": policy_id,
        "arm": arm,
        "epsilon": epsilon,
        "threshold": threshold,
    }
    if arm in {"T-fixed", "T-adaptive", "C-adaptive"}:
        policy["checkpoint"] = f"/tmp/{arm}.ckpt"
        policy["checkpoint_sha256"] = f"{arm}-hash"
    return policy


def _write_run(
    run_dir: Path,
    *,
    policies: list[dict[str, object]],
    work_by_policy: dict[str, float] | None = None,
    nfe_by_policy: dict[str, int] | None = None,
    manifest_split: str = "selection",
    source_sha256: str = "source-hash",
    examples: list[str] | None = None,
) -> None:
    examples = examples or ["a", "b"]
    manifest_path = run_dir / "selection.jsonl"
    manifest_records = [_manifest_record(example_id, manifest_split) for example_id in examples]
    manifest_hash = write_frozen_jsonl(manifest_path, manifest_records)
    config = {
        "schema_version": 1,
        "examples": [{"example_id": example_id, "dataset": "math"} for example_id in examples],
        "manifest_sha256": {str(manifest_path): manifest_hash},
        "policies": policies,
        "repeats": 1,
        "gen_length": 16,
        "block_length": 8,
        "checkpoint": "/tmp/model",
        "checkpoint_index_sha256": "model-index-hash",
        "source": {"sha256": source_sha256},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = []
    for example_id in examples:
        for policy in policies:
            policy_id = str(policy["policy_id"])
            nfe = int((nfe_by_policy or {}).get(policy_id, 12))
            rows.append(
                {
                    "example_id": example_id,
                    "dataset": "math",
                    "policy_id": policy_id,
                    "arm": policy["arm"],
                    "epsilon": policy.get("epsilon"),
                    "threshold": policy.get("threshold"),
                    "repeat": 1,
                    "nfe": nfe,
                    "forward_calls": nfe,
                    "latency_s": 1.0,
                    "token_ids": list(range(16)),
                    "generation": "1",
                    "correct": policy_id == "B0",
                    "scoring_status": "passed",
                    "final_mask_count": 0,
                    "immutable_token_violations": 0,
                    "prompt_length": 3,
                    "gen_length": 16,
                    "total_query_rows": 256,
                    "generated_row_updates_avoided": int(
                        round((work_by_policy or {}).get(policy_id, 0.0) * 256)
                    ),
                    "frozen_count": 0,
                    "gate_time_s": 0.0,
                    "scorer_time_s": 0.0,
                    "status": "ok",
                }
            )
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_selects_ties_dedups_and_pairs_operating_points(tmp_path):
    policies = [
        _policy("B0", "B0"),
        _policy("R0", "R-dense", epsilon=0.0),
        _policy("R01", "R-dense", epsilon=0.1),
        _policy("R02", "R-dense", epsilon=0.2),
        _policy("R03", "R-dense", epsilon=0.3),
        _policy("T-fixed", "T-fixed"),
        _policy("T90", "T-adaptive", threshold=0.90),
        _policy("T95", "T-adaptive", threshold=0.95),
        _policy("T80", "T-adaptive", threshold=0.80),
        _policy("C91", "C-adaptive", threshold=0.91),
        _policy("C70", "C-adaptive", threshold=0.70),
    ]
    _write_run(
        tmp_path / "run",
        policies=policies,
        work_by_policy={"R0": 0.0, "R01": 0.20, "R02": 0.30, "R03": 0.52},
        nfe_by_policy={"T90": 10, "T95": 10, "T80": 6, "C91": 10, "C70": 4},
    )

    summary = select_operating_points([tmp_path / "run"], tmp_path / "selected", nfe_targets=[10, 6])
    selected = json.loads((tmp_path / "selected" / "policies.json").read_text(encoding="utf-8"))
    ids = [policy["policy_id"] for policy in selected]

    assert "B0" in ids
    assert "R0" in ids
    assert "R01" in ids
    assert "R03" in ids
    assert "R02" not in ids
    assert "R-sparse-eps-0.1" in ids
    assert "R-sparse-eps-0.3" in ids
    trace_pairs = summary["trace_lock"]["targets"]
    assert trace_pairs[0]["t_policy_id"] == "T95"
    assert trace_pairs[0]["matched_within_5pct"] is True
    assert trace_pairs[1]["t_policy_id"] == "T80"
    selected_trace = [
        policy
        for policy in selected
        if policy["policy_id"] == "T95" or policy.get("source_policy_id") == "C91"
    ]
    assert {policy["operating_point"] for policy in selected_trace} == {"trace-op-1"}


def test_unmatched_confidence_pair_is_still_chosen_and_marked(tmp_path):
    policies = [
        _policy("B0", "B0"),
        _policy("R0", "R-dense", epsilon=0.0),
        _policy("R1", "R-dense", epsilon=0.1),
        _policy("T", "T-adaptive", threshold=0.9),
        _policy("C", "C-adaptive", threshold=0.9),
    ]
    _write_run(
        tmp_path / "run",
        policies=policies,
        work_by_policy={"R0": 0.0, "R1": 0.25},
        nfe_by_policy={"T": 10, "C": 15},
    )

    summary = select_operating_points([tmp_path / "run"], tmp_path / "out", nfe_targets=[10])

    target = summary["trace_lock"]["targets"][0]
    assert target["c_source_policy_id"] == "C"
    assert target["c_policy_id"] == "C-trace-op-1"
    assert target["matched_within_5pct"] is False


def test_reused_confidence_control_gets_unique_aliases_and_pairs_in_analyzer(tmp_path):
    policies = [
        _policy("B0", "B0"),
        _policy("R0", "R-dense", epsilon=0.0),
        _policy("R1", "R-dense", epsilon=0.1),
        _policy("T-high", "T-adaptive", threshold=0.9),
        _policy("T-low", "T-adaptive", threshold=0.8),
        _policy("C-shared", "C-adaptive", threshold=0.85),
    ]
    _write_run(
        tmp_path / "run",
        policies=policies,
        work_by_policy={"R0": 0.0, "R1": 0.25},
        nfe_by_policy={"T-high": 12, "T-low": 8, "C-shared": 10},
    )

    summary = select_operating_points(
        [tmp_path / "run"], tmp_path / "selected", nfe_targets=[12, 8]
    )
    selected = json.loads((tmp_path / "selected" / "policies.json").read_text(encoding="utf-8"))
    selected_ids = [policy["policy_id"] for policy in selected]

    assert len(selected_ids) == len(set(selected_ids))
    assert "C-shared-trace-op-1" in selected_ids
    assert "C-shared-trace-op-2" in selected_ids
    assert summary["trace_lock"]["targets"][0]["c_source_policy_id"] == "C-shared"
    assert summary["trace_lock"]["targets"][1]["c_source_policy_id"] == "C-shared"
    aliased_controls = [
        policy for policy in selected if policy.get("source_policy_id") == "C-shared"
    ]
    assert {policy["threshold"] for policy in aliased_controls} == {0.85}
    assert {policy["checkpoint"] for policy in aliased_controls} == {"/tmp/C-adaptive.ckpt"}

    analysis_policies = [
        policy
        for policy in selected
        if policy["policy_id"] in {"B0", "T-high", "T-low", "C-shared-trace-op-1", "C-shared-trace-op-2"}
    ]
    _write_run(tmp_path / "analysis_run", policies=analysis_policies)
    analyze_run(tmp_path / "analysis_run", tmp_path / "analysis_out", bootstrap_reps=5, seed=3)
    analysis_summary = json.loads(
        (tmp_path / "analysis_out" / "summary.json").read_text(encoding="utf-8")
    )
    pairs = {
        (comparison["baseline_policy_id"], comparison["candidate_policy_id"])
        for comparison in analysis_summary["comparisons"]
        if comparison["comparison_kind"] == "T_adaptive_vs_C_adaptive"
    }

    assert ("T-high", "C-shared-trace-op-1") in pairs
    assert ("T-low", "C-shared-trace-op-2") in pairs


def test_rejects_eval_manifest_hidden_limit_and_source_drift(tmp_path):
    policies = [_policy("B0", "B0"), _policy("R0", "R-dense", epsilon=0.0)]
    _write_run(tmp_path / "eval", policies=policies, manifest_split="eval")
    with pytest.raises(ValueError, match="expected 'selection'"):
        select_operating_points([tmp_path / "eval"], tmp_path / "eval_out", nfe_targets=[10])

    _write_run(tmp_path / "full", policies=policies, examples=["a", "b"])
    config_path = tmp_path / "full" / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["examples"] = config["examples"][:1]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        select_operating_points([tmp_path / "full"], tmp_path / "limit_out", nfe_targets=[10])

    _write_run(tmp_path / "run_a", policies=policies, source_sha256="a")
    _write_run(tmp_path / "run_b", policies=policies, source_sha256="b")
    with pytest.raises(ValueError, match="source provenance"):
        select_operating_points([tmp_path / "run_a", tmp_path / "run_b"], tmp_path / "drift_out")


def test_rejects_missing_dense_or_trace_grid_and_nonidentical_overwrite(tmp_path):
    no_dense = [_policy("B0", "B0"), _policy("T", "T-adaptive", threshold=0.9), _policy("C", "C-adaptive", threshold=0.9)]
    _write_run(tmp_path / "no_dense", policies=no_dense, nfe_by_policy={"T": 8, "C": 8})
    with pytest.raises(ValueError, match="R-dense"):
        select_operating_points([tmp_path / "no_dense"], tmp_path / "no_dense_out")

    no_trace = [_policy("B0", "B0"), _policy("R0", "R-dense", epsilon=0.0)]
    _write_run(tmp_path / "no_trace", policies=no_trace)
    with pytest.raises(ValueError, match="T-adaptive"):
        select_operating_points([tmp_path / "no_trace"], tmp_path / "no_trace_out")

    policies = [
        _policy("B0", "B0"),
        _policy("R0", "R-dense", epsilon=0.0),
        _policy("R1", "R-dense", epsilon=0.1),
        _policy("T", "T-adaptive", threshold=0.9),
        _policy("C", "C-adaptive", threshold=0.9),
    ]
    _write_run(tmp_path / "run", policies=policies, work_by_policy={"R1": 0.25}, nfe_by_policy={"T": 8, "C": 8})
    output = tmp_path / "selected"
    select_operating_points([tmp_path / "run"], output, nfe_targets=[8])
    (output / "policies.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonidentical"):
        select_operating_points([tmp_path / "run"], output, nfe_targets=[8])
