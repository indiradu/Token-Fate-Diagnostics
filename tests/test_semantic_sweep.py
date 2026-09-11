from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_ranker():
    path = Path(__file__).parents[1] / "scripts" / "rank_semantic_lock_sweep.py"
    spec = importlib.util.spec_from_file_location("rank_semantic_lock_sweep_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(policy: str, *, nfe: float, exact: float = 1.0, harm: float = 0.0) -> dict:
    return {
        "semantic_policy": policy,
        "pooled_lock_precision_vs_a0": 1.0 if exact == 1.0 else 0.9,
        "max_dataset_answer_harm_rate": harm,
        "example_weighted_exact_sequence_match_rate": exact,
        "example_weighted_answer_harm_rate": harm,
        "example_weighted_nfe_reduction": nfe,
        "mean_accelerated_locks_per_example": nfe * 64,
        "example_weighted_runtime_gain": nfe,
    }


def test_ranker_rejects_unsafe_speedup_and_selects_safe_improvement(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_ranker()
    summary = tmp_path / "summary.json"
    output_dir = tmp_path / "ranking"
    summary.write_text(
        json.dumps(
            [
                _row("baseline_c095_s2", nfe=0.02),
                _row("unsafe_fast", nfe=0.10, exact=0.9, harm=0.1),
                _row("safe_faster", nfe=0.04),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rank_semantic_lock_sweep.py",
            "--summary",
            str(summary),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    decision = json.loads(
        (output_dir / "semantic_candidate_decision.json").read_text(encoding="utf-8")
    )
    ranking = json.loads(
        (output_dir / "semantic_candidate_ranking.json").read_text(encoding="utf-8")
    )
    unsafe = next(row for row in ranking if row["semantic_policy"] == "unsafe_fast")
    assert decision["selected_for_frozen_confirmation"] == "safe_faster"
    assert not unsafe["safety_eligible"]
    assert "nonzero_per_dataset_answer_harm" in unsafe["rejection_reasons"]


def test_active_block_manifest_contains_complete_predeclared_grid() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "semantic_lock_v2b_active_block.json"
    )
    specs = json.loads(path.read_text(encoding="utf-8"))["semantic_policies"]
    labels = {spec["label"] for spec in specs}

    assert len(specs) == 25
    assert len(labels) == len(specs)
    assert {
        "baseline_c095_s2",
        "loose_c090_age0_s4",
        "warm_c090_age1_s4",
        "warm_c090_age2_s4",
        "warm_c090_age4_s8",
    } <= labels

    grid = {
        (float(spec["semantic_max_kl"]), int(spec["semantic_min_runlength"]))
        for spec in specs
        if spec["label"].startswith("active_kl")
    }
    assert grid == {
        (threshold, patience)
        for threshold in (0.002, 0.005, 0.01, 0.02, 0.05)
        for patience in (1, 2, 3, 4)
    }
    for spec in specs:
        if spec["label"].startswith("active_kl"):
            assert spec["semantic_min_confidence"] == 0.90
            assert spec["semantic_min_block_age"] == 1
            assert spec["semantic_optional_steps_per_block"] == 4
            assert spec["semantic_require_posterior_history"] is True


def test_frozen_baseline_manifest_is_one_exact_capped_confidence_arm() -> None:
    path = Path(__file__).parents[1] / "configs" / "semantic_lock_v1_baseline.json"
    specs = json.loads(path.read_text(encoding="utf-8"))["semantic_policies"]

    assert specs == [
        {
            "label": "baseline_c095_s2",
            "policy": "capped_confidence",
            "semantic_lock_fraction": 0.06,
            "semantic_min_confidence": 0.95,
            "semantic_min_margin": 0.0,
            "semantic_max_kl": None,
            "semantic_min_runlength": 1,
            "semantic_min_block_age": 0,
            "semantic_optional_steps_per_block": 2,
            "semantic_require_posterior_history": False,
        }
    ]


def test_confirmation_manifest_freezes_baseline_and_selected_candidate() -> None:
    path = Path(__file__).parents[1] / "configs" / "semantic_lock_v2_confirmation.json"
    specs = json.loads(path.read_text(encoding="utf-8"))["semantic_policies"]

    assert [spec["label"] for spec in specs] == [
        "baseline_c095_s2",
        "phaselock_semantic_v2",
    ]
    baseline, candidate = specs
    assert baseline["semantic_min_confidence"] == 0.95
    assert baseline["semantic_min_block_age"] == 0
    assert baseline["semantic_optional_steps_per_block"] == 2
    assert candidate["policy"] == "capped_confidence"
    assert candidate["semantic_lock_fraction"] == 0.06
    assert candidate["semantic_min_confidence"] == 0.90
    assert candidate["semantic_max_kl"] is None
    assert candidate["semantic_min_runlength"] == 1
    assert candidate["semantic_min_block_age"] == 4
    assert candidate["semantic_optional_steps_per_block"] == 8
