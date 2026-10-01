from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.paper_decode import PaperDecodeConfig, WINDOW_ARMS
from regret_remasking.paper_experiment import _validate_policy, make_paper_policies
from test_paper_token_kl_commitment import CountingRuntime, _decode

CONFIDENT_1 = [0.001, 0.95, 0.02, 0.01, 0.01, 0.009]
CONFIDENT_2 = [0.001, 0.01, 0.95, 0.02, 0.01, 0.009]
CONFIDENT_3 = [0.001, 0.01, 0.02, 0.95, 0.01, 0.009]


def test_c_block_commits_threshold_set_inside_current_block_only() -> None:
    # Prompt is position 0; generated positions 1-4 form two blocks of two.
    frames = [
        {1: CONFIDENT_1, 2: CONFIDENT_2, 3: CONFIDENT_3},
        {3: CONFIDENT_3},
        {},
    ]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(arm="C-block", gen_length=4, block_length=2, threshold=0.9, mask_id=0),
    )

    commits = [event for event in result.events if event["type"] == "token_commit"]
    by_step: dict[int, list[tuple[int, bool]]] = {}
    for event in commits:
        by_step.setdefault(event["global_step"], []).append((event["position"], event["fallback"]))
    # Position 3 is confident at step 0 but belongs to the next block.
    assert by_step == {0: [(1, False), (2, False)], 1: [(3, False)], 2: [(4, True)]}
    assert result.nfe == 3
    assert result.fallback_count == 1
    assert result.adaptive_accept_count == 4
    assert result.frozen_count == 0
    assert result.generated_row_updates_avoided == 0
    assert result.final_mask_count == 0


def test_c_block_is_not_a_window_arm_and_policy_factory_names_it() -> None:
    assert "C-block" not in WINDOW_ARMS
    policies = make_paper_policies(["B0", "C-block"], [0.005], [0.9])
    assert [policy["policy_id"] for policy in policies] == ["B0", "C-block-tau-0.9"]
    c_block = policies[1]
    assert c_block["threshold"] == 0.9
    _validate_policy(c_block, gen_length=256)


def test_all_unmasked_scope_can_freeze_prompt_rows() -> None:
    # Posteriors never change, so every revealed row has zero adjacent KL after step 0.
    frames = [{1: CONFIDENT_1, 2: CONFIDENT_2}]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(
            arm="S-sparse",
            gen_length=2,
            block_length=2,
            epsilon=0.005,
            freeze_scope="all_unmasked",
            mask_id=0,
            runtime_cls=CountingRuntime,
        ),
    )
    freezes = [event for event in result.events if event["type"] == "reference_freeze"]
    assert any(event["position"] == 0 for event in freezes)
    assert result.frozen_prompt_count >= 1
    assert not any(event["input_was_masked"] for event in freezes)


def test_generated_scope_never_freezes_prompt_rows() -> None:
    frames = [{1: CONFIDENT_1, 2: CONFIDENT_2}]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(
            arm="S-sparse",
            gen_length=2,
            block_length=2,
            epsilon=0.005,
            freeze_scope="generated",
            mask_id=0,
            runtime_cls=CountingRuntime,
        ),
    )
    freezes = [event for event in result.events if event["type"] == "reference_freeze"]
    assert all(event["position"] >= result.prompt_length for event in freezes)
    assert result.frozen_prompt_count == 0
