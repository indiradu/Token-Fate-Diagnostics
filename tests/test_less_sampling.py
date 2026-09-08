from __future__ import annotations

import pathlib
import sys

import torch

from regret_remasking.less_sampling import LessConfig, LessSelectionState, coarse_topk_js

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_settlement_fate_audit import less_replay_prefix_matches


def test_coarse_topk_js_is_zero_for_identical_summaries():
    ids = torch.tensor([[1, 2]])
    probs = torch.tensor([[0.7, 0.2]])
    jsd = coarse_topk_js(ids, probs, ids, probs)
    assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-7)


def test_coarse_topk_js_is_symmetric_and_bounded():
    current_ids = torch.tensor([[1, 2]])
    current_probs = torch.tensor([[0.7, 0.2]])
    previous_ids = torch.tensor([[2, 3]])
    previous_probs = torch.tensor([[0.6, 0.3]])
    forward = coarse_topk_js(current_ids, current_probs, previous_ids, previous_probs)
    reverse = coarse_topk_js(previous_ids, previous_probs, current_ids, current_probs)
    assert torch.allclose(forward, reverse, atol=1e-7)
    assert 0.0 < float(forward.item()) <= float(torch.log(torch.tensor(2.0)))


def test_released_less_rule_accepts_after_two_prior_matching_winners():
    config = LessConfig(confidence_threshold=0.75, jsd_threshold=0.04, top_k=2, history_length=2)
    state = LessSelectionState(sequence_length=2, device=torch.device("cpu"), config=config)
    state.reset_range(0, 2)
    probs = torch.tensor([[[0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]]])
    mask = torch.tensor([[True, True]])

    first = state.select(probs, mask, default_k=1)
    second = state.select(probs, mask, default_k=1)
    third = state.select(probs, mask, default_k=1)

    assert not first.accepted_index.any()
    assert not second.accepted_index.any()
    assert third.accepted_index.all()
    assert int(first.transfer_index.sum()) == 1
    assert int(second.transfer_index.sum()) == 1
    assert int(third.transfer_index.sum()) == 2


def test_less_confidence_gate_blocks_otherwise_stable_position():
    config = LessConfig(confidence_threshold=0.9, jsd_threshold=0.04, top_k=2, history_length=2)
    state = LessSelectionState(sequence_length=1, device=torch.device("cpu"), config=config)
    probs = torch.tensor([[[0.1, 0.8, 0.05, 0.05]]])
    mask = torch.tensor([[True]])
    decisions = [state.select(probs, mask, default_k=1) for _ in range(3)]
    assert all(not decision.accepted_index.any() for decision in decisions)
    assert all(decision.used_fallback for decision in decisions)


def test_full_replay_prefix_detects_non_target_divergence():
    baseline_trace = {
        3: {
            "input_tokens": torch.tensor([[10, 20, 30]]),
            "raw_top1": torch.tensor([[11, 21, 31]]),
            "transfer_index": torch.tensor([[False, True, False]]),
        }
    }
    assert less_replay_prefix_matches(
        baseline_trace,
        global_step=3,
        freeze_step=4,
        input_tokens=torch.tensor([[10, 20, 30]]),
        raw_top1=torch.tensor([[11, 21, 31]]),
        transfer_index=torch.tensor([[False, True, False]]),
    )
    assert not less_replay_prefix_matches(
        baseline_trace,
        global_step=3,
        freeze_step=4,
        input_tokens=torch.tensor([[10, 20, 30]]),
        raw_top1=torch.tensor([[11, 99, 31]]),
        transfer_index=torch.tensor([[False, True, False]]),
    )
    assert not less_replay_prefix_matches(
        baseline_trace,
        global_step=3,
        freeze_step=4,
        input_tokens=torch.tensor([[10, 20, 30]]),
        raw_top1=torch.tensor([[11, 21, 31]]),
        transfer_index=torch.tensor([[True, False, False]]),
    )


def test_freeze_step_checks_input_but_not_post_intervention_decisions():
    baseline_trace = {
        4: {
            "input_tokens": torch.tensor([[10, 20, 30]]),
            "raw_top1": torch.tensor([[11, 21, 31]]),
            "transfer_index": torch.tensor([[False, True, False]]),
        }
    }
    assert less_replay_prefix_matches(
        baseline_trace,
        global_step=4,
        freeze_step=4,
        input_tokens=torch.tensor([[10, 20, 30]]),
        raw_top1=torch.tensor([[99, 99, 99]]),
        transfer_index=torch.tensor([[True, False, True]]),
    )
    assert not less_replay_prefix_matches(
        baseline_trace,
        global_step=4,
        freeze_step=4,
        input_tokens=torch.tensor([[10, 99, 30]]),
        raw_top1=torch.tensor([[11, 21, 31]]),
        transfer_index=torch.tensor([[False, True, False]]),
    )
