from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.less_arm import LessBlockSelector, _SubsetLessState
from regret_remasking.less_sampling import LessConfig, LessSelectionState
from regret_remasking.paper_decode import PAPER_ARMS, PaperDecodeConfig, _validate
from test_paper_token_kl_commitment import _decode

# Four generated positions (1-4) with distinct confident winners; the model repeats this posterior at every call.
FRAME = {
    1: [0.001, 0.96, 0.02, 0.01, 0.005, 0.004],
    2: [0.001, 0.01, 0.95, 0.02, 0.01, 0.009],
    3: [0.001, 0.01, 0.02, 0.94, 0.02, 0.009],
    4: [0.001, 0.01, 0.02, 0.02, 0.93, 0.019],
}
CFG = dict(arm="LESS", gen_length=4, mask_id=0, less_config=LessConfig(top_k=3))


def test_less_is_a_registered_arm_and_validates() -> None:
    assert "LESS" in PAPER_ARMS
    _validate(PaperDecodeConfig(arm="LESS", block_length=32))


def test_subset_selection_matches_released_state_over_a_decode() -> None:
    """select_at on the blank subset must make the same decisions as LessSelectionState.select on full-length posteriors."""
    torch.manual_seed(5)
    length, vocab = 12, 30
    config = LessConfig(top_k=4)
    full = LessSelectionState(length, torch.device("cpu"), config)
    subset = _SubsetLessState(length, torch.device("cpu"), config)
    base = torch.randn(1, length, vocab) * 2.0
    base[:, :, 0] += 14.0 * (torch.arange(length) % 2 == 0)       # half the slots are sharply peaked, half are diffuse
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[0, :2] = False                                          # two non-blank slots that must stay untouched
    accepted_any = fallback_any = False
    for step in range(8):
        probs = torch.softmax(base + 0.05 * torch.randn_like(base), dim=-1)
        expected = full.select(probs, mask, default_k=1)
        positions = torch.nonzero(mask[0]).flatten()
        selected, accepted, used_fallback = subset.select_at(positions, probs[0, positions], 1)
        assert used_fallback == expected.used_fallback
        assert selected.tolist() == torch.nonzero(expected.transfer_index[0]).flatten().tolist()
        assert torch.equal(subset.previous_winners, full.previous_winners)
        assert torch.equal(subset.previous_topk_ids, full.previous_topk_ids)
        accepted_any |= not used_fallback
        fallback_any |= used_fallback
        mask[0, selected] = False
        if not mask.any():
            break
    assert accepted_any and fallback_any


def test_less_decode_falls_back_to_one_blank_then_commits_the_stable_rest_together() -> None:
    result, _ = _decode([FRAME], PaperDecodeConfig(block_length=4, **CFG))
    by_step: dict[int, list[tuple[int, bool]]] = {}
    for event in result.events:
        if event["type"] == "token_commit":
            by_step.setdefault(event["global_step"], []).append((event["position"], event["fallback"]))
    # Calls 1 and 2 have fewer than two prior winners: the single most-confident blank each. Call 3: both remaining blanks pass.
    assert by_step == {0: [(1, True)], 1: [(2, True)], 2: [(3, False), (4, False)]}
    assert result.nfe == result.forward_calls == 3
    assert result.fallback_count == 2
    assert result.final_mask_count == 0 and result.immutable_token_violations == 0
    assert result.frozen_count == 0
    assert result.total_query_rows == result.nfe * 5          # LESS computes every row at every call


def test_less_decode_is_identical_with_a_step_observer() -> None:
    plain, _ = _decode([FRAME], PaperDecodeConfig(block_length=4, **CFG))
    observed, _ = _decode([FRAME], PaperDecodeConfig(block_length=4, **CFG), observer=True)
    assert plain.token_ids == observed.token_ids and plain.nfe == observed.nfe


def test_confidence_floor_blocks_acceptance_so_every_call_commits_one_blank() -> None:
    config = PaperDecodeConfig(block_length=4, **{**CFG, "less_config": LessConfig(top_k=3, confidence_threshold=0.99)})
    result, _ = _decode([FRAME], config)
    assert result.nfe == 4 and result.fallback_count == 4


def test_selector_resets_history_when_the_block_changes() -> None:
    selector = LessBlockSelector(6, torch.device("cpu"), LessConfig(top_k=3))
    logp = torch.log(torch.tensor([[[0.01, 0.9, 0.04, 0.03, 0.02]]]))
    block_a, block_b = torch.tensor([1]), torch.tensor([3])
    assert selector.select(logp, block_a, 1, 3)[1] and selector.select(logp, block_a, 1, 3)[1]
    assert selector.select(logp, block_b, 3, 5)[1]                 # a new block starts with no history
    assert selector.select(logp, block_a, 1, 3)[1]                 # returning to block A is a reset: history must be rebuilt
