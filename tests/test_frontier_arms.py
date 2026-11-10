from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.factor_rule import factor_select
from regret_remasking.paper_decode import PAPER_ARMS, TOKEN_KL_ARMS, WINDOW_ARMS, PaperDecodeConfig, _validate
from test_paper_token_kl_commitment import _decode


def peaked(top: int, confidence: float) -> list[float]:
    """A posterior over 6 ids (0 = mask) whose top non-mask token is ``top`` with probability ``confidence``."""
    rest = (1.0 - confidence) / 5
    return [confidence if i == top else rest for i in range(6)]


# ------------------------------------------------------------------------------------------ factor rule
def test_factor_select_leading_run_of_admissible_words() -> None:
    idx, k = factor_select(torch.tensor([0.99, 0.9, 0.85, 0.3]), 1.0)
    # n=2: 3*0.1=0.3<=1, n=3: 4*0.15=0.6<=1, n=4: 5*0.7=3.5>1
    assert k == 3 and idx.tolist() == [0, 1, 2]


def test_factor_select_always_writes_one_and_stops_at_first_failure() -> None:
    idx, k = factor_select(torch.tensor([0.2]), 0.5)
    assert (k, idx.tolist()) == (1, [0])
    idx, k = factor_select(torch.tensor([0.2, 0.1]), 1.0)
    assert (k, idx.tolist()) == (1, [0])
    # word 3 would pass (4*(1-0.99)=0.04) but word 2 failed first (3*(1-0.5)=1.5 > 1): stop at the first failure.
    idx, k = factor_select(torch.tensor([0.995, 0.5, 0.99]), 1.0)
    assert k == 2 and idx.tolist() == [0, 2]


def test_factor_select_ranks_by_confidence_and_returns_ascending_positions() -> None:
    idx, k = factor_select(torch.tensor([0.3, 0.99, 0.9]), 1.0)
    assert k == 2 and idx.tolist() == [1, 2]
    assert idx.tolist() == sorted(idx.tolist())


def test_factor_select_ties_go_to_the_earlier_position() -> None:
    idx, k = factor_select(torch.tensor([0.5, 0.5, 0.5, 0.5]), 0.5)
    assert (k, idx.tolist()) == (1, [0])


def test_factor_select_all_pass_and_last_ranked_failure_quirk() -> None:
    idx, k = factor_select(torch.tensor([0.99, 0.99, 0.99]), 1.0)
    assert k == 3 and idx.tolist() == [0, 1, 2]
    # Only the last-ranked word fails (4*(1-0.1)=3.6 > 1). The clean rule writes 2 words; the release (top_i == len-1 -> top_i += 1)
    # would write all three. This is the documented deviation (factor_rule.py, PREREGISTERED.md).
    idx, k = factor_select(torch.tensor([0.99, 0.98, 0.1]), 1.0)
    assert k == 2 and idx.tolist() == [0, 1]


def test_factor_four_admits_the_first_three_words_whatever_the_confidence() -> None:
    idx, k = factor_select(torch.tensor([0.05, 0.04, 0.03, 0.02]), 4.0)
    assert k == 3 and idx.tolist() == [0, 1, 2]


def test_factor_select_is_monotone_in_factor_and_matches_threshold_form() -> None:
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):
        conf = torch.rand(int(torch.randint(1, 33, (1,), generator=gen)), generator=gen)
        counts = [factor_select(conf, f)[1] for f in (0.5, 1.0, 2.0, 4.0)]
        assert counts == sorted(counts)
        ranked = torch.sort(conf.double(), descending=True, stable=True).values
        for f, k in zip((0.5, 1.0, 2.0, 4.0), counts):
            expected = 1
            for n in range(2, ranked.numel() + 1):
                if float(ranked[n - 1]) >= 1 - f / (n + 1):
                    expected = n
                else:
                    break
            assert k == expected


def test_factor_select_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        factor_select(torch.tensor([]), 1.0)
    with pytest.raises(ValueError):
        factor_select(torch.tensor([0.5]), 0.0)
    with pytest.raises(ValueError):
        factor_select(torch.tensor([[0.5]]), 1.0)


def test_f_block_registered_default_validation_and_block_mode() -> None:
    assert "F-block" in PAPER_ARMS and "F-block" not in WINDOW_ARMS
    _validate(PaperDecodeConfig(arm="F-block", factor=2.0))
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="F-block", factor=0.0))
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="F-block", log_calls="everything"))
    assert PaperDecodeConfig().log_calls == "none"


def test_f_block_decode_writes_leading_run_inside_the_current_block_only() -> None:
    # Block 1 = positions 1-4, block 2 = positions 5-8. Position 5 is very confident but belongs to the next block.
    frame = {1: peaked(1, 0.99), 2: peaked(2, 0.9), 3: peaked(3, 0.85), 4: peaked(4, 0.3), 5: peaked(1, 0.999),
             6: peaked(2, 0.6), 7: peaked(3, 0.6), 8: peaked(4, 0.6)}
    result, _ = _decode([frame], PaperDecodeConfig(arm="F-block", gen_length=8, block_length=4, factor=1.0, mask_id=0, log_calls="writes"))
    by_step: dict[int, list[int]] = {}
    for event in result.events:
        if event["type"] == "token_commit":
            by_step.setdefault(event["global_step"], []).append(event["position"])
    # call 0: positions 1,2,3 (n=4 fails); call 1: the single remaining blank 4; then block 2.
    assert by_step[0] == [1, 2, 3] and by_step[1] == [4]
    assert all(p <= 4 for p in by_step[0] + by_step[1]) and min(by_step[2]) >= 5
    assert result.final_mask_count == 0 and result.nfe == len(result.call_log)
    assert result.call_log[0]["w"] == [0, 1, 2] and result.call_log[0]["fb"] is False
    assert result.call_log[1]["fb"] is True
    assert result.adaptive_accept_count == 8


# ------------------------------------------------------------------------------------------ K-block
def test_k_block_registered_in_block_mode() -> None:
    assert "K-block" in PAPER_ARMS and "K-block" in TOKEN_KL_ARMS and "K-block" not in WINDOW_ARMS
    assert "K-adaptive" in WINDOW_ARMS
    _validate(PaperDecodeConfig(arm="K-block", epsilon=0.005))


def test_k_block_needs_history_inside_the_block_and_never_crosses_a_block() -> None:
    steady = {1: peaked(1, 0.9), 2: peaked(2, 0.8), 3: peaked(3, 0.7), 4: peaked(4, 0.95)}
    # position 3 is stable but outside block 1: it must not be written at call 1.
    frames = [steady, steady, steady, steady]
    result, _ = _decode(frames, PaperDecodeConfig(arm="K-block", gen_length=4, block_length=2, epsilon=0.005, mask_id=0))
    commits = [(e["global_step"], e["position"], e["fallback"], e["had_immediate_history"]) for e in result.events if e["type"] == "token_commit"]
    # call 0: block 1 has no history -> single fallback at the most confident blank (1); call 1: blank 2 has history, KL 0 -> written;
    # call 2: block 2 starts, no history -> fallback at the most confident (4); call 3: blank 3 has history -> written.
    assert commits == [(0, 1, True, False), (1, 2, False, True), (2, 4, True, False), (3, 3, False, True)]
    assert result.nfe == 4 and result.fallback_count == 2 and result.adaptive_accept_count == 4
    assert result.frozen_count == 0 and result.final_mask_count == 0


def test_k_block_writes_a_burst_when_everything_is_stable_and_logs_the_blank_table() -> None:
    steady = {i: peaked((i % 5) + 1, 0.9 - 0.01 * i) for i in range(1, 7)}
    result, _ = _decode([steady] * 3, PaperDecodeConfig(arm="K-block", gen_length=6, block_length=6, epsilon=0.005, mask_id=0, log_calls="blanks"))
    log = result.call_log
    assert [len(c["w"]) for c in log] == [1, 5]          # fallback, then the five blanks that have history in one burst
    assert log[0]["kl"] is None and log[0]["pos"] == [0, 1, 2, 3, 4, 5] and log[0]["w"] == [0]
    assert log[1]["pos"] == [1, 2, 3, 4, 5] and log[1]["w"] == [1, 2, 3, 4, 5] and log[1]["fb"] is False
    assert all(v is not None and 0 <= v <= 0.005 for v in log[1]["kl"])
    assert len(log[1]["conf"]) == len(log[1]["tok"]) == 5


def test_k_block_large_kl_blocks_the_write_and_falls_back() -> None:
    first = {1: peaked(1, 0.9), 2: peaked(2, 0.8)}
    second = {1: peaked(1, 0.9), 2: peaked(3, 0.8)}      # position 2 moved: KL well above 0.005
    result, _ = _decode([first, second, second], PaperDecodeConfig(arm="K-block", gen_length=2, block_length=2, epsilon=0.005, mask_id=0))
    commits = [(e["position"], e["fallback"]) for e in result.events if e["type"] == "token_commit"]
    assert commits == [(1, True), (2, True)]
    assert result.nfe == 2


def test_logging_is_off_by_default_and_does_not_change_the_decode() -> None:
    steady = {i: peaked((i % 5) + 1, 0.9 - 0.01 * i) for i in range(1, 7)}
    base = PaperDecodeConfig(arm="K-block", gen_length=6, block_length=6, epsilon=0.005, mask_id=0)
    plain, _ = _decode([steady] * 3, base)
    logged, _ = _decode([steady] * 3, PaperDecodeConfig(arm="K-block", gen_length=6, block_length=6, epsilon=0.005, mask_id=0, log_calls="blanks"))
    assert plain.call_log == [] and plain.token_ids == logged.token_ids and plain.nfe == logged.nfe
