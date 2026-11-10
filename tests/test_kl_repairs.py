"""Default-off repairs of the K-block KL word commit: confidence floor (R1), adjacent-identical veto (R2), persistence (R3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.kl_repairs import veto_adjacent_identical
from regret_remasking.paper_decode import PaperDecodeConfig, _validate
from test_frontier_arms import peaked
from test_paper_token_kl_commitment import _decode


def writes_by_call(result) -> list[list[int]]:
    by: dict[int, list[int]] = {}
    for e in result.events:
        if e["type"] == "token_commit":
            by.setdefault(e["global_step"], []).append(e["position"])
    return [by[k] for k in sorted(by)]


def cfg(**kw) -> PaperDecodeConfig:
    base = dict(arm="K-block", gen_length=6, block_length=6, epsilon=0.005, mask_id=0, log_calls="writes")
    base.update(kw)
    return PaperDecodeConfig(**base)


# ------------------------------------------------------------------------------------------ veto helper
def test_veto_drops_the_lower_confidence_of_an_adjacent_identical_pair() -> None:
    keep = veto_adjacent_identical([3, 4, 9], [7, 7, 7], [0.6, 0.9, 0.99])
    assert keep == [1, 2]                      # position 3 dropped; 9 is not adjacent to anything


def test_veto_tie_keeps_the_earlier_position() -> None:
    assert veto_adjacent_identical([3, 4], [7, 7], [0.8, 0.8]) == [0]


def test_veto_repeats_along_a_chain_until_no_adjacent_identical_pair_remains() -> None:
    # positions 1..4 all propose 7; confidences .5 .9 .6 .6: (1,2) -> drop 1; (2,3) -> drop 3; (2,4) not adjacent. 2 and 4 survive.
    keep = veto_adjacent_identical([1, 2, 3, 4], [7, 7, 7, 7], [0.5, 0.9, 0.6, 0.6])
    assert keep == [1, 3]
    surv = [(p, 7) for p, i in zip([1, 2, 3, 4], range(4)) if i in keep]
    assert all(b[0] - a[0] != 1 for a, b in zip(surv, surv[1:]))


def test_veto_ignores_different_tokens_exempt_padding_and_unsorted_input() -> None:
    assert veto_adjacent_identical([5, 4], [3, 2], [0.9, 0.9]) == [1, 0]       # returned ascending by position
    assert veto_adjacent_identical([4, 5, 6], [9, 9, 9], [0.5, 0.5, 0.5], exempt={9}) == [0, 1, 2]
    assert veto_adjacent_identical([], [], []) == []


def test_veto_always_keeps_at_least_the_most_confident() -> None:
    keep = veto_adjacent_identical([1, 2, 3], [4, 4, 4], [0.7, 0.99, 0.7])
    assert keep == [1]


def test_veto_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        veto_adjacent_identical([1, 2], [3], [0.5, 0.5])
    with pytest.raises(ValueError):
        veto_adjacent_identical([1, 1], [3, 3], [0.5, 0.5])


# ------------------------------------------------------------------------------------------ validation / defaults
def test_repair_defaults_are_off_and_only_k_block_accepts_them() -> None:
    c = PaperDecodeConfig()
    assert (c.confidence_floor, c.persist_calls, c.veto_adjacent) == (0.0, 0, False)
    _validate(cfg(confidence_floor=0.5, persist_calls=2, veto_adjacent=True))
    for bad in (dict(confidence_floor=1.5), dict(confidence_floor=-0.1), dict(persist_calls=-1)):
        with pytest.raises(ValueError):
            _validate(cfg(**bad))
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="C-block", confidence_floor=0.5))
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="K-adaptive", veto_adjacent=True))


def test_defaults_give_identical_k_block_output() -> None:
    steady = {i: peaked((i % 5) + 1, 0.9 - 0.01 * i) for i in range(1, 7)}
    plain, _ = _decode([steady] * 4, cfg())
    explicit, _ = _decode([steady] * 4, cfg(confidence_floor=0.0, persist_calls=0, veto_adjacent=False))
    assert plain.token_ids == explicit.token_ids and plain.nfe == explicit.nfe and plain.call_log == explicit.call_log


# ------------------------------------------------------------------------------------------ R1 floor
def test_floor_blocks_a_stable_but_weak_blank_and_keeps_the_fallback() -> None:
    # stable positions: 1 (0.9), 2 (0.45), 3 (0.8). K-block writes all stable ones at call 1; the floor keeps only conf >= 0.5.
    frame = {1: peaked(1, 0.9), 2: peaked(2, 0.45), 3: peaked(3, 0.8)}
    base, _ = _decode([frame] * 6, cfg(gen_length=3, block_length=3))
    floor, _ = _decode([frame] * 6, cfg(gen_length=3, block_length=3, confidence_floor=0.5))
    assert writes_by_call(base) == [[1], [2, 3]]         # call 0 fallback (most confident = 1), call 1: stable 2 and 3
    assert writes_by_call(floor) == [[1], [3], [2]]      # 2 is below the floor: call 1 writes only 3; 2 falls back later
    assert floor.nfe == 3 and floor.final_mask_count == 0


def test_floor_with_nobody_qualifying_writes_the_single_most_confident_blank() -> None:
    frame = {1: peaked(1, 0.4), 2: peaked(2, 0.3)}
    r, _ = _decode([frame] * 4, cfg(gen_length=2, block_length=2, confidence_floor=0.9))
    assert writes_by_call(r) == [[1], [2]] and r.fallback_count == 2


# ------------------------------------------------------------------------------------------ R2 veto
def test_veto_decode_stops_adjacent_identical_writes_in_one_call() -> None:
    # blanks 1..4 all propose token 2 and are stable: K-block writes 1..4 together at call 1, the veto leaves 2 and 4 (confidence descending).
    frame = {1: peaked(2, 0.5), 2: peaked(2, 0.9), 3: peaked(2, 0.6), 4: peaked(2, 0.6)}
    base, _ = _decode([frame] * 8, cfg(gen_length=4, block_length=4))
    vet, _ = _decode([frame] * 8, cfg(gen_length=4, block_length=4, veto_adjacent=True))
    assert writes_by_call(base) == [[2], [1, 3, 4]]
    # call 0 fallback (position 2, most confident). Call 1: qualifying 1,3,4 ; 3 and 4 adjacent same token tie -> drop later (4). 1 not adjacent to 3.
    assert writes_by_call(vet)[:2] == [[2], [1, 3]]
    assert vet.final_mask_count == 0 and vet.nfe > base.nfe


def test_veto_leaves_different_adjacent_tokens_alone() -> None:
    frame = {i: peaked(i, 0.9 - 0.01 * i) for i in range(1, 5)}
    base, _ = _decode([frame] * 4, cfg(gen_length=4, block_length=4))
    vet, _ = _decode([frame] * 4, cfg(gen_length=4, block_length=4, veto_adjacent=True))
    assert writes_by_call(base) == writes_by_call(vet) and base.token_ids == vet.token_ids


# ------------------------------------------------------------------------------------------ R3 persistence
def test_persistence_needs_three_equal_proposals_inside_the_block() -> None:
    frame = {1: peaked(1, 0.9), 2: peaked(2, 0.8), 3: peaked(3, 0.7)}
    r, _ = _decode([frame] * 8, cfg(gen_length=3, block_length=3, persist_calls=2))
    # call 0: no history -> fallback (1). call 1: blanks 2,3 have KL history but only one earlier proposal -> fallback (2).
    # call 2: blank 3 has proposals at calls 0, 1, 2 -> persistent -> written (as a non-fallback write).
    assert writes_by_call(r) == [[1], [2], [3]]
    assert [e["fallback"] for e in r.events if e["type"] == "token_commit"] == [True, True, False]
    plain, _ = _decode([frame] * 8, cfg(gen_length=3, block_length=3))
    assert writes_by_call(plain) == [[1], [2, 3]]


def test_persistence_breaks_when_the_proposal_flips_with_a_tiny_kl() -> None:
    # Position 3 is a near tie between tokens 3 and 4. With the tie steady it passes persistence at call 2 as a normal write; when it
    # flips at call 2 the KL is still far below epsilon (plain K-block would accept it) but the top-1 is not unchanged, so it is a fallback.
    head = {1: peaked(1, 0.9), 2: peaked(2, 0.8)}
    a = {**head, 3: [0.001, 0.05, 0.05, 0.2501, 0.2500, 0.001]}
    b = {**head, 3: [0.001, 0.05, 0.05, 0.2500, 0.2501, 0.001]}
    flag = lambda r: [e["fallback"] for e in r.events if e["type"] == "token_commit" and e["position"] == 3][0]
    token = lambda r: [e["token"] for e in r.events if e["type"] == "token_commit" and e["position"] == 3][0]
    steady, _ = _decode([a] * 4, cfg(gen_length=3, block_length=3, persist_calls=2))
    flipped, _ = _decode([a, a, b, b], cfg(gen_length=3, block_length=3, persist_calls=2))
    plain_flipped, _ = _decode([a, a, b, b], cfg(gen_length=3, block_length=3))
    assert writes_by_call(steady) == [[1], [2], [3]] and flag(steady) is False and token(steady) == 3
    assert writes_by_call(flipped) == [[1], [2], [3]] and flag(flipped) is True and token(flipped) == 4
    # plain K-block: position 3 is written at call 1 together with 2, as the frame there is the unflipped one.
    assert writes_by_call(plain_flipped) == [[1], [2, 3]]


def test_persistence_history_is_not_carried_from_one_block_to_the_next() -> None:
    # Block 2 (positions 3-4) starts at call 2 with no history for any of its blanks, so its first call is a fallback and its second
    # call cannot yet satisfy two previous proposals.
    frame = {1: peaked(1, 0.9), 2: peaked(2, 0.8), 3: peaked(3, 0.9), 4: peaked(4, 0.7)}
    r, _ = _decode([frame] * 8, cfg(gen_length=4, block_length=2, persist_calls=2))
    commits = [(e["global_step"], e["position"], e["fallback"]) for e in r.events if e["type"] == "token_commit"]
    assert commits == [(0, 1, True), (1, 2, True), (2, 3, True), (3, 4, True)]


def test_floor_and_persistence_combine_as_r3() -> None:
    frame = {1: peaked(1, 0.9), 2: peaked(2, 0.45), 3: peaked(3, 0.8)}
    r, _ = _decode([frame] * 8, cfg(gen_length=3, block_length=3, confidence_floor=0.5, persist_calls=2))
    assert writes_by_call(r) == [[1], [3], [2]]          # call 0 fallback (1); call 1 fallback (3, most confident of 2,3); call 2: 2 fails the floor -> fallback
    assert r.nfe == 3
