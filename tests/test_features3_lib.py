"""Features of results/lctx_features3_20261115 (features3_lib.compute) on a hand-built one-word-per-call trace: every feature checked against a manual reading of the STATE."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "results/lctx_features3_20261115/orchestration/features3_lib.py"
if _P.exists():
    _spec = importlib.util.spec_from_file_location("features3_lib_test", _P)
    F3 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(F3)
else:
    F3 = None
pytestmark = pytest.mark.skipif(F3 is None, reason="results/lctx_features3_20261115 not present")

# vocabulary: 0 pad, 1 "x", 2 "=", 3 " ", 4 "7", 5 "3", 6 "$", 7 "{", 8 "}", 9 "+"
V = 10
INFO = {k: np.zeros(V, np.int8) for k in ("digit", "eq", "op", "ws", "dollar", "lbrace", "rbrace")}
INFO["digit"][[4, 5]] = 1
INFO["eq"][2] = 1
INFO["op"][9] = 1
INFO["ws"][3] = 1
INFO["dollar"][6] = 1
INFO["lbrace"][7] = 1
INFO["rbrace"][8] = 1
PROMPT = np.array([1, 2, 4, 1], np.int64)     # "x = 7 x"
G, B = 8, 4                                     # response length 8, blocks of 4


def _trace():
    """Block 0 (r 0..3) final tokens [6, 1, 2, 5] = "$ x = 3"; block 1 (r 4..7) final [3, 4, 9, 8].
    Calls 0-3 write r = 0, 1, 2, 3 in that order; calls 4-7 write r = 4, 5, 6, 7. Proposals equal the final tokens except call 0 at r 3 (proposes 4, '7')."""
    final = np.array([6, 1, 2, 5, 3, 4, 9, 8])
    rows = []
    for c in range(8):
        blk = c // B
        for p in range(blk * B + c % B, blk * B + B):
            t = 4 if (c == 0 and p == 3) else final[p]
            conf = 0.9 if p == c else 0.3 + 0.1 * (p % B)
            rows.append((c, p, t, conf, p == c, final[p]))
    a = np.array(rows, dtype=float)
    return {"call": a[:, 0].astype(np.int32), "position": (a[:, 1] + len(PROMPT)).astype(np.int32), "token": a[:, 2].astype(np.int64), "confidence": a[:, 3].astype(np.float32),
            "written_now": a[:, 4].astype(bool), "final_token": a[:, 5].astype(np.int64), "y": (a[:, 2] == a[:, 5]).astype(np.int8)}


def _row(rows, out, c, p):
    i = np.nonzero((rows["call"] == c) & (rows["position"] == p + len(PROMPT)))[0][0]
    return {k: (None if np.isnan(v[i]) else float(v[i])) for k, v in out.items()}


def test_state_features():
    rows = _trace()
    out, meta = F3.compute(rows, PROMPT, INFO, gen_length=G, block=B)
    assert meta["v1"] == {"positions_written": 8, "writes": 8, "written_equals_final": True} and meta["calls"] == 8
    # call 0, r 3 proposes '7': in the prompt, digit, not novel; left neighbour r 2 is a blank -> bigram NaN; no written token within 3 back except prompt? r0..2 blanks -> NaN
    f = _row(rows, out, 0, 3)
    assert f["h_in_prompt"] == 1 and f["i_digit"] == 1 and f["h_num_novel"] == 0 and f["h_bigram_prompt"] is None and f["i_after_eq"] is None
    # call 0, r 0 proposes '$': left is the last prompt token 'x' (bigram (x, $) not in prompt); after_eq reads prompt 'x' -> 0
    f = _row(rows, out, 0, 0)
    assert f["h_bigram_prompt"] == 0 and f["i_after_eq"] == 0 and f["h_in_written"] == 0
    # call 3, r 3 proposes '3': state "$ x =" -> after '=', inside math ($ count 1), novel number, bigram (=, 3) not in prompt; ('=' , '7') would be
    f = _row(rows, out, 3, 3)
    assert f["i_after_eq"] == 1 and f["i_in_math"] == 1 and f["h_num_novel"] == 1 and f["h_bigram_prompt"] == 0 and f["h_bigram_written"] == 0
    # call 2, r 2 proposes '=': bigram (x, =) is in the prompt; '=' not yet written
    f = _row(rows, out, 2, 2)
    assert f["h_bigram_prompt"] == 1 and f["h_in_written"] == 0 and f["h_in_prompt"] == 1
    # call 5, r 5 proposes '7': left r 4 is ' ' (whitespace, skipped), r 3 '3' -> not eq, not op; bigram (' ', 7) not written
    f = _row(rows, out, 5, 5)
    assert f["i_after_eq"] == 0 and f["i_after_op"] == 0 and f["h_bigram_written"] == 0
    # call 7, r 7 proposes '}': written "$ x = 3 _ 7 +" -> after '+' (op); block 1 has 3 of 4 written; brace depth 0
    f = _row(rows, out, 7, 7)
    assert f["i_after_op"] == 1 and f["j_block_written"] == 0.75 and f["i_brace_depth"] == 0 and f["k_written_w4"] == 4


def test_window_rank_lookahead():
    rows = _trace()
    out, _ = F3.compute(rows, PROMPT, INFO, gen_length=G, block=B)
    # call 0: blanks r0..3, conf [0.9, 0.4, 0.5, 0.6]
    f1 = _row(rows, out, 0, 1)
    assert f1["k_min_w2"] == pytest.approx(0.5) and f1["k_mean_w4"] == pytest.approx((0.9 + 0.5 + 0.6) / 3)
    assert f1["k_run"] == 0                                   # own confidence 0.4 < 0.5
    f2 = _row(rows, out, 0, 2)
    assert f2["k_run"] == 2                                   # r2, r3 eligible >= 0.5; r1 breaks the run; r4 is not eligible at call 0
    f0 = _row(rows, out, 0, 0)
    assert f0["j_rank_frac"] == 1.0 and f1["j_rank_frac"] == 0.25 and f1["j_conf_minus_med"] == pytest.approx(0.4 - 0.55)
    # lookahead: r1 at call 1 has confidence 0.9 and the same proposal; r3 changes its proposal from '7' to '3' at call 1
    assert f1["o_conf_next"] == pytest.approx(0.9) and f1["o_same_next"] == 1
    f3 = _row(rows, out, 0, 3)
    assert f3["o_same_next"] == 0 and f3["o_nbr_y_r"] is None and f3["o_nbr_y_l"] == 1
    assert _row(rows, out, 0, 2)["o_nbr_y_r"] == 0             # r3's proposal at call 0 is not final
    assert _row(rows, out, 3, 3)["o_conf_next"] is None        # last blank of block 0, written at call 3
    assert f1["_dist_l"] == 1 and f1["_dist_r"] == 1


def test_prompt_mismatch_raises():
    rows = _trace()
    with pytest.raises(ValueError):
        F3.compute(rows, PROMPT[:2], INFO, gen_length=G, block=B)
