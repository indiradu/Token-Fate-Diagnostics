"""Features of PREREGISTERED.md (groups h, i, k, j and the lookahead reference O), computed from one question's trace (the npz columns call, position, token, confidence,
written_now, final_token / y), the question's prompt ids and the token table of build_assets.py. No model, no forward pass.

STATE at call t = the response tokens written at calls < t (each written row's token). The rows of call t are the call's eligible blanks.
Positions in the traces are absolute (prompt length + response index); r = position - len(prompt)."""
from __future__ import annotations

import time

import numpy as np

H = ["h_in_prompt", "h_in_written", "h_bigram_prompt", "h_bigram_written", "h_num_novel"]
I = ["i_digit", "i_after_eq", "i_after_op", "i_in_math", "i_brace_depth"]
K = ["k_min_w2", "k_mean_w4", "k_run", "k_written_w4"]
J = ["j_rank_frac", "j_conf_minus_med", "j_block_written"]
O = ["o_conf_next", "o_same_next", "o_nbr_y_l", "o_nbr_y_r"]
GROUPS3 = {"h": H, "i": I, "k": K, "j": J, "O": O}
NEW3 = H + I + K + J
AUX = ["_dist_l", "_dist_r", "_r"]     # distance to the nearest eligible blank on each side at the same call, the response index (checks only)
BACK = 3


def compute(rows: dict[str, np.ndarray], prompt: np.ndarray, info: dict[str, np.ndarray], gen_length: int = 512, block: int = 32) -> tuple[dict[str, np.ndarray], dict]:
    """Returns ({feature: float32 array aligned with the rows}, {"ms_per_call": decode-time features h/i/k/j, "v1": state reconstruction check})."""
    call = rows["call"].astype(np.int64)
    r = rows["position"].astype(np.int64) - len(prompt)
    tok = rows["token"].astype(np.int64)
    conf = rows["confidence"].astype(np.float64)
    wn = rows["written_now"].astype(bool)
    y = rows["y"].astype(np.float64) if "y" in rows else (rows["token"] == rows["final_token"]).astype(np.float64)
    n = len(call)
    if r.min() < 0 or r.max() >= gen_length:
        raise ValueError(f"response index out of range [{r.min()}, {r.max()}]: prompt length {len(prompt)} does not match the trace")
    V = len(info["digit"])
    digit, eq, op, ws = (info[k].astype(bool) for k in ("digit", "eq", "op", "ws"))
    dollar, brace = info["dollar"].astype(np.int64), info["lbrace"].astype(np.int64) - info["rbrace"].astype(np.int64)
    in_prompt = np.zeros(V, bool)
    in_prompt[prompt] = True
    prompt_bi = set((prompt[:-1] * V + prompt[1:]).tolist())
    in_written = np.zeros(V, bool)
    written_bi: set[int] = set()
    state = np.full(gen_length, -1, np.int64)
    out = {f: np.full(n, np.nan, np.float32) for f in NEW3 + O + AUX}
    out["_r"][:] = r
    order = np.lexsort((r, call))
    starts = np.r_[0, np.nonzero(np.diff(call[order]))[0] + 1, n]
    t_feat = 0.0
    n_calls = len(starts) - 1
    for c in range(n_calls):
        idx = order[starts[c]:starts[c + 1]]
        rr, tk, cf = r[idx], tok[idx], conf[idx]
        t0 = time.perf_counter()
        wmask = state >= 0
        # (h) provenance
        out["h_in_prompt"][idx] = in_prompt[tk]
        out["h_in_written"][idx] = in_written[tk]
        lt = np.where(rr > 0, state[np.maximum(rr - 1, 0)], prompt[-1])
        has = lt >= 0
        key = lt * V + tk
        out["h_bigram_prompt"][idx] = np.where(has, [k in prompt_bi for k in key.tolist()], np.nan)
        out["h_bigram_written"][idx] = np.where(has, [k in written_bi for k in key.tolist()], np.nan)
        out["h_num_novel"][idx] = digit[tk] & ~in_prompt[tk]
        # (i) structure
        out["i_digit"][idx] = digit[tk]
        aeq, aop = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
        for j, p in enumerate(rr.tolist()):
            for k in range(1, BACK + 1):
                q = p - k
                t_ = state[q] if q >= 0 else prompt[len(prompt) + q]
                if t_ < 0 or ws[t_]:
                    continue
                aeq[j], aop[j] = eq[t_], op[t_]
                break
        out["i_after_eq"][idx], out["i_after_op"][idx] = aeq, aop
        dol = np.concatenate([[0], np.cumsum(np.where(wmask, dollar[np.maximum(state, 0)], 0))])
        brc = np.concatenate([[0], np.cumsum(np.where(wmask, brace[np.maximum(state, 0)], 0))])
        out["i_in_math"][idx] = dol[rr] % 2
        out["i_brace_depth"][idx] = np.maximum(brc[rr], 0)
        # (k) neighbour window (other eligible blanks of the same call)
        D = np.abs(rr[:, None] - rr[None, :])
        other = D > 0
        for f, w, agg in (("k_min_w2", 2, np.min), ("k_mean_w4", 4, np.mean)):
            m = other & (D <= w)
            vals = np.where(m, cf[None, :], np.nan)
            okrow = m.any(1)
            res = np.full(len(idx), np.nan)
            if okrow.any():
                res[okrow] = (np.nanmin if agg is np.min else np.nanmean)(vals[okrow], axis=1)
            out[f][idx] = res
        ok = wmask.copy()
        ok[rr[cf >= 0.5]] = True
        seg = np.cumsum(~ok)
        runlen = np.bincount(seg[ok], minlength=seg.max() + 1)
        out["k_run"][idx] = np.where(cf >= 0.5, runlen[seg[rr]], 0)
        wc = np.concatenate([[0], np.cumsum(wmask)])
        lo, hi = np.maximum(rr - 4, 0), np.minimum(rr + 4, gen_length - 1)
        out["k_written_w4"][idx] = wc[hi + 1] - wc[lo]
        # (j) call-relative
        nn = len(idx)
        rank0 = np.empty(nn, np.int64)
        rank0[np.argsort(-cf, kind="stable")] = np.arange(nn)
        out["j_rank_frac"][idx] = (nn - rank0) / nn
        out["j_conf_minus_med"][idx] = cf - np.median(cf)
        b0 = (rr // block) * block
        out["j_block_written"][idx] = (wc[np.minimum(b0 + block, gen_length)] - wc[b0]) / block
        t_feat += time.perf_counter() - t0
        # (O) neighbours' fates at the same call; aux distances
        yy = y[idx]
        out["o_nbr_y_l"][idx[1:]], out["o_nbr_y_r"][idx[:-1]] = yy[:-1], yy[1:]
        out["_dist_l"][idx[1:]], out["_dist_r"][idx[:-1]] = np.diff(rr), np.diff(rr)
        # update the STATE with this call's writes
        for p, t_ in zip(rr[wn[idx]].tolist(), tk[wn[idx]].tolist()):
            state[p] = t_
            in_written[t_] = True
            if p > 0 and state[p - 1] >= 0:
                written_bi.add(int(state[p - 1]) * V + t_)
            if p + 1 < gen_length and state[p + 1] >= 0:
                written_bi.add(t_ * V + int(state[p + 1]))
    # (O) lookahead: the same blank at call t+1
    key_now = call * gen_length + r
    pos_of = {k: i for i, k in enumerate(key_now.tolist())}
    nxt = np.array([pos_of.get(k + gen_length, -1) for k in key_now.tolist()])
    have = nxt >= 0
    out["o_conf_next"][have] = conf[nxt[have]]
    out["o_same_next"][have] = tok[nxt[have]] == tok[have]
    # V1: every response position written exactly once (G1), and the written token is final
    wpos = r[wn]
    v1 = {"positions_written": int(len(np.unique(wpos))), "writes": int(wn.sum()), "written_equals_final": bool(np.all(tok[wn] == rows["final_token"][wn]))}
    return out, {"ms_per_call": 1000.0 * t_feat / max(n_calls, 1), "calls": n_calls, "v1": v1}
