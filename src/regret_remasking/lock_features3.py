"""Decode-time inputs h/i/k/j of results/lctx_features3_20261115 for a learned context word lock (Lctx), computed live inside the decoder.

The arithmetic is a line-by-line copy of the per-call body of results/lctx_features3_20261115/orchestration/features3_lib.compute (which computed them offline from G1
traces); tests/test_lock_features3.py checks that driving F3State call by call reproduces compute() exactly on a hand-built trace, and the comparison grid's validity
check V-F3 repeats that on the real G1 traces. Default off: the engine computes these only when a lock == "rule" rule names one of FEATURES3 (wants_f3).

STATE at a call = the response tokens written at earlier calls. Rows = the call's eligible blanks in ascending position order; r = position - prompt length.
  (h) provenance   h_in_prompt, h_in_written, h_bigram_prompt, h_bigram_written, h_num_novel
  (i) structure    i_digit, i_after_eq, i_after_op, i_in_math, i_brace_depth
  (k) window       k_min_w2, k_mean_w4, k_run, k_written_w4
  (j) call-relative j_rank_frac, j_conf_minus_med, j_block_written
`info` = the token table of features3's build_assets.py: int8 [vocab] arrays digit, eq, op, ws, dollar, lbrace, rbrace."""
from __future__ import annotations

import numpy as np

H = ["h_in_prompt", "h_in_written", "h_bigram_prompt", "h_bigram_written", "h_num_novel"]
I = ["i_digit", "i_after_eq", "i_after_op", "i_in_math", "i_brace_depth"]
K = ["k_min_w2", "k_mean_w4", "k_run", "k_written_w4"]
J = ["j_rank_frac", "j_conf_minus_med", "j_block_written"]
GROUPS3 = {"h": H, "i": I, "k": K, "j": J}
FEATURES3 = H + I + K + J
BACK = 3
INFO_KEYS = ("digit", "eq", "op", "ws", "dollar", "lbrace", "rbrace")


def wants_f3(cfg) -> bool:
    rule = getattr(cfg, "rule", None)
    return getattr(cfg, "lock", None) == "rule" and rule is not None and bool(set(getattr(rule, "features", []) or []) & set(FEATURES3))


class F3State:
    """Per-decode state (written tokens, written-token and written-bigram sets). features() for the rows of a call, then update() with that call's writes."""

    def __init__(self, prompt: np.ndarray, info: dict, gen_length: int = 512, block: int = 32):
        missing = [k for k in INFO_KEYS if k not in info]
        if missing:
            raise ValueError(f"token table lacks {missing}")
        self.prompt = np.asarray(prompt, dtype=np.int64)
        self.gen_length, self.block = int(gen_length), int(block)
        self.V = V = len(info["digit"])
        self.digit, self.eq, self.op, self.ws = (np.asarray(info[k]).astype(bool) for k in ("digit", "eq", "op", "ws"))
        self.dollar = np.asarray(info["dollar"]).astype(np.int64)
        self.brace = np.asarray(info["lbrace"]).astype(np.int64) - np.asarray(info["rbrace"]).astype(np.int64)
        self.in_prompt = np.zeros(V, bool)
        self.in_prompt[self.prompt] = True
        self.prompt_bi = set((self.prompt[:-1] * V + self.prompt[1:]).tolist())
        self.in_written = np.zeros(V, bool)
        self.written_bi: set[int] = set()
        self.state = np.full(self.gen_length, -1, np.int64)

    def features(self, rr: np.ndarray, tk: np.ndarray, cf: np.ndarray) -> dict[str, np.ndarray]:
        rr = np.asarray(rr, dtype=np.int64)
        tk = np.asarray(tk, dtype=np.int64)
        cf = np.asarray(cf, dtype=np.float64)
        if len(rr) > 1 and np.any(np.diff(rr) <= 0):
            raise ValueError("rows must be in strictly ascending position order")
        n, V, state, prompt, gen_length, block = len(rr), self.V, self.state, self.prompt, self.gen_length, self.block
        out = {f: np.full(n, np.nan, np.float32) for f in FEATURES3}
        wmask = state >= 0
        # (h) provenance
        out["h_in_prompt"][:] = self.in_prompt[tk]
        out["h_in_written"][:] = self.in_written[tk]
        lt = np.where(rr > 0, state[np.maximum(rr - 1, 0)], prompt[-1])
        has = lt >= 0
        key = lt * V + tk
        out["h_bigram_prompt"][:] = np.where(has, [k in self.prompt_bi for k in key.tolist()], np.nan)
        out["h_bigram_written"][:] = np.where(has, [k in self.written_bi for k in key.tolist()], np.nan)
        out["h_num_novel"][:] = self.digit[tk] & ~self.in_prompt[tk]
        # (i) structure
        out["i_digit"][:] = self.digit[tk]
        aeq, aop = np.full(n, np.nan), np.full(n, np.nan)
        for j, p in enumerate(rr.tolist()):
            for k in range(1, BACK + 1):
                q = p - k
                t_ = state[q] if q >= 0 else prompt[len(prompt) + q]
                if t_ < 0 or self.ws[t_]:
                    continue
                aeq[j], aop[j] = self.eq[t_], self.op[t_]
                break
        out["i_after_eq"][:], out["i_after_op"][:] = aeq, aop
        dol = np.concatenate([[0], np.cumsum(np.where(wmask, self.dollar[np.maximum(state, 0)], 0))])
        brc = np.concatenate([[0], np.cumsum(np.where(wmask, self.brace[np.maximum(state, 0)], 0))])
        out["i_in_math"][:] = dol[rr] % 2
        out["i_brace_depth"][:] = np.maximum(brc[rr], 0)
        # (k) neighbour window (other eligible blanks of the same call)
        D = np.abs(rr[:, None] - rr[None, :])
        other = D > 0
        for f, w, agg in (("k_min_w2", 2, np.min), ("k_mean_w4", 4, np.mean)):
            m = other & (D <= w)
            vals = np.where(m, cf[None, :], np.nan)
            okrow = m.any(1)
            res = np.full(n, np.nan)
            if okrow.any():
                res[okrow] = (np.nanmin if agg is np.min else np.nanmean)(vals[okrow], axis=1)
            out[f][:] = res
        ok = wmask.copy()
        ok[rr[cf >= 0.5]] = True
        seg = np.cumsum(~ok)
        runlen = np.bincount(seg[ok], minlength=seg.max() + 1)
        out["k_run"][:] = np.where(cf >= 0.5, runlen[seg[rr]], 0)
        wc = np.concatenate([[0], np.cumsum(wmask)])
        lo, hi = np.maximum(rr - 4, 0), np.minimum(rr + 4, gen_length - 1)
        out["k_written_w4"][:] = wc[hi + 1] - wc[lo]
        # (j) call-relative
        rank0 = np.empty(n, np.int64)
        rank0[np.argsort(-cf, kind="stable")] = np.arange(n)
        out["j_rank_frac"][:] = (n - rank0) / n
        out["j_conf_minus_med"][:] = cf - np.median(cf)
        b0 = (rr // block) * block
        out["j_block_written"][:] = (wc[np.minimum(b0 + block, gen_length)] - wc[b0]) / block
        return out

    def update(self, rr_w: np.ndarray, tk_w: np.ndarray) -> None:
        """Record a call's writes, in ascending position order (the order compute() uses)."""
        rr_w = np.asarray(rr_w, dtype=np.int64)
        tk_w = np.asarray(tk_w, dtype=np.int64)
        order = np.argsort(rr_w, kind="stable")
        state, V, gen_length = self.state, self.V, self.gen_length
        for p, t_ in zip(rr_w[order].tolist(), tk_w[order].tolist()):
            state[p] = t_
            self.in_written[t_] = True
            if p > 0 and state[p - 1] >= 0:
                self.written_bi.add(int(state[p - 1]) * V + t_)
            if p + 1 < gen_length and state[p + 1] >= 0:
                self.written_bi.add(t_ * V + int(state[p + 1]))

    def check_state(self, answer: np.ndarray, mask_id: int) -> None:
        """Raise if the tracked state disagrees with the decoder's sequence (a write that bypassed update())."""
        answer = np.asarray(answer, dtype=np.int64)[: self.gen_length]
        written = answer != mask_id
        if not (np.array_equal(written, self.state >= 0) and np.array_equal(answer[written], self.state[written])):
            raise RuntimeError("lock_features3 state out of sync with the decoder's sequence")
