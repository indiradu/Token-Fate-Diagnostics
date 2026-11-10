"""Oracle representation-freeze gate (results/repfreeze_tests_20261029, Test 2).

Same duck-typed surface as `LearnedFreezeGate` (reset / observe / score / decide / sha256 / threshold / min_age), but the decision is read from a table
built from a dense Confidence-0.9 decode of the same question: a row freezes at the first call where its age (calls since written; prompt rows
step + 1) is at least the age from which the dense decode shows the row permanently settled (relative feature distance to the final state <= 0.25
at that call and at every later recorded call). No hidden state is projected; `observe` returns the row positions in place of features.

`settled_ages` turns recorder rows (call, pos, age, d_feat_final) into that per-position age; positions that never qualify get `NEVER`.
"""
from __future__ import annotations

import numpy as np
import torch

NEVER = 10**9
TAU_SETTLED = 0.25


def settled_ages(calls: np.ndarray, pos: np.ndarray, age: np.ndarray, dist: np.ndarray, size: int, *, tau: float = TAU_SETTLED) -> np.ndarray:
    """Per position (0..size-1): the age of the first recorded row from which every recorded row (later calls) has dist <= tau, else NEVER.

    Rows are the recorder's rows: calls 1..T-2 (the reference call T-1 is not a row). A position without rows never freezes."""
    out = np.full(size, NEVER, dtype=np.int64)
    if len(pos) == 0:
        return out
    order = np.lexsort((calls, pos))
    calls, pos, age, dist = calls[order], pos[order], age[order], dist[order]
    bounds = np.flatnonzero(np.diff(pos)) + 1
    for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(pos)]])):
        ok = dist[lo:hi] <= tau
        if not ok[-1]:
            continue
        bad = np.flatnonzero(~ok)
        first = 0 if len(bad) == 0 else int(bad[-1]) + 1
        out[int(pos[lo])] = int(age[lo + first])
    return out


class OracleGate:
    """rows: "answer" or "prompt" (which row type this instance decides). `never=True` keeps the table but never freezes (determinism check)."""

    def __init__(self, rows: str, *, never: bool = False, min_age: int = 1, sha256: str = "oracle") -> None:
        if rows not in ("answer", "prompt"):
            raise ValueError("rows must be answer or prompt")
        self.rows, self.never = rows, bool(never)
        self.threshold, self.min_age, self.sha256 = 0.5, int(min_age), sha256
        self._ages: torch.Tensor | None = None
        self._table: dict | None = None
        self._offset = 0

    def set_example(self, answer_ages: np.ndarray, prompt_ages: np.ndarray, prompt_length: int) -> None:
        """answer_ages: (gen_length,) indexed by answer position; prompt_ages: (prompt_length,) indexed by absolute prompt position."""
        self._table = {"answer": np.asarray(answer_ages, dtype=np.int64), "prompt": np.asarray(prompt_ages, dtype=np.int64)}
        self._offset = int(prompt_length)

    def reset(self, length: int, device: torch.device) -> None:
        if self._table is None:
            raise RuntimeError("call set_example() before decoding")
        src = self._table[self.rows]
        if self.rows == "answer" and len(src) != length - self._offset:
            raise ValueError("oracle table does not match the decode shape")
        if self.rows == "prompt" and len(src) != self._offset:
            raise ValueError("oracle prompt table does not match the prompt length")
        ages = torch.full((length,), NEVER, dtype=torch.long)
        if not self.never:
            if self.rows == "answer":
                ages[self._offset:] = torch.from_numpy(src)
            else:
                ages[: self._offset] = torch.from_numpy(src)
        self._ages = ages.to(device)

    def observe(self, hidden3, positions: torch.Tensor):
        feats = positions.to(torch.float32)[:, None]
        return feats, torch.zeros(positions.numel(), dtype=torch.float32, device=positions.device)

    def score(self, feats, rc_feat, vkl, echo_p, entropy, top1_match, age, *, return_logit: bool = False, progress=None):
        assert self._ages is not None
        need = self._ages.index_select(0, feats[:, 0].long())
        p = (age.long() >= need).float()
        if return_logit:
            return p, torch.where(p > 0, torch.zeros_like(p), torch.full_like(p, -torch.inf))
        return p

    def decide(self, scores: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        return (scores >= self.threshold) & (age >= self.min_age)
