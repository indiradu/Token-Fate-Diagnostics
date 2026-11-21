"""Learned refresh controller for the representation cache (default off; ``SelectiveConfig.refresh_ctrl``). Pre-registration: results/refresh_family_20261111/PREREGISTERED.md, "Stage C".

At every non-dense call the controller scores the RISK that the cache is stale enough for this call to commit a word the dense pass would not propose (a logistic model on the stage G features F1: the clock
and free signals of the previous call; frozen, fitted on KodCode-tune plain-S32 decodes) and picks a refresh rung from the risk: no refresh, a partial refresh of the blank rows ("blank"), of all cached rows
("all") from the start layer, or a dense call ("dense"). A refresh is never made before ``i_min`` calls after the last refresh, and is forced after ``i_max`` (the lowest rung). All thresholds are quantile
thresholds of the training risk scores (``top_share_thresholds`` of the model file), so one number per rung sets a refresh budget.

Use: ``RefreshController(model_json, rungs=("blank","all","dense"), top_shares=(0.2, 0.1, 0.03), i_min=3, i_max=8)``; ``decide(...)`` before the forward of a call, ``observe(...)`` after the word lock wrote.
A fresh controller per decode (it keeps the previous two calls' proposals).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

RUNG_ORDER = ("blank", "all", "dense")


def window_proposals(logits: torch.Tensor, mask_id: int):
    """(argmax token, its probability) per decoding row; the mask token suppressed; float32 softmax (as the confidence word lock)."""
    lg = logits[0].float().clone()
    lg[:, mask_id] = float("-inf")
    conf, tok = torch.softmax(lg, dim=-1).max(dim=-1)
    return tok.tolist(), conf.tolist()


class RefreshController:
    def __init__(self, model_json: str | Path | dict, rungs: tuple[str, ...] = ("blank", "all", "dense"), top_shares: tuple[float, ...] = (0.2, 0.1, 0.03), i_min: int = 3, i_max: int = 8,
                 block_length: int = 32) -> None:
        spec = model_json if isinstance(model_json, dict) else json.loads(Path(model_json).read_text())
        if len(rungs) != len(top_shares) or not rungs or any(r not in RUNG_ORDER for r in rungs) or list(rungs) != sorted(rungs, key=RUNG_ORDER.index):
            raise ValueError("rungs must be a non-empty ascending subset of (blank, all, dense), one top share each")
        if any(b >= a for a, b in zip(top_shares, top_shares[1:])):
            raise ValueError("top_shares must decrease with the rung (a higher rung fires on a rarer, riskier call)")
        if not 1 <= i_min <= i_max:
            raise ValueError("1 <= i_min <= i_max")
        self.features, self.mean, self.scale, self.coef, self.intercept = spec["features"], spec["mean"], spec["scale"], spec["coef"], spec["intercept"]
        qs = spec["top_share_thresholds"]
        self.thresholds = []
        for q in top_shares:
            if str(q) not in qs:
                raise ValueError(f"no stored threshold for top share {q}; available {sorted(qs)}")
            self.thresholds.append(qs[str(q)])
        self.rungs, self.top_shares, self.i_min, self.i_max, self.block_length = tuple(rungs), tuple(top_shares), i_min, i_max, block_length
        self.log: list[dict] = []
        self.reset_decode()

    # ------------------------------------------------------------------ state
    def reset_decode(self) -> None:
        self.prev = None            # (positions, tokens, probs, written positions) of call t-1
        self.prev2 = None           # (positions, tokens) of call t-2
        self.prev_kind = "dense"
        self.prev2_kind = "dense"
        self.calls_since_refresh = 0
        self.calls_since_dense = 0
        self.words_since_dense = 0
        self.words_block = 0
        self.block = None

    def new_block(self, block: int) -> None:
        if block != self.block:
            self.block, self.words_block = block, 0

    # ------------------------------------------------------------------ features and risk
    def _features(self, n_window: int) -> dict[str, float]:
        f = {"j": float(self.calls_since_dense + 1), "words_since_dense": float(self.words_since_dense), "progress": self.words_block / self.block_length, "n_window": float(n_window),
             "b": 1.0, "mean_p": 0.0, "min_p": 0.0, "n_written_prev": 0.0, "written_p_prev": 0.0}
        if self.prev is not None:
            pos, tok, prob, wr = self.prev
            f["mean_p"], f["min_p"], f["n_written_prev"] = sum(prob) / len(prob), min(prob), float(len(wr))
            w = [p for q, p in zip(pos, prob) if q in set(wr)]
            f["written_p_prev"] = sum(w) / len(w) if w else 0.0
            if self.prev2 is not None and self.prev_kind == "none" and self.prev2_kind == "none":       # a clean pair of cached calls (a refresh in between would contaminate the shift)
                p2 = dict(zip(self.prev2[0], self.prev2[1]))
                common = [(t, p2[q]) for q, t in zip(pos, tok) if q in p2]
                if common:
                    f["b"] = sum(a == b for a, b in common) / len(common)
        return f

    def risk(self, n_window: int) -> float:
        f = self._features(n_window)
        z = self.intercept + sum(c * (f[n] - m) / s for n, c, m, s in zip(self.features, self.coef, self.mean, self.scale))
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------------------------------------------ decide / observe
    def decide(self, n_window: int, forced_dense: bool = False) -> tuple[str | None, float]:
        """Before the forward of a call that is not already dense: None (no refresh) or a rung. Returns (action, risk)."""
        if forced_dense or self.prev is None:
            return None, float("nan")
        r = self.risk(n_window)
        if self.calls_since_refresh + 1 < self.i_min:
            return None, r
        action = None
        for rung, thr in zip(self.rungs, self.thresholds):
            if r >= thr:
                action = rung
        if action is None and self.calls_since_refresh + 1 >= self.i_max:
            action = self.rungs[0]
        return action, r

    def observe(self, window_pos: list[int], tokens: list[int], probs: list[float], written: list[int], kind: str, call: int, risk: float | None = None) -> None:
        """After the word lock wrote at this call. kind in {first, dense, blank, all, none} (what this call was)."""
        dense = kind in ("first", "dense")
        self.log.append({"call": call, "kind": kind, "risk": None if risk is None or risk != risk else round(risk, 4), "since_refresh": self.calls_since_refresh + 1})
        self.prev2, self.prev2_kind = (self.prev[0], self.prev[1]) if self.prev is not None else None, self.prev_kind
        self.prev, self.prev_kind = (list(window_pos), list(tokens), list(probs), list(written)), kind
        self.words_block += len(written)
        if dense:
            self.calls_since_dense, self.words_since_dense, self.calls_since_refresh = 0, len(written), 0
        else:
            self.calls_since_dense += 1
            self.words_since_dense += len(written)
            self.calls_since_refresh = 0 if kind in ("blank", "all") else self.calls_since_refresh + 1
