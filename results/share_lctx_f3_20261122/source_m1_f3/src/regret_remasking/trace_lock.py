"""TraceLock (the published learned word lock, paper_trace.PaperTraceController over paper_trace.HiddenProjector) inside the selective-cache engine.

WORD LOCK only (which blanks are written per call). What the controller reads (paper_decode.decode_paper_policy, arms T-adaptive / T-block): for EVERY row of positions
[0, right) -- prompt rows, written rows and blank rows alike -- the three hidden states ``hidden_states[30], [31], [32]`` of the current forward (LLaDA: the input of
block 31, the input of block 32, and the final-normalised output), projected row by row to 832 features; the controller attends over all ``right`` rows and a blank is written
iff sigmoid(margin) > cutoff (else the single best blank). No time history.

Under a representation cache only the computed rows have a fresh forward. The projector is a pure row function, so this scorer keeps the PROJECTED features of every position in a
(T, 832) fp16 store: each call projects the computed rows below ``right`` and overwrites their features; rows the cache did not recompute keep the features of the last call that
computed them (prompt rows and older written rows: the block-opening dense call). In a dense decode (refresh = every_call) every row is recomputed, so the store equals the dense
features exactly. ``SelectiveEngine.forward`` exposes the three hidden tensors (``info["hid"]``) when ``SelectiveConfig.trace`` is set.
"""
from __future__ import annotations

import torch


class TraceScorer:
    def __init__(self, controller: torch.nn.Module, projector: torch.nn.Module, feature_dim: int = 832) -> None:
        self.controller, self.projector, self.feature_dim = controller, projector, int(feature_dim)
        self.controller.eval()
        self.projector.eval()
        self.store: torch.Tensor | None = None
        self.have: torch.Tensor | None = None
        self.last_proj_rows = 0

    def reset(self, total_len: int, device: torch.device | str) -> None:
        self.store = torch.zeros(int(total_len), self.feature_dim, dtype=torch.float16, device=device)
        self.have = torch.zeros(int(total_len), dtype=torch.bool, device=device)
        self.last_proj_rows = 0

    @torch.no_grad()
    def observe(self, hid, right: int) -> int:
        """hid = ((rows30, h30), (rows31, h31), (rows32, h32)), each h of shape (1, n, H) for the computed rows ``rows``. Project the rows below ``right`` and store their features."""
        (r30, h30), (r31, h31), (r32, h32) = hid
        if not (r30.shape == r31.shape == r32.shape):
            raise ValueError("the three hidden-state captures cover different row sets (trace needs refresh in {block, every_call})")
        idx = torch.nonzero(r30 < right, as_tuple=False).flatten()
        self.last_proj_rows = int(idx.numel())
        if idx.numel():
            feats = self.projector.encode_hidden((h30[0].index_select(0, idx), h31[0].index_select(0, idx), h32[0].index_select(0, idx)))
            rows = r30.index_select(0, idx)
            self.store[rows] = feats.to(torch.float16)
            self.have[rows] = True
        return self.last_proj_rows

    @torch.no_grad()
    def score(self, hid, right: int, eligible: torch.Tensor) -> torch.Tensor:
        """sigmoid(margin) of the eligible blanks (float32, aligned with ``eligible``), from the stored features of positions [0, right)."""
        self.observe(hid, right)
        if self.store is None:
            raise RuntimeError("TraceScorer.reset() was not called")
        if not bool(self.have[:right].all()):
            raise RuntimeError("the controller prefix contains a row that no call has computed (trace needs a dense call at every block start)")
        features = self.store[:right].float().unsqueeze(0)
        valid = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        position_ids = torch.arange(right, device=features.device).unsqueeze(0)
        scored = self.controller(features, valid_mask=valid, position_ids=position_ids)
        scores = torch.sigmoid(scored["margins"])[0].index_select(0, eligible)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("nonfinite learned commitment probabilities")
        return scores


def pick_threshold(target_calls: float, by_threshold: dict[float, float]) -> tuple[float, dict]:
    """Confidence threshold whose mean call count is nearest to ``target_calls`` among the evaluated thresholds (the E2 procedure: nearest on a grid; calls rise with the threshold).
    Returns (threshold, info). info["status"]: "bracketed" (the target lies inside the evaluated range) or "unmatched_low" / "unmatched_high" (the nearest end of the grid is used)."""
    if not by_threshold:
        raise ValueError("no thresholds evaluated")
    ts = sorted(by_threshold)
    lo, hi = min(by_threshold.values()), max(by_threshold.values())
    status = "bracketed" if lo <= target_calls <= hi else ("unmatched_low" if target_calls < lo else "unmatched_high")
    best = min(ts, key=lambda t: (abs(by_threshold[t] - target_calls), t))
    return best, {"status": status, "target_calls": target_calls, "chosen": best, "chosen_calls": by_threshold[best], "range_calls": [lo, hi]}


def bracket(target_calls: float, by_threshold: dict[float, float]) -> tuple[float, float] | None:
    """Adjacent evaluated thresholds (t_i, t_j) whose mean calls enclose the target; None if the target lies outside the evaluated range."""
    ts = sorted(by_threshold)
    for a, b in zip(ts, ts[1:]):
        lo, hi = sorted((by_threshold[a], by_threshold[b]))
        if lo <= target_calls <= hi:
            return a, b
    return None
