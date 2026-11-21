"""Word-lock rules for the X-block arm of paper_decode: write the blanks of the current block whose rule score clears a frozen threshold.

A rule maps the features of the blanks in the current block to one score per blank; a blank is written iff score >= lam (at least the best-scoring blank is
always written). Kinds: "conf" (score = confidence; with lam = c this reproduces C-block), "simple" (estimated precision by confidence within five
distance-to-written-text bins, one isotonic table per bin) and "gbm" (a fitted sklearn classifier on a named feature list). Feature definitions match the
B0 trace collector (locking_policy._top_features) so a rule fit on B0 traces sees the same quantities at decode time:
  confidence        top-1 probability of the blank's proposal
  entropy, margin   entropy of the full next-token distribution; top-1 minus top-2 probability
  dist_written      positions to the nearest answer token already written (cap = gen_length when none)
  local_mask_ratio  share of masked positions in a centred +-cpv_window window of the whole sequence (prompt counts as not masked)
  pos_in_block      share of the current block already written (B0: calls made in the block / block_length), clamped to >= 1/block_length
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

DIST_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10**9)]
CPV_WINDOW = 4
KINDS = {"conf", "simple", "gbm"}
FEATURES = {"confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"}
FEATURES |= {"run_words", "flip", "kl_prev", "nbr_conf_l", "nbr_conf_r", "nbr_cand_l", "nbr_cand_r", "same_l", "same_r", "p_eos", "tok_newline", "tok_space", "tok_punct",
             "dist_leftmost", "rank_left"}      # lock_features.EXTRA_FEATURES (names listed here to keep this module import-free; a test checks the two lists agree)
FEATURES |= {"emb_spread", "emb_velocity", "hid_velocity", "att_masked", "att_cand", "n_cand", "ll_agree", "ll_p", "ll_kl"}      # lock_features2.FEATURES2 (same arrangement, same test)
FEATURES |= {"h_in_prompt", "h_in_written", "h_bigram_prompt", "h_bigram_written", "h_num_novel", "i_digit", "i_after_eq", "i_after_op", "i_in_math", "i_brace_depth",
             "k_min_w2", "k_mean_w4", "k_run", "k_written_w4", "j_rank_frac", "j_conf_minus_med", "j_block_written"}      # lock_features3.FEATURES3 (same arrangement, same test)


@dataclass
class ContextRule:
    kind: str
    lam: float
    features: list[str] = field(default_factory=list)
    payload: Any = None   # simple: {bin index: (x_thresholds, y_thresholds)}; gbm: a fitted sklearn estimator with predict_proba
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown rule kind {self.kind}")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must lie in [0,1]")
        bad = set(self.features) - FEATURES
        if bad:
            raise ValueError(f"unsupported features {sorted(bad)}")

    def scores(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        conf = feats["confidence"].astype(np.float64)
        if self.kind == "conf":
            return conf
        if self.kind == "simple":
            out = np.zeros_like(conf)
            dist = feats["dist_written"]
            for k, (lo, hi) in enumerate(DIST_BINS):
                m = (dist >= lo) & (dist <= hi)
                if m.any():
                    x, y = self.payload[k]
                    out[m] = np.interp(conf[m], x, y)
            return out
        cols = np.stack([feats[f] for f in self.features], axis=1).astype(np.float32)
        return self.payload.predict_proba(cols)[:, 1]

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "ContextRule":
        return pickle.loads(Path(path).read_bytes())


def simple_payload(isotonic_by_bin: dict[int, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Breakpoints of fitted sklearn IsotonicRegression objects (np.interp reproduces predict inside the fitted range, clipped outside)."""
    return {k: (np.asarray(iso.X_thresholds_, dtype=np.float64), np.asarray(iso.y_thresholds_, dtype=np.float64)) for k, iso in isotonic_by_bin.items()}


@torch.no_grad()
def block_features(x: torch.Tensor, prompt_length: int, eligible: torch.Tensor, eligible_conf: torch.Tensor, eligible_logp: torch.Tensor,
                   mask_id: int, block_length: int, gen_length: int, cpv_window: int = CPV_WINDOW) -> dict[str, np.ndarray]:
    """Features of the blanks `eligible` (absolute positions, ascending, all inside one block) from the sequence state at the start of the call.
    eligible_logp: [n, vocab] log-probabilities of those blanks (full vocabulary, as the collector uses them)."""
    probs = eligible_logp.exp()
    entropy = -(probs * eligible_logp).sum(-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    length = x.shape[1]
    masked = (x[0] == mask_id).float()
    cs = torch.cat([masked.new_zeros(1), masked.cumsum(0)])
    start = (eligible - cpv_window).clamp_min(0)
    end = (eligible + cpv_window + 1).clamp_max(length)
    local = (cs[end] - cs[start]) / (end - start).float()
    answer = x[0, prompt_length:]
    written = torch.nonzero(answer != mask_id, as_tuple=False).flatten()
    rel = eligible - prompt_length
    if written.numel():
        dist = (rel.unsqueeze(1) - written.unsqueeze(0)).abs().min(dim=1).values.float()
    else:
        dist = torch.full(rel.shape, float(gen_length), device=x.device)
    # the block's blanks are all masked; written-in-block count = block_length - number of blanks left (one-word-per-call B0 convention)
    pos_in_block = max(block_length - int(eligible.numel()), 1) / block_length
    out = {"confidence": eligible_conf.float(), "entropy": entropy, "margin": margin, "dist_written": dist, "local_mask_ratio": local,
           "pos_in_block": torch.full_like(dist, pos_in_block)}
    return {k: v.detach().cpu().numpy() for k, v in out.items()}
