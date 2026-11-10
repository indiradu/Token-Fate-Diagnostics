"""Extra decode-time inputs for a learned context word lock (Lctx), computed from the call's own logits, the sequence state and ONE stored copy of the previous call's log-probabilities.

Pre-registration: results/lctx_features_20261110/PREREGISTERED.md. Default off: nothing here runs unless ``SelectiveConfig.collect_extra`` or a ``lock == "rule"`` rule whose feature
list names one of EXTRA_FEATURES asks for it (see selective_cache._rule_features). All features are per ELIGIBLE blank (the blanks the call computes).

Notation: W = number of words written before this call (``SelectiveResult.written_total``), the proposal = the mask-suppressed argmax, confidence = its probability.
Kinds: S = function of the sequence state at the call (like the six original inputs); H = function of history, counted in WORDS written so that it equals the call count of the
one-word-per-call training traces; P = per-call quantity.

  (a) stability, H   run_words   words written since this blank's proposal last changed (0 if it changed since the previous call or there is no previous observation)
                     flip        1 if the argmax changed since the previous call, 0 if not, NaN if the blank has no observation at the previous call
                     kl_prev     KL(p_now || p_prev) in nats, divided by max(1, words written by the previous call) (approximation); NaN without previous observation
  (b) neighbours, S  nbr_conf_l / nbr_conf_r   confidence of the nearest ELIGIBLE blank on the left / right (NaN if none)
                     nbr_cand_l / nbr_cand_r   1 if that blank's confidence >= 0.5, else 0 (NaN if none)
                     same_l / same_r           1 if the proposal equals the token at i-1 / i+1 (written token, or the proposal of an eligible blank), NaN if that position is an uncomputed blank
  (c) proposal, P    p_eos       probability mass on EOS + EOT at this position
                     tok_newline / tok_space / tok_punct   class of the proposal token from a [vocab] int8 bit table (bit 0 newline, bit 1 starts with whitespace, bit 2 punctuation-only)
  (d) frontier, S    dist_leftmost   position distance to the leftmost eligible blank (the leftmost blank of the block in the block shape)
                     rank_left       number of eligible blanks to the left

"Previous observation" = the blank was eligible on the immediately preceding call (the first call of its block, or after a resume, has none). The state is the previous call only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

EOT_ID = 126348       # equals llada_trace.EOT_ID (tests check); not imported to keep this module dependency-free
GROUPS: dict[str, list[str]] = {
    "a": ["run_words", "flip", "kl_prev"],
    "b": ["nbr_conf_l", "nbr_conf_r", "nbr_cand_l", "nbr_cand_r", "same_l", "same_r"],
    "c": ["p_eos", "tok_newline", "tok_space", "tok_punct"],
    "d": ["dist_leftmost", "rank_left"],
}
EXTRA_FEATURES: list[str] = [f for g in GROUPS.values() for f in g]
CANDIDATE_CONFIDENCE = 0.5
TOKEN_CLASS_BITS = {"tok_newline": 1, "tok_space": 2, "tok_punct": 4}


@dataclass
class ExtraState:
    """Per-decode memory of the previous call's eligible blanks (positions ascending), kept on the device."""
    prev_call: int = -10
    prev_words: int = 0
    prev_pos: Any = None
    prev_tok: Any = None
    prev_logp: Any = None
    prev_start: Any = None            # W at the call where the blank's current identical-proposal stretch began
    token_class: Any = None           # device copy of the class table


def wants_extra(cfg) -> bool:
    """Extras are computed iff a collection asks for them or the rule's feature list names one (a rule with only the six inputs never triggers this)."""
    if getattr(cfg, "collect_extra", False):
        return True
    rule = getattr(cfg, "rule", None)
    return cfg.lock == "rule" and rule is not None and bool(set(getattr(rule, "features", []) or []) & set(EXTRA_FEATURES))


@torch.no_grad()
def extra_features(state: ExtraState, x: torch.Tensor, eligible: torch.Tensor, conf: torch.Tensor, logp: torch.Tensor, tokens: torch.Tensor, call: int, words_before: int,
                   mask_id: int, eos_id: int, token_class: Any) -> dict[str, np.ndarray]:
    """Extra inputs of the eligible blanks. x: (1, T) state at the start of the call; eligible: ascending absolute positions; conf, tokens: (n,) confidence and proposal;
    logp: (n, V) log-probabilities of those blanks; call: index of this call; words_before: words written before this call. Updates ``state`` (previous call) and returns float32 arrays of length n."""
    n, dev = int(eligible.numel()), eligible.device
    nan = torch.full((n,), float("nan"), device=dev)
    probs = logp.exp()
    conf = conf.float()
    # (a) stability
    start = torch.full((n,), int(words_before), dtype=torch.long, device=dev)
    flip, kl = nan.clone(), nan.clone()
    if state.prev_pos is not None and state.prev_call == call - 1 and state.prev_pos.numel() > 0:
        idx = torch.searchsorted(state.prev_pos, eligible).clamp_max(state.prev_pos.numel() - 1)
        has = state.prev_pos[idx] == eligible
        same = has & (tokens == state.prev_tok[idx])
        start = torch.where(same, state.prev_start[idx], start)
        flip = torch.where(has, (~same).float(), nan)
        words_gap = max(1, int(words_before) - int(state.prev_words))
        kl_all = (probs * (logp - state.prev_logp[idx])).sum(-1).clamp_min(0.0) / words_gap
        kl = torch.where(has, kl_all, nan)
    run = (int(words_before) - start).float()
    # (b) neighbours: nearest eligible blank on each side; token at i-1 / i+1
    nbr_l, nbr_r = nan.clone(), nan.clone()
    if n > 1:
        nbr_l[1:], nbr_r[:-1] = conf[:-1], conf[1:]
    cand = lambda v: torch.where(torch.isnan(v), nan, (v >= CANDIDATE_CONFIDENCE).float())
    full = x[0].clone()
    full[eligible] = tokens
    T = int(full.numel())
    left_tok = full[(eligible - 1).clamp_min(0)]
    right_tok = full[(eligible + 1).clamp_max(T - 1)]
    same_l = torch.where((left_tok != mask_id) & (eligible >= 1), (tokens == left_tok).float(), nan)
    same_r = torch.where((right_tok != mask_id) & (eligible + 1 < T), (tokens == right_tok).float(), nan)
    # (c) proposal
    p_eos = probs[:, eos_id] + (probs[:, EOT_ID] if EOT_ID < probs.shape[-1] else 0.0)       # (toy vocabularies are smaller than the real EOT id)
    if state.token_class is None:
        if token_class is None:
            raise ValueError("the token-class features need SelectiveConfig.token_class (an int8 [vocab] table; see results/lctx_features_20261110/orchestration/build_token_class.py)")
        state.token_class = torch.as_tensor(np.asarray(token_class), dtype=torch.int16, device=dev)
    tc = state.token_class[tokens]
    bit = lambda name: ((tc & TOKEN_CLASS_BITS[name]) != 0).float()
    # (d) frontier
    dist_left = (eligible - eligible[0]).float()
    rank = torch.arange(n, device=dev, dtype=torch.float32)
    out = {"run_words": run, "flip": flip, "kl_prev": kl, "nbr_conf_l": nbr_l, "nbr_conf_r": nbr_r, "nbr_cand_l": cand(nbr_l), "nbr_cand_r": cand(nbr_r), "same_l": same_l, "same_r": same_r,
           "p_eos": p_eos, "tok_newline": bit("tok_newline"), "tok_space": bit("tok_space"), "tok_punct": bit("tok_punct"), "dist_leftmost": dist_left, "rank_left": rank}
    state.prev_call, state.prev_words = int(call), int(words_before)
    state.prev_pos, state.prev_tok, state.prev_logp, state.prev_start = eligible.clone(), tokens.clone(), logp, start
    mat = torch.stack([out[k].float() for k in EXTRA_FEATURES]).cpu().numpy()
    return {k: mat[i] for i, k in enumerate(EXTRA_FEATURES)}
