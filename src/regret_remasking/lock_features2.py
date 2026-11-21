"""Model-internal decode-time inputs for a learned context word lock (Lctx): embedding geometry of the candidate set, attention of the blank's query, and a logit lens on a middle layer.

Pre-registration: results/lctx_features2_20261111/PREREGISTERED.md. Default off: nothing here runs unless ``SelectiveConfig.collect_probe`` or a ``lock == "rule"`` rule whose feature list
names one of FEATURES2 asks for it (see selective_cache._rule_features). The engine then records three probes in the forward's ``info`` (``probe_mid``: the eligible rows' hidden at the
input of block ``probe_mid_layer``; ``probe_attn``: head-mean float32 attention weights of the eligible rows' queries at layer ``probe_attn_layer`` over every position; ``probe_final``: the
eligible rows' final-norm output) and this module turns them into per-blank features. All features are per ELIGIBLE blank (the blanks the call computes), in eligible order.

Kinds as in lock_features: S = function of the state at the call, H = function of history (previous call), P = per-call quantity. "Previous observation" = eligible at the immediately preceding call.

  (e) continuous, S/H  emb_spread    sum_k p~_k (1 - cos(E_k, E_proposal)) over the K most probable tokens (p~ renormalised over the K); 0 if one token holds all mass; in [0, 2]
                       emb_velocity  1 - cos(mu_now, mu_prev), mu = sum_k p~_k E_k; NaN without previous observation
                       hid_velocity  ||h_now - h_prev|| / ||h_prev|| of the blank's final hidden state (relative L2, premise A's drift metric); NaN without previous observation
  (f) joint-write, P   att_masked    attention mass of the blank's query (layer probe_attn_layer, head mean) on masked positions (itself included)
                       att_cand      attention mass on the OTHER eligible blanks with confidence >= 0.5
                       n_cand        number of other eligible blanks with confidence >= 0.5
  (g) logit lens, P    ll_agree      1 if the mask-suppressed argmax of head(ln_f(hidden at block probe_mid_layer)) equals the final proposal
                       ll_p          probability that distribution gives the final proposal
                       ll_kl         KL(p_final || p_mid) in nats
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

GROUPS2: dict[str, list[str]] = {
    "e": ["emb_spread", "emb_velocity", "hid_velocity"],
    "f": ["att_masked", "att_cand", "n_cand"],
    "g": ["ll_agree", "ll_p", "ll_kl"],
}
FEATURES2: list[str] = [f for g in GROUPS2.values() for f in g]
CANDIDATE_CONFIDENCE = 0.5          # = lock_features.CANDIDATE_CONFIDENCE
TOPK = 32


@dataclass
class ProbeState:
    """Per-decode memory of the previous call's eligible blanks (positions ascending), on the device."""
    prev_call: int = -10
    prev_pos: Any = None
    prev_mu: Any = None               # (n, d) expected embedding over the top-K
    prev_h: Any = None                # (n, d) final hidden


def wants_probe(cfg) -> bool:
    """Probes are computed iff a collection asks for them or the rule's feature list names one (a six-input or a-d rule never triggers this)."""
    if getattr(cfg, "collect_probe", False):
        return True
    rule = getattr(cfg, "rule", None)
    return cfg.lock == "rule" and rule is not None and bool(set(getattr(rule, "features", []) or []) & set(FEATURES2))


def probe_layers(cfg, n_layers: int) -> tuple[int, int]:
    """(mid layer of the logit lens, attention layer): the config's values, or 3L/4 and L/2 when they are -1. mid == n_layers means the final norm's output (tests)."""
    mid = cfg.probe_mid_layer if cfg.probe_mid_layer >= 0 else (3 * n_layers) // 4
    att = cfg.probe_attn_layer if cfg.probe_attn_layer >= 0 else n_layers // 2
    if not 0 <= mid <= n_layers or not 0 <= att < n_layers:
        raise ValueError(f"probe layers out of range: mid {mid} (0..{n_layers}), attn {att} (0..{n_layers - 1})")
    return mid, att


@torch.no_grad()
def lens_logits(engine, h_mid: torch.Tensor) -> torch.Tensor:
    """The vocabulary head applied to the final norm of a hidden state (1, n, d) -> (n, V) float32 logits; the same path the engine's final logits take."""
    h = engine.ln_f(h_mid)
    out = engine.head(h) if engine.head is not None else F.linear(h, engine.wte.weight)
    return out[0].float()


@torch.no_grad()
def probe_features(state: ProbeState, engine, probe: dict, eligible: torch.Tensor, conf: torch.Tensor, logp: torch.Tensor, tokens: torch.Tensor, blank_mask: torch.Tensor,
                   call: int, mask_id: int, topk: int = TOPK) -> dict[str, np.ndarray]:
    """The nine features of the eligible blanks. probe: the forward's info (probe_mid (1, n, d), probe_attn (n, T), probe_final (1, n, d)); logp: (n, V) final log-probabilities;
    conf, tokens: (n,) confidence and proposal; blank_mask: (T,) every masked position at the call. Updates ``state`` (previous call)."""
    n, dev = int(eligible.numel()), eligible.device
    nan = torch.full((n,), float("nan"), device=dev)
    probs = logp.exp()
    E = engine.wte.weight
    # (e) embedding geometry over the top-K candidates
    k = min(int(topk), probs.shape[-1])
    pk, ik = probs.topk(k, dim=-1)
    pk = pk / pk.sum(-1, keepdim=True).clamp_min(1e-30)
    Ek = E.index_select(0, ik.flatten()).view(n, k, -1).float()
    Ep = E.index_select(0, tokens).float()
    cos = F.cosine_similarity(Ek, Ep.unsqueeze(1), dim=-1)                                   # (n, k)
    emb_spread = (pk.float() * (1.0 - cos)).sum(-1).clamp_(0.0, 2.0)
    mu = (pk.float().unsqueeze(-1) * Ek).sum(1)                                              # (n, d)
    h = probe["probe_final"][0].float()                                                       # (n, d)
    emb_vel, hid_vel = nan.clone(), nan.clone()
    if state.prev_pos is not None and state.prev_call == call - 1 and state.prev_pos.numel() > 0:
        idx = torch.searchsorted(state.prev_pos, eligible).clamp_max(state.prev_pos.numel() - 1)
        has = state.prev_pos[idx] == eligible
        ev = 1.0 - F.cosine_similarity(mu, state.prev_mu[idx], dim=-1)
        hp = state.prev_h[idx]
        hv = (h - hp).norm(dim=-1) / hp.norm(dim=-1).clamp_min(1e-12)
        emb_vel = torch.where(has, ev, nan)
        hid_vel = torch.where(has, hv, nan)
    # (f) attention of the blank's query
    A = probe["probe_attn"].float()                                                           # (n, T)
    att_masked = (A * blank_mask.float().unsqueeze(0)).sum(-1)
    cand = conf.float() >= CANDIDATE_CONFIDENCE
    cand_cols = torch.zeros(A.shape[-1], dtype=torch.float32, device=dev)
    cand_cols[eligible[cand]] = 1.0
    self_w = A.gather(1, eligible.unsqueeze(1)).squeeze(1)
    att_cand = (A * cand_cols.unsqueeze(0)).sum(-1) - self_w * cand.float()
    n_cand = (cand.sum() - cand.float()).float()
    # (g) logit lens at the middle layer
    lg = lens_logits(engine, probe["probe_mid"])                                              # (n, V)
    lp_mid = torch.log_softmax(lg, dim=-1)
    sup = lg.clone()
    if 0 <= mask_id < sup.shape[-1]:
        sup[:, mask_id] = -torch.inf
    ll_agree = (sup.argmax(-1) == tokens).float()
    ll_p = lp_mid.gather(1, tokens.unsqueeze(1)).squeeze(1).exp()
    ll_kl = (probs * (logp - lp_mid)).sum(-1).clamp_min(0.0)
    state.prev_call, state.prev_pos, state.prev_mu, state.prev_h = int(call), eligible.clone(), mu, h
    out = {"emb_spread": emb_spread, "emb_velocity": emb_vel, "hid_velocity": hid_vel, "att_masked": att_masked, "att_cand": att_cand.clamp_min(0.0), "n_cand": n_cand,
           "ll_agree": ll_agree, "ll_p": ll_p, "ll_kl": ll_kl}
    mat = torch.stack([out[k_].float() for k_ in FEATURES2]).cpu().numpy()
    return {k_: mat[i] for i, k_ in enumerate(FEATURES2)}
