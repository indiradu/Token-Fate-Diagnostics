"""Selective K/V caching for a diffusion LM: Elastic-Cache's three ingredients plus our own row-selective refresh, inside our own decoder.

REPRESENTATION FREEZE side only (which rows have their K/V cached and skipped); the WORD LOCK here is the official Elastic-Cache one (Conf-threshold decoding
inside a sliding window of the next W blank rows, optional first-EOS truncation). Everything is switchable in ``SelectiveConfig``:

 1. ``window`` / ``far_mask_cache``: the call computes (query rows) = tracked rows + rows written on the previous call + the next ``window`` blank rows;
    every other row (prompt, older written rows, blanks beyond the window) keeps the K/V of its last refresh. ``far_mask_cache=False`` computes every blank row.
 2. ``eos_truncation``: after the first EOS is written, blanks beyond it are dropped (leftover blanks at the end become EOS).
 3. ``refresh="trigger"``: the adaptive layerwise trigger. Per layer, shallow to deep, the tracked rows' attention now is compared with their attention on the
    previous call (``trigger_cos`` = "elastic_heads": cosine across the head axis, exactly what upstream computes; "keys": across the key axis). If the mean
    similarity is below ``gamma`` at layer l, every remaining row joins the computation at layer l+1 (from its cached layer input) and all caches deeper than l are refreshed.
 4. ``refresh="selective"`` (ours): at ``score_layer`` the exposure of every cached row is scored (attention mass the row's CACHED query puts on positions written
    since the row's last refresh, accumulated over calls); rows above ``exposure_theta`` join at the next layer, all other rows stay cached.
 Other schedules: "every_call" (a dense call, window decode: the control), "block" (a dense call whenever the leading blank enters a new block of
 ``block_length``), "none" (dense first call only).

The engine runs each LLaDA layer by hand (like ``PaperReferenceRuntime._sparse_forward``) because the computed row set changes MID-forward: rows join at the trigger layer.
Layer-input (x), query, key and value caches are kept per layer as upstream does. Refresh semantics follow upstream: a trigger at layer l makes layer l+1 start from
the cached layer-l+1 inputs of the non-query rows, so their layer-(l+1) K/V are unchanged and the K/V actually refreshed start at layer l+2 (``refresh_from`` below
is the first layer at which rows join; ``refresh="every_call"`` is refresh_from = 0 and equals a dense forward).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from regret_remasking.locking_reference import _apply_rope_at_positions, transformer_blocks

EOS_ID = 126081
MASK_ID = 126336


@dataclass(frozen=True)
class SelectiveConfig:
    gen_length: int = 256
    threshold: float = 0.9              # confidence word lock inside the window (official: min(threshold, max confidence) so one row is always written)
    window: int = 16                    # blank rows decoded per call (official window_length)
    far_mask_cache: bool = True         # False: all blank rows are computed every call (official block_caching=False)
    eos_truncation: bool = True
    refresh: str = "trigger"            # trigger | selective | every_call | block | none
    gamma: float = 0.9
    trigger_cos: str = "elastic_heads"  # elastic_heads | keys
    track_num: int = 1
    block_length: int = 32              # refresh == "block"
    attn_impl: str = "sdpa"             # sdpa (the model's fused kernel) | manual (upstream's explicit bf16 softmax)
    head_rows: str = "window"           # window | all (logits of every row; parity debugging only)
    mask_id: int = MASK_ID
    eos_id: int = EOS_ID
    # row-selective refresh (refresh == "selective")
    score_layer: int = 8                # exposure is scored on this layer's cached queries and the keys it just updated
    exposure_theta: float = 0.5         # accumulated exposure above which a cached row rejoins at score_layer + 1
    score_heads: int = 8                # heads averaged in the exposure score (cost control; 0 = all)
    track_always: bool = False          # also compute tracked rows in schedules that never use the trigger (default: only when needed)
    force_trigger_layer: int = -1       # tests: act as if the trigger fired at this layer on every call after the first (-1 = off)
    # Query shape / word lock (default = upstream's: window of `window` blanks, Elastic's confidence rule)
    shape: str = "window"               # window: next `window` blanks (S16) | block: the blanks of the current fixed block of `block_length` (S32, Fast-dLLM dual-cache-like)
    lock: str = "elastic"               # elastic: upstream get_decoded_token_confident | conf: paper_decode's C-block rule (mask-suppressed proposal, float32) | rule: a ContextRule (X-block)
    rule: Any = None                    # lock == "rule": a context_rules.ContextRule
    refresh_every_k: int = 0            # > 0: with refresh == 'block', ALSO a dense call every k-th call (diagnostic: refresh more often than every block); 0 = off
    block_rows: bool = False            # lock == 'rule' with a block-vector rule (Learn2PD): also compute the rows of EVERY slot of the current block each call (<= 32 rows) so the filter sees real confidences
    # Shared staleness state (one per-row score, read by the refresh AND the word lock). Defaults = off, so every earlier cell is reproduced exactly.
    #   score_i = prior(type_i) * sum_p [token p first appeared after row i's last refresh] * 1/(1 + |p - i| / stale_tau)   ("full": how much of the context near row i changed since i was computed)
    #   "age": score_i = calls since row i was last computed.  "random": the refresh rows are uniformly random cached rows (control; same budget).
    stale_budget: int = 0               # > 0: on every cached call also recompute the `stale_budget` highest-score cached rows (a true refresh at every layer)
    stale_score: str = "full"           # full | age | random
    stale_prior: tuple = (1.0, 0.25, 0.25)   # row-type prior (blank, written, prompt): blank rows drift 3 to 10 times more per call than written or prompt rows
    stale_tau: float = 32.0
    stale_seed: int = 0                 # stale_score == "random"
    lock_stale_beta: float = 0.0        # > 0: the word lock's cutoff is raised by beta * (mean score of the cached blank rows) on cached calls (the lock reads the same state)
    log_stale: bool = False             # also record the staleness level per call (and the full vector under log_detail) even when nothing reads it
    refresh_on_newline: bool = False    # refresh == 'block': ALSO a dense call after a call that wrote a token containing a newline (structure-event refresh)
    newline_token_ids: tuple = ()
    log_calls: bool = True              # cheap per-call counters (rows per layer, refresh kind, joined rows)
    log_detail: bool = True             # also written positions/tokens, query sets, tracked rows, similarities (device->host copies; off for timed runs)


@dataclass
class SelectiveResult:
    tokens: list[int]
    nfe: int
    latency_s: float
    prompt_length: int
    rows_per_call: list[int] = field(default_factory=list)          # query rows at layer 0 per call
    layer_rows: list[list[int]] = field(default_factory=list)       # rows computed per layer per call
    refresh_layers: list[int] = field(default_factory=list)         # per call: first layer rows joined (-1 none; 0 dense call)
    refresh_kind: list[str] = field(default_factory=list)           # per call: first / dense / trigger / selective / none
    written: list[list[int]] = field(default_factory=list)          # per call: absolute positions written
    written_tokens: list[list[int]] = field(default_factory=list)
    query_sets: list[list[int]] = field(default_factory=list)       # per call: initial query positions (tracked, new, window)
    tracked: list[list[int]] = field(default_factory=list)          # per call: union of tracked rows chosen on this call (used on the next)
    sims: list[list[float]] = field(default_factory=list)           # per call: trigger similarity per layer checked
    joined_rows: list[int] = field(default_factory=list)            # per call: rows that joined mid-forward
    eos_position: int = -1
    final_mask_count: int = 0
    rule_time_s: float = 0.0            # lock == rule: features + rule scoring (device-to-host included)
    fallback_calls: int = 0             # calls on which no blank passed the lock and the best blank was written
    written_total: int = 0
    stale_level: list[float] = field(default_factory=list)          # per call: mean score of the cached blank rows (0 on dense calls)
    stale_rows_n: list[int] = field(default_factory=list)           # per call: rows recomputed because of their staleness score
    lock_shift: list[float] = field(default_factory=list)           # per call: amount added to the word lock's cutoff
    stale_scores: list[list[float]] = field(default_factory=list)   # log_detail only: the score of every position at the start of the call
    newline_dense_calls: int = 0


def _modules(model: torch.nn.Module):
    tr = model.model.transformer
    head = getattr(tr, "ff_out", None)
    return tr.wte, tr.ln_f, head


class SelectiveEngine:
    """Per-layer caches and a forward whose computed row set can grow mid-forward."""

    def __init__(self, model: torch.nn.Module, total_len: int, config: SelectiveConfig) -> None:
        self.model, self.cfg, self.T = model, config, int(total_len)
        self.blocks = transformer_blocks(model)
        self.L = len(self.blocks)
        self.wte, self.ln_f, self.head = _modules(model)
        self.device = next(model.parameters()).device
        self.x_cache: list[torch.Tensor | None] = [None] * self.L
        self.q_cache: list[torch.Tensor | None] = [None] * self.L
        self.k_cache: list[torch.Tensor | None] = [None] * self.L
        self.v_cache: list[torch.Tensor | None] = [None] * self.L
        self.track_position = torch.empty(0, dtype=torch.long, device=self.device)
        # exposure state (selective refresh)
        self.token_since = torch.zeros(self.T, dtype=torch.long, device=self.device)    # call at which each position's current token first appeared as input
        self.snap = torch.zeros(self.T, dtype=torch.long, device=self.device)           # call of each row's last refresh
        self.exposure = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.call = 0
        self._prox: torch.Tensor | None = None

    # ------------------------------------------------------------------ shared staleness state
    @torch.no_grad()
    def stale_scores(self, x_row: torch.Tensor, prompt_length: int) -> torch.Tensor:
        """Score (T,) float64 of every position at the start of the current call (before its forward). snap = call of the row's last refresh, token_since = call at
        which each position's token first appeared as input; a position written after row i's last refresh is a change row i has not seen."""
        cfg, T = self.cfg, self.T
        age = (self.call - self.snap).to(torch.float64)
        if cfg.stale_score == "age":
            return age
        if self._prox is None:
            pos = torch.arange(T, device=self.device, dtype=torch.float32)
            self._prox = 1.0 / (1.0 + (pos[:, None] - pos[None, :]).abs() / float(cfg.stale_tau))
        changed = self.token_since[None, :] > self.snap[:, None]
        near = (changed.to(torch.float32) * self._prox).sum(-1).to(torch.float64)
        pb, pw, pp = cfg.stale_prior
        pos_i = torch.arange(T, device=self.device)
        prior = torch.where(pos_i < prompt_length, torch.full_like(near, pp), torch.where(x_row == cfg.mask_id, torch.full_like(near, pb), torch.full_like(near, pw)))
        return prior * near

    @torch.no_grad()
    def pick_stale(self, score: torch.Tensor, query: torch.Tensor, budget: int, dropped: torch.Tensor | None = None) -> torch.Tensor:
        """The `budget` cached positions (not in `query`) to recompute: highest score first, ties to the position nearer the front (closer to the decoding frontier);
        'random' draws them uniformly from the cached positions. Rows whose score is 0 (nothing changed since their refresh) are never picked by the scored modes.
        `dropped` (bool (T,)): blank rows beyond the first EOS (truncated; they end as EOS whatever happens) are never refreshed."""
        T = self.T
        cached = torch.ones(T, dtype=torch.bool, device=self.device)
        cached[query] = False
        if dropped is not None:
            cached &= ~dropped
        if self.cfg.stale_score == "random":
            cand = torch.nonzero(cached, as_tuple=False).flatten()
            g = torch.Generator().manual_seed(int(self.cfg.stale_seed) * 1_000_003 + int(self.call))
            perm = torch.randperm(int(cand.numel()), generator=g).to(self.device)
            return cand[perm[:budget]].sort().values
        pos = torch.arange(T, device=self.device, dtype=torch.float64)
        rank = score + 1e-9 * (T - pos) / T
        rank = torch.where(cached & (score > 0), rank, torch.full_like(rank, -1.0))
        k = min(int(budget), int((rank >= 0).sum()))
        if k <= 0:
            return query[:0]
        return rank.topk(k).indices.sort().values

    @torch.no_grad()
    def stale_level(self, score: torch.Tensor, x_row: torch.Tensor, query: torch.Tensor, dropped: torch.Tensor | None = None) -> float:
        """Mean score of the cached blank rows (those the next blocks' words will be conditioned on): the one number the word lock reads."""
        cached_blank = (x_row == self.cfg.mask_id)
        cached_blank[query] = False
        if dropped is not None:
            cached_blank &= ~dropped
        return float(score[cached_blank].mean()) if bool(cached_blank.any()) else 0.0

    # ------------------------------------------------------------------ helpers
    def _project(self, block, x_normed):
        if hasattr(block, "att_proj"):
            q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
        else:
            q, k, v = block.q_proj(x_normed), block.k_proj(x_normed), block.v_proj(x_normed)
        if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
            q = block.q_norm(q).to(dtype=k.dtype)
            k = block.k_norm(k).to(dtype=k.dtype)
        return q, k, v

    def _ffn(self, block, x):
        residual = x
        ff = block.ff_norm(x)
        if hasattr(block, "up_proj"):
            ff, up = block.ff_proj(ff), block.up_proj(ff)
            ff = block.act(ff) * up
        else:
            ff = block.act(block.ff_proj(ff))
        return residual + block.dropout(block.ff_out(ff))

    @staticmethod
    def _weights(q_rows, k_full, scale):
        """Upstream's attention weights for the given query rows: bf16 q @ k^T * scale, softmax (same dtype as the inputs)."""
        return torch.softmax(q_rows @ k_full.transpose(-2, -1) * scale, dim=-1)

    def _cos(self, past, now):
        axis = 1 if self.cfg.trigger_cos == "elastic_heads" else -1
        return F.cosine_similarity(past, now, dim=axis).mean()

    # ------------------------------------------------------------------ one forward
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, rows: torch.Tensor, n_track: int, dec_local: torch.Tensor, mass_local: torch.Tensor, blank_mask: torch.Tensor, *,
                refresh_from: int | None = None, fresh: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """tokens (1,T) current ids; rows = initial computed positions ordered [tracked (first n_track), ...]; blank_mask (T,) bool: every blank row at this call.
        dec_local: indices INTO ``rows`` of the rows whose logits are returned (the decoding window). mass_local: indices into ``rows`` whose attention defines the
        tracked rows (upstream: the blank query rows; the window when far blanks are cached).

        fresh=True: first call or a dense call: every layer recomputes all rows and refreshes its caches. refresh_from=l: all non-query rows join at layer l (tests).
        Returns (logits (1, len(dec_local), V), info)."""
        cfg, T = self.cfg, self.T
        nq = int(rows.numel())
        win_local, trk_local = dec_local, torch.arange(0, n_track, device=self.device)
        x = self.wte(tokens[:, rows])
        cur_rows = rows
        info: dict[str, Any] = {"layer_rows": [], "refresh_layer": -1, "sims": [], "joined": 0, "tracked": None}
        join_at = None if fresh else refresh_from
        all_in = bool(fresh) or nq == T      # every row already computed (no further trigger / joins)
        track_tokens: list[torch.Tensor] = []
        want_track = cfg.refresh in {"trigger"} or cfg.track_always
        selective_pending: torch.Tensor | None = None
        for l, block in enumerate(self.blocks):
            # rows joining at this layer (decided on the previous layer)
            if not all_in and ((join_at is not None and l == join_at) or selective_pending is not None):
                if selective_pending is not None:
                    joiners = selective_pending
                    selective_pending = None
                else:
                    in_set = torch.zeros(T, dtype=torch.bool, device=self.device)
                    in_set[cur_rows] = True
                    joiners = torch.nonzero(~in_set, as_tuple=False).flatten()
                if joiners.numel():
                    self.x_cache[l][:, cur_rows] = x                       # layer inputs of the rows already computed
                    x = torch.cat([x, self.x_cache[l].index_select(1, joiners)], dim=1)
                    cur_rows = torch.cat([cur_rows, joiners])
                    info["joined"] += int(joiners.numel())
                    if info["refresh_layer"] < 0:
                        info["refresh_layer"] = l
                    if int(cur_rows.numel()) == T:
                        all_in = True
                if join_at is not None and l == join_at:
                    join_at = None
            n_cur = int(cur_rows.numel())
            xn = block.attn_norm(x)
            q, k, v = self._project(block, xn)
            nh, nkv = block.config.n_heads, block.config.effective_n_kv_heads
            hd = q.shape[-1] // nh
            q = q.view(1, n_cur, nh, hd).transpose(1, 2)
            k = k.view(1, n_cur, nkv, hd).transpose(1, 2)
            v = v.view(1, n_cur, nkv, hd).transpose(1, 2)
            rotary = getattr(block, "rotary_emb", None)
            if rotary is not None:
                q = _apply_rope_at_positions(rotary, q, cur_rows, T)
                k = _apply_rope_at_positions(rotary, k, cur_rows, T)
            scale = 1.0 / math.sqrt(hd)
            if self.k_cache[l] is None:                                  # first call: allocate
                self.k_cache[l] = torch.zeros(1, nkv, T, hd, dtype=k.dtype, device=self.device)
                self.v_cache[l] = torch.zeros_like(self.k_cache[l])
                self.q_cache[l] = torch.zeros(1, nh, T, hd, dtype=q.dtype, device=self.device)
                self.x_cache[l] = torch.zeros(1, T, x.shape[-1], dtype=x.dtype, device=self.device)
            kc, vc, qc, xc = self.k_cache[l], self.v_cache[l], self.q_cache[l], self.x_cache[l]
            # trigger comparison needs the previous call's attention of the tracked rows (cached q and k BEFORE this call's update)
            past_att = None
            check_trigger = (not all_in and cfg.refresh == "trigger" and n_track > 0 and join_at is None and not fresh)
            if check_trigger:
                past_q = qc[:, :, rows[:n_track]]
                past_att = self._weights(past_q, kc, scale)
            xc[:, cur_rows] = x
            kc.index_copy_(2, cur_rows, k)
            vc.index_copy_(2, cur_rows, v)
            qc.index_copy_(2, cur_rows, q)
            k_full, v_full = kc, vc
            if cfg.attn_impl == "manual":
                aw = self._weights(q, k_full, scale)
                att = aw @ v_full
            else:
                att = block._scaled_dot_product_attention(q, k_full, v_full, attn_mask=None)
                aw = None
            if want_track:
                # rows whose attention defines the tracked rows are among the INITIAL query rows (joiners are appended after them, local indices unchanged)
                waw = aw[:, :, mass_local, :] if aw is not None else self._weights(q[:, :, mass_local, :], k_full, scale)
                if check_trigger:
                    now = aw[:, :, trk_local, :] if aw is not None else self._weights(q[:, :, trk_local, :], k_full, scale)
                    sim = self._cos(past_att, now)
                    info["sims"].append(float(sim))
                    if bool(sim < cfg.gamma):
                        join_at = l + 1
                        if info["refresh_layer"] < 0:
                            info["trigger_layer"] = l
                if cfg.force_trigger_layer == l and join_at is None and not all_in and not fresh:
                    join_at = l + 1
                masked_w = waw.sum(dim=(0, 1, 2))
                masked_w[blank_mask] = 0.0
                track_tokens.append(masked_w.topk(k=cfg.track_num, dim=0, largest=True)[1])
            elif cfg.force_trigger_layer == l and join_at is None and not all_in and not fresh:
                join_at = l + 1
            # ---- our row-selective refresh: score every cached row at score_layer; the exposed ones join at l + 1
            if cfg.refresh == "selective" and not all_in and l == cfg.score_layer and not fresh:
                in_set = torch.zeros(T, dtype=torch.bool, device=self.device)
                in_set[cur_rows] = True
                cand = torch.nonzero(~in_set, as_tuple=False).flatten()
                if cand.numel():
                    hs = nh if cfg.score_heads in (0, nh) else min(cfg.score_heads, nh)
                    qs = qc[:, :hs].index_select(2, cand)                                  # cached queries of the cached rows
                    w = torch.softmax(qs.float() @ k_full[:, :hs].float().transpose(-2, -1) * scale, dim=-1).mean(1)[0]      # (cand, T)
                    written_since = self.token_since[None, :] > self.snap[cand][:, None]      # column p changed after the row's last refresh
                    mass = (w * written_since).sum(-1)
                    self.exposure[cand] += mass
                    sel = cand[self.exposure[cand] > cfg.exposure_theta]
                    if sel.numel():
                        selective_pending = sel
            att = att.transpose(1, 2).contiguous().view(1, n_cur, -1)
            x = x + block.dropout(block.attn_out(att))
            x = self._ffn(block, x)
            info["layer_rows"].append(n_cur)
        info["tracked"] = torch.cat(track_tokens).unique() if track_tokens else None
        win_x = x[:, win_local] if cfg.head_rows == "window" else x
        h = self.ln_f(win_x)
        logits = self.head(h) if self.head is not None else F.linear(h, self.wte.weight)
        if cfg.head_rows == "all":
            logits = logits[:, win_local]
        info["final_rows"] = cur_rows
        self.snap[cur_rows] = self.call                     # last-refresh call of every computed row (all schedules: the staleness state)
        if cfg.refresh == "selective":
            self.exposure[cur_rows] = 0.0
        self.call += 1
        return logits, info


# ---------------------------------------------------------------------------------------------- decode loop (upstream's decode, switchable parts)
def _confident_write(logits: torch.Tensor, window_positions: torch.Tensor, x: torch.Tensor, threshold: float, eos_id: int):
    """Upstream get_decoded_token_confident: float64 softmax, write every window row with confidence >= min(threshold, max confidence)."""
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, tok = torch.max(p, dim=-1)
    keep = conf >= min(threshold, conf.max())
    pos = window_positions[keep[0]]
    tok_keep = tok[:, keep[0]]
    x[:, pos] = tok_keep
    return pos, tok_keep[0], pos[tok_keep.eq(eos_id)[0]]


@torch.no_grad()
def _lock_write(logits: torch.Tensor, eligible: torch.Tensor, x: torch.Tensor, cfg: SelectiveConfig, prompt_length: int, blanks_in_first_block: int, res: SelectiveResult,
                shift: float = 0.0):
    """paper_decode's word locks on the eligible blanks (all rows of `logits`): C-block rule (lock == conf) or a ContextRule (lock == rule), incl. the best-blank fallback."""
    from regret_remasking.paper_decode import _argmax_stable
    cur = logits.float()
    logp = F.log_softmax(cur, dim=-1)
    proposal = cur.clone()
    if 0 <= cfg.mask_id < proposal.shape[-1]:
        proposal[..., cfg.mask_id] = -torch.inf
    proposed = proposal.argmax(dim=-1)
    conf = logp.gather(-1, proposed.unsqueeze(-1)).squeeze(-1).exp()[0]
    tokens = proposed[0]
    fallback = False
    if cfg.lock == "conf":
        selected = eligible[conf >= cfg.threshold + shift]
        if selected.numel() == 0:
            selected = _argmax_stable(conf, eligible)
            fallback = True
    else:
        from regret_remasking.context_rules import block_features
        t0 = time.perf_counter()
        feats = block_features(x, prompt_length, eligible, conf, logp[0], cfg.mask_id, cfg.block_length, cfg.gen_length)
        if cfg.shape == "window":      # the rule's block feature = share of the 32-block of the leftmost blank already written (B0 convention), not of the window
            feats["pos_in_block"] = feats["pos_in_block"] * 0 + max(cfg.block_length - blanks_in_first_block, 1) / cfg.block_length
        scores = cfg.rule.scores(feats)
        accept = torch.as_tensor(scores >= min(cfg.rule.lam + shift, 1.0), device=x.device)
        selected = eligible[accept]
        if selected.numel() == 0:
            selected = _argmax_stable(torch.as_tensor(scores, device=x.device, dtype=torch.float32), eligible)
            fallback = True
        res.rule_time_s += time.perf_counter() - t0
    idx = torch.searchsorted(eligible, selected)
    chosen = tokens.index_select(0, idx)
    x[0, selected] = chosen
    res.fallback_calls += int(fallback)
    return selected, chosen, selected[chosen.eq(cfg.eos_id)]


@torch.no_grad()
def _lock_write_l2pd(logits: torch.Tensor, eligible: torch.Tensor, block_start: int, x: torch.Tensor, cfg: SelectiveConfig, res: SelectiveResult):
    """Learn2PD word lock (l2pd_rule.Learn2PDRule): logits are those of EVERY slot of the current block (in block order); the filter sees all 32 confidences and writes the
    blanks it accepts plus the most confident blank. Confidence/proposal conventions as in paper_decode (mask-suppressed argmax, float32 softmax)."""
    cur = logits.float()
    logp = F.log_softmax(cur, dim=-1)
    proposal = cur.clone()
    if 0 <= cfg.mask_id < proposal.shape[-1]:
        proposal[..., cfg.mask_id] = -torch.inf
    proposed = proposal.argmax(dim=-1)
    conf = logp.gather(-1, proposed.unsqueeze(-1)).squeeze(-1).exp()[0]
    if conf.device.type == "cuda":
        torch.cuda.synchronize(conf.device)          # the timed region below is the filter, not the forward pass still in flight
    t0 = time.perf_counter()
    slots = eligible - block_start
    keep = cfg.rule.accept_mask(conf, slots)
    selected = eligible[keep]
    chosen = proposed[0].index_select(0, slots[keep])
    res.rule_time_s += time.perf_counter() - t0
    res.fallback_calls += int(cfg.rule.last_empty)
    x[0, selected] = chosen
    return selected, chosen, selected[chosen.eq(cfg.eos_id)]


@torch.no_grad()
def decode_selective(model: torch.nn.Module, prompt_ids: torch.Tensor, config: SelectiveConfig, *, engine: Any | None = None,
                     sync: bool = True) -> SelectiveResult:
    """Decode one prompt (ids shape (1, P)) with the selective-cache engine. ``engine`` may be injected (tests)."""
    cfg = config
    if cfg.lock == "rule" and getattr(cfg.rule, "needs_block_vector", False) and not cfg.block_rows:
        raise ValueError("a block-vector rule (Learn2PD) needs SelectiveConfig.block_rows=True")
    if cfg.stale_budget > 0 and not cfg.far_mask_cache:
        raise ValueError("stale_budget needs far_mask_cache=True (without it every blank row is computed anyway)")
    if cfg.stale_score not in {"full", "age", "random"}:
        raise ValueError(f"unknown stale_score {cfg.stale_score}")
    if cfg.refresh_on_newline and cfg.refresh != "block":
        raise ValueError("refresh_on_newline adds dense calls to refresh == 'block'")
    stale_on = cfg.stale_budget > 0 or cfg.lock_stale_beta > 0 or cfg.log_stale
    device = prompt_ids.device
    P = int(prompt_ids.shape[1])
    T = P + cfg.gen_length
    nl_ids = torch.tensor(list(cfg.newline_token_ids), dtype=torch.long, device=device)
    x = torch.full((1, T), cfg.mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    eng = engine if engine is not None else SelectiveEngine(model, T, cfg)
    if sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    all_pos = torch.arange(T, device=device)
    masked_position = all_pos[P:].clone()
    new_decoded = all_pos[:0].clone()
    track_position = all_pos[:0].clone()
    res = SelectiveResult(tokens=[], nfe=0, latency_s=0.0, prompt_length=P)
    decoded_eos = False
    block_of_first = -1
    call = 0
    while True:
        first_blank = int(masked_position[0])
        blk = (first_blank - P) // cfg.block_length
        if cfg.shape == "block":
            window_pos = masked_position[((masked_position - P) // cfg.block_length) == blk]
        else:
            window_pos = masked_position[: cfg.window]
        mass_pos = window_pos if cfg.far_mask_cache else masked_position     # rows whose attention defines the tracked rows (upstream)
        dense_call = (call == 0 or cfg.refresh == "every_call" or (cfg.refresh == "block" and (blk != block_of_first or (cfg.refresh_every_k > 0 and call % cfg.refresh_every_k == 0))))
        nl_event = bool(cfg.refresh_on_newline and not dense_call and call > 0 and new_decoded.numel() and nl_ids.numel() and bool(torch.isin(x[0, new_decoded], nl_ids).any()))
        if nl_event:
            dense_call = True
            res.newline_dense_calls += 1
        block_of_first = blk
        if call > 0 and hasattr(eng, "token_since"):
            eng.token_since[new_decoded] = call
        l2pd = cfg.block_rows and cfg.lock == "rule" and getattr(cfg.rule, "needs_block_vector", False)
        if l2pd:
            if cfg.shape != "block" or cfg.refresh == "trigger" or not cfg.far_mask_cache:
                raise ValueError("block_rows (Learn2PD) needs shape='block', far_mask_cache=True and a non-trigger refresh schedule")
            block_start = P + blk * cfg.block_length
            block_all = all_pos[block_start:block_start + cfg.block_length]
            window_pos = masked_position[((masked_position - P) // cfg.block_length) == blk]      # the blanks (after EOS truncation)
        if dense_call and l2pd:
            rows = all_pos
            n_track = 0
            dec_local = mass_local = block_all
            kind = "first" if call == 0 else "dense"
        elif l2pd:
            outside = new_decoded[(new_decoded < block_start) | (new_decoded >= block_start + cfg.block_length)]
            rows = torch.cat([outside, block_all])
            n_track = 0
            dec_local = mass_local = outside.numel() + torch.arange(block_all.numel(), device=device)
            kind = None
        elif dense_call:
            rows = all_pos
            n_track = 0
            dec_local, mass_local = window_pos, mass_pos                    # local index == position when rows = arange(T)
            kind = "first" if call == 0 else "dense"
        else:
            use_track = cfg.refresh == "trigger" or cfg.track_always
            tr = track_position if use_track else all_pos[:0]
            n_track = int(tr.numel())
            if cfg.far_mask_cache:
                rows = torch.cat([tr, new_decoded, window_pos])
                dec_local = torch.arange(rows.numel() - window_pos.numel(), rows.numel(), device=device)
                mass_local = dec_local
            else:
                rows = torch.cat([tr, new_decoded, masked_position])
                start = n_track + int(new_decoded.numel())
                dec_local = start + torch.arange(window_pos.numel(), device=device)
                mass_local = start + torch.arange(masked_position.numel(), device=device)
            kind = None
        stale_level, stale_n, shift = 0.0, 0, 0.0
        if stale_on and not dense_call and not l2pd:
            sscore = eng.stale_scores(x[0], P)
            dropped = None
            if decoded_eos:                                                    # blanks beyond the first EOS are truncated: not candidates, not part of the level
                live = torch.zeros(T, dtype=torch.bool, device=device)
                live[masked_position] = True
                dropped = (x[0] == cfg.mask_id) & ~live
            if cfg.stale_budget > 0:
                stale_rows = eng.pick_stale(sscore, rows, cfg.stale_budget, dropped)
                stale_n = int(stale_rows.numel())
                if stale_n:
                    nw = int(window_pos.numel())
                    rows = torch.cat([rows[:-nw], stale_rows, rows[-nw:]])              # the decoding window stays last
                    dec_local = torch.arange(rows.numel() - nw, rows.numel(), device=device)
                    mass_local = dec_local
            stale_level = eng.stale_level(sscore, x[0], rows, dropped)
            shift = cfg.lock_stale_beta * stale_level
            if cfg.log_calls and cfg.log_detail:
                res.stale_scores.append(sscore.tolist())
        elif stale_on and cfg.log_calls and cfg.log_detail:
            res.stale_scores.append([])                                                # dense call: everything was just refreshed
        blank_mask = torch.zeros(T, dtype=torch.bool, device=device)
        blank_mask[masked_position] = True
        logits, info = eng.forward(x, rows, n_track, dec_local, mass_local, blank_mask, fresh=dense_call)
        if cfg.lock == "elastic":
            written, tok, eos_pos = _confident_write(logits, window_pos, x, cfg.threshold + shift, cfg.eos_id)
        elif l2pd:
            written, tok, eos_pos = _lock_write_l2pd(logits, window_pos, block_start, x, cfg, res)
        else:
            blanks_first_block = int((((masked_position - P) // cfg.block_length) == blk).sum())
            written, tok, eos_pos = _lock_write(logits, window_pos, x, cfg, P, blanks_first_block, res, shift)
        res.written_total += int(written.numel())
        new_decoded = written
        masked_position = masked_position[~torch.isin(masked_position, written)]
        res.nfe += 1
        if cfg.log_calls:
            res.rows_per_call.append(int(rows.numel()))
            res.layer_rows.append(list(info["layer_rows"]))
            if kind is None:
                kind = ("trigger" if cfg.refresh == "trigger" and info["refresh_layer"] >= 0 else "selective" if cfg.refresh == "selective" and info["refresh_layer"] >= 0 else "none")
            res.refresh_layers.append(int(info["refresh_layer"]) if not dense_call else 0)
            res.refresh_kind.append(kind)
            res.joined_rows.append(int(info["joined"]))
            res.stale_level.append(stale_level)
            res.stale_rows_n.append(stale_n)
            res.lock_shift.append(shift)
            if cfg.log_detail:
                res.written.append(written.tolist())
                res.written_tokens.append(tok.tolist())
                res.query_sets.append(rows.tolist())
                res.tracked.append(info["tracked"].tolist() if info["tracked"] is not None else [])
                res.sims.append(info["sims"])
        if info["tracked"] is not None:
            track_position = info["tracked"]
        if cfg.eos_truncation and not decoded_eos and eos_pos.numel() > 0:
            eos_min = int(eos_pos.min())
            decoded_eos = True
            res.eos_position = eos_min
            masked_position = masked_position[masked_position <= eos_min]
        if masked_position.numel() == 0:
            break
        call += 1
    x[x.eq(cfg.mask_id)] = cfg.eos_id
    if sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    res.latency_s = time.perf_counter() - t0
    res.tokens = x[0, P:].tolist()
    res.final_mask_count = 0
    return res
