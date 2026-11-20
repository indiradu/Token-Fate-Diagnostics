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
    partial_every_k: int = 0            # > 0 (refresh == 'block' only): every k-th call after a dense call is a PARTIAL refresh: the selected cached rows join at layer `partial_layer` from their stored layer inputs
                                        # (layers below keep the cached K/V of those rows, layers from partial_layer on are recomputed and written back). 0 = off (default)
    partial_layer: int = 11             # first layer at which the joining rows are recomputed (0 = the same rows as a dense call from layer 0)
    partial_join: str = "all"           # which cached rows join: all | blank (live blank rows outside the query set) | context (prompt + written rows outside the query set)
    topk: int = 1                       # lock == 'topk' (NAIVE DIFFUSION): write the `topk` most confident eligible blanks every call, no threshold (topk = 1: one token per call = the repo's B0 baseline)
    refresh_every_words: int = 0        # > 0 (refresh == 'block' only): refresh when at least this many words have been WRITTEN since the last refresh (words written by the refreshing call itself count), in addition to the
                                        # block-start dense calls. A count of changes to the conditioning context, not of calls. 0 = off (default)
    words_refresh: str = "dense"        # what the word-count trigger does: dense (a dense call) | partial (a partial refresh with partial_layer / partial_join)
    # Relative-unit refresh family (results/../refresh_family.md). All default off; they ADD to partial_every_k / refresh_every_words (a partial refresh happens if any of them is due).
    partial_alpha: float = -1.0         # >= 0: the partial-refresh start layer is floor(alpha * L) (L = layers of the model) instead of the absolute partial_layer
    refresh_m: int = 0                  # > 0: m partial refreshes per block at progress points of the block (u = fraction of the block's slots already written); rows that join: partial_join
    refresh_schedule: str = "uniform"   # progress points of the m refreshes: uniform (i/(m+1)) | front (2**-(m+1-i): m=3 -> 1/8, 1/4, 1/2)
    refresh_gate: float = 0.0           # mu in [0, 1): suppress ANY partial refresh when the fraction of the block still blank (1 - u) <= mu. 0 = never suppress
    near_blocks: int = 0                # delta > 0: split the blank rows into NEAR (rest of the current block and the next delta blocks) and FAR (all later blanks); each group has its own m (below). Needs partial_join == 'blank'
    refresh_m_near: int = 0             # near_blocks > 0 only: refreshes per block of the near blank rows
    refresh_m_far: int = 0              # near_blocks > 0 only: refreshes per block of the far blank rows
    query_width: float = 1.0            # nu (shape == 'block' only): nu < 1: only the first ceil(nu * block_length) blanks of the current block are decoded; nu > 1: those of the current block are decoded and the
                                        # blanks of the following slots up to ceil(nu * block_length) from the block start are also COMPUTED fresh every call (written only in the current block)
    refresh_ctrl: Any = None            # default off. refresh_controller.RefreshController: a LEARNED, state-dependent refresh schedule (risk score of the previous calls -> no refresh / partial refresh of the blank rows / of all rows / dense call).
                                        # Needs refresh='block', is exclusive with partial_every_k, refresh_every_k, refresh_every_words, refresh_m, near_blocks, refresh_gate, query_width. The start layer is partial_alpha or partial_layer
    block_rows: bool = False            # lock == 'rule' with a block-vector rule (Learn2PD): also compute the rows of EVERY slot of the current block each call (<= 32 rows) so the filter sees real confidences
    log_calls: bool = True              # cheap per-call counters (rows per layer, refresh kind, joined rows)
    log_detail: bool = True             # also written positions/tokens, query sets, tracked rows, similarities (device->host copies; off for timed runs)
    collect_rows: bool = False          # default off. Trace collection for fitting a word-lock rule: log, per call and per ELIGIBLE blank, the six rule inputs (the same context_rules.block_features path
                                        # and window pos_in_block override that lock == 'rule' uses) and the proposal, in SelectiveResult.collected. Works with any lock; intended with lock == 'topk', topk = 1
    collect_extra: bool = False         # default off (needs collect_rows or a rule that names them). lock_features.EXTRA_FEATURES (stability / neighbours / proposal / frontier) are added to the rule inputs and to
                                        # SelectiveResult.collected; lock == 'rule' computes them anyway when the rule's feature list names one. Pre-registration: results/lctx_features_20261110/PREREGISTERED.md
    token_class: Any = None             # int8 [vocab] bit table (bit 0 newline, 1 starts with whitespace, 2 punctuation-only) for the tok_* extra features; only read when they are computed
    trace: Any = None                   # default off. lock == 'trace': a trace_lock.TraceScorer (TraceLock's controller + projector). Exposes the hidden states of layers L-2, L-1 and the final norm for all computed rows
                                        # (info['hid']); needs shape='block' and refresh in {block, every_call}. The controller reads the stored features of every row below the end of the current block.
    ctrl: Any = None                    # default off. stale_cause.CacheControl: per row class (prompt / prior_written / inblock_old / next_blank / far_blank) take the CACHED rows' K/V
                                        # from the cache ("stale"), from a dense pass at this call ("fresh"), or fresh + a random perturbation ("noise"), or COMPUTE the class's rows in the call


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
    call_blocks: list[int] = field(default_factory=list)            # log_detail: per call: block index (of the leftmost blank) the call decodes
    start_call: int = 0                 # resume: absolute index of the first call of this result (lists above are relative to it)
    extra_state: Any = field(default=None, repr=False)              # collect_extra / extra-feature rules: lock_features.ExtraState (previous call's eligible blanks)
    collected: list[dict] = field(default_factory=list)             # collect_rows: per call {call, position, token, n_first_block, six rule inputs} (numpy arrays over the eligible blanks)
    ctrl_prefix: list[int] = field(default_factory=list)            # lock == trace: per call the controller's prefix length (rows it attends over)
    ctrl_proj_rows: list[int] = field(default_factory=list)         # lock == trace: per call the rows the projector ran on (computed rows below the prefix end)
    ctrl_scores: list[list[float]] = field(default_factory=list)    # lock == trace and log_detail: per call the controller's sigmoid score of every eligible blank (parity post-mortem)
    ctl_log: list = field(default_factory=list)                     # refresh_ctrl: per call {call, kind, risk, since_refresh}
    ctl_time_s: float = 0.0                                         # refresh_ctrl: time in decide / observe (inside latency_s)
    ctrl_stats: dict = field(default_factory=dict)                  # ctrl: per class and layer: relative K/V distance stale-vs-fresh, realized noise ratio, rows


@dataclass
class ResumeState:
    """State at the START of a dense call: the cache is rebuilt from ``tokens`` alone, so the decode can be continued from here exactly."""
    tokens: list[int]                   # (prompt + gen) ids, MASK where nothing has been written
    call: int                           # absolute index of the first call to run
    eos_position: int = -1              # absolute position of the first-EOS truncation point (-1: none yet)


def replay_state(prompt_ids: list[int], gen_length: int, written: list[list[int]], written_tokens: list[list[int]], upto_call: int, cfg: SelectiveConfig) -> ResumeState:
    """State before call ``upto_call`` of a decode whose per-call writes are logged (log_detail): apply the writes of the calls before it, no forward pass.
    EOS truncation follows the live loop: the truncation point is the smallest EOS position written by the FIRST call that wrote any EOS."""
    P = len(prompt_ids)
    tokens = list(prompt_ids) + [cfg.mask_id] * gen_length
    eos_position = -1
    for i in range(upto_call):
        for pos, tok in zip(written[i], written_tokens[i]):
            tokens[pos] = tok
        if cfg.eos_truncation and eos_position < 0:
            e = [pos for pos, tok in zip(written[i], written_tokens[i]) if tok == cfg.eos_id]
            if e:
                eos_position = min(e)
    return ResumeState(tokens=tokens, call=upto_call, eos_position=eos_position)


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
                refresh_from: int | None = None, fresh: bool = False, kv_override: Any = None, join_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """tokens (1,T) current ids; rows = initial computed positions ordered [tracked (first n_track), ...]; blank_mask (T,) bool: every blank row at this call.
        dec_local: indices INTO ``rows`` of the rows whose logits are returned (the decoding window). mass_local: indices into ``rows`` whose attention defines the
        tracked rows (upstream: the blank query rows; the window when far blanks are cached).

        fresh=True: first call or a dense call: every layer recomputes all rows and refreshes its caches. refresh_from=l: all non-query rows join at layer l (tests).
        kv_override (default None): callable (layer, k_cache, v_cache) -> (k_full, v_full) giving the K/V the attention of this call reads; it must not modify the caches
        (stale_cause.KVOverride). join_mask (default None = every non-query row): (T,) bool, restricts the rows that join at refresh_from to those where it is True.
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
        hid: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for l, block in enumerate(self.blocks):
            # rows joining at this layer (decided on the previous layer)
            if not all_in and ((join_at is not None and l == join_at) or selective_pending is not None):
                if selective_pending is not None:
                    joiners = selective_pending
                    selective_pending = None
                else:
                    in_set = torch.zeros(T, dtype=torch.bool, device=self.device)
                    in_set[cur_rows] = True
                    joiners = torch.nonzero(~in_set if join_mask is None else (~in_set & join_mask), as_tuple=False).flatten()
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
            if cfg.trace is not None and l >= self.L - 2:
                hid[l] = (cur_rows, x)            # hidden_states[L-2], [L-1] of the HF model: the input of block l, for every computed row
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
            k_full, v_full = (kc, vc) if kv_override is None else kv_override(l, kc, vc)
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
        if cfg.trace is not None:
            info["hid"] = (hid[self.L - 2], hid[self.L - 1], (cur_rows, self.ln_f(x)))      # + hidden_states[L] = the final-normalised output of every computed row
        if cfg.refresh == "selective":
            self.snap[cur_rows] = self.call
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
def _rule_features(x: torch.Tensor, prompt_length: int, eligible: torch.Tensor, conf: torch.Tensor, logp: torch.Tensor, cfg: SelectiveConfig, blanks_in_first_block: int,
                   res: SelectiveResult | None = None, tokens: torch.Tensor | None = None) -> dict:
    """The six rule inputs of the eligible blanks, exactly as lock == 'rule' scores them (also used by collect_rows, so a fitted rule sees at decode time what it was fitted on).
    With res/tokens given and lock_features.wants_extra(cfg): plus the extra inputs of lock_features (called exactly once per call, they carry the previous-call state)."""
    from regret_remasking.context_rules import block_features
    feats = block_features(x, prompt_length, eligible, conf, logp[0], cfg.mask_id, cfg.block_length, cfg.gen_length)
    if cfg.shape == "window":      # the rule's block feature = share of the 32-block of the leftmost blank already written (B0 convention), not of the window
        feats["pos_in_block"] = feats["pos_in_block"] * 0 + max(cfg.block_length - blanks_in_first_block, 1) / cfg.block_length
    if res is not None and tokens is not None:
        from regret_remasking import lock_features as LF
        if LF.wants_extra(cfg):
            if res.extra_state is None:
                res.extra_state = LF.ExtraState()
            feats.update(LF.extra_features(res.extra_state, x, eligible, conf, logp[0], tokens, res.nfe, res.written_total, cfg.mask_id, cfg.eos_id, cfg.token_class))
    return feats


@torch.no_grad()
def _lock_write(logits: torch.Tensor, eligible: torch.Tensor, x: torch.Tensor, cfg: SelectiveConfig, prompt_length: int, blanks_in_first_block: int, res: SelectiveResult,
                trace_scores: torch.Tensor | None = None):
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
    feats = None
    if cfg.collect_rows:
        feats = _rule_features(x, prompt_length, eligible, conf, logp, cfg, blanks_in_first_block, res, tokens)
        res.collected.append(dict(feats, call=res.nfe, position=eligible.cpu().numpy(), token=tokens.cpu().numpy(), n_first_block=int(blanks_in_first_block)))
    if cfg.lock == "conf":
        selected = eligible[conf >= cfg.threshold]
        if selected.numel() == 0:
            selected = _argmax_stable(conf, eligible)
            fallback = True
    elif cfg.lock == "trace":
        # paper_decode T-adaptive / T-block: strictly above the cutoff writes; else the single best-scoring blank
        selected = eligible[trace_scores > cfg.threshold]
        if selected.numel() == 0:
            selected = _argmax_stable(trace_scores, eligible)
            fallback = True
    elif cfg.lock == "topk":
        if cfg.topk <= 1 or eligible.numel() <= 1:
            selected = _argmax_stable(conf, eligible)
        else:
            order = torch.argsort(conf, descending=True, stable=True)[: cfg.topk]
            selected = eligible[order].sort().values
    else:
        t0 = time.perf_counter()
        if feats is None:
            feats = _rule_features(x, prompt_length, eligible, conf, logp, cfg, blanks_in_first_block, res, tokens)
        scores = cfg.rule.scores(feats)
        accept = torch.as_tensor(scores >= cfg.rule.lam, device=x.device)
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
def refresh_points(m: int, schedule: str) -> list[float]:
    """Progress points (fractions of a block already written) at which a group is refreshed: m of them in (0, 1)."""
    if m <= 0:
        return []
    if schedule == "uniform":
        return [i / (m + 1) for i in range(1, m + 1)]
    if schedule == "front":
        return [2.0 ** -(m + 1 - i) for i in range(1, m + 1)]
    raise ValueError(f"unknown refresh_schedule {schedule!r}")


class ProgressTrigger:
    """Fires once the progress u of the current block has reached the next unfired point; every point passed since the last call is consumed by that one firing. reset() at a new block."""

    def __init__(self, points: list[float]) -> None:
        self.points = list(points)
        self.next = 0

    def reset(self) -> None:
        self.next = 0

    def due(self, u: float) -> bool:
        fired = False
        while self.next < len(self.points) and u >= self.points[self.next] - 1e-9:
            self.next += 1
            fired = True
        return fired


def partial_start_layer(cfg: SelectiveConfig, n_layers: int) -> int:
    """The first layer a partial refresh recomputes: floor(alpha * L) when partial_alpha >= 0, else the absolute partial_layer."""
    if cfg.partial_alpha >= 0:
        return int(math.floor(cfg.partial_alpha * n_layers + 1e-9))
    return cfg.partial_layer


def decode_selective(model: torch.nn.Module, prompt_ids: torch.Tensor, config: SelectiveConfig, *, engine: Any | None = None,
                     sync: bool = True, resume: ResumeState | None = None, shadow_engine: Any | None = None) -> SelectiveResult:
    """Decode one prompt (ids shape (1, P)) with the selective-cache engine. ``engine`` may be injected (tests).
    ``resume`` (refresh block / every_call only): continue from a dense call's state (see ``replay_state``); the first call is dense, lists in the result start at ``resume.call``.
    ``shadow_engine``: second engine for the dense pass that ``cfg.ctrl`` needs (created on demand)."""
    cfg = config
    if cfg.lock == "rule" and getattr(cfg.rule, "needs_block_vector", False) and not cfg.block_rows:
        raise ValueError("a block-vector rule (Learn2PD) needs SelectiveConfig.block_rows=True")
    if cfg.lock == "trace":
        if cfg.trace is None or cfg.shape != "block" or cfg.refresh not in ("block", "every_call") or cfg.block_rows or cfg.ctrl is not None or resume is not None:
            raise ValueError("lock='trace' needs SelectiveConfig.trace, shape='block', refresh in {block, every_call}, no block_rows / ctrl / resume")
    device = prompt_ids.device
    P = int(prompt_ids.shape[1])
    T = P + cfg.gen_length
    if cfg.lock == "trace":
        cfg.trace.reset(T, device)
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
    if resume is not None:
        if cfg.refresh not in ("block", "every_call") or cfg.block_rows:
            raise ValueError("resume needs refresh in {block, every_call}: only then does a dense call rebuild the whole state")
        x[0] = torch.as_tensor(resume.tokens, dtype=torch.long, device=device)
        masked_position = all_pos[P:][x[0, P:] == cfg.mask_id]
        if resume.eos_position >= 0 and cfg.eos_truncation:
            decoded_eos = True
            res.eos_position = resume.eos_position
            masked_position = masked_position[masked_position <= resume.eos_position]
        call = res.start_call = int(resume.call)
    since_dense = 0
    words_since = 0
    if cfg.refresh_every_words > 0 and (cfg.refresh != "block" or cfg.block_rows or cfg.ctrl is not None or cfg.words_refresh not in ("dense", "partial")
                                        or (cfg.words_refresh == "partial" and (cfg.partial_join not in ("all", "blank", "context") or not 0 <= (partial_start_layer(cfg, len(eng.blocks)) if cfg.partial_alpha >= 0 else cfg.partial_layer) < len(eng.blocks)))):
        raise ValueError("refresh_every_words needs refresh='block', no block_rows / ctrl, words_refresh in {dense, partial} (partial: a valid partial_layer and partial_join)")
    layer0 = partial_start_layer(cfg, len(eng.blocks)) if cfg.partial_alpha >= 0 else cfg.partial_layer
    if cfg.partial_every_k > 0 and (cfg.refresh != "block" or cfg.block_rows or cfg.ctrl is not None or cfg.partial_join not in ("all", "blank", "context")
                                    or not 0 <= layer0 < len(eng.blocks)):
        raise ValueError("partial_every_k needs refresh='block', no block_rows / ctrl, partial_join in {all, blank, context} and a valid partial_layer")
    triggers: dict[str, ProgressTrigger] = {}
    uses_family = (cfg.refresh_m > 0 or cfg.near_blocks > 0 or cfg.refresh_m_near > 0 or cfg.refresh_m_far > 0 or cfg.refresh_gate > 0 or cfg.query_width != 1.0)
    if uses_family:
        if cfg.refresh != "block" or cfg.block_rows or cfg.ctrl is not None or cfg.lock == "trace":
            raise ValueError("the refresh family (refresh_m / near_blocks / refresh_gate / query_width) needs refresh='block', no block_rows / ctrl / trace")
        if cfg.refresh_schedule not in ("uniform", "front") or not 0.0 <= cfg.refresh_gate < 1.0 or cfg.query_width <= 0 or cfg.partial_join not in ("all", "blank", "context"):
            raise ValueError("refresh_schedule in {uniform, front}, refresh_gate in [0, 1), query_width > 0, partial_join in {all, blank, context}")
        if (cfg.refresh_m > 0 or cfg.refresh_m_near > 0 or cfg.refresh_m_far > 0 or cfg.partial_every_k > 0 or (cfg.refresh_every_words > 0 and cfg.words_refresh == "partial")) and not 0 <= layer0 < len(eng.blocks):
            raise ValueError("the partial-refresh start layer (partial_alpha * L or partial_layer) must lie in [0, L)")
        if cfg.near_blocks > 0:
            if cfg.partial_join != "blank" or cfg.refresh_m > 0:
                raise ValueError("near_blocks > 0 needs partial_join == 'blank' and uses refresh_m_near / refresh_m_far, not refresh_m")
            triggers = {g: ProgressTrigger(refresh_points(m, cfg.refresh_schedule)) for g, m in (("near", cfg.refresh_m_near), ("far", cfg.refresh_m_far)) if m > 0}
        else:
            if cfg.refresh_m_near > 0 or cfg.refresh_m_far > 0:
                raise ValueError("refresh_m_near / refresh_m_far need near_blocks > 0")
            triggers = {"main": ProgressTrigger(refresh_points(cfg.refresh_m, cfg.refresh_schedule))} if cfg.refresh_m > 0 else {}
        if cfg.query_width != 1.0 and (cfg.shape != "block" or cfg.lock not in ("elastic", "conf", "topk") or (cfg.query_width > 1.0 and not cfg.far_mask_cache)):
            raise ValueError("query_width != 1 needs shape='block', lock in {elastic, conf, topk} (and far_mask_cache for nu > 1)")
    rc = cfg.refresh_ctrl
    if rc is not None:
        if (cfg.refresh != "block" or cfg.block_rows or cfg.ctrl is not None or cfg.lock == "trace" or cfg.partial_every_k > 0 or cfg.refresh_every_k > 0 or cfg.refresh_every_words > 0 or uses_family
                or not 0 <= layer0 < len(eng.blocks)):
            raise ValueError("refresh_ctrl needs refresh='block', a valid start layer, and none of block_rows / ctrl / trace / partial_every_k / refresh_every_k / refresh_every_words / the refresh family options")
        rc.reset_decode()
    recorder = shadow = None
    if cfg.ctrl is not None:
        from regret_remasking.stale_cause import CtrlRecorder
        if not cfg.far_mask_cache or cfg.refresh not in ("block", "every_call") or cfg.block_rows:
            raise ValueError("ctrl needs far_mask_cache=True, refresh in {block, every_call}, no block_rows")
        recorder = CtrlRecorder(cfg.ctrl, len(eng.blocks) if hasattr(eng, "blocks") else 0, device)
    while True:
        first_blank = int(masked_position[0])
        blk = (first_blank - P) // cfg.block_length
        if cfg.shape == "block":
            window_pos = masked_position[((masked_position - P) // cfg.block_length) == blk]
        else:
            window_pos = masked_position[: cfg.window]
        lookahead = all_pos[:0]
        if cfg.query_width != 1.0:
            n_q = max(1, math.ceil(cfg.query_width * cfg.block_length - 1e-9))
            in_blk = ((masked_position - P) // cfg.block_length) == blk
            if cfg.query_width < 1.0:
                window_pos = masked_position[in_blk][:n_q]
            else:
                lookahead = masked_position[(~in_blk) & ((masked_position - P) < blk * cfg.block_length + n_q)]
        mass_pos = window_pos if cfg.far_mask_cache else masked_position     # rows whose attention defines the tracked rows (upstream)
        ctrl_classes = None
        new_block = blk != block_of_first
        if new_block:
            for trg in triggers.values():
                trg.reset()
        dense_call = (call == 0 or cfg.refresh == "every_call" or (cfg.refresh == "block" and (blk != block_of_first or (cfg.refresh_every_k > 0 and call % cfg.refresh_every_k == 0)
                                                                                or (cfg.refresh_every_words > 0 and cfg.words_refresh == "dense" and words_since >= cfg.refresh_every_words))))
        block_of_first = blk
        since_dense = 0 if dense_call else since_dense + 1
        ctl_action, ctl_risk = None, float("nan")
        if rc is not None:
            t_c = time.perf_counter()
            rc.new_block(blk)
            if not dense_call:
                ctl_action, ctl_risk = rc.decide(int(window_pos.numel()))
                if ctl_action == "dense":
                    dense_call, since_dense, ctl_action = True, 0, None
            res.ctl_time_s += time.perf_counter() - t_c
        partial_due = (not dense_call) and ((cfg.partial_every_k > 0 and since_dense % cfg.partial_every_k == 0)
                                            or (cfg.refresh_every_words > 0 and cfg.words_refresh == "partial" and words_since >= cfg.refresh_every_words))
        period_due = partial_due
        sched_due: set[str] = set()
        if uses_family:
            slots = min(cfg.block_length, T - P - blk * cfg.block_length)
            u = 1.0 - int((((masked_position - P) // cfg.block_length) == blk).sum()) / max(1, slots)       # progress of the current block before this call's writes
            for g, trg in triggers.items():
                if trg.due(u) and not dense_call:
                    sched_due.add(g)
            if cfg.refresh_gate > 0 and (1.0 - u) <= cfg.refresh_gate:
                period_due, sched_due = False, set()
            partial_due = (not dense_call) and (period_due or bool(sched_due))
        if ctl_action in ("blank", "all"):
            partial_due = True
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
            extra = all_pos[:0]
            if cfg.ctrl is not None and cfg.ctrl.active(call, blk):
                from regret_remasking.stale_cause import row_classes
                ctrl_classes = row_classes(x[0], torch.cat([tr, new_decoded, window_pos]), P, blk, cfg.block_length, cfg.mask_id)
                extra = cfg.ctrl.compute_rows(ctrl_classes)
            if lookahead.numel():
                extra = torch.cat([extra, lookahead])
            if cfg.far_mask_cache:
                rows = torch.cat([tr, new_decoded, extra, window_pos])
                dec_local = torch.arange(rows.numel() - window_pos.numel(), rows.numel(), device=device)
                mass_local = dec_local
            else:
                rows = torch.cat([tr, new_decoded, masked_position])
                start = n_track + int(new_decoded.numel())
                dec_local = start + torch.arange(window_pos.numel(), device=device)
                mass_local = start + torch.arange(masked_position.numel(), device=device)
            kind = None
        blank_mask = torch.zeros(T, dtype=torch.bool, device=device)
        blank_mask[masked_position] = True
        override = None
        if ctrl_classes is not None:
            if shadow is None:
                from dataclasses import replace
                shadow = shadow_engine if shadow_engine is not None else SelectiveEngine(model, T, replace(cfg, ctrl=None))
            override = cfg.ctrl.make_override(shadow, x, ctrl_classes, recorder, T)
        extra_kw = {} if override is None else {"kv_override": override}      # engine stubs in the tests do not take the keyword
        if partial_due and ctl_action is not None:
            extra_kw = dict(extra_kw, refresh_from=layer0, join_mask=blank_mask if ctl_action == "blank" else None)
        elif partial_due:
            base = {"all": None, "blank": blank_mask, "context": x[0] != cfg.mask_id}[cfg.partial_join]
            parts = []
            if not uses_family or period_due:
                parts.append(base)
            if sched_due:
                if cfg.near_blocks > 0:
                    near = blank_mask & (((all_pos - P) // cfg.block_length) <= blk + cfg.near_blocks)
                    if "near" in sched_due:
                        parts.append(near)
                    if "far" in sched_due:
                        parts.append(blank_mask & ~near)
                else:
                    parts.append(base)
            join = None if any(q is None for q in parts) else torch.stack(parts).any(dim=0)
            extra_kw = dict(extra_kw, refresh_from=layer0, join_mask=join)
        logits, info = eng.forward(x, rows, n_track, dec_local, mass_local, blank_mask, fresh=dense_call, **extra_kw)
        trace_scores = None
        if cfg.lock == "trace":
            if device.type == "cuda":
                torch.cuda.synchronize(device)           # the timed region below is the projector + controller, not the forward pass still in flight (this sync is inside latency_s)
            t_ctrl = time.perf_counter()
            right = min(T, P + (blk + 1) * cfg.block_length)
            trace_scores = cfg.trace.score(info["hid"], right, window_pos)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            res.rule_time_s += time.perf_counter() - t_ctrl
            res.ctrl_prefix.append(right)
            res.ctrl_proj_rows.append(cfg.trace.last_proj_rows)
            if cfg.log_detail:
                res.ctrl_scores.append(trace_scores.tolist())
        if cfg.lock == "elastic":
            written, tok, eos_pos = _confident_write(logits, window_pos, x, cfg.threshold, cfg.eos_id)
        elif l2pd:
            written, tok, eos_pos = _lock_write_l2pd(logits, window_pos, block_start, x, cfg, res)
        else:
            blanks_first_block = int((((masked_position - P) // cfg.block_length) == blk).sum())
            written, tok, eos_pos = _lock_write(logits, window_pos, x, cfg, P, blanks_first_block, res, trace_scores)
        res.written_total += int(written.numel())
        if rc is not None:
            from regret_remasking.refresh_controller import window_proposals
            t_c = time.perf_counter()
            tk, pr = window_proposals(logits, cfg.mask_id)
            rc.observe(window_pos.tolist(), tk, pr, written.tolist(), "first" if call == 0 else "dense" if dense_call else (ctl_action or "none"), call, ctl_risk)
            res.ctl_time_s += time.perf_counter() - t_c
        words_since = int(written.numel()) if (dense_call or partial_due) else words_since + int(written.numel())
        new_decoded = written
        masked_position = masked_position[~torch.isin(masked_position, written)]
        res.nfe += 1
        if cfg.log_calls:
            res.rows_per_call.append(int(rows.numel()))
            res.layer_rows.append(list(info["layer_rows"]))
            if kind is None and partial_due:
                kind = "partial"
            if kind is None:
                kind = ("trigger" if cfg.refresh == "trigger" and info["refresh_layer"] >= 0 else "selective" if cfg.refresh == "selective" and info["refresh_layer"] >= 0 else "none")
            res.refresh_layers.append(int(info["refresh_layer"]) if not dense_call else 0)
            res.refresh_kind.append(kind)
            res.joined_rows.append(int(info["joined"]))
            if cfg.log_detail:
                res.written.append(written.tolist())
                res.written_tokens.append(tok.tolist())
                res.query_sets.append(rows.tolist())
                res.tracked.append(info["tracked"].tolist() if info["tracked"] is not None else [])
                res.sims.append(info["sims"])
                res.call_blocks.append(int(blk))
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
    if recorder is not None:
        res.ctrl_stats = recorder.summary()
    res.tokens = x[0, P:].tolist()
    if rc is not None:
        res.ctl_log = list(rc.log)
    res.final_mask_count = 0
    return res
