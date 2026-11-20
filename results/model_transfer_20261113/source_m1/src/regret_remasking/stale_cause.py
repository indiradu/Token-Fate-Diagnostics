"""Cause-of-damage controls for the selective (far-blank) K/V cache: is the damage STALENESS, or any perturbation of the same size, and which row class carries it?

REPRESENTATION CACHE side only (which rows' K/V a call reads from the cache). The WORD LOCK is whatever ``SelectiveConfig`` says (C-0.9 in the studies).

Row classes of the CACHED rows (every row outside the call's query set), in the S32 query shape (query = rows written on the previous call + the blanks of the current block):
  prompt         positions < P
  prior_written  written rows of earlier blocks
  inblock_old    written rows of the current block that were not written on the previous call (written >= 2 calls ago)
  next_blank     blank rows of the next block (blk + 1)
  far_blank      all other blank rows: blocks >= blk + 2, and the current block's blank rows that are outside the query set (only beyond a first-EOS truncation point)
Classes are by block index relative to the block being decoded; first-EOS truncation does not move a row between classes. (blank = still the mask token; a truncated-away blank is still an input row, so it is attended and cached.)

``CacheControl`` (SelectiveConfig.ctrl, default None = off) picks, per class, where the K/V that class contributes at a call comes from:
  stale  the cache (what S32 does)                      -> with every class "stale" the decode is exactly S32
  fresh  K/V of a dense pass over the CURRENT tokens at this call (an oracle: what a dense call would give)
  noise  fresh + a random perturbation: per row, per layer, K and V separately, a Gaussian direction over the flattened (kv heads x head dim) vector rescaled to
         ``noise_scale`` x ||stale - fresh|| of that row. K is the post-RoPE key that the cache stores. Computed in float32, cast to the cache dtype.
and ``compute`` lists classes whose rows are COMPUTED in the call (appended to the query rows, not read from the cache): the real, realizable repair.
The caches themselves are never written by the override (the override builds K/V for the attention only), so "stale" stays the S32 stale cache.

``layers`` (default () = every layer = the behaviour above) restricts the fresh / noise replacement to the listed transformer layers (0-based); at every other layer the class keeps the CACHE.
Statistics (stale-vs-fresh distances) are still recorded at every layer. ``layers`` is an ORACLE band: the fresh K/V of the listed layers come from a dense pass whose LOWER layers were fresh too, which is
not the same object as a partial refresh that recomputes the upper layers over stale lower activations. Not defined for ``compute`` (rows computed in the call run every layer): the combination is refused.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch

CLASSES = ("prompt", "prior_written", "inblock_old", "next_blank", "far_blank")
BLANKS = ("next_blank", "far_blank")
MODES = ("stale", "fresh", "noise")


def seed_for(*parts) -> int:
    """Stable (cross-process) 31-bit seed from arbitrary parts."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class CacheControl:
    sources: tuple = ()                 # ((class, mode), ...); classes not listed are "stale"
    compute: tuple = ()                 # classes computed in the call instead of read from the cache
    noise_scale: float = 1.0
    noise_seed: int = 0
    block: int = -1                     # >= 0: only in calls decoding this block index (a single-block repair); -1: every call
    stats: bool = False                 # record relative stale-vs-fresh distances per class and layer (needs the dense pass)
    layers: tuple = ()                  # () = fresh/noise replacement at every layer; else only at these layers (stats still at every layer)

    def __post_init__(self):
        for c, m in self.sources:
            if c not in CLASSES or m not in MODES:
                raise ValueError(f"bad source {(c, m)}")
        for c in self.compute:
            if c not in CLASSES:
                raise ValueError(f"bad compute class {c}")
        if any(isinstance(l, bool) or not isinstance(l, int) or l < 0 for l in self.layers):
            raise ValueError(f"bad layers {self.layers}: non-negative ints expected")
        if self.layers and self.compute:
            raise ValueError("layers restricts the fresh / noise replacement; it is not defined for computed (compute) classes")

    def replaces(self, layer: int) -> bool:
        """True when the fresh / noise replacement is applied at this layer."""
        return not self.layers or layer in self.layers

    def mode(self, cls: str) -> str:
        return dict(self.sources).get(cls, "stale")

    def active(self, call: int, blk: int) -> bool:
        return self.block < 0 or blk == self.block

    def compute_rows(self, classes: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [classes[c] for c in CLASSES if c in self.compute]
        base = next(iter(classes.values()))[:0]
        return torch.cat(parts) if parts else base

    def groups(self, classes: dict[str, torch.Tensor]):
        return [(c, classes[c], self.mode(c)) for c in CLASSES if c not in self.compute and classes[c].numel()]

    def make_override(self, shadow, x: torch.Tensor, classes: dict[str, torch.Tensor], recorder: "CtrlRecorder", total_len: int):
        """The callable for ``SelectiveEngine.forward(kv_override=...)``, or None when nothing differs from the cache and no statistics are wanted."""
        groups = self.groups(classes)
        needs_dense = any(m != "stale" for _, _, m in groups) or (self.stats and groups)
        if not needs_dense:
            return None
        T = total_len
        all_pos = torch.arange(T, device=x.device)
        blank = x[0] == shadow.cfg.mask_id
        shadow.forward(x, all_pos, 0, all_pos[:1], all_pos[:1], blank, fresh=True)      # fills shadow.k_cache / v_cache with a dense pass over the current tokens
        return KVOverride(shadow, groups, self, recorder)


def row_classes(x_row: torch.Tensor, query_rows: torch.Tensor, P: int, blk: int, block_length: int, mask_id: int) -> dict[str, torch.Tensor]:
    """Partition of the rows OUTSIDE ``query_rows`` into the five classes (positions, ascending). x_row: (T,) current ids."""
    T = int(x_row.numel())
    pos = torch.arange(T, device=x_row.device)
    in_q = torch.zeros(T, dtype=torch.bool, device=x_row.device)
    in_q[query_rows] = True
    outside = ~in_q
    is_blank = (x_row == mask_id) & (pos >= P)
    written = (x_row != mask_id) & (pos >= P)
    b = torch.div(pos - P, block_length, rounding_mode="floor")
    m = {
        "prompt": (pos < P) & outside,
        "prior_written": written & outside & (b < blk),
        "inblock_old": written & outside & (b >= blk),
        "next_blank": is_blank & outside & (b == blk + 1),
        "far_blank": is_blank & outside & ((b >= blk + 2) | (b <= blk)),
    }
    return {c: torch.nonzero(m[c], as_tuple=False).flatten() for c in CLASSES}


class CtrlRecorder:
    """Accumulates, per class and layer, on the device (one host copy at the end): mean relative stale-vs-fresh K / V distance over rows, the realized noise norm ratio, and row counts."""

    def __init__(self, ctrl: CacheControl, n_layers: int, device) -> None:
        self.ctrl, self.L, self.device = ctrl, n_layers, device
        self.acc = {c: torch.zeros(5, max(n_layers, 1), dtype=torch.float64, device=device) for c in CLASSES}   # relK, relV, ratioK, ratioV, rows
        self.calls = {c: 0 for c in CLASSES}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(ctrl.noise_seed))

    def add(self, cls: str, layer: int, rel_k, rel_v, ratio_k, ratio_v, n_rows: int) -> None:
        a = self.acc[cls]
        a[0, layer] += rel_k.sum().double()
        a[1, layer] += rel_v.sum().double()
        a[2, layer] += ratio_k.sum().double()
        a[3, layer] += ratio_v.sum().double()
        a[4, layer] += n_rows
        if layer == 0:
            self.calls[cls] += 1

    def summary(self) -> dict:
        out = {}
        for c, a in self.acc.items():
            n = a[4].clamp_min(1)
            out[c] = {"rows_per_layer": a[4].tolist(), "calls_nonempty": self.calls[c], "rel_k": (a[0] / n).tolist(), "rel_v": (a[1] / n).tolist(),
                      "ratio_k": (a[2] / n).tolist(), "ratio_v": (a[3] / n).tolist()}
        return out


def _row_norm(t: torch.Tensor) -> torch.Tensor:
    """(1, nkv, R, hd) -> (R,) Frobenius norm of each row's flattened (kv heads x head dim) vector, float32."""
    return t.float().norm(dim=(1, 3))[0]


class KVOverride:
    """K/V the attention of one cached call reads: the cache, with the rows of each class replaced as configured. Never modifies the caches."""

    def __init__(self, shadow, groups, ctrl: CacheControl, recorder: CtrlRecorder) -> None:
        self.shadow, self.groups, self.ctrl, self.rec = shadow, groups, ctrl, recorder

    def _noisy(self, stale: torch.Tensor, fresh: torch.Tensor):
        s, f = stale.float(), fresh.float()
        target = self.ctrl.noise_scale * (s - f).norm(dim=(1, 3), keepdim=True)                        # (1, 1, R, 1)
        g = torch.randn(f.shape, generator=self.rec.gen, device=f.device, dtype=torch.float32)
        g = g / g.norm(dim=(1, 3), keepdim=True).clamp_min(1e-30) * target
        return (f + g).to(stale.dtype)

    def __call__(self, layer: int, kc: torch.Tensor, vc: torch.Tensor):
        fk, fv = self.shadow.k_cache[layer], self.shadow.v_cache[layer]
        k_full, v_full, cloned = kc, vc, False
        apply = self.ctrl.replaces(layer)                                                       # layers outside the band keep the cache; statistics are recorded regardless
        for cls, rows, mode in self.groups:
            ks, ke = kc[:, :, rows], fk[:, :, rows]
            vs, ve = vc[:, :, rows], fv[:, :, rows]
            watch = self.ctrl.stats or mode == "noise"
            if watch:
                nk, nv = _row_norm(ke).clamp_min(1e-30), _row_norm(ve).clamp_min(1e-30)
                dk, dv = _row_norm(ks.float() - ke.float()), _row_norm(vs.float() - ve.float())
            out_k = out_v = None
            if not apply:
                pass
            elif mode == "fresh":
                out_k, out_v = ke, ve
            elif mode == "noise":
                out_k, out_v = self._noisy(ks, ke), self._noisy(vs, ve)
            if out_k is not None:
                if not cloned:
                    k_full, v_full, cloned = kc.clone(), vc.clone(), True
                k_full[:, :, rows] = out_k
                v_full[:, :, rows] = out_v
            if watch:
                if mode == "noise" and out_k is not None:
                    rk, rv = _row_norm(out_k.float() - ke.float()) / dk.clamp_min(1e-30), _row_norm(out_v.float() - ve.float()) / dv.clamp_min(1e-30)
                    rk, rv = torch.where(dk > 0, rk, torch.zeros_like(rk)), torch.where(dv > 0, rv, torch.zeros_like(rv))
                else:
                    rk = rv = torch.zeros_like(dk)
                self.rec.add(cls, layer, dk / nk, dv / nv, rk, rv, int(rows.numel()))
        return k_full, v_full


# ---------------------------------------------------------------------------------------------- analysis helpers (pure python, no model)
def events(written: list[list[int]], tokens: list[list[int]]) -> list[dict[int, int]]:
    return [dict(zip(w, t)) for w, t in zip(written, tokens)]


def first_divergence(ev_a: list[dict[int, int]], ev_b: list[dict[int, int]], offset_a: int = 0) -> dict | None:
    """First call at which two runs' write events differ. ev_a[i] is call ``offset_a + i`` of run A; run B is indexed from call 0 (absolute).
    Returns None when A (to its end) equals B over A's calls AND they have the same number of remaining calls; else
    {call, pos, kind, tok_a, tok_b}: kind = token (same position, different token) | a_only | b_only | length (one run ended)."""
    n = max(len(ev_a) + offset_a, len(ev_b))
    for call in range(offset_a, n):
        a = ev_a[call - offset_a] if call - offset_a < len(ev_a) else None
        b = ev_b[call] if call < len(ev_b) else None
        if a is None or b is None:
            return {"call": call, "pos": None, "kind": "length", "tok_a": None, "tok_b": None}
        if a != b:
            diff = sorted({p for p in a.keys() | b.keys() if a.get(p) != b.get(p)})
            p = diff[0]
            kind = "token" if p in a and p in b else ("a_only" if p in a else "b_only")
            return {"call": call, "pos": p, "kind": kind, "tok_a": a.get(p), "tok_b": b.get(p)}
    return None


def first_token_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def dense_calls(refresh_kind: list[str]) -> list[int]:
    return [i for i, k in enumerate(refresh_kind) if k in ("first", "dense")]


def last_dense_before(refresh_kind: list[str], call: int) -> int:
    """Index of the last block-start dense call at or before ``call``."""
    ds = [i for i in dense_calls(refresh_kind) if i <= call]
    if not ds:
        raise ValueError("no dense call before the divergence")
    return ds[-1]


REPAIR_CLASSES = {
    "next_blank": ("next_blank",),                       # (a) blank rows of the next block
    "far2_blank": ("far_blank",),                        # (b) blank rows beyond the next block
    "inblock_old": ("inblock_old",),                     # (c) in-block written rows older than one step
    "prompt": ("prompt",),                               # (d) prompt rows
    "prior_written": ("prior_written",),                 # (f) written rows of earlier blocks (completes the partition)
    "all_blank": BLANKS,                                 # (e) every cached blank row (positive control)
    "all_cached": CLASSES,                               # (g) every cached row (dense-equivalent control)
}


def repair_ctrl(variant: str, mode: str, block: int, *, stats: bool = False, layers: tuple = ()) -> CacheControl | None:
    """variant: None | key of REPAIR_CLASSES. mode: oracle (fresh dense-pass K/V) | compute (rows computed in the call). layers: oracle only; () = every layer, else the layers made fresh."""
    if variant is None:
        return CacheControl(stats=stats, block=block) if stats else None
    cls = REPAIR_CLASSES[variant]
    if mode == "oracle":
        return CacheControl(sources=tuple((c, "fresh") for c in cls), block=block, stats=stats, layers=tuple(layers))
    if mode == "compute":
        return CacheControl(compute=tuple(cls), block=block, stats=stats, layers=tuple(layers))
    raise ValueError(mode)
