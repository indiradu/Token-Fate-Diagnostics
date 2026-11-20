"""Rule R0: a zero-search default for the representation-cache refresh, with every hyperparameter DERIVED from the setting (default off; library function, no decode).
Pre-registration: results/refresh_family_20261111/PREREGISTERED.md, "Stage R0".

Inputs (the setting): L layers; B0 the base decoder's block size; G generated length; P prompt length; calls_per_block (measured on a handful of prompts with the base cached decoder: calls / blocks);
the per-model, per-GPU cost parameters F (fixed ms per call: reading the weights) and c (ms per row per layer, in 'mean rows over layers' units, as fitted by analyze_depth_cells.py).
Two universal design constants, fixed before any evaluation and never tuned per setting: ALPHA0 = 1/3 (the refresh recomputes layers floor(L/3)..L-1) and BETA = 0.10 (refreshes may add at most 10% to the time of a block).
Derivation. Rows that rejoin: the blank rows of later blocks ('blank'): on average (G - B0) / 2 rows; or every cached row ('all'): P + G - (rows computed anyway). The extra time of one refresh is
delta = c * n_join * (L - l0) / L (ms). A cached call costs t_c = F + c * rows_cached with rows_cached = 1.25 * B0 (the query rows). Refreshes per block m = max(1, floor(BETA * calls_per_block * t_c / delta)) for 'blank';
'all' is used instead if its own m >= 1 under the same budget. The refresh clock is the call period k = max(2, round(calls_per_block / (m + 1))), the form the stage R results favoured over progress points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

ALPHA0 = 1.0 / 3.0
BETA = 0.10


@dataclass(frozen=True)
class Setting:
    n_layers: int
    block_length: int
    gen_length: int
    prompt_len: float
    calls_per_block: float
    fixed_ms: float
    row_ms: float


def derive(s: Setting, alpha0: float = ALPHA0, beta: float = BETA) -> dict:
    if not (0 < alpha0 < 1 and 0 < beta < 1 and s.calls_per_block >= 1 and s.n_layers >= 2):
        raise ValueError("alpha0, beta in (0, 1); calls_per_block >= 1; at least 2 layers")
    l0 = int(math.floor(alpha0 * s.n_layers + 1e-9))
    t_cached = s.fixed_ms + s.row_ms * 1.25 * s.block_length
    budget = beta * s.calls_per_block * t_cached
    out = {}
    for join, n_join in (("blank", max(1.0, (s.gen_length - s.block_length) / 2.0)), ("all", s.prompt_len + s.gen_length - 1.25 * s.block_length)):
        delta = s.row_ms * n_join * (s.n_layers - l0) / s.n_layers
        out[join] = {"n_join": n_join, "delta_ms": delta, "m": int(math.floor(budget / delta)) if delta > 0 else 0}
    join = "all" if out["all"]["m"] >= 1 else "blank"
    m = max(1, out[join]["m"])
    k = max(2, int(round(s.calls_per_block / (m + 1))))
    return {"partial_alpha": alpha0, "partial_layer_start": l0, "partial_join": join, "partial_every_k": k, "refreshes_per_block": m, "t_cached_ms": t_cached, "budget_ms": budget, "detail": out}


def config_kwargs(d: dict) -> dict:
    """SelectiveConfig keyword arguments of a derived rule."""
    return {"partial_every_k": d["partial_every_k"], "partial_alpha": d["partial_alpha"], "partial_join": d["partial_join"]}
