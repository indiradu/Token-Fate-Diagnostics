"""Autofit v0: find fast representation-cache refresh settings on YOUR model, GPU, length and word lock (representation cache only; the word lock is fixed). Spec: refresh_family.md section 6;
pre-registration of its parts: results/refresh_family_20261111/PREREGISTERED.md (stages S1, R0, C).

The procedure is pure orchestration over a ``decode_fn`` the caller supplies, so it runs on any model/engine and is testable without a GPU:

    decode_fn(policy, prompt) -> {"correct": bool, "latency_s": float, "nfe": int, ...}     policy is None for the dense decode (the reference), else a dict of cache options.

Steps: (1) dense and plain-cache decodes of the first ``headroom_n`` prompts; if the plain cache loses less than ``headroom_frac`` of the answers, STOP and return the plain cache (nothing to repair);
(2) candidates: the derived rule (refresh_rule) as the seed, four near-identical neighbours of it that measure the NOISE FLOOR, dense refreshes, optional learned-controller settings, and a grid around the seed
(start layer, rows that rejoin, period), at most ``max_candidates``; (3) successive halving over the rounds: after each round a setting is pruned when another one is at least as fast and has more than
max(margin_frac * n, noise_k * SD) more correct answers (SD: standard deviation of the pooled answers across the seed and its neighbours), the Pareto front of (speed, answers) is always kept, at most ``max_survivors``;
(4) the report: the Pareto set with speed against dense, net correct answers (with an approximate paired interval), the secondary metrics the decoder returns, the headroom statement, the noise floor, and, if a tolerance
``tau`` (share of questions) is given, the fastest setting within tau of dense. Deterministic given the decode function. Speeds are ratios of mean measured latency (dense / policy) on the same prompts.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from regret_remasking.refresh_rule import Setting, config_kwargs, derive

LAYER_FRACTIONS = (0.25, 1.0 / 3.0, 0.5, 0.75)


@dataclass(frozen=True)
class AutofitConfig:
    rounds: tuple[int, ...] = (10, 25, 50)           # cumulative number of prompts per round (offline benchmark on the recorded KodCode screen: halving on 10-25-50 prompts is within 0.02 speed units of the full grid at half its decodes)
    space: str = "environment"                       # environment: the candidate space is PROPOSED by the sensed setting (propose_space) | grid: the fixed screen grid (candidates_for)
    beta_max: float = 0.25                            # environment space: a setting may add at most this share to a block's time by its refreshes
    selection: str = "halving"                       # halving: keep the top keep_frac of the settings by score after each round but the last | margin: the screen's pruning rule (pairwise margin and Pareto front)
    keep_frac: float = 0.5
    rank_tau: float = 0.04                            # accepted loss of answers (share of the prompts) used only to RANK settings in halving mode: score = speed - 10 x shortfall
    headroom_n: int = 50
    headroom_frac: float = 0.02                       # plain cache must lose at least this share of the answers for a refresh to be worth searching
    margin_frac: float = 0.03                         # pruning margin as a share of the prompts used so far (the screen used 6 of 200)
    noise_k: float = 2.0
    max_survivors: int = 12
    max_candidates: int = 45
    tau: float | None = None                          # accepted loss of answers against dense, as a share of the prompts; None = no pick, the Pareto set only
    periods: tuple[int, ...] = (2, 4, 8)
    dense_periods: tuple[int, ...] = (4,)             # K2 and K8 never reached the top 12 on either recorded screen


@dataclass
class Report:
    stopped_at_headroom: bool
    headroom: dict
    candidates: list[str]
    rounds: list[dict]
    pareto: list[dict]
    survivors: list[str]
    noise_sd: float
    pick: dict | None
    n_prompts_used: int
    decodes: int
    notes: list[str] = field(default_factory=list)


def _policy(**kw) -> dict:
    return dict(kw)


def propose_space(setting: Setting, cfg: AutofitConfig, controllers: dict[str, dict] | None = None) -> dict[str, dict]:
    """The candidate space PROPOSED BY THE ENVIRONMENT (deterministic; every axis is derived from the sensed setting, nothing from a fixed grid):
    layers floor(L/4), floor(L/3), floor(L/2) (the recorded screen: 3L/4 never won); refreshes per block m in 1..4 limited by the cost model (m x extra time of a refresh <= beta_max x block time), as a call
    period k = round(calls per block / (m + 1)) and as m progress points; rows that rejoin: blank rows, and all rows only at one period (m = 1) on two layers and only if one such refresh fits the budget; the near-block
    group (m = 2); dense refreshes K4 and K8 (K2 never won); one PROBE for each dropped family (layer 3L/4, the gate, query width 2) so that a prior that fails on a new model can be discovered; the seed (derived rule),
    four near-identical neighbours of it (noise floor) and any learned controllers handed in. Names and configs are unique; at most cfg.max_candidates, in this priority order."""
    rule = derive(setting)
    seed_kw = config_kwargs(rule)
    L, row_ms, G, B0 = setting.n_layers, setting.row_ms, setting.gen_length, setting.block_length
    l0, k0, cpb = rule["partial_layer_start"], seed_kw["partial_every_k"], setting.calls_per_block
    block_ms = cpb * rule["t_cached_ms"]
    n_blank, n_all = rule["detail"]["blank"]["n_join"], rule["detail"]["all"]["n_join"]
    out: dict[str, dict] = {"SEED": seed_kw}
    out["NF_layer_minus"] = dict(seed_kw, partial_alpha=max(0, l0 - 1) / L)
    out["NF_layer_plus"] = dict(seed_kw, partial_alpha=min(L - 1, l0 + 1) / L)
    out["NF_period_minus"] = dict(seed_kw, partial_every_k=max(1, k0 - 1))
    out["NF_period_plus"] = dict(seed_kw, partial_every_k=k0 + 1)
    for name, kw in (controllers or {}).items():
        out[name] = kw
    layers = sorted({max(0, int(math.floor(f * L + 1e-9))) for f in (0.25, 1.0 / 3.0, 0.5)})
    extra = lambda l, n: row_ms * n * (L - l) / L
    ms = lambda l, n: [m for m in (1, 2, 3, 4) if m * extra(l, n) <= cfg.beta_max * block_ms] or [1]
    cand: list[tuple[str, dict]] = []
    for l in layers:
        for m in ms(l, n_blank):
            cand.append((f"P_m{m}_a{l}_blank", {"partial_every_k": max(2, int(round(cpb / (m + 1)))), "partial_alpha": l / L, "partial_join": "blank"}))
    for l in layers:
        for m in [m for m in ms(l, n_blank) if m in (2, 4)]:           # one progress point per block (M1) never reached the top 12
            cand.append((f"M_m{m}_a{l}_blank", {"refresh_m": m, "partial_alpha": l / L, "partial_join": "blank"}))
    for l in (l0, int(math.floor(0.5 * L))):
        if extra(l, n_all) <= cfg.beta_max * block_ms:
            cand.append((f"P_m1_a{l}_all", {"partial_every_k": max(2, int(round(cpb / 2))), "partial_alpha": l / L, "partial_join": "all"}))
    if extra(l0, n_all) * 2 <= cfg.beta_max * block_ms:
        cand.append((f"M_m2_a{l0}_all", {"refresh_m": 2, "partial_alpha": l0 / L, "partial_join": "all"}))      # all rows with two progress points won on GSM8K
    for l in (l0, int(math.floor(0.5 * L))):
        cand.append((f"NEAR_m2_a{l}", {"near_blocks": 1, "refresh_m_near": 2, "partial_alpha": l / L, "partial_join": "blank"}))
    for k in cfg.dense_periods:
        cand.append((f"DENSE_k{k}", {"refresh_every_k": k}))
    cand.append((f"PROBE_layer{int(0.75 * L)}", dict(seed_kw, partial_alpha=int(0.75 * L) / L)))
    cand.append(("PROBE_gate", dict(seed_kw, refresh_gate=0.25)))
    cand.append(("PROBE_query2", dict(seed_kw, query_width=2.0)))
    # priority: the seed family first, then the structured space; unique configs only
    for name, kw in cand:
        if kw not in out.values() and name not in out:
            out[name] = kw
    return dict(list(out.items())[: cfg.max_candidates])


def candidates_for(setting: Setting, cfg: AutofitConfig, controllers: dict[str, dict] | None = None) -> dict[str, dict]:
    """Ordered candidate policies (name -> cache options), deterministic. space='environment' (default): propose_space. space='grid': the fixed screen grid, ordered by distance from the seed."""
    if cfg.space == "environment":
        return propose_space(setting, cfg, controllers)
    seed_rule = derive(setting)
    seed_kw = config_kwargs(seed_rule)
    L = setting.n_layers
    l0 = seed_rule["partial_layer_start"]
    k0 = seed_kw["partial_every_k"]
    out: dict[str, dict] = {"SEED": seed_kw}
    out["NF_layer_minus"] = dict(seed_kw, partial_alpha=max(0, l0 - 1) / L)
    out["NF_layer_plus"] = dict(seed_kw, partial_alpha=min(L - 1, l0 + 1) / L)
    out["NF_period_minus"] = dict(seed_kw, partial_every_k=max(1, k0 - 1))
    out["NF_period_plus"] = dict(seed_kw, partial_every_k=k0 + 1)
    for k in cfg.dense_periods:
        out[f"DENSE_k{k}"] = {"refresh_every_k": k}
    for name, kw in (controllers or {}).items():
        out[name] = kw
    grid = []
    for a in LAYER_FRACTIONS:
        for j in ("blank", "all"):
            for k in sorted(set(cfg.periods) | {k0}):
                kw = {"partial_every_k": k, "partial_alpha": a, "partial_join": j}
                dist = abs(math.log(max(a, 1e-6) / max(seed_kw["partial_alpha"], 1e-6))) + abs(math.log(k / k0)) + (0.0 if j == seed_kw["partial_join"] else 0.5)
                grid.append((dist, f"P_k{k}_a{int(round(a * L))}_{j}", kw))
    for _, name, kw in sorted(grid, key=lambda t: (t[0], t[1])):
        if len(out) >= cfg.max_candidates:
            break
        if kw not in out.values():
            out[name] = kw
    return dict(list(out.items())[: cfg.max_candidates])


def _pareto(points: dict[str, tuple[float, float]]) -> set[str]:
    """names not dominated in (speed, net answers): no other point is at least as good on both and better on one."""
    keep = set()
    for a, (sa, na) in points.items():
        if not any((sb >= sa and nb >= na) and (sb > sa or nb > na) for b, (sb, nb) in points.items() if b != a):
            keep.add(a)
    return keep


def run(decode_fn: Callable[[dict | None, Any], dict], prompts: list, setting: Setting, cfg: AutofitConfig = AutofitConfig(), controllers: dict[str, dict] | None = None,
        cache: dict | None = None) -> Report:
    """Run the procedure. ``cache`` (optional dict) makes it resumable: decodes are stored under (policy name, prompt index) and not repeated."""
    cache = {} if cache is None else cache
    decodes = [0]

    def get(name: str, policy: dict | None, i: int) -> dict:
        key = (name, i)
        if key not in cache:
            cache[key] = decode_fn(policy, prompts[i])
            decodes[0] += 1
        return cache[key]

    if len(prompts) < max(cfg.rounds[0], cfg.headroom_n):
        raise ValueError(f"need at least {max(cfg.rounds[0], cfg.headroom_n)} calibration prompts, got {len(prompts)}")
    hn = cfg.headroom_n
    dense = [get("DENSE", None, i) for i in range(hn)]
    plain = [get("PLAIN", {}, i) for i in range(hn)]
    d_c, p_c = sum(bool(r["correct"]) for r in dense), sum(bool(r["correct"]) for r in plain)
    flips = sum(bool(a["correct"]) != bool(b["correct"]) for a, b in zip(dense, plain))
    loss = d_c - p_c
    headroom = {"n": hn, "dense_correct": d_c, "plain_correct": p_c, "loss": loss, "loss_share": loss / hn, "paired_halfwidth": 1.96 * math.sqrt(flips), "threshold_share": cfg.headroom_frac}
    if loss / hn < cfg.headroom_frac:
        speed = sum(r["latency_s"] for r in dense) / sum(r["latency_s"] for r in plain)
        pick = {"name": "PLAIN", "policy": {}, "speed_vs_dense": speed, "net_correct": -loss}
        return Report(True, headroom, ["PLAIN"], [], [pick], ["PLAIN"], float("nan"), pick, hn, decodes[0],
                      [f"the plain cache loses {loss} of {hn} answers ({loss / hn:.1%} < {cfg.headroom_frac:.0%}): no refresh worth searching for; plain cache returned"])
    cands = candidates_for(setting, cfg, controllers)
    alive = list(cands)
    rounds: list[dict] = []
    stats: dict[str, tuple[float, float, float]] = {}
    used = 0
    for r, n_r in enumerate(cfg.rounds):
        n_r = min(n_r, len(prompts))
        for i in range(n_r):
            get("DENSE", None, i)
            for name in alive:
                get(name, cands[name], i)
        used = n_r
        d_lat = sum(cache[("DENSE", i)]["latency_s"] for i in range(n_r))
        d_cor = [bool(cache[("DENSE", i)]["correct"]) for i in range(n_r)]
        stats = {}
        for name in alive:
            cor = [bool(cache[(name, i)]["correct"]) for i in range(n_r)]
            flips = sum(a != b for a, b in zip(cor, d_cor))
            stats[name] = (d_lat / sum(cache[(name, i)]["latency_s"] for i in range(n_r)), float(sum(cor) - sum(d_cor)), 1.96 * math.sqrt(flips))
        nf = [stats[k][1] for k in ("SEED", "NF_layer_minus", "NF_layer_plus", "NF_period_minus", "NF_period_plus") if k in stats]
        sd = statistics.stdev(nf) if len(nf) > 1 else (rounds[-1]["noise_sd"] if rounds else 0.0)
        margin = max(cfg.margin_frac * n_r, cfg.noise_k * sd)
        pts = {n: (s[0], s[1]) for n, s in stats.items()}
        front = _pareto(pts)
        if cfg.selection == "halving":
            sc = {n: pts[n][0] - 10.0 * max(0.0, -pts[n][1] - cfg.rank_tau * n_r) / n_r for n in alive}
            last = r == len(cfg.rounds) - 1
            keep = list(alive) if last else sorted(alive, key=lambda n: -sc[n])[: max(2, math.ceil(cfg.keep_frac * len(alive)))]
            pruned = {n for n in alive if n not in keep}
        else:
            pruned = {n for n in alive if n not in front and any(pts[o][0] >= pts[n][0] and pts[o][1] - pts[n][1] > margin for o in alive if o != n)}
            keep = [n for n in alive if n not in pruned]
            if r < len(cfg.rounds) - 1 and len(keep) > cfg.max_survivors:
                ranked = sorted(keep, key=lambda n: (n not in front, -pts[n][0]))
                keep = ranked[: cfg.max_survivors]
        rounds.append({"round": r, "n": n_r, "alive": list(alive), "noise_sd": sd, "margin": margin, "pruned": sorted(pruned), "kept": list(keep)})
        if r < len(cfg.rounds) - 1:
            alive = [n for n in keep if n in cands]
    kept = [n for n in rounds[-1]["kept"] if n in stats]
    final = {n: (stats[n][0], stats[n][1]) for n in kept}
    front = _pareto(final)
    pareto = [{"name": n, "policy": cands[n], "speed_vs_dense": final[n][0], "net_correct": final[n][1], "halfwidth": stats[n][2],
               "nfe": statistics.fmean(cache[(n, i)].get("nfe", float("nan")) for i in range(used))} for n in sorted(front, key=lambda n: -final[n][0])]
    pick = None
    if cfg.tau is not None:
        ok = [p for p in pareto if p["net_correct"] >= -cfg.tau * used]
        pick = max(ok, key=lambda p: p["speed_vs_dense"]) if ok else None
    return Report(False, headroom, list(cands), rounds, pareto, kept, rounds[-1]["noise_sd"], pick, used, decodes[0],
                  [] if pick is not None or cfg.tau is None else ["no Pareto setting is within the tolerance of dense"])


def format_report(rep: Report) -> str:
    L = ["# Autofit report\n", f"Headroom ({rep.headroom['n']} prompts): dense {rep.headroom['dense_correct']}, plain cache {rep.headroom['plain_correct']} "
         f"(loss {rep.headroom['loss']}, +-{rep.headroom['paired_halfwidth']:.1f}). " + ("STOPPED: " + "; ".join(rep.notes) if rep.stopped_at_headroom else "A refresh is worth searching for.")]
    if not rep.stopped_at_headroom:
        L.append(f"\nPrompts used {rep.n_prompts_used}; decodes {rep.decodes}; noise floor SD {rep.noise_sd:.2f} answers; {len(rep.candidates)} candidates; survivors {', '.join(rep.survivors)}.\n")
        L += ["| setting | speed vs dense | net correct vs dense | +- | calls |", "|---|---|---|---|---|"]
        L += [f"| {p['name']} | {p['speed_vs_dense']:.2f}x | {p['net_correct']:+.0f} | {p['halfwidth']:.1f} | {p['nfe']:.1f} |" for p in rep.pareto]
        if rep.pick:
            L.append(f"\nPick within the tolerance: {rep.pick['name']} ({rep.pick['speed_vs_dense']:.2f}x, {rep.pick['net_correct']:+.0f} answers).")
        L += [f"\n{n}" for n in rep.notes]
    return "\n".join(L) + "\n"
