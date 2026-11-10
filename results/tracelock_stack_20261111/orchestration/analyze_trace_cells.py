"""CPU analysis of the TraceLock-under-caches records. Usage: analyze_trace_cells.py [RECORDS_DIR] -> results_<stage>.md for each stage found (M512, P256, H512), results_pooled512.md (MATH 108 + HumanEval 164) and
results_selection.md. Reads records/stage<S>.scored.jsonl if present (HumanEval rescored with bwrap) else records/stage<S>.jsonl. Bootstrap over QUESTIONS (10,000 resamples, seed 23), never over calls.
Definitions (the conventions of scripts/locking_flop_accounting.py and results/depth_refresh_20261109/orchestration/metrics_extra.py; none is a hardware counter):
 - backbone TFLOPs: per layer and per computed row attention 4*d*T, projections 4*d*d + 4*d*d_kv, gated MLP 6*d*d_ff, with the ACTUAL per-layer row counts of every call; dense arms: T rows in all 32 layers at every call;
   LM head and elementwise operations excluded.
 - controller TFLOPs: projector 2*(3*(h*s_h + s_h*s_d) + 2*(h*d_h + d_h*d_d)) per PROJECTED row (rows computed on that call below the prefix end; dense arms: every row of the prefix) + the controller's input projection,
   transformer and heads on the prefix of m rows (scripts/locking_flop_accounting.controller_flops with its projector term replaced by the projected-row count: under a cache fewer rows are projected than the prefix holds).
   Prefix lengths of the dense arms are reconstructed by replaying the logged per-call writes (moving window for D, 32-block for DB).
 - controller ms/call: the segment from the end of the backbone forward to the scores (projector + controller, synchronised); it is INSIDE latency_s for every TraceLock arm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from locking_flop_accounting import ControllerConfig, ProjectorConfig, controller_flops  # noqa: E402

D, L, DKV, DFF = 4096, 32, 4096, 12288
PC, CC = ProjectorConfig(hidden_size=4096, state_hidden=1024, state_dim=256, delta_hidden=256, delta_dim=32), ControllerConfig(model_dim=384, num_layers=3, ffn_dim=768)
PROJ_ROW = 3 * (2 * PC.hidden_size * PC.state_hidden + 2 * PC.state_hidden * PC.state_dim) + 2 * (2 * PC.hidden_size * PC.delta_hidden + 2 * PC.delta_hidden * PC.delta_dim)
B, SEED = 10000, 23
REF = "D:C-0.9"


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def backbone_tflops(r):
    if r["layer_rows"] is None:
        T = r["rows_per_call_mean"]
        return r["nfe"] * L * T * (4 * D * T + 4 * D * D + 4 * D * DKV + 6 * D * DFF) / 1e12
    T = r["layer_rows"][0][0]
    return sum(sum(x * (4 * D * T + 4 * D * D + 4 * D * DKV + 6 * D * DFF) for x in lr) for lr in r["layer_rows"]) / 1e12


def dense_prefixes(r, block):
    """Prefix lengths of a dense TraceLock decode from its per-call writes (relative positions): window = [first blank, first blank + 32), block = the 32-block of the first blank."""
    P, G = r["prompt_length"], r["gen_length"]
    written = [False] * G
    first, out = 0, []
    for w in r["call_writes"]:
        while first < G and written[first]:
            first += 1
        out.append(P + (min(G, (first // 32 + 1) * 32) if block else min(G, first + 32)))
        for p in w:
            written[p] = True
    return out


def controller_tflops(r):
    if not r["arm"].endswith(":TraceLock"):
        return 0.0
    if r["layer_rows"] is None:
        ms = dense_prefixes(r, block=r["arm"].startswith("DB:"))
        proj = ms
    else:
        ms, proj = r["ctrl_prefix"], r["ctrl_proj_rows"]
    tot = 0
    for m, n in zip(ms, proj):
        tot += controller_flops([m], PC, CC)["controller_total_flops"] - PROJ_ROW * m + PROJ_ROW * n
    return tot / 1e12


def load(stage, rec_dir):
    sc = rec_dir / f"stage{stage}.scored.jsonl"
    rs = rows(sc if sc.exists() else rec_dir / f"stage{stage}.jsonl")
    return [r for r in rs if r.get("status") == "ok"]


class Panel:
    def __init__(self, rs, ids=None):
        self.arms = []
        for r in rs:
            if r["arm"] not in self.arms:
                self.arms.append(r["arm"])
        self.by = {a: {r["example_id"]: r for r in rs if r["arm"] == a} for a in self.arms}
        self.ids = sorted(set.intersection(*[set(v) for v in self.by.values()])) if ids is None else ids
        n = len(self.ids)
        self.idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        self.scored = all(self.by[a][i].get("correct") is not None for a in self.arms for i in self.ids)

    def arr(self, a, k):
        return np.array([float(self.by[a][i][k]) for i in self.ids])

    def correct(self, a):
        return self.arr(a, "correct")

    def paired(self, a, b):
        d = self.correct(a) - self.correct(b)
        bo = d[self.idx].sum(1)
        lost, gained = int(((self.correct(b) == 1) & (self.correct(a) == 0)).sum()), int(((self.correct(b) == 0) & (self.correct(a) == 1)).sum())
        return int(d.sum()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), lost, gained

    def speed(self, a, b):
        """latency of b over latency of a (a faster: > 1), with bootstrap CI over questions."""
        la, lb = self.arr(a, "latency_s"), self.arr(b, "latency_s")
        r = lb[self.idx].mean(1) / la[self.idx].mean(1)
        return float(lb.mean() / la.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def calls(self, a):
        return float(self.arr(a, "nfe").mean())


def fmt_pair(p):
    return f"{p[0]:+d} [{p[1]:+.0f}, {p[2]:+.0f}] ({p[3]}/{p[4]})"


def arm_table(P: Panel, title):
    lines = [f"### {title}", "", f"n = {len(P.ids)} questions common to all arms. Reference D:C-0.9 (dense confidence 0.9, 32-blocks, in the same process). Net = correct answers minus the reference's, 95% bootstrap over questions, (lost/gained).", ""]
    if not P.scored:
        lines += ["**Scoring is deferred for these records (HumanEval): run rescore_trace_records.py with bubblewrap on PATH first; accuracy columns omitted.**", ""]
    lines += ["| arm | correct | net vs D:C-0.9 | speed vs D:C-0.9 [CI] | calls | dense / partial calls | ms per call | controller ms per call | fallback calls | peak GPU GiB | backbone TFLOPs | controller TFLOPs | total vs D:C-0.9 | text = D:C-0.9 up to first EOS |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = None
    ref_tok = {i: P.by[REF][i]["tokens"] for i in P.ids} if REF in P.by else None

    def upto(t):
        e = next((k for k, x in enumerate(t) if x in (126081, 126348, 2)), len(t))
        return t[:e]

    for a in P.arms:
        bb = np.mean([backbone_tflops(P.by[a][i]) for i in P.ids])
        ct = np.mean([controller_tflops(P.by[a][i]) for i in P.ids])
        if a == REF:
            base = bb + ct
        cr = f"{int(P.correct(a).sum())}/{len(P.ids)}" if P.scored else "-"
        net = fmt_pair(P.paired(a, REF)) if P.scored else "-"
        sp = P.speed(a, REF) if REF in P.by else (float("nan"),) * 3
        ms = 1e3 * P.arr(a, "latency_s").sum() / P.arr(a, "nfe").sum()
        cms = 1e3 * P.arr(a, "rule_time_s").sum() / P.arr(a, "nfe").sum() if a.endswith(":TraceLock") else float("nan")
        same = np.mean([upto(P.by[a][i]["tokens"]) == upto(ref_tok[i]) for i in P.ids]) if ref_tok else float("nan")
        red = f"{100 * (1 - (bb + ct) / base):.1f}%" if base else "-"
        lines.append(f"| {a} | {cr} | {net} | {sp[0]:.2f}x [{sp[1]:.2f}, {sp[2]:.2f}] | {P.calls(a):.1f} | {P.arr(a, 'dense_calls').mean():.1f} / {P.arr(a, 'partial_calls').mean():.1f} | {ms:.1f} | "
                     f"{cms:.1f} | {P.arr(a, 'fallback_calls').mean():.1f} | {max(P.by[a][i].get('peak_mem_gib', 0) for i in P.ids):.1f} | {bb:,.0f} | {ct:.2f} | {red} | {same:.2f} |")
    return lines


PAIRS = [  # (label, a, b): "a minus b" in correct answers; speed of a relative to b
    ("geometry: DB:TraceLock (32-block) vs D:TraceLock (published moving window)", "DB:TraceLock", "D:TraceLock"),
    ("TraceLock vs D:C-0.9", "DB:TraceLock", "D:C-0.9"),
    ("cache effect, S32: S32:TraceLock vs DB:TraceLock", "S32:TraceLock", "DB:TraceLock"),
    ("cache effect, K4: K4:TraceLock vs DB:TraceLock", "K4:TraceLock", "DB:TraceLock"),
    ("cache effect, P4: P4:TraceLock vs DB:TraceLock", "P4:TraceLock", "DB:TraceLock"),
    ("P4 vs K4 under TraceLock", "P4:TraceLock", "K4:TraceLock"),
    ("P4 vs S32 under TraceLock", "P4:TraceLock", "S32:TraceLock"),
    ("TraceLock vs matched confidence, dense (DB:TraceLock vs D:C-matched)", "DB:TraceLock", "D:C-matched"),
    ("TraceLock vs matched confidence, S32", "S32:TraceLock", "S32:C-matched"),
    ("TraceLock vs matched confidence, P4", "P4:TraceLock", "P4:C-matched"),
    ("TraceLock vs the E2 threshold, dense (256)", "DB:TraceLock", "D:C-adaptive"),
    ("TraceLock vs the E2 threshold, S32 (256)", "S32:TraceLock", "S32:C-0.9567"),
    ("TraceLock vs the E2 threshold, P4 (256)", "P4:TraceLock", "P4:C-0.9567"),
]


def pair_table(P: Panel):
    lines = ["| comparison (a vs b) | paired net a - b [CI] (lost/gained) | speed of a over b [CI] | calls a / b | calls ratio |", "|---|---|---|---|---|"]
    for label, a, b in PAIRS:
        if a in P.by and b in P.by:
            pr = fmt_pair(P.paired(a, b)) if P.scored else "-"
            sp = P.speed(a, b)
            ca, cb = P.calls(a), P.calls(b)
            lines.append(f"| {label} | {pr} | {sp[0]:.2f}x [{sp[1]:.2f}, {sp[2]:.2f}] | {ca:.1f} / {cb:.1f} | {ca / cb:.3f} |")
    return lines


def decisions(P: Panel, name):
    """The pre-registered decision rules (PREREGISTERED.md), evaluated on one panel."""
    out = [f"**Decision rules, {name}** (resolution about +-6 answers at these sizes; 'resolved' only if the 95% interval excludes 0)", ""]
    if not P.scored:
        return out + ["- not evaluated: records are unscored."]
    if all(a in P.by for a in ("P4:TraceLock", "K4:TraceLock")):
        p, s = P.paired("P4:TraceLock", "K4:TraceLock"), P.speed("P4:TraceLock", "K4:TraceLock")
        out.append(f"- R1 (P4 beats K4 under TraceLock: more than 5% faster at paired answers >= -2): speed {s[0]:.3f}x, paired {p[0]:+d} -> {'HOLDS' if s[0] > 1.05 and p[0] >= -2 else 'does not hold'}.")
    if all(a in P.by for a in ("S32:TraceLock", "DB:TraceLock")):
        rs = []
        for c in ("S32", "K4", "P4"):
            if f"{c}:TraceLock" in P.by:
                p, s = P.paired(f"{c}:TraceLock", "DB:TraceLock"), P.speed(f"{c}:TraceLock", "DB:TraceLock")
                rs.append(f"{c}: paired {p[0]:+d} [{p[1]:+.0f}, {p[2]:+.0f}], {s[0]:.2f}x -> {'composes' if p[0] >= -4 and s[0] > 1.3 else 'does not compose'}")
        out.append("- R2 (a cell composes with TraceLock if paired >= -4 against the dense block twin DB:TraceLock and more than 1.3x faster): " + "; ".join(rs) + ".")
    for c, a, b in (("dense, 32-block", "DB:TraceLock", "D:C-matched"), ("S32", "S32:TraceLock", "S32:C-matched"), ("P4", "P4:TraceLock", "P4:C-matched"),
                    ("dense, published window, vs the E2 threshold", "D:TraceLock", "D:C-adaptive"), ("S32, vs the E2 threshold", "S32:TraceLock", "S32:C-0.9567"), ("P4, vs the E2 threshold", "P4:TraceLock", "P4:C-0.9567")):
        if a in P.by and b in P.by:
            p = P.paired(a, b)
            ratio = P.calls(a) / P.calls(b)
            out.append(f"- R3 ({c}) (TraceLock beats the matched confidence threshold at equal calls: paired >= +4, calls within 5%): paired {p[0]:+d} [{p[1]:+.0f}, {p[2]:+.0f}], calls ratio {ratio:.3f} -> "
                       f"{'HOLDS' if p[0] >= 4 and abs(ratio - 1) <= 0.05 else 'does not hold' if abs(ratio - 1) <= 0.05 else 'not at equal calls (ratio outside 5%): no verdict'}"
                       f"{' (resolved)' if p[1] > 0 or p[2] < 0 else ' (interval contains 0)'}.")
    if all(a in P.by for a in ("D:TraceLock", REF)):
        p = P.paired("D:TraceLock", REF)
        bound = -12 if "HumanEval 164" in name else -8 if "MATH 108" in name else None
        verdict = "n/a for this panel (bounds are stated for HumanEval 164: -12 and MATH 108: -8)" if bound is None else ("TRIGGERED: the 512 composition grid is uninformative, the 256 cell is primary" if p[0] <= bound else "not triggered")
        out.append(f"- R4 (length sentinel): D:TraceLock net vs D:C-0.9 = {p[0]:+d} [{p[1]:+.0f}, {p[2]:+.0f}] -> {verdict}.")
    return out


def pooled(rs_by_stage):
    """MATH 108 + HumanEval 164 at 512 as one panel (ids are disjoint across the two sets)."""
    rs = [r for s in ("M512", "H512") for r in rs_by_stage.get(s, [])]
    return Panel(rs) if rs else None


def selection_report(rec_dir):
    p = rec_dir / "selection.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    lines = ["# Matched-confidence selection (written by the SEL stage)", "", f"- c* = {s['c_star']} ({s.get('status')}; usable: {s.get('usable')}); target = mean calls of DB:TraceLock on the {s.get('n_questions')} selection questions = "
             f"{s.get('target_calls_DB_TraceLock', float('nan')):.2f}; window D:TraceLock on the same questions: {s.get('mean_calls_D_TraceLock_window', float('nan')):.2f}; controller collapsed: {s.get('controller_collapsed')}.",
             f"- questions: {s.get('datasets')}, {s.get('gen_length')} tokens; fallback: {s.get('fallback')}", "", "| threshold | mean calls |", "|---|---|"]
    lines += [f"| {t} | {c:.2f} |" for t, c in s.get("mean_calls_by_threshold", {}).items()]
    return lines


def main():
    rec_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "records"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    rs_by = {s: load(s, rec_dir) for s in ("M512", "P256", "H512")}
    sel = selection_report(rec_dir)
    if sel:
        (out_dir / "results_selection.md").write_text("\n".join(sel) + "\n")
    names = {"M512": "MATH 108 (54 level-5 + 54 level-1-4), 512 tokens", "H512": "HumanEval 164, 512 tokens", "P256": "the earlier 124-case panel (HumanEval 0-69 + MATH level-5 54), 256 tokens"}
    combined = ["# TraceLock under our representation caches: results (computed by analyze_trace_cells.py; read PREREGISTERED.md for the predictions and decision rules)", ""]
    for s, rs in rs_by.items():
        if not rs:
            continue
        P = Panel(rs)
        lines = [f"# {names[s]}", ""] + arm_table(P, names[s]) + ["", "#### Paired comparisons", ""] + pair_table(P) + [""] + decisions(P, names[s])
        e2 = [(a, [P.by[a][i].get("eq_e2_tokens") for i in P.ids]) for a in ("D:TraceLock", "D:C-adaptive") if a in P.by]
        if e2:
            lines += ["", "Informational: share of questions whose dense 256-token decode equals the earlier E2 record token for token: " + "; ".join(f"{a} {sum(1 for x in v if x)}/{sum(1 for x in v if x is not None)}" for a, v in e2) + "."]
        (out_dir / f"results_{s}.md").write_text("\n".join(lines) + "\n")
        combined += lines + [""]
    PP = pooled(rs_by)
    if PP is not None and "M512" in rs_by and "H512" in rs_by and rs_by["M512"] and rs_by["H512"]:
        lines = ["# Pooled MATH 108 + HumanEval 164 at 512 tokens", ""] + arm_table(PP, "pooled 272 questions") + ["", "#### Paired comparisons", ""] + pair_table(PP) + [""] + decisions(PP, "pooled")
        (out_dir / "results_pooled512.md").write_text("\n".join(lines) + "\n")
        combined += lines
    if sel:
        combined += [""] + sel
    (out_dir / "results.md").write_text("\n".join(combined) + "\n")
    print("\n".join(combined))


if __name__ == "__main__":
    main()
