"""Analysis 2 (CPU): oracle repair at the first divergence. Reads records/oracle_census_<lock>.jsonl and records/oracle_repair_<lock>.jsonl.
Usage: <python> analyze_oracle_repair.py [--records DIR] [--out DIR]   -> oracle_repair_results.md, oracle_repair_summary.json
Counts with Wilson 95% intervals (few tasks: no bootstrap); a task is "lost" when the dense decode with the same word lock passes the HumanEval tests and the S32 cached decode fails."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
VARIANT_ORDER = ["next_blank", "far2_blank", "inblock_old", "prompt", "prior_written", "all_blank", "all_cached"]
LABEL = {"next_blank": "(a) blank rows of the next block", "far2_blank": "(b) blank rows beyond the next block (and past-EOS blanks)", "inblock_old": "(c) written rows of the current block older than one step",
         "prompt": "(d) prompt rows", "prior_written": "(f) written rows of earlier blocks", "all_blank": "(e) all cached blank rows (positive control)", "all_cached": "(g) all cached rows (dense-equivalent control)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})" if n else "0/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT / "records"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    md = ["# Analysis 2: oracle repair at the first divergence", "",
          "HumanEval 0-163, 512 tokens. S32 representation cache (dense call at every block start) vs the dense decode of the same word lock (S32 engine, dense call every call, first-EOS truncation = the dense decoder's tokens up to the first EOS).",
          "A task's FIRST DIVERGENCE is the first call at which the cached decode's write event (which blank positions were given which token) differs from the dense decode's. The cached decode is resumed from the last block-start dense call before it with one row class repaired.",
          "`oracle`: the class's cached K/V are replaced by those of a dense pass over the current tokens. `compute`: the class's rows are computed in the call instead of read from the cache. `seg`: repaired only in the calls of the block of the divergence; `end`: from there to the end of the decode.",
          "RESTORED = the repaired decode's write events equal the dense decode's through the first-divergence call (its own first divergence from dense is later than the original, or none). PASSES = the final program passes the tests.", ""]
    summ: dict = {}
    for lock in ("C-0.9", "Lctx-0.8"):
        cen = {r["example_id"]: r for r in rows(Path(a.records) / f"oracle_census_{lock}.jsonl") if r.get("status") == "ok"}
        rep = [r for r in rows(Path(a.records) / f"oracle_repair_{lock}.jsonl") if r.get("status") == "ok"]
        if not cen:
            continue
        lost = [q for q, r in cen.items() if r["dense_correct_stage1"] and not r["cached_correct_stage1"]]
        retained = [q for q, r in cen.items() if q not in lost]
        md += [f"## Word lock {lock}", "", f"Census tasks: {len(cen)}; lost tasks (dense passes, S32 fails): {len(lost)}. Census parity with stage 1: dense decode equal to the stage-1 dense tokens up to the first EOS on {sum(r['eq_stage1_dense_to_eos'] for r in cen.values())}/{len(cen)}, cached decode equal to the stage-1 S32 tokens on {sum(r['eq_stage1_cached'] for r in cen.values())}/{len(cen)}.", ""]
        s = summ.setdefault(lock, {"census": len(cen), "lost": len(lost)})

        def divstats(qs, name):
            ds = [cen[q]["divergence"] for q in qs if cen[q]["divergence"] is not None]
            if not ds:
                return
            kinds = collections.Counter(d["kind"] for d in ds)
            rel = [cen[q]["div_rel_pos"] for q in qs if "div_rel_pos" in cen[q]]
            inb = [cen[q]["div_pos_in_block"] for q in qs if "div_pos_in_block" in cen[q]]
            fr = [cen[q]["div_rel_pos"] / max(cen[q]["dense_first_eos"] or 512, 1) for q in qs if "div_rel_pos" in cen[q]]
            blk = [cen[q]["div_block"] for q in qs if "div_block" in cen[q]]
            after = [cen[q]["div_calls_after_block_start"] for q in qs if "div_calls_after_block_start" in cen[q]]
            q1 = lambda v: (f"{np.median(v):.0f} [{np.quantile(v, .25):.0f}, {np.quantile(v, .75):.0f}]" if len(v) else "-")
            q1f = lambda v: (f"{np.median(v):.2f} [{np.quantile(v, .25):.2f}, {np.quantile(v, .75):.2f}]" if len(v) else "-")
            md.append(f"| {name} | {len(qs)} | {len(ds)} | {dict(kinds)} | {q1(rel)} | {q1f(fr)} | {q1(inb)} | {q1(blk)} | {q1(after)} |")
            s.setdefault("divergence", {})[name] = {"tasks": len(qs), "diverged": len(ds), "kinds": dict(kinds), "rel_pos_median": float(np.median(rel)) if rel else None}

        md += ["### First divergence of the cached decode from the dense decode", "",
               "Position = answer position (token index in the generated text) of the lowest position whose write differs; fraction = position / dense answer length (first EOS); calls after block start = divergence call - dense call that opened the block. kind: token = the two decodes wrote different tokens at the same position; a_only / b_only = the cached / the dense decode wrote a position the other did not (word-lock set difference).", "",
               "| tasks | n | diverged | kinds | answer position | fraction of dense answer | position in block | block index | calls after block start |", "|---|---|---|---|---|---|---|---|---|"]
        divstats(lost, "lost")
        divstats(retained, "retained (all other census tasks)")
        md.append("")
        if not rep:
            continue
        # ---- repair table
        by = collections.defaultdict(list)
        for r in rep:
            by[(r["variant"], r["mode"], r["window"])].append(r)
        null = by.get(("null", "-", "-"), [])
        md += [f"Null control (resume from the dense call, nothing repaired, must equal the cached decode): {sum(r['equals_cached'] for r in null)}/{len(null)}.", ""]
        tasks = sorted({r["example_id"] for r in rep})
        md += [f"### Repair of one row class, {len(tasks)} lost tasks", "",
               "RESTORED: write events equal the dense decode's through the original divergence call. POS TOKEN: the token at the first divergent position equals the dense token (defined only when the divergence is a position that the cached decode wrote: kind token or a_only). IDENTICAL: final tokens equal dense to the first EOS. PASSES: HumanEval tests pass. `class non-empty`: tasks where the repaired class had rows at some repaired call (oracle) or added rows to the query set (compute). Wilson 95% intervals.", "",
               "| class | mode | window | tasks | class non-empty | RESTORED | POS TOKEN | IDENTICAL | PASSES |", "|---|---|---|---|---|---|---|---|---|"]
        tab = {}
        for v in VARIANT_ORDER:
            for m in ("oracle", "compute"):
                for w in ("seg", "end"):
                    rs = by.get((v, m, w), [])
                    if not rs:
                        continue
                    n = len(rs)
                    nullrows = {r["example_id"]: r["rows_per_call"] for r in null}

                    def nonempty(r):
                        if m == "oracle":
                            cl = {"next_blank": ["next_blank"], "far2_blank": ["far_blank"], "inblock_old": ["inblock_old"], "prompt": ["prompt"], "prior_written": ["prior_written"],
                                  "all_blank": ["next_blank", "far_blank"], "all_cached": ["prompt", "prior_written", "inblock_old", "next_blank", "far_blank"]}[v]
                            return any(r.get("class_rows", {}).get(c, {}).get("calls_nonempty", 0) > 0 for c in cl)
                        nr = nullrows.get(r["example_id"])
                        return bool(nr) and sum(r["rows_per_call"]) > sum(nr[:len(r["rows_per_call"])])
                    ne = sum(nonempty(r) for r in rs)
                    k_rest = sum(r["restored_through_d"] for r in rs)
                    pt = [r["pos_token_restored"] for r in rs if "pos_token_restored" in r]
                    k_id = sum(r["identical_to_dense_to_eos"] for r in rs)
                    k_pass = sum(r["correct"] for r in rs)
                    md.append(f"| {LABEL[v]} | {m} | {w} | {n} | {ne} | {frac(k_rest, n)} | {frac(sum(pt), len(pt))} | {frac(k_id, n)} | {frac(k_pass, n)} |")
                    tab[f"{v}|{m}|{w}"] = {"n": n, "nonempty": ne, "restored": k_rest, "pos_token": [sum(pt), len(pt)], "identical": k_id, "passes": k_pass}
        s["repair"] = tab
        md.append("")
        # per-task view of the oracle/seg runs
        md += ["### Per task (oracle, window = seg): which classes restore the divergence", "", "| task | first-divergence call | kind | answer position | " + " | ".join(v for v in VARIANT_ORDER) + " |", "|---|---|---|---|" + "---|" * len(VARIANT_ORDER)]
        for q in tasks:
            d = cen[q]["divergence"]
            cells = []
            for v in VARIANT_ORDER:
                rr = [r for r in by.get((v, "oracle", "seg"), []) if r["example_id"] == q]
                cells.append("-" if not rr else ("R" if rr[0]["restored_through_d"] else ".") + ("P" if rr[0]["correct"] else "f"))
            md.append(f"| {q} | {d['call']} | {d['kind']} | {cen[q].get('div_rel_pos', '-')} | " + " | ".join(cells) + " |")
        md += ["", "R = restored through the divergence call, . = not; P = final program passes, f = fails.", ""]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "oracle_repair_results.md").write_text("\n".join(md) + "\n")
    (Path(a.out) / "oracle_repair_summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
