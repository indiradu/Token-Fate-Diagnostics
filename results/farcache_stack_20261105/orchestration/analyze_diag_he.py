"""Analysis of the ADDED HumanEval diagnostics (records/diag_he.jsonl) and of the stage-1 HumanEval records: which tasks lose answers, is it degenerate text, and which ingredient causes it.
Usage: <abs python> analyze_diag_he.py   -> diag_he_results.md, diag_he_summary.json"""
import ast
import collections
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
NB = 10000
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def code(t):
    m = re.search(r"```(?:python)?\n(.*?)(```|$)", t, re.S)
    return m.group(1) if m else t


def valid(t):
    try:
        ast.parse(code(t))
        return True
    except Exception:
        return False


def runlen(tokens, fe):
    body = tokens[:fe] if fe is not None else tokens
    best = run = 1
    for a, b in zip(body, body[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best, len(body)


def ci(v):
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def table(by, arms, ref, md, title, summ):
    ids = sorted(set.intersection(*[set(by[a]) for a in arms]))
    n = len(ids)
    rng = np.random.default_rng(23)
    idx = rng.integers(0, n, size=(NB, n))
    cor = {a: np.array([int(by[a][i]["correct"]) for i in ids]) for a in arms}
    lat = {a: np.array([by[a][i]["latency_s"] for i in ids]) for a in arms}
    nfe = {a: np.array([by[a][i]["nfe"] for i in ids]) for a in arms}
    md.append(f"\n### {title} (n = {n}; reference {ref}: {int(cor[ref].sum())} correct)\n")
    md.append("| cell | speed vs ref [CI] | calls | dense calls | rows/call | correct | net vs ref [CI] (lost/gained) | syntactically valid | max token run >= 8 | first EOS earlier than ref by >= 20 (of lost) | text differs from ref |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        sp = lat[ref].sum() / lat[a].sum()
        lo, hi = ci(lat[ref][idx].sum(1) / lat[a][idx].sum(1))
        d = (cor[a] - cor[ref]).astype(float)
        nlo, nhi = ci(d[idx].sum(1))
        lost = [i for i in ids if by[ref][i]["correct"] and not by[a][i]["correct"]]
        gained = [i for i in ids if not by[ref][i]["correct"] and by[a][i]["correct"]]
        v = sum(valid(by[a][i]["text"]) for i in ids)
        rl = sum(runlen(by[a][i]["tokens"], by[a][i]["first_eos"])[0] >= 8 for i in ids)
        early = sum(1 for i in lost if (by[a][i]["first_eos"] or 512) < (by[ref][i]["first_eos"] or 512) - 20)
        ch = float(np.mean([by[a][i]["text"] != by[ref][i]["text"] for i in ids]))
        dc = np.mean([by[a][i].get("dense_calls", 0) for i in ids])
        rpc = np.mean([by[a][i].get("rows_per_call_mean", 0) for i in ids])
        md.append(f"| {a} | {sp:.3f} [{lo:.3f}, {hi:.3f}] | {nfe[a].mean():.1f} | {dc:.1f} | {rpc:.0f} | {int(cor[a].sum())}/{n} | {int(d.sum()):+d} [{nlo:+.0f}, {nhi:+.0f}] ({len(lost)}/{len(gained)}) | {v}/{n} | {rl} | {early}/{len(lost)} | {ch:.2f} |")
        summ[a] = {"speed": [float(sp), lo, hi], "calls": float(nfe[a].mean()), "correct": int(cor[a].sum()), "net": [int(d.sum()), nlo, nhi, len(lost), len(gained)], "valid": v, "n": n,
                   "rows_per_call": float(rpc), "dense_calls": float(dc)}
    return ids


def main():
    md = ["# Added after seeing interim HumanEval: why does the cache lose answers on code?\n",
          "(Not part of PREREGISTERED.md. WORD LOCK = C-0.9 everywhere here; the cells change the cache, the EOS truncation or the refresh frequency one at a time. HumanEval 0-163, gen 512, dense reference D:C-0.9 in the same process.)\n"]
    summ = {}
    # (1) stage-1 records, six cache cells: which tasks lose, lost-answer anatomy
    s1 = rows(OUT / "records/stage1.jsonl")
    by = collections.defaultdict(dict)
    for r in s1:
        if r["set"] == "he164" and r["status"] == "ok":
            by[r["arm"]][r["example_id"]] = r
    if by:
        arms = [a for a in by if a != "D:C-0.9"]
        md.append("## 1. Anatomy of the lost answers in the stage-1 HumanEval cells\n")
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        base = by["D:C-0.9"]
        md.append(f"Of {sum(base[i]['correct'] for i in ids)} tasks that dense C-0.9 solves, the number lost by each cell; and how many cells lose each task.\n")
        cnt = collections.Counter()
        for a in arms:
            if a.startswith("S"):
                for i in ids:
                    if base[i]["correct"] and not by[a][i]["correct"]:
                        cnt[i] += 1
        md.append(f"Tasks lost by at least one cache cell: {len(cnt)}; by all six: {sum(c == 6 for c in cnt.values())}; by at least four: {sum(c >= 4 for c in cnt.values())} ({', '.join(i for i, c in cnt.most_common() if c >= 4)}).\n")
        table(by, ["D:C-0.9"] + arms, "D:C-0.9", md, "Stage-1 cells, all HumanEval tasks", summ.setdefault("stage1", {}))
        # lost answers: failure type from the scorer status
        st = collections.Counter(by[a][i]["scoring_status"] for a in arms if a.startswith("S") for i in ids if base[i]["correct"] and not by[a][i]["correct"])
        md.append(f"\nScorer status of the lost answers (all cache cells): {dict(st)}.\n")
    d = rows(OUT / "records/diag_he.jsonl")
    byd = collections.defaultdict(dict)
    for r in d:
        if r.get("status") == "ok":
            byd[r["arm"]][r["example_id"]] = r
    if byd:
        arms = [a for a in ["D:C-0.9", "S32:main", "S32:every+eos", "S32:block-noeos", "S32:nofar", "S32:block+k2", "S32:block+k4", "S16:every+eos", "S16:block-noeos"] if a in byd]
        md.append("\n## 2. One-ingredient-at-a-time cells (own process, same 164 tasks)\n")
        table(byd, arms, "D:C-0.9", md, "Diagnostic process", summ.setdefault("diag", {}))
        if "S32:main" in byd:
            # parity with the stage-1 S32:C-0.9 cell
            same = [i for i in byd["S32:main"] if i in by.get("S32:C-0.9", {})]
            eq = sum(byd["S32:main"][i]["tokens"] == by["S32:C-0.9"][i]["tokens"] for i in same)
            md.append(f"\nParity: S32:main tokens equal the stage-1 S32:C-0.9 tokens on {eq}/{len(same)} tasks (same question, same code path).\n")
        # task overlap: which tasks does each cell lose
        ids = sorted(set.intersection(*[set(byd[a]) for a in arms]))
        for a in arms[1:]:
            lost = {i for i in ids if byd["D:C-0.9"][i]["correct"] and not byd[a][i]["correct"]}
            main = {i for i in ids if byd["D:C-0.9"][i]["correct"] and not byd["S32:main"][i]["correct"]} if "S32:main" in byd else set()
            md.append(f"- {a}: lost {len(lost)}; overlap with the tasks lost by S32:main {len(lost & main)}")
    Path(OUT / "diag_he_results.md").write_text("\n".join(md) + "\n")
    Path(OUT / "diag_he_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    main()
