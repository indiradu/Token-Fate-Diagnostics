"""Study 5 table per (model, set): every representation cache under the shared Conf-0.9 word lock against the original decoder.
Usage: python analyze_study5.py MODEL_TAG [SET ...]  ->  results/replock_clean_20261117/results_study5_<MODEL_TAG>.md
Columns: speed vs the original decoder (ratio of total time over the prompts, bootstrap CI over prompts), speed vs the SAME STACK's dense Conf-0.9 run (isolates the cache gain from implementation overhead),
calls, ms per call, FLOP cut (mean total FLOPs executed, GEMM + attention including the vocabulary head, over the first 20 prompts vs the original decoder's; the head cannot be separated for the engine arms, whose final projection is not a named module), correct and paired delta vs the original (bootstrap CI over prompts, whether it includes 0).
Sorted by speed. No tolerance is applied."""
import json
import sys
from pathlib import Path

import numpy as np

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
rng = np.random.default_rng(0)


def rows(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p)]


def pick(name):                                                 # the scored twin when it exists and is complete, else the raw file
    s, r = REC / name.replace(".jsonl", ".scored.jsonl"), REC / name
    rs, rr = rows(s), rows(r)
    return rs if len(rs) >= len(rr) else rr


def load(tag, ds, prefix=""):
    arms, stack = {}, {}
    for r in pick(f"{prefix}pd_{tag}_{ds}.jsonl"):
        arms.setdefault(r["arm"], {})[r["example_id"]] = r
    for r in pick(f"{prefix}eng_{tag}_{ds}.jsonl"):
        arms.setdefault(r["arm"] if r["arm"] != "DENSE_C09" else "DENSE_C09_eng", {})[r["example_id"]] = r
    for r in pick(f"{prefix}eng_{tag}b32_{ds}.jsonl"):                  # Dream: the original decoder defined as for LLaDA (one token per step, block 32), run through our engine
        arms.setdefault("ORIG", {})[r["example_id"]] = r
    for a in ("DENSE_SDPA", "FD_DUAL", "DLLM", "DENSE_EAGER", "D2CACHE", "ORIG_NATIVE"):
        for r in pick(f"{prefix}d2c_{tag}_{ds}_{a}.jsonl"):
            arms.setdefault(a, {})[r["example_id"]] = r
    for a in arms:
        stack[a] = {"ORIG": "pd", "DENSE_C09": "pd", "SURELOCK": "pd", "DENSE_C09_eng": "eng", "ELASTIC": "eng", "R0": "eng", "CTL": "eng"}.get(a, "d2c")
    return arms, stack


LABEL = {"ORIG": "Original decoder", "ORIG_NATIVE": "Dream native decoding (entropy order, whole length, one token per step; extra reference)", "DENSE_C09": "Dense Conf-0.9 (pd engine)", "SURELOCK": "SureLock (ICLR'26, our reproduction)", "DENSE_C09_eng": "Dense Conf-0.9 (selective engine anchor)",
         "ELASTIC": "Elastic-Cache (ICLR'26, our reproduction)", "R0": "R0 (ours, derived rule)", "CTL": "Learned controller (ours)", "DENSE_SDPA": "Dense Conf-0.9 (d2Cache stack anchor)",
         "FD_DUAL": "Fast-dLLM dual cache (ICLR'26, d2Cache repo)", "DLLM": "dLLM-Cache (ICML'26, d2Cache repo)", "DENSE_EAGER": "Dense Conf-0.9 (d2Cache stack, eager)", "D2CACHE": "d2Cache (ICLR'26)"}
OWN_DENSE = {"pd": "DENSE_C09", "eng": "DENSE_C09_eng", "d2c": "DENSE_SDPA"}


def ratio_ci(num, den, B=2000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    pt = num.sum() / den.sum()
    bs = [num[i].sum() / den[i].sum() for i in (rng.integers(0, len(num), len(num)) for _ in range(B))]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def table(tag, ds):
    arms, stack = load(tag, ds)
    fl, _ = load(tag, ds, "flops_")
    bn = "ORIG" if "ORIG" in arms else "ORIG_NATIVE"
    if bn not in arms:
        return f"\n### {tag} / {ds}\n(original decoder not run yet)\n"
    base = arms[bn]
    ids = sorted(base)
    fids = sorted(fl.get(bn, {}))
    bf = np.mean([fl[bn][i]["flops_total"] for i in fids]) if fids else None
    out = []
    for a, d in arms.items():
        ii = [i for i in ids if i in d]
        if len(ii) < 0.9 * len(ids) and a != bn:
            out.append((0, f"| {LABEL.get(a, a)} | incomplete ({len(ii)}/{len(ids)}) | | | | | | |"))
            continue
        L = [d[i]["latency_s"] for i in ii]
        sp, lo, hi = ratio_ci([base[i]["latency_s"] for i in ii], L)
        own = arms.get(OWN_DENSE[stack[a]])
        so = ""
        if own and a != bn and a != OWN_DENSE[stack[a]] and all(i in own for i in ii):
            p2, l2, h2 = ratio_ci([own[i]["latency_s"] for i in ii], L)
            so = f"{p2:.2f}x [{l2:.2f}, {h2:.2f}]"
        calls = np.mean([d[i]["nfe"] for i in ii])
        mspc = 1000 * sum(L) / sum(d[i]["nfe"] for i in ii)
        c = [d[i].get("correct") for i in ii]
        acc = "pending" if any(x is None for x in c) else None
        delta = ""
        if acc is None:
            cc, bb = np.array(c, float), np.array([base[i]["correct"] for i in ii], float)
            diff = cc - bb
            bs = [diff[rng.integers(0, len(ii), len(ii))].sum() for _ in range(2000)]
            lo_d, hi_d = np.percentile(bs, [2.5, 97.5])
            delta = f"{int(diff.sum()):+d} [{lo_d:+.0f}, {hi_d:+.0f}]" + ("" if (lo_d > 0 or hi_d < 0) else " (incl. 0)")
            acc = f"{int(cc.sum())}/{len(ii)}"
        d2 = ""
        if own and a != bn and a != OWN_DENSE[stack[a]] and all(i in own for i in ii) and all(own[i].get("correct") is not None for i in ii) and acc is not None and not acc.startswith("pending"):
            dd = np.array(c, float) - np.array([own[i]["correct"] for i in ii], float)
            bs2 = [dd[rng.integers(0, len(ii), len(ii))].sum() for _ in range(2000)]
            l2, h2 = np.percentile(bs2, [2.5, 97.5])
            d2 = f"{int(dd.sum()):+d} [{l2:+.0f}, {h2:+.0f}]" + ("" if (l2 > 0 or h2 < 0) else " (incl. 0)")
        fc = "n/a"
        if bf and a in fl and all(i in fl[a] for i in fids):
            af = np.mean([fl[a][i]["flops_total"] for i in fids])
            ah = np.mean([fl[a][i]["flops_head"] for i in fids])
            fc = f"{100 * (1 - af / bf):.1f}% ({af / 1e12:.0f} TF)"
        pk = np.mean([d[i].get("peak_mem_gib", float("nan")) for i in ii])
        out.append((sp, f"| {LABEL.get(a, a)} | {sp:.1f}x [{lo:.1f}, {hi:.1f}] | {so} | {calls:.1f} | {mspc:.0f} | {fc} | {acc} | {delta} | {d2} | {pk:.1f} |"))
    out.sort(key=lambda x: -x[0])
    head = (f"\n### {tag} / {ds} (n={len(ids)} prompts; FLOP pass n={len(fids)})\n| arm | speed vs original [95% CI] | speed vs own stack's dense | calls | ms/call | FLOP cut (total GEMM+attention incl. vocabulary head; mean TFLOPs per prompt) | correct | paired delta vs original | paired delta vs same-stack dense Conf-0.9 | peak GiB |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(x[1] for x in out) + "\n"


if __name__ == "__main__":
    tag = sys.argv[1]
    sets = sys.argv[2:] or ["he164", "kod_held", "gsm_held", "math108"]
    md = f"# Study 5: representation caches under the shared Conf-0.9 word lock, {tag}\n(generated by analyze_study5.py; sorted by speed; no tolerance applied; correct = paired over prompts)\n"
    for ds in sets:
        md += table(tag, ds)
    (REC.parent / f"results_study5_{tag}.md").write_text(md)
    print(md)
