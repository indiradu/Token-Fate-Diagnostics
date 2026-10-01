"""Score the collaborator's outputs with OUR scorer and compare them to the same-job B0 (our environment).

smoke: hard checks (her dense == our B0 token-for-token, our B0 == yesterday's B0, every method completes).
full:  validation, parity report, pooled paired statistics with example-clustered bootstrap, results.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

CAMPAIGN = Path(__file__).resolve().parents[1]
ALIGNED = CAMPAIGN.parent / "aligned_standard_g256_20260929"
CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
os.environ["HF_MODULES_CACHE"] = str(ALIGNED / "runtime_cache/model_modules")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(ALIGNED / "source/src"))
sys.path.insert(0, str(ALIGNED / "orchestration"))
import run_aligned as ra  # noqa: E402
from regret_remasking.paper_decode import _decode_response  # noqa: E402
from regret_remasking.paper_experiment import score_paper_generation  # noqa: E402


def load_entries(names):
    entries = {}
    for name in names:
        for line in (ALIGNED / "manifests" / name).read_text().splitlines():
            e = json.loads(line)
            entries[(e["dataset"], e["example_id"])] = e
    return entries


def score_collab(records, entries, tokenizer):
    """Decode token ids exactly as our decoder does, then score with our scorer (repeat 1)."""
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["dataset"], r["example_id"], r["policy_id"])].append(r)
    scored = []
    for key, group in grouped.items():
        if len({json.dumps(g["token_ids"]) for g in group}) != 1:
            raise ValueError(f"repeats differ in output for {key}")
        text = _decode_response(tokenizer, group[0]["token_ids"])
        correct, status = score_paper_generation(text, entries[key[:2]])
        if correct is None:
            raise ValueError(f"unscorable output for {key}: {status}")
        for g in group:
            g.update(correct=bool(correct), scoring_status=status, text=text)
            scored.append(g)
    return scored


def pooled(rows, order, baseline="B0"):
    latency = defaultdict(list)
    first = {}
    for r in rows:
        latency[(r["dataset"], r["example_id"], r["policy_id"])].append(r["latency_s"])
        if r["repeat"] == 1:
            first[(r["dataset"], r["example_id"], r["policy_id"])] = r
    cases = sorted({(d, e) for d, e, _ in first})
    out = {}
    for policy in order:
        paired = []
        for d, e in cases:
            b, c = first[(d, e, baseline)], first[(d, e, policy)]
            paired.append({"dataset": d, "base_correct": int(b["correct"]), "cand_correct": int(c["correct"]),
                           "base_time": statistics.median(latency[(d, e, baseline)]), "cand_time": statistics.median(latency[(d, e, policy)]),
                           "changed": int(b["token_ids"] != c["token_ids"]), "nfe": c["nfe"]})
        acc = lambda s: sum(p["cand_correct"] - p["base_correct"] for p in s) / len(s)
        spd = lambda s: 100.0 * (sum(p["base_time"] for p in s) / sum(p["cand_time"] for p in s) - 1.0)
        res = {}
        for scope in ["math500", "humaneval", "overall"]:
            sub = [p for p in paired if scope == "overall" or p["dataset"] == scope]
            res[scope] = {"n": len(sub), "correct": sum(p["cand_correct"] for p in sub), "baseline_correct": sum(p["base_correct"] for p in sub),
                          "accuracy_delta_pp": 100 * acc(sub), "accuracy_delta_pp_ci95": [100 * v for v in ra._bootstrap(sub, acc)] if policy != baseline else [0.0, 0.0],
                          "seconds_per_example": sum(p["cand_time"] for p in sub) / len(sub), "speedup_pct": spd(sub),
                          "speedup_pct_ci95": list(ra._bootstrap(sub, spd)) if policy != baseline else [0.0, 0.0],
                          "mean_nfe": sum(p["nfe"] for p in sub) / len(sub), "sequence_changed": sum(p["changed"] for p in sub),
                          "correct_to_wrong": sum(1 for p in sub if p["base_correct"] and not p["cand_correct"]),
                          "wrong_to_correct": sum(1 for p in sub if p["cand_correct"] and not p["base_correct"])}
        out[policy] = res
    return out


def token_parity(left, right, left_id, right_id):
    a = {(r["dataset"], r["example_id"]): r["token_ids"] for r in left if r["policy_id"] == left_id and r["repeat"] == 1}
    b = {(r["dataset"], r["example_id"]): r["token_ids"] for r in right if r["policy_id"] == right_id and r["repeat"] == 1}
    keys = sorted(set(a) & set(b))
    mism = []
    for k in keys:
        if a[k] != b[k]:
            first = next(i for i, (x, y) in enumerate(zip(a[k], b[k])) if x != y)
            mism.append({"case": "/".join(k), "first_diff_position": first, "tokens_differ": sum(x != y for x, y in zip(a[k], b[k]))})
    return {"compared": len(keys), "matched": len(keys) - len(mism), "mismatched": mism}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--collab-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    names = ["smoke_pair.jsonl"] if args.stage == "smoke" else ["eval_math.jsonl", "eval_code.jsonl"]
    entries = load_entries(names)
    b0 = ra.rows(args.b0_dir / "records.jsonl")
    for r in b0:
        if r["status"] != "ok" or r["nfe"] != 256 or r["final_mask_count"]:
            raise ValueError("invalid same-job B0 record")
    collab = score_collab(ra.rows(args.collab_dir / "records.jsonl"), entries, tokenizer)
    methods = sorted({r["policy_id"] for r in collab}, key=lambda m: ra_order(m))
    cases = set(entries)
    for m in methods:
        got = {(r["dataset"], r["example_id"]) for r in collab if r["policy_id"] == m}
        if got != cases:
            raise ValueError(f"{m} incomplete: {len(got)}/{len(cases)}")
    yesterday = [r for r in ra.rows(ALIGNED / ("smoke_timed/records.jsonl" if args.stage == "smoke" else "full/records.jsonl"))]
    parity = {
        "same_job_B0_vs_aligned_campaign_B0": token_parity(b0, yesterday, "B0", "B0"),
        "collab_dense_vs_same_job_B0": token_parity(collab, b0, "collab-dense", "B0"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    rows = b0 + collab
    stats = pooled(rows, ["B0"] + methods)
    dense_stats = pooled(rows, ["collab-dense"] + methods[1:] if methods and methods[0] == "collab-dense" else methods, baseline="collab-dense") if "collab-dense" in methods else {}
    (args.out / "pooled_vs_B0.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.out / "pooled_vs_collab_dense.json").write_text(json.dumps(dense_stats, indent=2) + "\n")
    if args.stage == "smoke":
        if parity["same_job_B0_vs_aligned_campaign_B0"]["matched"] != len(cases):
            raise ValueError(f"same-job B0 differs from the aligned campaign: {parity}")
        by = {(r["example_id"], r["policy_id"]): r for r in collab if r["repeat"] == 1}
        for (eid, pid), r in by.items():
            if "SureLock" in pid and not r["lock_count"]:
                raise ValueError(f"{pid} never locked on {eid}")
            if "H2" in pid and not r["nfe"] < 256:
                raise ValueError(f"{pid} never committed in parallel on {eid}")
        summary = {"passed": True, "parity": parity, "per_record": [{k: r[k] for k in ("example_id", "policy_id", "nfe", "latency_s", "internal_latency_s", "lock_count", "extra_commit_count", "correct")} for r in collab]}
        (args.out / "smoke_pass.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["parity"], indent=1))
        return
    lines = ["# Collaborator methods under the aligned protocol", "",
             "Same job and GPU as the B0 anchor. Batch 1, 256/256/32, our 124 prompts and our scorer, 1 warm-up + 3 timed repeats (rotating order). Her decoders from commit 24cd51b; timing is a wall timer around each call (includes tokenization, like ours). Intervals: 10,000-sample bootstrap over questions.", "",
             "| Method | MATH /54 | Code /70 | Total /124 | Δ correct vs B0 (pp) [95% CI] | s / example | Speedup vs B0 [95% CI] | Mean calls | Outputs changed vs B0 | Broken / fixed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in ["B0"] + methods:
        s = stats[p]; o = s["overall"]; lo, hi = o["accuracy_delta_pp_ci95"]; slo, shi = o["speedup_pct_ci95"]
        lines.append(f"| {p} | {s['math500']['correct']} | {s['humaneval']['correct']} | {o['correct']} | {o['accuracy_delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {o['seconds_per_example']:.3f} | {o['speedup_pct']:+.2f}% [{slo:+.2f}, {shi:+.2f}] | {o['mean_nfe']:.1f} | {o['sequence_changed']}/124 | {o['correct_to_wrong']} / {o['wrong_to_correct']} |")
    lines += ["", f"Parity: same-job B0 vs the aligned campaign's B0: {parity['same_job_B0_vs_aligned_campaign_B0']['matched']}/{parity['same_job_B0_vs_aligned_campaign_B0']['compared']}; her dense vs our B0: {parity['collab_dense_vs_same_job_B0']['matched']}/{parity['collab_dense_vs_same_job_B0']['compared']} (details in parity.json)."]
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    (args.out / "scored_collab_records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in collab))
    (args.out / "completion.json").write_text(json.dumps({"complete": True, "methods": methods, "collab_records": len(collab), "b0_records": len(b0), "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    print("\n".join(lines))


def ra_order(method):
    order = ["collab-dense", "collab-SureLock-5e-4", "collab-SureLock-5e-3", "collab-capped-confidence", "collab-H2-guard90-delay1", "collab-TraceLock-LLaDA", "collab-open-confidence", "collab-safety-full"]
    return order.index(method) if method in order else len(order)


if __name__ == "__main__":
    main()
