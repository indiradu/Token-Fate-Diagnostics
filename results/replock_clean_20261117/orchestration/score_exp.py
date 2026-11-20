"""Score the deferred code records of the follow-up experiments (records/{thr,abl,par,noise}_*.jsonl for the code sets he164, kod_held, kod_ext) in the bwrap sandbox -> .scored.jsonl twins (CPU job; rescored when the source has more rows)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
import run_par
import run_rule as R

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
by = {}
for ds in ("he164", "kod_held"):
    for e in R.questions(ds):
        by[(ds, e["example_id"])] = e
for e in run_par.questions("kod_ext"):
    by[("kod_ext", e["example_id"])] = e
    by[("kod_held", e["example_id"])] = e                      # kod_ext prompts also come from the kod_held manifest
cache = {}
for src in sorted(REC.glob("*.jsonl")):
    if src.name.startswith("smoke_") or src.name.endswith(".scored.jsonl") or not src.name.startswith(("thr_", "abl_", "par_", "noise_")):
        continue
    dst = src.with_suffix(".scored.jsonl")
    rows = C.rows(src)
    if dst.exists() and len(C.rows(dst)) >= len(rows):
        continue
    out = []
    for r in rows:
        if r.get("set") in ("he164", "kod_held", "kod_ext"):
            key = (r["set"], r["example_id"], r["text"])
            if key not in cache:
                cache[key] = C.score_paper_generation(r["text"], by[(r["set"], r["example_id"])])
            correct, status = cache[key]
            if status in ("sandbox_error", "sandbox_unavailable"):
                raise SystemExit(f"scoring failure {src.name} {r['example_id']} ({status})")
            r = dict(r, correct=bool(correct), scoring_status=status)
        out.append(r)
    dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
    n = [r for r in out if r.get("correct") is not None]
    print(f"{src.name}: {sum(bool(r['correct']) for r in n)}/{len(n)} correct", flush=True)
