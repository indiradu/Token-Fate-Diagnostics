"""Fill in `correct` for records/layer_band.jsonl, whose scoring was deferred (no bwrap sandbox on the Slurm nodes). Needs bubblewrap (bwrap) on PATH (the scorer guard aborts otherwise).
Usage: <abs path python with torch> rescore_records.py  -> records/layer_band.scored.jsonl (same records, `correct` / `scoring_status` recomputed from `text`).
Scoring is per distinct text (identical programs are scored once). Aborts without writing on any sandbox_error / sandbox_unavailable."""
from __future__ import annotations

import json
import sys

import common as C

C.scorer_guard()
by_id = {e["example_id"]: e for e in C.he164()}
src, dst = C.OUT / "records/layer_band.jsonl", C.OUT / "records/layer_band.scored.jsonl"
if len(sys.argv) > 1:
    src, dst = C.OUT / f"records/{sys.argv[1]}.jsonl", C.OUT / f"records/{sys.argv[1]}.scored.jsonl"
cache, out = {}, []
for r in C.rows(src):
    key = (r["example_id"], r["text"])
    if key not in cache:
        cache[key] = C.score_paper_generation(r["text"], by_id[r["example_id"]])
    correct, status = cache[key]
    if status in ("sandbox_error", "sandbox_unavailable"):
        raise SystemExit(f"scoring infrastructure failure on {r['example_id']} ({status}); not writing")
    out.append(dict(r, correct=bool(correct), scoring_status=status))
dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
print(f"rescored {len(out)} records ({len(cache)} distinct programs) -> {dst}")
