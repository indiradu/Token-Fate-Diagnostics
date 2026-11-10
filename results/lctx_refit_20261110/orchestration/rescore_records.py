"""Fill in `correct` for records/stage<STAGE>.jsonl whose scoring was deferred (no bwrap sandbox on the decoding host). Needs bwrap on PATH (the scorer guard aborts otherwise).
Usage: <python with torch> rescore_records.py G_he164  -> records/stageG_he164.scored.jsonl (same records, `correct` / `scoring_status` recomputed from `text` for every record)."""
from __future__ import annotations

import json
import sys

import common as C

stage = sys.argv[1]
C.scorer_guard()
by_id = {e["example_id"]: e for e in C.rows(C.OUT / "manifests/he164.jsonl")}
src = C.OUT / f"records/stage{stage}.jsonl"        # stage = G_he164
dst = C.OUT / f"records/stage{stage}.scored.jsonl"
out, changed = [], 0
for r in C.rows(src):
    correct, status = C.score_paper_generation(r["text"], by_id[r["example_id"]])
    if status in ("sandbox_error", "sandbox_unavailable"):
        raise SystemExit(f"scoring infrastructure failure on {r['example_id']} ({status}); not writing")
    if r.get("correct") is not None and bool(r["correct"]) != bool(correct):
        changed += 1
    out.append(dict(r, correct=bool(correct), scoring_status=status))
dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
print(f"rescored {len(out)} records; {changed} differ from the scores recorded at decode time")
