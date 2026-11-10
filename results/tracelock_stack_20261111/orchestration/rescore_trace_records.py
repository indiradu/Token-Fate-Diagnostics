"""Fill in `correct` for the HumanEval records of records/stage<STAGE>.jsonl whose scoring was deferred (no bwrap sandbox on the decoding host). Needs bwrap on PATH, e.g.
  PATH=/tmp/claude-3430/-home-rui-gao-Desktop-Token-Fate-Diagnostics/a6c9e73d-5398-4784-8f1d-2dd0b921122b/scratchpad/bwrap_env/bin:$PATH <abs python with torch> rescore_trace_records.py H512
(the scorer guard aborts otherwise). Usage: rescore_trace_records.py STAGE -> records/stage<STAGE>.scored.jsonl (MATH records are copied unchanged; every HumanEval record is rescored from its `text`)."""
from __future__ import annotations

import json
import sys

import common as C

stage = sys.argv[1]
C.scorer_guard()
by_id = {e["example_id"]: e for e in C.manifest("he164")}
src, dst = C.REC / f"stage{stage}.jsonl", C.REC / f"stage{stage}.scored.jsonl"
out, changed, n = [], 0, 0
for r in C.rows(src):
    if r.get("dataset") == "humaneval":
        correct, status = C.score_paper_generation(r["text"], by_id[r["example_id"]])
        if status in ("sandbox_error", "sandbox_unavailable"):
            raise SystemExit(f"scoring infrastructure failure on {r['example_id']} ({status}); not writing")
        n += 1
        if r.get("correct") is not None and bool(r["correct"]) != bool(correct):
            changed += 1
        r = dict(r, correct=bool(correct), scoring_status=status)
    out.append(r)
dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
print(f"rescored {n} HumanEval records of {len(out)}; {changed} differ from the scores recorded at decode time")
