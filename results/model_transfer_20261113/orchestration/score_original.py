"""Score deferred records of stage O / stage B (original/records/{original,published}_*.jsonl) in the bwrap sandbox -> same file name with .scored.jsonl.
Usage: PATH=<bwrap_env/bin>:$PATH MODEL_TAG=original <abs python> score_original.py   (scores every file without a .scored twin; per distinct (question, text) once; aborts on sandbox failure)"""
from __future__ import annotations

import json
import sys

import common as C
from run_rule import questions

by = {"he164": {e["example_id"]: e for e in questions("he164")}, "kod_held": {e["example_id"]: e for e in questions("kod_held")}}
cache = {}
for src in sorted((C.OUT / "records").glob("*.jsonl")):
    if src.name.startswith("smoke_") or src.name.endswith(".scored.jsonl") or not src.name.startswith(("original_", "published_")):
        continue
    dst = src.with_suffix(".scored.jsonl")
    if dst.exists():
        continue
    out = []
    for r in C.rows(src):
        key = (r["set"], r["example_id"], r["text"])
        if key not in cache:
            cache[key] = C.score_paper_generation(r["text"], by[r["set"]][r["example_id"]])
        correct, status = cache[key]
        if status in ("sandbox_error", "sandbox_unavailable"):
            raise SystemExit(f"scoring infrastructure failure {src.name} {r['example_id']} ({status})")
        out.append(dict(r, correct=bool(correct), scoring_status=status))
    dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
    print(f"{src.name}: {sum(r['correct'] for r in out)}/{len(out)} correct -> {dst.name}", flush=True)
