"""Score the deferred code records of Study 5 (records/{pd,d2c,eng}_<tag>_<set>[_<arm>].jsonl for he164 / kod_held) in the bwrap sandbox -> <name>.scored.jsonl.
GSM8K / MATH records were scored in process and are copied through. Run on the LOGIN node (compute nodes lack bwrap):
  PATH=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/refresh_family_20261111/tools/bwrap_env/bin:$PATH FAMILY_SRC=<model_transfer>/source_m1 MODEL_TAG=original <venv python> score_study5.py
Re-scores a file only when its scored twin has fewer rows than the source (jobs still running are scored partially and refreshed on the next call); one sandbox run per distinct (set, question, text)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
from run_rule import questions

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
by = {"he164": {e["example_id"]: e for e in questions("he164")}, "kod_held": {e["example_id"]: e for e in questions("kod_held")}}
cache = {}
for src in sorted(REC.glob("*.jsonl")):
    if src.name.startswith("smoke_") or src.name.endswith(".scored.jsonl") or not src.name.startswith(("pd_", "d2c_", "eng_", "flops_")):
        continue
    dst = src.with_suffix(".scored.jsonl")
    rows = C.rows(src)
    if dst.exists() and len(C.rows(dst)) >= len(rows):
        continue
    out = []
    for r in rows:
        if r.get("set") in by:
            key = (r["set"], r["example_id"], r["text"])
            if key not in cache:
                cache[key] = C.score_paper_generation(r["text"], by[r["set"]][r["example_id"]])
            correct, status = cache[key]
            if status in ("sandbox_error", "sandbox_unavailable"):
                raise SystemExit(f"scoring infrastructure failure {src.name} {r['example_id']} ({status})")
            r = dict(r, correct=bool(correct), scoring_status=status)
        out.append(r)
    dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
    n = [r for r in out if r.get("correct") is not None]
    print(f"{src.name}: {sum(bool(r['correct']) for r in n)}/{len(n)} correct -> {dst.name}", flush=True)
