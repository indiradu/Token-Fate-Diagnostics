"""Score the code-set records of all ciai comparison files (he164, kod_held) in the bwrap sandbox -> .scored.jsonl (CPU job on the RTX cluster)."""
import json, sys
sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
from run_rule import questions
from pathlib import Path
REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/ciai/records")
cache = {}
for ds in ("he164", "kod_held"):
    by = {e["example_id"]: e for e in questions(ds)}
    for src in sorted(REC.glob(f"transfer_*_{ds}.jsonl")):
        out = []
        for r in C.rows(src):
            key = (ds, r["example_id"], r["text"])
            if key not in cache:
                cache[key] = C.score_paper_generation(r["text"], by[r["example_id"]])
            correct, status = cache[key]
            if status in ("sandbox_error", "sandbox_unavailable"):
                raise SystemExit(f"scoring failure {src.name} {r['example_id']} ({status})")
            out.append(dict(r, correct=bool(correct), scoring_status=status))
        src.with_suffix(".scored.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
        print(src.name, sum(r["correct"] for r in out), "/", len(out), flush=True)
