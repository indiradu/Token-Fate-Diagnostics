"""Score the HumanEval records of the ciai comparison (records/transfer_a100_he164.jsonl) in the bwrap sandbox -> .scored.jsonl. Run as a CPU job on the RTX cluster (ciai has no bwrap)."""
import json, sys
sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
from run_rule import questions
from pathlib import Path
REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/ciai/records")
by = {e["example_id"]: e for e in questions("he164")}
cache = {}
src = REC / "transfer_a100_he164.jsonl"
out = []
for r in C.rows(src):
    key = (r["example_id"], r["text"])
    if key not in cache:
        cache[key] = C.score_paper_generation(r["text"], by[r["example_id"]])
    correct, status = cache[key]
    if status in ("sandbox_error", "sandbox_unavailable"):
        raise SystemExit(f"scoring infrastructure failure {r['example_id']} ({status})")
    out.append(dict(r, correct=bool(correct), scoring_status=status))
(REC / "transfer_a100_he164.scored.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
print(f"{sum(r['correct'] for r in out)}/{len(out)} correct over {len(out)} records")
