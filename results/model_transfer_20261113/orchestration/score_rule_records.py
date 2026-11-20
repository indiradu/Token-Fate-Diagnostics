"""Score the deferred stage-R0 records of LLaDA-8B (refresh_family_20261111/records/rule_{he164,kod_held}_C-0.9.jsonl) -> original/records/rule_llada8b_<set>.scored.jsonl (arms D:C-0.9, S32, K4, P4, R0, CTL)."""
import json
from pathlib import Path
import common as C
from run_rule import questions
RF = Path(__file__).resolve().parents[2] / "refresh_family_20261111/records"
for ds in ("he164", "kod_held"):
    by = {e["example_id"]: e for e in questions(ds)}
    cache, out = {}, []
    for r in C.rows(RF / f"rule_{ds}_C-0.9.jsonl"):
        k = (r["example_id"], r["text"])
        if k not in cache: cache[k] = C.score_paper_generation(r["text"], by[r["example_id"]])
        c, s = cache[k]
        if s in ("sandbox_error", "sandbox_unavailable"): raise SystemExit(f"scoring failure {ds} {r['example_id']}")
        out.append(dict(r, correct=bool(c), scoring_status=s))
    dst = C.OUT / f"records/rule_llada8b_{ds}.scored.jsonl"
    dst.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in out))
    print(ds, len(out), "records", len(cache), "distinct programs ->", dst.name, flush=True)
