"""Manifests for the far-blank-cache stacking campaign.
  math108      = longlen_wordlock manifests math_l5 (54) + math_l14 (54), copied verbatim (disjoint from the Lctx rule fit/threshold questions: longlen_wordlock validation/disjoint.json)
  math54half   = the even-indexed half of each (27 L5 + 27 L1-4), the same 54 as the earlier 1024-token records
  he164        = HumanEval tasks 0-163 from data/humaneval.jsonl (dataset 'humaneval'; the Lctx rules were fitted on MATH only; HumanEval 0-39 = the earlier he40 prompts)
  gsm8k200     = the first 200 GSM8K TEST questions (openai/gsm8k main, parquet). Every GSM8K collection of this repo (locking_original_plan fit/calibration/selection/smoke) drew from the
                 TRAIN split (checked below), and the Lctx rules were fitted on MATH, so none of these 200 was used for any fit or threshold.
Run: <abs python> build_manifests.py
"""
import glob
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
WL = ROOT / "results/longlen_wordlock_20261103/manifests"
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
M = OUT / "manifests"
M.mkdir(exist_ok=True)
wr = lambda name, recs: (M / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))

l5, l14 = rows(WL / "math_l5.jsonl"), rows(WL / "math_l14.jsonl")
assert (len(l5), len(l14)) == (54, 54)
wr("math_l5.jsonl", l5)
wr("math_l14.jsonl", l14)

hd = rows(ROOT / "data/humaneval.jsonl")
he = [{"dataset": "humaneval", "example_id": h["task_id"], "question": h["prompt"], "gold_answer": h["canonical_solution"],
       "metadata": {"canonical_solution": h["canonical_solution"], "entry_point": h["entry_point"], "test": h["test"]}} for h in hd]
assert len(he) == 164
wr("he164.jsonl", he)

f = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"))[0]
df = pd.read_parquet(f)
gs = [{"dataset": "gsm8k", "example_id": f"gsm8k-test-{i}", "question": df.iloc[i]["question"], "gold_answer": df.iloc[i]["answer"], "metadata": {"source_index": i}} for i in range(200)]
wr("gsm8k200.jsonl", gs)

# provenance: every GSM8K record in the old locking campaign is a train-split record
splits = set()
for p in glob.glob(str(ROOT / "results/locking_original_plan_20260917_204619/manifests/*.jsonl")):
    for r in rows(p):
        if r.get("dataset") == "gsm8k":
            splits.add(r["example_id"].rsplit("-", 2)[0] + "-" + r["example_id"].split("-")[1])
audit = {"math108": 108, "he164": 164, "gsm8k200": 200, "gsm8k_example_id_prefixes_in_old_locking_manifests": sorted(splits),
         "gsm8k_test_used_in_old_manifests": any("test" in s for s in splits),
         "lctx_rule_fit_questions": "MATH only (results/longlen_wordlock_20261103/rules/math512/rules.json: 60 fit + 20 threshold questions, disjoint from the 108)"}
(M / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
print(json.dumps(audit, indent=1))
