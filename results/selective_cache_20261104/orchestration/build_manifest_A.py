"""Part A training manifest: 20 MATH L5 + 20 MATH L1-4 questions disjoint from every held-out / reported pool.

Why not the collect_mc math_train split as requested: its 54 L1-4 questions are exactly the held-out longlen math_l14 set and 36 of its 80 L5 questions are
the Test-4 fresh set, so only 44 L5 and zero L1-4 questions of that split are clean. We therefore draw from MATH-500 (the only MATH data on disk) after
excluding every question text that appears in any manifest listed in EXCLUDE. L5 comes from the 44 clean L5 questions of the collect_mc math_train split
(deterministic pick, seed 23); L1-4 comes from the MATH-500 L1-4 questions that are in no listed manifest (seed 23).
Run: <abs python> build_manifest_A.py
"""
import glob
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
R = ROOT / "results"
norm = lambda s: re.sub(r"\s+", " ", s).strip()
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

EXCLUDE = {
    "longlen_wordlock math_l14+math_l5 (108 held-out, Part C set)": [R / "longlen_wordlock_20261103/manifests/math_l14.jsonl", R / "longlen_wordlock_20261103/manifests/math_l5.jsonl"],
    "repfreeze Test 4 fresh (math_l14, math_l5_new, humaneval54)": [R / "repfreeze_tests_20261029/manifests/math_l14.jsonl", R / "repfreeze_tests_20261029/manifests/math_l5_new.jsonl",
                                                                     R / "repfreeze_tests_20261029/manifests/humaneval54.jsonl"],
    "compare_stage panel124": [R / "repfreeze_tests_20261029/compare_stage/manifests/panel124.jsonl"],
    "collect_mc math_report (54 L5 panel)": sorted(map(Path, glob.glob(str(R / "kl_repair_cpu_20261028/collect_mc/manifests/math_report_part_*.jsonl")))),
    "data/math500_levels1to4_54_seed23": [ROOT / "data/math500_levels1to4_54_seed23.jsonl"],
}
excluded = {}
for name, paths in EXCLUDE.items():
    for p in paths:
        for r in rows(p):
            q = r.get("question") or r.get("problem") or r.get("prompt")
            if q:
                excluded[norm(q)] = name
math = rows(next(iter(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")))))
assert len(math) == 500
train_l5 = [r for p in sorted(glob.glob(str(R / "kl_repair_cpu_20261028/collect_mc/manifests/math_train_part_*.jsonl"))) for r in rows(p) if r["metadata"]["level"] == 5]
clean_l5 = [r for r in train_l5 if norm(r["question"]) not in excluded]
by_text = {norm(r["problem"]): (i, r) for i, r in enumerate(math)}
l14_pool = [(i, r) for i, r in enumerate(math) if int(r["level"]) <= 4 and norm(r["problem"]) not in excluded]
rng = random.Random(23)
pick_l5 = rng.sample(clean_l5, 20)
pick_l14 = rng.sample(l14_pool, 20)
entries = []
for r in pick_l5:
    i, m = by_text[norm(r["question"])]
    entries.append({"example_id": f"mathA-l5-{i}", "dataset": "math500", "question": m["problem"], "gold_answer": m["answer"],
                    "metadata": {"level": 5, "subject": m["subject"], "source_index": i, "stratum": "L5"}})
for i, m in pick_l14:
    entries.append({"example_id": f"mathA-l14-{i}", "dataset": "math500", "question": m["problem"], "gold_answer": m["answer"],
                    "metadata": {"level": int(m["level"]), "subject": m["subject"], "source_index": i, "stratum": "L1-4"}})
used = {norm(e["question"]) for e in entries}
overlap = {name: sum(1 for q in used if excluded.get(q) == name) for name in EXCLUDE}
assert not (used & set(excluded)), "overlap with an excluded pool"
(OUT / "manifests").mkdir(exist_ok=True)
(OUT / "manifests/math40_train.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
audit = {"n": len(entries), "L5": 20, "L1-4": 20, "pool_clean_L5_in_collect_mc_train": len(clean_l5), "pool_clean_L1-4_in_MATH500": len(l14_pool),
         "excluded_question_texts": len(excluded), "overlap_with_each_excluded_pool": overlap, "seed": 23,
         "levels": {str(l): sum(1 for e in entries if e["metadata"]["level"] == l) for l in range(1, 6)},
         "deviation": "collect_mc math_train has 0 clean L1-4 and 44 clean L5 questions (its L1-4 are the held-out longlen math_l14 set); L1-4 drawn from unused MATH-500 L1-4 questions instead"}
(OUT / "manifests/audit_A.json").write_text(json.dumps(audit, indent=1) + "\n")
print(json.dumps(audit, indent=1))
