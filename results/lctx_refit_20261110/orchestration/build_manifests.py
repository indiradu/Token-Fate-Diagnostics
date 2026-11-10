"""CPU. Builds manifests/fit_thr.jsonl (the SAME 60 fit + 20 threshold questions as the original Lctx rule, taken from rules.json, text from data/math500_level5.jsonl) and writes
validation/disjoint.json: id / normalised-text / 200-character-substring overlap of those 80 questions with the evaluation sets (MATH 108 = math_l5 + math_l14, HumanEval 164).
Usage: <python> build_manifests.py   (exit code 1 if anything overlaps)"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
WL = ROOT / "results/longlen_wordlock_20261103"
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    meta = json.loads((WL / "rules/math512/rules.json").read_text())
    fit, thr = list(meta["fit_questions"]), list(meta["thr_questions"])
    src = {r["id"]: r for r in rows(ROOT / "data/math500_level5.jsonl")}
    out = []
    for split, ids in (("fit", fit), ("thr", thr)):
        for i in ids:
            r = src[i]
            out.append({"dataset": "math500", "example_id": i, "question": r["question"], "gold_answer": r["answer"], "split": split,
                        "metadata": {"level": r["level"], "subject": r["subject"], "source_index": r["source_index"], "collect_split": split}})
    (OUT / "manifests/fit_thr.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    math108 = rows(OUT / "manifests/math_l5.jsonl") + rows(OUT / "manifests/math_l14.jsonl")
    he = rows(OUT / "manifests/he164.jsonl")
    tn = [norm(r["question"]) for r in out]
    report = {"n_fit": len(fit), "n_thr": len(thr), "n_fit_and_thr_unique": len({r["example_id"] for r in out}), "fit_thr_ids_equal_original_rules_json": True,
              "source": "data/math500_level5.jsonl (sha256 %s)" % sha(ROOT / "data/math500_level5.jsonl"), "manifest_sha256": {p.name: sha(p) for p in sorted((OUT / "manifests").glob("*.jsonl"))}}
    ok = True
    for name, ev in (("math108", math108), ("he164", he)):
        id_over = sorted({r["example_id"] for r in ev} & {r["example_id"] for r in out})
        txt, sub = [], []
        for r in ev:
            n = norm(r["question"])
            if n in tn:
                txt.append(r["example_id"])
            if len(n) >= 40 and any(n[:200] in t or t[:200] in n for t in tn):
                sub.append(r["example_id"])
        report[name] = {"n_eval": len(ev), "n_unique_ids": len({r["example_id"] for r in ev}), "id_overlap": id_over, "text_overlap": txt, "substring_overlap": sub, "disjoint": not (id_over or txt or sub)}
        ok &= report[name]["disjoint"]
    report["disjoint"] = ok
    (OUT / "validation/disjoint.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "n_unique_ids"}) for k, v in report.items() if k != "manifest_sha256"}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
