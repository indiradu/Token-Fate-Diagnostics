"""CPU. Build the TRAINING-split question lists for the retrained Learn2PD filter and prove they are disjoint from every test question.

Usage: <abs python> build_train_manifests.py
Inputs (downloaded once, online, outside the project): /home/rui.gao/Desktop/l2pd_untrusted/data/{math_train_all.parquet, main/train-00000-of-00001.parquet}
  = EleutherAI/hendrycks_math train (7,500) and openai/gsm8k main train (7,473). MATH-500 is a subset of the MATH TEST split, so the train split is disjoint by construction;
  the exact- and normalised-text checks below are run anyway against every test manifest of this campaign and against the Lctx fit/threshold questions.
Outputs: manifests/train_math.jsonl (alternating Level 5 / Levels 1-4, so every prefix is balanced; seed 23), manifests/train_gsm8k.jsonl, validation/train_disjoint.json.
A question's role is fixed by its index in the list: index % 4 == 3 -> 'thr' (held out: threshold choice / matching), else 'fit'. Fixed here, before any trace is collected.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
DATA = Path("/home/rui.gao/Desktop/l2pd_untrusted/data")
N_MATH, N_GSM = 120, 80          # lists are long; the collector stops by budget and uses the collected prefix


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def main():
    rng = np.random.default_rng(23)
    m = pd.read_parquet(DATA / "math_train_all.parquet")
    m = m[m.level.isin([f"Level {i}" for i in range(1, 6)])].reset_index(drop=True)
    l5 = m[m.level == "Level 5"].sample(frac=1.0, random_state=23).reset_index(drop=True)
    l14 = m[m.level != "Level 5"].sample(frac=1.0, random_state=24).reset_index(drop=True)
    math = []
    for i in range(N_MATH // 2):
        for tag, df in (("l5", l5), ("l14", l14)):
            r = df.iloc[i]
            math.append({"dataset": "math", "example_id": f"mathtrain-{tag}-{i}", "question": r["problem"], "gold_answer": "", "metadata": {"level": r["level"], "subject": r["subject"], "role": None}})
    g = pd.read_parquet(DATA / "main/train-00000-of-00001.parquet").sample(frac=1.0, random_state=23).reset_index(drop=True)
    gsm = [{"dataset": "gsm8k", "example_id": f"gsm8ktrain-{i}", "question": g.iloc[i]["question"], "gold_answer": g.iloc[i]["answer"], "metadata": {"role": None}} for i in range(N_GSM)]
    for lst in (math, gsm):
        for i, e in enumerate(lst):
            e["metadata"]["role"] = "thr" if i % 4 == 3 else "fit"
    # disjointness against every test manifest and the Lctx rule questions
    test_q = {}
    for p in sorted((OUT.parent / "farcache_stack_20261105/manifests").glob("*.jsonl")):
        for ln in p.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                test_q[norm(e["question"])] = f"{p.name}:{e['example_id']}"
    for p in [ROOT / "data/math500_level5.jsonl", ROOT / "data/math500_levels1to4_54_seed23.jsonl", ROOT / "data/humaneval.jsonl"]:
        for ln in p.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                q = e.get("question") or e.get("problem") or e.get("prompt") or ""
                test_q[norm(q)] = f"{p.name}:{e.get('example_id', e.get('task_id', '?'))}"
    test_keys = list(test_q)
    exact, substring = [], []
    for e in math + gsm:
        k = norm(e["question"])
        if k in test_q:
            exact.append((e["example_id"], test_q[k]))
        elif len(k) > 40:
            for t in test_keys:
                if len(t) > 40 and (k in t or t in k):
                    substring.append((e["example_id"], test_q[t]))
                    break
    # drop colliding train questions (exact or substring) from the lists
    bad = {a for a, _ in exact} | {a for a, _ in substring}
    math = [e for e in math if e["example_id"] not in bad]
    gsm = [e for e in gsm if e["example_id"] not in bad]
    for name, lst in (("train_math", math), ("train_gsm8k", gsm)):
        (OUT / "manifests" / f"{name}.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in lst) + "\n")
    rep = {"n_math": len(math), "n_gsm8k": len(gsm), "test_questions_checked": len(test_q), "exact_overlap": exact, "substring_overlap": substring, "dropped": sorted(bad),
           "disjoint_after_drop": True, "note": "colliding train questions (if any) were dropped from the lists; the lists written are disjoint from all test questions",
           "sha256": {n: hashlib.sha256((OUT / "manifests" / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("train_math", "train_gsm8k")}}
    (OUT / "validation/train_disjoint.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
