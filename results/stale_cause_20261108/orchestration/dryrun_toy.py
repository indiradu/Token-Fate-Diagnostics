"""CPU dry run of both GPU runners on the toy model (no LLaDA, no GPU, fake scorer): checks the record contracts, resumability, and that the analyzers read what the runners write.
Usage: <python> dryrun_toy.py OUTDIR       (OUTDIR receives records/; nothing in the result directory is touched)
The fake scorer is a hash of the tokens, so "lost" tasks exist by construction; this tests plumbing, not science."""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import torch

import common as C
import run_noise_control as N
import run_oracle_repair as R
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

MASK, EOS, V, P = 3, 2, 40, 6
C.GEN, C.BLOCK = 24, 8
out = Path(sys.argv[1])
(out / "records").mkdir(parents=True, exist_ok=True)
C.OUT = out
model = ToyLLaDA(vocab=V, d=32, n_heads=4, layers=3, ffn=48, seed=2)
EX = [{"example_id": f"HumanEval/{i}", "question": "q", "metadata": {}} for i in range(7)]
IDS = {e["example_id"]: torch.randint(4, V, (1, P), generator=torch.Generator().manual_seed(i)) for i, e in enumerate(EX)}
fake_correct = lambda tokens: zlib.crc32(str(tokens).encode()) % 2 == 0


def cfg(lock, ctx, *, dense=False, ctrl=None, log_detail=False):
    return SelectiveConfig(gen_length=24, window=8, far_mask_cache=True, mask_id=MASK, eos_id=EOS, attn_impl="sdpa", log_calls=True, log_detail=log_detail, block_length=8,
                           refresh="every_call" if dense else "block", eos_truncation=True, shape="block", lock="conf", threshold=float(lock.split("-")[1]), ctrl=ctrl)


C.gpu_guard = lambda: None
C.scorer_guard = lambda: None
C.load_model = lambda: (model, None)
C.load_rule = lambda: (None, "sha")
C.cached_cfg = cfg
C.he164 = lambda: EX
C.prompt_ids = lambda ctx, e: IDS[e["example_id"]]
C.score = lambda ctx, e, tokens: (str(tokens), fake_correct(tokens), "passed")
C.first_eos = lambda tokens: next((i for i, t in enumerate(tokens) if t == EOS), None)
R.LOCKS = ["C-0.9", "C-0.5"]


def stage1():
    by = {}
    ctx = {"model": model}
    for lock in R.LOCKS:
        for pref, dense in (("D:", True), ("S32:", False)):
            for e in EX:
                _, res = C.run_cached(ctx, e, cfg(lock, ctx, dense=dense))
                by.setdefault(pref + lock, {})[e["example_id"]] = {"tokens": res.tokens, "correct": fake_correct(res.tokens)}
    return by


S1 = stage1()
import json
with (out / "stage1_fake.jsonl").open("w") as h:
    for arm, d in S1.items():
        for q, r in d.items():
            h.write(json.dumps({"set": "he164", "status": "ok", "arm": arm if arm.split(":")[1] == "C-0.9" else arm + "_other", "example_id": q, "tokens": r["tokens"], "correct": r["correct"],
                                "text": str(r["tokens"]), "first_eos": C.first_eos(r["tokens"])}) + "\n")
C.stage1_by_arm = lambda: S1
N.main()
sys.argv = ["x"]
R.main()
print("dry run finished; lost tasks per lock:", {l: sum(S1["D:" + l][q]["correct"] and not S1["S32:" + l][q]["correct"] for q in S1["D:" + l]) for l in R.LOCKS})
