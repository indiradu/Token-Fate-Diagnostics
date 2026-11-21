"""GPU check of the hand-over: the collaborator's frozen engine + engine_f3.patch (source_m1_f3) runs Lctx-F3 and reproduces our sweep's D:Lctx-F3 answers token for token.

Arms per question:
  D    refresh every call, no early stop (eos_truncation=False): the setting of results/wordlock_sweep_20261120 (D:Lctx-F3); tokens compared to its records
  S32  the collaborator's cached_cfg defaults (refresh at each block start, first-EOS truncation)
  P4   S32 + blank-row partial refresh every 4th call from layer floor(11/32 * L)
Logs calls per block (call_blocks, log_detail=True) and words per call for every arm.
Usage: <python> check_f3_engine.py <sweep dir> [N per set]    writes validation/check_f3_engine.json next to this file"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source_m1_f3/src"))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from load_lctx_f3 import load_lctx_f3  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _prepare_prompt  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

SWEEP = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CKPT = Path(os.environ.get("TFD_CHECKPOINT", Path.home() / ".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"))


def cfg(arm, rule, info):
    c = dict(gen_length=512, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=True, block_length=32, shape="block",
             lock="rule", rule=rule, threshold=0.9, token_info=info)
    if arm == "D":
        c.update(refresh="every_call", eos_truncation=False)
    else:
        c.update(refresh="block", eos_truncation=True)
        if arm == "P4":
            c.update(partial_every_k=4, partial_alpha=11 / 32, partial_join="blank")
    return SelectiveConfig(**c)


def main():
    rule, sha, info = load_lctx_f3()
    model, tok = load_local_llada(CKPT, "cuda", "bf16")
    model.eval()
    ref = {}
    for f in sorted((SWEEP / "records").glob("*_s*of*.jsonl")):
        if f.name.startswith("smoke"):
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] == "D:Lctx-F3":
                ref[r["example_id"]] = r["tokens"]
    out = {"rule_sha256": sha, "rows": []}
    for ds in ("math108", "he164"):
        qs = [json.loads(x) for x in (SWEEP / f"manifests/{ds}.jsonl").read_text().splitlines() if x.strip()][:N]
        for e in qs:
            ids = _prepare_prompt(tok, _resolve_prompt(e), "cuda")[0]
            for arm in ("D", "S32", "P4"):
                res = decode_selective(model, ids, cfg(arm, rule, info))
                per_block = collections.Counter(res.call_blocks)
                row = {"set": ds, "example_id": e["example_id"], "arm": arm, "calls": res.nfe, "words_per_call": len(res.tokens) / res.nfe,
                       "calls_per_block": [per_block.get(b, 0) for b in range(16)], "fallback_calls": res.fallback_calls, "latency_s": res.latency_s}
                if arm == "D":
                    want = ref.get(e["example_id"])
                    row["tokens_match_sweep"] = None if want is None else list(res.tokens) == list(want)
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
    d = [r["tokens_match_sweep"] for r in out["rows"] if r["arm"] == "D"]
    out["all_D_match"] = all(x is True for x in d)
    (HERE / "validation").mkdir(exist_ok=True)
    (HERE / "validation/check_f3_engine.json").write_text(json.dumps(out, indent=1) + "\n")
    print("all D tokens match the sweep:", out["all_D_match"])


if __name__ == "__main__":
    main()
