"""Shared setup for the two stale-cause GPU runners (imports the FROZEN source/ first). WORD LOCK C-0.9 / Lctx-0.8 and the S32 representation cache are as in farcache_stack_20261105."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(ROOT / "results/repfreeze_tests_20261029/runtime_cache/model_modules")
SRC = OUT / "source/src" if (OUT / "source/src").exists() and not os.environ.get("STALE_CAUSE_LIVE_SRC") else ROOT / "src"
sys.path.insert(0, str(SRC))
import torch  # noqa: E402

torch.set_num_threads(1)
from regret_remasking.context_rules import ContextRule  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
WL = ROOT / "results/longlen_wordlock_20261103"
RULE = WL / "rules/math512/Lctx_c0.8.pkl"
STACK1 = ROOT / "results/farcache_stack_20261105/records/stage1.jsonl"
GEN, BLOCK = 512, 32
OTHER_MARKERS = [ROOT / "results/learn2pd_control_20261108/GPU_IN_USE", ROOT / "results/farcache_stack_20261105/GPU_IN_USE", ROOT / "results/repfreeze_tests_20261029/GPU_IN_USE",
                 ROOT / "results/longlen_wordlock_20261103/GPU_IN_USE", ROOT / "results/selective_cache_20261104/GPU_IN_USE"]


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as h:
        h.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def he164():
    return [dict(e, _set="he164", _stratum="he164") for e in rows(OUT / "manifests/he164.jsonl")]


def stage1_by_arm():
    by = {}
    for r in rows(STACK1):
        if r.get("set") == "he164" and r.get("status") == "ok":
            by.setdefault(r["arm"], {})[r["example_id"]] = r
    return by


def gpu_guard():
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    out = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [ln for ln in out.stdout.splitlines() if ln.split(",")[0].strip().isdigit() and int(ln.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    for m in OTHER_MARKERS:
        if m.exists() and Path(f"/proc/{m.read_text().split()[0]}").exists():
            raise RuntimeError(f"another agent's marker is live: {m}")


def scorer_guard():
    he = rows(OUT / "manifests/he164.jsonl")[0]
    correct, status = score_paper_generation("```python\n" + he["question"] + he["metadata"]["canonical_solution"] + "\n```", he)
    print("[scorer guard]", correct, status, sys.executable, flush=True)
    if status == "sandbox_error" or not correct or not sys.executable.startswith("/"):
        raise SystemExit("HumanEval scorer guard failed: start with an absolute interpreter path")


def load_rule():
    rules = json.loads((WL / "rules/math512/rules.json").read_text())["rules"]["Lctx_c0.8"]
    sha = hashlib.sha256(RULE.read_bytes()).hexdigest()
    assert sha == rules["sha256"], f"Lctx rule file does not match rules.json: {sha}"
    rule = ContextRule.load(RULE)
    assert abs(rule.lam - rules["lambda"]) < 1e-12
    return rule, sha


def load_model():
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    return model, tok


def cached_cfg(lock: str, ctx, *, dense: bool = False, ctrl=None, log_detail: bool = False) -> SelectiveConfig:
    """S32 representation cache (dense call at every block start, first-EOS truncation) with word lock `lock` in {C-0.9, Lctx-0.8}; dense=True: a dense call every call (the dense decode,
    token-identical to the dense word-lock decoder up to the first EOS: farcache_stack diag_he)."""
    c = dict(gen_length=GEN, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=log_detail, block_length=BLOCK,
             refresh="every_call" if dense else "block", eos_truncation=True, shape="block", ctrl=ctrl)
    if lock.startswith("Lctx"):
        c.update(lock="rule", rule=ctx["rule"], threshold=0.9)
    else:
        c.update(lock="conf", threshold=float(lock.split("-")[1]))
    return SelectiveConfig(**c)


def prompt_ids(ctx, e):
    return _prepare_prompt(ctx["tok"], _resolve_prompt(e), "cuda")[0]


def run_cached(ctx, e, cfg, resume=None):
    ids = prompt_ids(ctx, e)
    res = decode_selective(ctx["model"], ids, cfg, resume=resume)
    return ids, res


def score(ctx, e, tokens):
    text = _decode_response(ctx["tok"], list(tokens))
    correct, status = score_paper_generation(text, e)
    if status == "sandbox_error":
        raise SystemExit("sandbox_error while scoring: abort")
    return text, bool(correct), status


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
