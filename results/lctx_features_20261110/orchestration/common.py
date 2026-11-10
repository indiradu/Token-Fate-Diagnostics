"""Shared setup for the Lctx-features GPU runners (collection and stage H). Copied from results/lctx_refit_20261110/orchestration/common.py (itself from depth_refresh_20261109) and adapted: imports the
FROZEN source/ of THIS directory first (the engine changes relative to the refit's frozen source are the default-off collect_extra / token_class options and lock_features.py)."""
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
(OUT / "runtime_cache").mkdir(exist_ok=True)
os.environ["HF_MODULES_CACHE"] = str(OUT / "runtime_cache/model_modules")          # own cache: the other stages' jobs may be running at the same time
SRC = OUT / "source/src" if (OUT / "source/src").exists() and not os.environ.get("FEATURES_LIVE_SRC") else ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

torch.set_num_threads(1)
from regret_remasking.context_rules import ContextRule  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
WL = ROOT / "results/longlen_wordlock_20261103"
RULE0 = WL / "rules/math512/Lctx_c0.8.pkl"                                          # the ORIGINAL rule (never modified)
GEN, BLOCK = 512, 32
P4 = dict(partial_every_k=4, partial_layer=11, partial_join="blank")                # the blank-row refresh: every 4th call after the block-start dense call, blank rows from layer 11
RULE_X = OUT / "rules/X/Lctx_X_c0.8.pkl"                                            # the extended rule (written by fit_ablation.py ONLY if the gate passes)
RULE_X_JSON = OUT / "rules/X/rules.json"
REFIT_DIR = ROOT / "results/lctx_refit_20261110"                                    # parity references and the control's reference numbers (read only)
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


def load_rule(path=RULE0, meta_path=WL / "rules/math512/rules.json", key="Lctx_c0.8"):
    """Load a ContextRule and check it against its rules.json entry (sha256 and lambda)."""
    meta = json.loads(Path(meta_path).read_text())["rules"][key]
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha == meta["sha256"], f"rule file does not match {meta_path}: {sha}"
    rule = ContextRule.load(path)
    assert abs(rule.lam - meta["lambda"]) < 1e-12
    return rule, sha


def load_rule_x(dry: bool = False):
    """The extended rule, sha256-checked. dry=True: rules/dry (the six+all fit written by fit_ablation.py whether or not the gate passed; ONLY for the smoke of the decode path, never for results)."""
    path, meta = (OUT / "rules/dry/Lctx_dry_c0.8.pkl", OUT / "rules/dry/rules.json") if dry else (RULE_X, RULE_X_JSON)
    m = json.loads(meta.read_text())
    if not dry:
        assert m["token_class_sha256"] == json.loads((OUT / "assets/token_class.json").read_text())["sha256"], "the rule was fitted with a different token-class table than assets/token_class.npy"
    return load_rule(path, meta, m["name"])


def token_class_table():
    """The int8 [vocab] token-class bit table (assets/token_class.npy), checked against assets/token_class.json."""
    import numpy as np
    meta = json.loads((OUT / "assets/token_class.json").read_text())
    path = OUT / "assets/token_class.npy"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], "token_class.npy does not match token_class.json"
    return np.load(path)


def load_model():
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    return model, tok


def prompt_ids(ctx, e):
    return _prepare_prompt(ctx["tok"], _resolve_prompt(e), "cuda")[0]


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)


def collect_cfg(**extra) -> SelectiveConfig:
    """Trace collection in the R-S32c geometry of the refit: ONE WORD PER CALL (lock topk, topk 1), NO first-EOS truncation, 32-block shape, dense call at every block start and the blank-row refresh
    every 4th call from layer 11 (P4). collect_rows logs the six rule inputs; collect_extra adds the extra features of lock_features (token_class = the table of assets/)."""
    c = dict(gen_length=GEN, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=BLOCK,
             refresh="block", eos_truncation=False, lock="topk", topk=1, collect_rows=True, collect_extra=True, token_class=token_class_table(), shape="block", **P4)
    c.update(extra)
    return SelectiveConfig(**c)


def eval_cfg(lock: str, rule=None, **extra) -> SelectiveConfig:
    """Evaluation arms as in depth_refresh stage E / the refit's stage G (P4 = 32-block shape + blank-row refresh): first-EOS truncation, block-start dense call, 512 tokens.
    'C-0.8' = lock 'conf' at 0.8; a rule (the original six-input rule or the extended one) = lock 'rule' (the extended rule needs the token-class table on the config)."""
    c = dict(gen_length=GEN, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=BLOCK,
             refresh="block", eos_truncation=True, shape="block", **P4)
    if rule is not None:
        c.update(lock="rule", rule=rule, threshold=0.9)
        if set(rule.features) - set(SIX):
            c.update(token_class=token_class_table())
    else:
        c.update(lock="conf", threshold=float(lock.split("-")[1]))
    c.update(extra)
    return SelectiveConfig(**c)


SIX = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]
