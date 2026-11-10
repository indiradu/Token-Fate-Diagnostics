"""Shared setup for the TraceLock-under-representation-caches runner. Adapted from results/depth_refresh_20261109/orchestration/common.py; imports the FROZEN source/ of THIS directory first
(TRACELOCK_LIVE_SRC=1 uses the live src/ instead: for CPU dry runs only)."""
from __future__ import annotations

import hashlib
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
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
SRC = OUT / "source/src" if (OUT / "source/src").exists() and not os.environ.get("TRACELOCK_LIVE_SRC") else ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

torch.set_num_threads(1)
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
E2 = ROOT / "results/locking_paper_redesign_20260920_161812"
CONTROLLER = E2 / "e2_train/full/attempt_01/best_controller.pt"
PROJECTOR = E2 / "projector/projector.pt"
CONTROLLER_SHA = "a36a47e8ac8f88874c024150ec008b3a26a0b0de27a17bd8a789ac4e602f93c8"
PROJECTOR_SHA = "6cba72b5a2a2c01a2982308136280a3c68855d85f5218b96e7c1d4522431155d"
E2_RECORDS = E2 / "e2_eval/merged_records/records.jsonl"       # the earlier 124-case panel (HumanEval 0-69 + MATH level-5 54, 256 tokens): informational parity of the dense TraceLock / C-adaptive arms
BLOCK = 32
REC = Path(os.environ.get("TRACELOCK_RECORDS", str(OUT / "records")))      # dry runs write elsewhere
DEVICE = os.environ.get("TRACELOCK_DEVICE", "cuda")
P4 = dict(partial_every_k=4, partial_layer=11, partial_join="blank")                # the blank-row refresh: every 4th call after the block-start dense call, blank rows from layer 11
CACHE = {"S32": {}, "K4": dict(refresh_every_k=4), "P4": P4}
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


def manifest(name):
    return rows(OUT / f"manifests/{name}.jsonl")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_guard():
    if DEVICE != "cuda":
        return
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    out = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [ln for ln in out.stdout.splitlines() if ln.split(",")[0].strip().isdigit() and int(ln.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    for m in OTHER_MARKERS:
        if m.exists() and Path(f"/proc/{m.read_text().split()[0]}").exists():
            raise RuntimeError(f"another agent's marker is live: {m}")


def scorer_guard():
    he = manifest("he164")[0]
    correct, status = score_paper_generation("```python\n" + he["question"] + he["metadata"]["canonical_solution"] + "\n```", he)
    print("[scorer guard]", correct, status, sys.executable, flush=True)
    if status == "sandbox_error" or not correct or not sys.executable.startswith("/"):
        raise SystemExit("HumanEval scorer guard failed: start with an absolute interpreter path")


def load_model():
    model, tok = load_local_llada(CHECKPOINT, DEVICE, "bf16")
    model.eval()
    return model, tok


def load_trace():
    """The frozen TraceLock controller + projector (hash-checked), on the decoding device in fp32, as paper_experiment loads them."""
    from regret_remasking.paper_trace import HiddenProjector, PaperTraceController
    assert sha256(CONTROLLER) == CONTROLLER_SHA and sha256(PROJECTOR) == PROJECTOR_SHA, "TraceLock checkpoints differ from the frozen hashes"
    payload = torch.load(CONTROLLER, map_location="cpu", weights_only=False)
    assert payload.get("projection_sha256") == PROJECTOR_SHA, "controller checkpoint was not trained on this projector"
    return PaperTraceController.load(CONTROLLER, device=DEVICE), HiddenProjector.load(PROJECTOR, device=DEVICE)


def prompt_ids(ctx, e):
    return _prepare_prompt(ctx["tok"], _resolve_prompt(e), DEVICE)[0]


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
