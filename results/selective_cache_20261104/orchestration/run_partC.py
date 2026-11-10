"""Part C (GPU): which ingredient of Elastic-Cache buys the speed? One process, gen 512, MATH 54 (even-indexed half of the long-length 108: 27 L5 + 27 L1-4).

Usage: <abs venv python> run_partC.py run [SMOKE]      records -> partC/records.jsonl (resumable); SMOKE = first 2 questions, 1 arm pass, records -> partC/smoke.jsonl
Cells (REPRESENTATION FREEZE side; word lock named per cell):
  Conf-0.9                   dense, our block-32 Confidence-0.9 word lock, every row every call (paper_decode C-block)
  SPARSE-none                the same word lock on our sparse row runtime with nothing cached (paper_decode CS-sparse, freeze_rule none): runtime-overhead control
  W16-dense                  Elastic's word lock (Confidence-0.9 in a window of 16 blanks) + first-EOS truncation, every row every call, selective_cache engine: control that
                             separates the word-lock change (window 16 + truncation) from the cache
  W16-cache-block            window + far-blank cache, dense refresh at every block start (our FAST-prefix-like schedule), no trigger, no EOS truncation
  W16-cache-block+EOS        + first-EOS truncation
  W16-cache+EOS+trigger      Elastic's adaptive layerwise trigger (gamma 0.9, heads cosine as coded, 1 tracked row) instead of the block schedule
  W16-cache+EOS+trigger-keys the same with the across-keys cosine
  W16-cache+EOS+selective-rXX our row-selective refresh: exposure scored at one layer, rows above theta rejoin at the next layer; score layer and thetas (one cell per
                             recall target XX of stale rows) from partC/params.json, fit on the Part A train questions only
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(ROOT / "results/repfreeze_tests_20261029/runtime_cache/model_modules")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / "source/src"))   # frozen copy first
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
LONG = ROOT / "results/longlen_wordlock_20261103/manifests"
GEN, BLOCK, THR = 512, 32, 0.9
BASE_ARMS = ["Conf-0.9", "SPARSE-none", "W16-dense", "W16-cache-block", "W16-cache-block+EOS", "W16-cache+EOS+trigger", "W16-cache+EOS+trigger-keys"]
ARMS = list(BASE_ARMS)       # + one selective cell per exposure threshold in partC/params.json (set in main)
OTHER_MARKERS = [ROOT / "results/repfreeze_tests_20261029/GPU_IN_USE", ROOT / "results/longlen_wordlock_20261103/GPU_IN_USE"]


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as h:
        h.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def questions():
    l5, l14 = rows(LONG / "math_l5.jsonl"), rows(LONG / "math_l14.jsonl")
    assert (len(l5), len(l14)) == (54, 54)
    return [dict(e, _set="L5") for e in l5[::2]] + [dict(e, _set="L1-4") for e in l14[::2]]


def gpu_guard():
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    out = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [ln for ln in out.stdout.splitlines() if ln.split(",")[0].strip().isdigit() and int(ln.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    for m in OTHER_MARKERS:
        if m.exists():
            pid = m.read_text().split()[0]
            if Path(f"/proc/{pid}").exists():
                raise RuntimeError(f"another agent's marker is live: {m}")


def cell_config(arm, params):
    c = dict(gen_length=GEN, threshold=THR, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=126081, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=BLOCK)
    if arm == "W16-dense":
        return SelectiveConfig(**c, refresh="every_call", eos_truncation=True)
    if arm == "W16-cache-block":
        return SelectiveConfig(**c, refresh="block", eos_truncation=False)
    if arm == "W16-cache-block+EOS":
        return SelectiveConfig(**c, refresh="block", eos_truncation=True)
    if arm == "W16-cache+EOS+trigger":
        return SelectiveConfig(**c, refresh="trigger", eos_truncation=True, gamma=0.9, trigger_cos="elastic_heads", track_num=1)
    if arm == "W16-cache+EOS+trigger-keys":
        return SelectiveConfig(**c, refresh="trigger", eos_truncation=True, gamma=0.9, trigger_cos="keys", track_num=1)
    if arm.startswith("W16-cache+EOS+selective-"):
        return SelectiveConfig(**c, refresh="selective", eos_truncation=True, score_layer=params["score_layer"], exposure_theta=params["thetas"][arm.rsplit("-", 1)[1]],
                               score_heads=params["score_heads"])
    raise KeyError(arm)


def decode(ctx, arm, e):
    model, tok = ctx["model"], ctx["tok"]
    ids, _ = _prepare_prompt(tok, _resolve_prompt(e), "cuda")
    torch.cuda.reset_peak_memory_stats()
    if arm in ("Conf-0.9", "SPARSE-none"):
        spec = dict(arm="C-block", threshold=THR) if arm == "Conf-0.9" else dict(arm="CS-sparse", threshold=THR, epsilon=0.005, confidence_percentile=100.0, freeze_scope="all_unmasked",
                                                                               freeze_gate="kl", freeze_rule="none")
        cfg = PaperDecodeConfig(gen_length=GEN, block_length=BLOCK, mask_id=MASK_ID, **spec)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, _resolve_prompt(e), e["example_id"], cfg)
        tokens = list(r.token_ids)
        extra = {"latency_s": r.latency_s, "nfe": r.nfe, "prompt_len": r.prompt_length, "text": r.text,
                 "rows_per_layer_per_call_mean": (r.total_query_rows / r.nfe) if r.nfe else None, "refreshes": 0, "layer_rows_sum": None}
    else:
        res = decode_selective(model, ids, cell_config(arm, ctx["params"]))
        tokens = list(res.tokens)
        L = len(res.layer_rows[0])
        lr = [sum(c[l] for c in res.layer_rows) for l in range(L)]
        kinds = res.refresh_kind
        extra = {"latency_s": res.latency_s, "nfe": res.nfe, "prompt_len": res.prompt_length, "text": _decode_response(tok, tokens),
                 "rows_per_layer_per_call_mean": sum(lr) / L / res.nfe, "layer_rows_sum": lr, "refreshes": sum(1 for k in kinds if k in ("trigger", "selective")),
                 "dense_calls": sum(1 for k in kinds if k in ("first", "dense")), "refresh_layers": [x for x in res.refresh_layers if x > 0],
                 "joined_rows_total": sum(res.joined_rows), "eos_position": res.eos_position}
    first_eos = next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
    return {"example_id": e["example_id"], "set": e["_set"], "arm": arm, "tokens": tokens, "first_eos": first_eos, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, **extra}


def main():
    assert sys.argv[1] == "run"
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    params = json.loads((OUT / "partC/params.json").read_text())
    ARMS.extend(f"W16-cache+EOS+selective-{k}" for k in params["thetas"])
    es = questions()
    path = OUT / ("partC/smoke.jsonl" if smoke else "partC/records.jsonl")
    if smoke:
        es = es[:1] + es[27:28]
    gpu_guard()
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    ctx = {"model": model, "tok": tok, "params": params}
    free, total = torch.cuda.mem_get_info(0)
    print(f"[partC] loaded; free {free / 2**30:.1f} of {total / 2**30:.1f} GiB; params {params}", flush=True)
    done = {(r["example_id"], r["arm"]) for r in rows(path) if r.get("status") == "ok"}
    for arm in ARMS:
        decode(ctx, arm, es[0])      # warm-up, not recorded
    started = time.time()
    for i, e in enumerate(es):
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (e["example_id"], arm) in done:
                continue
            gpu_guard()
            r = decode(ctx, arm, e)
            correct, status = score_paper_generation(r["text"], e)
            r.update({"correct": bool(correct), "scoring_status": status, "status": "ok", "gen": GEN})
            append(path, r)
        print(f"[partC] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print("[partC] done", flush=True)


if __name__ == "__main__":
    main()
