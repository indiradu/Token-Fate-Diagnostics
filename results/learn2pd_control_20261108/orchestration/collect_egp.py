"""GPU. Collect Learn2PD training traces with this repo's LLaDA harness (one process, resumable, one npz per question).

Usage: <abs venv python> collect_egp.py SET GEN [LIMIT]       SET in {train_math, train_gsm8k}; the manifest is manifests/<SET>.jsonl (training split only)

Follows the official generate_training_data.py (github ims-kdks/Learning-to-Parallel-Decoding, llada/, commit 06bfd7f, read, not run):
  pass 1  reference answer y = the standard one-word-per-call decode (block 32, low-confidence, temperature 0): this is also the B0 reference of the Lctx labels.
          Logged per call: the 32 block confidences, the 32 block proposals and which block slots were still blank (this is the B0-state data used to match tau to C-0.8's precision).
  pass 2  Extremely Greedy Parallel (EGP) replay: per step the proposals of all 32 block slots are compared with y; every slot whose proposal equals y is written
          (already-written slots included, as in the official code), the rest stay blank; label vector = (proposal == y) over all 32 slots; official stuck-guard
          (block proposals unchanged from the previous step -> write the reference block). Logged per step: the 32 confidences, the 32 labels, which slots were blank.
Deliberate differences from the official collector: (a) chat-template prompts of our harness (the official collector tokenises raw FLAN text without the template) and
  MATH-train / GSM8K-train questions instead of 2,640 FLAN questions; (b) proposals are the argmax with the mask token suppressed and the confidence is its float32
  softmax probability (the repo's convention; the official code uses the raw bf16 argmax); (c) both passes stop after the block that follows the first block
  holding a written EOS/EOT in the reference (later blocks are all padding; they cost 32 calls each and add only trivial rows).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(ROOT / "results/repfreeze_tests_20261029/runtime_cache/model_modules")
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _prepare_prompt  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
BLOCK = 32


@torch.no_grad()
def block_view(model, x, attn, left, right):
    """Confidence (float32 softmax prob of the mask-suppressed argmax) and proposal of the 32 block slots."""
    out = model(x, attention_mask=attn)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0, left:right].float()
    logp = F.log_softmax(logits, dim=-1)
    prop_logits = logits.clone()
    prop_logits[..., MASK_ID] = -torch.inf
    prop = prop_logits.argmax(-1)
    conf = logp.gather(-1, prop.unsqueeze(-1)).squeeze(-1).exp()
    return conf, prop


@torch.no_grad()
def collect_question(model, ids, gen):
    P = int(ids.shape[1])
    T = P + gen
    attn = torch.ones((1, T), dtype=torch.long, device=ids.device)
    n_blocks = gen // BLOCK
    # ---- pass 1: reference, one word per call
    x = torch.full((1, T), MASK_ID, dtype=torch.long, device=ids.device)
    x[:, :P] = ids
    b0_vec, b0_prop, b0_mask, b0_block = [], [], [], []
    stop = n_blocks - 1
    eos_block = None
    for b in range(n_blocks):
        left, right = P + b * BLOCK, P + (b + 1) * BLOCK
        while bool((x[0, left:right] == MASK_ID).any()):
            conf, prop = block_view(model, x, attn, left, right)
            blank = x[0, left:right] == MASK_ID
            pick = int(torch.where(blank, conf, torch.full_like(conf, -1.0)).argmax())
            b0_vec.append(conf.cpu().numpy()); b0_prop.append(prop.cpu().numpy().astype(np.int32)); b0_mask.append(blank.cpu().numpy()); b0_block.append(b)
            x[0, left + pick] = prop[pick]
        blk = x[0, left:right]
        if eos_block is None and bool(((blk == EOS_ID) | (blk == EOT_ID)).any()):
            eos_block = b
            stop = min(n_blocks - 1, b + 1)
        if b >= stop:
            break
    y = x[0, P:P + (stop + 1) * BLOCK].clone()
    # ---- pass 2: EGP replay on the same blocks
    x = torch.full((1, T), MASK_ID, dtype=torch.long, device=ids.device)
    x[:, :P] = ids
    egp_vec, egp_ok, egp_mask, egp_block = [], [], [], []
    for b in range(stop + 1):
        left, right = P + b * BLOCK, P + (b + 1) * BLOCK
        ref = y[b * BLOCK:(b + 1) * BLOCK]
        prev = None
        while bool((x[0, left:right] == MASK_ID).any()):
            conf, prop = block_view(model, x, attn, left, right)
            blank = x[0, left:right] == MASK_ID
            ok = prop == ref
            egp_vec.append(conf.cpu().numpy()); egp_ok.append(ok.cpu().numpy()); egp_mask.append(blank.cpu().numpy()); egp_block.append(b)
            if prev is not None and bool((prop == prev).all()):
                x[0, left:right] = ref
                continue
            x[0, left:right] = torch.where(ok, prop, x[0, left:right])
            prev = prop
    return dict(prompt_len=P, ref=y.cpu().numpy().astype(np.int32), eos_block=-1 if eos_block is None else eos_block, stop_block=stop,
                b0_vec=np.stack(b0_vec).astype(np.float32), b0_prop=np.stack(b0_prop), b0_mask=np.stack(b0_mask), b0_block=np.array(b0_block, dtype=np.int16),
                egp_vec=np.stack(egp_vec).astype(np.float32), egp_ok=np.stack(egp_ok), egp_mask=np.stack(egp_mask), egp_block=np.array(egp_block, dtype=np.int16))


def main():
    name, gen = sys.argv[1], int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    entries = [json.loads(x) for x in (OUT / f"manifests/{name}.jsonl").read_text().splitlines() if x.strip()][:limit]
    dest = OUT / "training_data" / name
    dest.mkdir(parents=True, exist_ok=True)
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    t_all = time.time()
    for i, e in enumerate(entries):
        path = dest / f"{e['example_id']}.npz"
        if path.exists():
            continue
        ids, _ = _prepare_prompt(tok, _resolve_prompt(e), "cuda")
        t0 = time.time()
        r = collect_question(model, ids, gen)
        tmp = dest / f"{e['example_id']}.tmp.npz"
        np.savez_compressed(tmp, gen=gen, role=e["metadata"]["role"], **r)
        os.replace(tmp, path)
        print(f"[egp {name}] {i + 1}/{len(entries)} {e['example_id']} role {e['metadata']['role']} b0 calls {len(r['b0_block'])} egp steps {len(r['egp_block'])} "
              f"stop block {r['stop_block']} {time.time() - t0:.0f}s total {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("[egp] done", flush=True)


if __name__ == "__main__":
    main()
