"""CPU (login node, tokenizer only). PREREGISTERED.md (Features): builds
  assets/token_info.npz  per vocabulary id: digit, eq, op, ws (0/1), dollar, lbrace, rbrace (counts); special and partial-UTF-8 tokens are 0 everywhere
  assets/prompt_ids.npz  per question of lctx_features2_20261111/manifests/fit_thr.jsonl: the prompt ids of the collection's chat template (locking_policy._prepare_prompt)
  assets/assets.json     sha256 of both files, counts, the per-question prompt sha256
Usage: <python with transformers + torch> build_assets.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
F2 = OUT.parent / "lctx_features2_20261111"
sys.path.insert(0, str(F2 / "source/src" if (F2 / "source/src").exists() else ROOT / "src"))
from regret_remasking.locking_policy import _prepare_prompt  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt  # noqa: E402

SNAPSHOT = "08b83a6feb34df1a6011b80c3c00c7563e963b07"     # as lctx_features2_20261111/orchestration/common.py
CHECKPOINT = Path(os.environ.get("TFD_CHECKPOINT", Path.home() / f".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/{SNAPSHOT}"))
VOCAB = 126464
OPS_CHARS = set("+-*/^<>")
OPS_SUB = ("\\times", "\\cdot", "\\frac")


def classify(s: str) -> tuple[int, int, int, int, int, int, int]:
    """(digit, eq, op, ws, dollar, lbrace, rbrace) of one decoded token string."""
    return (int(any(c in "0123456789" for c in s)), int("=" in s), int(any(c in OPS_CHARS for c in s) or any(x in s for x in OPS_SUB)),
            int(bool(s) and not s.strip()), s.count("$"), s.count("{"), s.count("}"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=True)
    special = set(tok.all_special_ids)
    cols = {k: np.zeros(VOCAB, np.int8) for k in ("digit", "eq", "op", "ws", "dollar", "lbrace", "rbrace")}
    for i in range(min(len(tok), VOCAB)):
        if i in special:
            continue
        s = tok.decode([i])
        if "�" in s:
            continue
        for k, v in zip(cols, classify(s)):
            cols[k][i] = min(v, 127)
    adir = OUT / "assets"
    adir.mkdir(exist_ok=True)
    np.savez(adir / "token_info.npz", **cols)
    entries = [json.loads(x) for x in (F2 / "manifests/fit_thr.jsonl").read_text().splitlines() if x.strip()]
    prompts, psha = {}, {}
    for e in entries:
        ids = _prepare_prompt(tok, _resolve_prompt(e), "cpu")[0][0].numpy().astype(np.int64)
        prompts[e["example_id"]] = ids
        psha[e["example_id"]] = hashlib.sha256(ids.tobytes()).hexdigest()
    np.savez(adir / "prompt_ids.npz", **prompts)
    meta = {"token_info_sha256": sha(adir / "token_info.npz"), "prompt_ids_sha256": sha(adir / "prompt_ids.npz"), "checkpoint": str(CHECKPOINT),
            "counts": {k: int((v > 0).sum()) for k, v in cols.items()}, "n_questions": len(prompts), "prompt_sha256": psha,
            "prompt_lengths": {k: int(len(v)) for k, v in prompts.items()}}
    (adir / "assets.json").write_text(json.dumps(meta, indent=1))
    print(f"[build_assets] token_info {meta['counts']}; {len(prompts)} prompts", flush=True)


if __name__ == "__main__":
    main()
