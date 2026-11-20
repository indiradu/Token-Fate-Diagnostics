"""LLaDA-MoE-7B-A1B-Instruct pieces for the Study 5 runs (same role as dream_common.py): weights through the adapter, chat-template prompt, engine configuration, response decoding.
Word lock Conf-0.9 inside 32-blocks, as for every other model. Token ids: mask 156895 (<|mask|>), eos 156892 (<|endoftext|>), role end 156900 (<|role_end|>)."""
from __future__ import annotations

import os

import sys
sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
os.environ.setdefault("MODEL_TAG", "dream")                    # manifests (same prompts for every model)
import common as C  # noqa: E402
import torch  # noqa: E402
from moe_adapter import load_llada_moe  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig  # noqa: E402

PATH = os.environ.get("MODEL_CHECKPOINT", "/l/users/rui.gao/models/LLaDA-MoE-7B-A1B-Instruct")
MASK, EOS, ROLE_END = 156895, 156892, 156900


def load():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    return load_llada_moe(PATH, "cuda", torch.bfloat16), tok


def prompt_ids(tok, e):
    text = tok.apply_chat_template([{"role": "user", "content": C._resolve_prompt(e)}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt")["input_ids"].to("cuda")


def cfg_for(gen: int, threshold: float = 0.9, **extra) -> SelectiveConfig:
    c = dict(gen_length=gen, window=16, far_mask_cache=True, mask_id=MASK, eos_id=EOS, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32, refresh="block",
             eos_truncation=True, shape="block", lock="conf", threshold=threshold)
    c.update(extra)
    return SelectiveConfig(**c)


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in (EOS, ROLE_END)), None)


def response_text(tok, tokens):
    cut = first_eos(tokens)
    return tok.decode(tokens[:cut] if cut is not None else tokens, skip_special_tokens=True)
