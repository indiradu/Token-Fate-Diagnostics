"""Shared pieces for the Dream-7B runs (MODEL_TAG=dream): weights from safetensors through the adapter, the chat-template prompt, the engine configuration, response decoding. The word lock is the same C-block rule as for LLaDA."""
from __future__ import annotations

import os

os.environ.setdefault("MODEL_TAG", "dream")
import common as C  # noqa: E402
import torch  # noqa: E402
from regret_remasking.model_adapters import load_dream  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig  # noqa: E402

PATH = os.environ.get("MODEL_CHECKPOINT", "/l/users/rui.gao/models/Dream-v0-Instruct-7B")
MASK, EOS, IM_END = 151666, 151643, 151645


def load():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    return load_dream(PATH, "cuda", torch.bfloat16), tok


def prompt_ids(tok, e):
    text = tok.apply_chat_template([{"role": "user", "content": C._resolve_prompt(e)}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt")["input_ids"].to("cuda")


def cfg_for(gen: int, threshold: float = 0.9, **extra) -> SelectiveConfig:
    c = dict(gen_length=gen, window=16, far_mask_cache=True, mask_id=MASK, eos_id=EOS, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32, refresh="block",
             eos_truncation=True, shape="block", lock="conf", threshold=threshold)
    c.update(extra)
    return SelectiveConfig(**c)


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in (EOS, IM_END)), None)


def response_text(tok, tokens):
    cut = first_eos(tokens)
    return tok.decode(tokens[:cut] if cut is not None else tokens, skip_special_tokens=True)
