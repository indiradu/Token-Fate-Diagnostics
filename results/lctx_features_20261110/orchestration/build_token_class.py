"""CPU. Build the token-class bit table used by the tok_* extra features (PREREGISTERED.md): assets/token_class.npy (int8 [vocab]) and assets/token_class.json (sha256, counts, examples).
Bits: 1 = the decoded token contains a line break; 2 = it begins with whitespace; 4 = it is non-empty after stripping whitespace and made only of characters that are neither letters nor digits.
Special tokens and tokens whose decode is a partial UTF-8 sequence (contains U+FFFD) are 0. Usage: <python with transformers> build_token_class.py"""
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parents[1]
CHECKPOINT = "/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
VOCAB = 126464                      # model output size (the tokenizer has fewer ids; the rest are 0)
tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
special = set(tok.all_special_ids)
table = np.zeros(VOCAB, dtype=np.int8)
for i in range(min(len(tok), VOCAB)):
    if i in special:
        continue
    s = tok.decode([i])
    if "�" in s:
        continue
    b = 0
    if "\n" in s or "\r" in s:
        b |= 1
    if s[:1].isspace():
        b |= 2
    core = s.strip()
    if core and all((not c.isalnum()) and (not c.isspace()) for c in core):
        b |= 4
    table[i] = b
(OUT / "assets").mkdir(exist_ok=True)
np.save(OUT / "assets/token_class.npy", table)
ex = {name: [tok.decode([int(i)]) for i in np.nonzero(table & bit)[0][:12]] for name, bit in (("newline", 1), ("space", 2), ("punct", 4))}
meta = {"vocab": VOCAB, "tokenizer_len": len(tok), "sha256": hashlib.sha256((OUT / "assets/token_class.npy").read_bytes()).hexdigest(),
        "counts": {name: int(((table & bit) != 0).sum()) for name, bit in (("newline", 1), ("space", 2), ("punct", 4))}, "examples": ex}
(OUT / "assets/token_class.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(meta, ensure_ascii=False)[:800])
