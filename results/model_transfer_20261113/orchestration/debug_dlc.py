"""Debug: dLLM-Cache wrapper on HumanEval/0 at 128 tokens. (a) cache refreshing every step (K_p=K_r=1, rho=0: must equal the model's plain decode), (b) published K_p 25, K_r 5, rho 0.25; compared to the plain B0 decode."""
import sys, time
import common as C
import torch
sys.path.insert(0, "/l/users/rui.gao/tools/dLLM-cache")
from dataclasses import asdict
from transformers import AutoModel, AutoTokenizer
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import register_cache_LLaDA
from utils import generate as dgen
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
ck = str(C.CHECKPOINT)
tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
model = AutoModel.from_pretrained(ck, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
e = C.he164()[0]
ctx = {"tok": tok}
p = C.prompt_ids(ctx, e).reshape(1, -1)
print("prompt shape", p.shape, "dtype", p.dtype, flush=True)
G = 128
ref = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], PaperDecodeConfig(gen_length=G, block_length=32, mask_id=C.MASK_ID, arm="B0"))
rt = list(ref.token_ids)
print("REF", repr(C._decode_response(tok, rt)[:150]), flush=True)
register_cache_LLaDA(model, "model.transformer.blocks")
for name, kp, kr, rho in [("every-step", 1, 1, 0.0), ("published", 25, 5, 0.25), ("kr1", 25, 1, 0.0)]:
    dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=kp, gen_interval_steps=kr, transfer_ratio=rho)))
    x = dgen(p, torch.ones_like(p), model, steps=G, gen_length=G, block_length=32, temperature=0., cfg_scale=0., remasking="low_confidence", mask_id=C.MASK_ID)
    t = [int(v) for v in x[0, p.shape[1]:].tolist()]
    eq = sum(a == b for a, b in zip(t, rt))
    print(name, "match with plain", eq, "/", G, repr(C._decode_response(tok, t)[:150]), flush=True)
