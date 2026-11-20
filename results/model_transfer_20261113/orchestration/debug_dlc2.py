"""Debug 2: dLLM-Cache hooks vs plain forward on the same input (logits difference), per refresh setting."""
import sys
import common as C
import torch
sys.path.insert(0, "/l/users/rui.gao/tools/dLLM-cache")
from dataclasses import asdict
from transformers import AutoModel, AutoTokenizer
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import register_cache_LLaDA, logout_cache_LLaDA
ck = str(C.CHECKPOINT)
tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
model = AutoModel.from_pretrained(ck, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
p = C.prompt_ids({"tok": tok}, C.he164()[0]).reshape(1, -1)
P = p.shape[1]
x = torch.full((1, P + 64), C.MASK_ID, dtype=torch.long, device="cuda"); x[:, :P] = p
with torch.no_grad():
    plain = model(x).logits.float()
    plain_m = model(x, attention_mask=torch.ones_like(x)).logits.float()
print("plain vs plain+mask maxdiff", (plain - plain_m).abs().max().item(), "nan", torch.isnan(plain).any().item(), flush=True)
print("plain argmax first gen:", plain[0, P:P+5].argmax(-1).tolist(), flush=True)
register_cache_LLaDA(model, "model.transformer.blocks")
dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=1, gen_interval_steps=1, transfer_ratio=0.0)))
fc = dLLMCache(); fc.reset_cache(P)
with torch.no_grad():
    h = model(x, attention_mask=torch.ones_like(x)).logits.float()
print("hooked(every step) maxdiff to plain", (h - plain).abs().max().item(), "nan", torch.isnan(h).any().item(), "argmax gen:", h[0, P:P+5].argmax(-1).tolist(), flush=True)
fc.reset_cache(P)
with torch.no_grad():
    h2 = model(x).logits.float()
print("hooked no-mask maxdiff to plain", (h2 - plain).abs().max().item(), flush=True)
