"""Dream-7B: the adapter against the model's OWN forward (its remote code under transformers 4.46.2, the version it targets, installed in /l/users/rui.gao/envs/tf446). Standalone: no repo imports except the adapter.
Inputs: 3 HumanEval prompts (chat template) + 96 masks, and a variant with half of the masks replaced by random tokens. Reports, over all positions, max/mean abs logit difference, relative to the mean |logit|, and argmax agreement."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source_ad/src/regret_remasking"))
import model_adapters as MA  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

PATH, MASK = "/l/users/rui.gao/models/Dream-v0-Instruct-7B", 151666
rows = [json.loads(x) for x in (HERE.parent / "dream/manifests/he164.jsonl").read_text().splitlines()[:3]]
tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
g = torch.Generator().manual_seed(0)
inputs = []
for e in rows:
    text = tok.apply_chat_template([{"role": "user", "content": e["question"]}], add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    x = torch.cat([ids, torch.full((1, 96), MASK)], 1)
    y = x.clone()
    half = torch.randperm(96, generator=g)[:48] + ids.shape[1]
    y[0, half] = torch.randint(1000, 100000, (48,), generator=g)
    inputs += [x, y]
hf = AutoModel.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
ref = []
with torch.no_grad():
    for x in inputs:
        ref.append(hf(x.to("cuda"), "full").logits.float().cpu())
del hf
torch.cuda.empty_cache()
model = MA.load_dream(PATH, "cuda", torch.bfloat16)
ad = model.cache_adapter
out = []
with torch.no_grad():
    for x, r in zip(inputs, ref):
        xc = x.to("cuda")
        pos = torch.arange(xc.shape[1], device="cuda")
        h = ad.embed(xc)
        for l in range(ad.n_layers):
            q, k, v = ad.qkv(l, ad.attn_norm(l, h), pos, xc.shape[1])
            att = ad.attend(l, q, k, v).transpose(1, 2).contiguous().view(1, xc.shape[1], -1)
            h = ad.ffn(l, h + ad.attn_out(l, att))
        mine = ad.final(h).float().cpu()
        d = (mine - r).abs()
        out.append({"len": int(x.shape[1]), "max_abs": float(d.max()), "mean_abs": float(d.mean()), "rel_mean": float(d.mean() / r.abs().mean()), "argmax_agree": float((mine.argmax(-1) == r.argmax(-1)).float().mean())})
print(json.dumps(out, indent=1))
(HERE.parent / "dream/records/hf_parity.json").write_text(json.dumps(out, indent=1))
