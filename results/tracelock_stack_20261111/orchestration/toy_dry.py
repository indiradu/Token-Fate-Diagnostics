"""CPU dry-run context for run_trace_cells.py (TRACELOCK_TOY=1): a random 32-layer toy LLaDA (LLaDA hidden-state convention), a random projector and a deterministic feature-dependent controller.
Tests the orchestration (arms, records, selection, parity, analysis) without the GPU or the real model; its numbers mean nothing."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from regret_remasking.paper_trace import HiddenProjector, ProjectorConfig
from regret_remasking.toy_llada import ToyLLaDA
from regret_remasking.trace_lock import TraceScorer

V, D = 40, 16


class ToyHid(ToyLLaDA):
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **_unused):
        tr = self.model.transformer
        x = tr.wte(input_ids)
        hs = []
        for block in tr.blocks:
            hs.append(x)
            x, _ = block(x)
        x = tr.ln_f(x)
        hs.append(x)
        return SimpleNamespace(logits=tr.ff_out(x), hidden_states=tuple(hs) if output_hidden_states else None)


class Tok:
    padding_side = "left"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = torch.tensor([[5 + (ord(c) % 30) for c in text[:8]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x".join(map(str, ids[0]))]


class FeatureController(torch.nn.Module):
    def forward(self, features, valid_mask, position_ids=None):
        return {"margins": features[..., :16].mean(-1) * 40.0 + features.mean(dim=(1, 2))[:, None] * 5.0 - 0.5}


def ctx():
    torch.manual_seed(0)
    model = ToyHid(vocab=V, d=D, n_heads=4, layers=32, ffn=24, seed=0)
    torch.manual_seed(1)
    projector = HiddenProjector(ProjectorConfig(hidden_dim=D)).eval()
    controller = FeatureController().eval()
    return {"model": model, "tok": Tok(), "scorer": TraceScorer(controller, projector), "controller": controller, "projector": projector}
