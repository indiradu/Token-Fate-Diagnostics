"""LLaDA-MoE diagnosis (premise P1 on a new architecture): in the plain S32 cached decode (word lock Conf-0.9, no refresh) of N HumanEval prompts, at every call compare the cached pass with a DENSE pass over the same tokens (shadow_tap):
 (a) expert routing: for the rows the cached call computes, the share of the top-8 experts that agree with the dense pass, per layer, split by row class (blank rows / rows written on the previous call / other);
 (b) proposals: for decoding rows whose cached confidence >= 0.9 (the rows the word lock would write), the share whose token differs from the dense pass's argmax ('wrong writes'), per call index within the block.
Output records/moe_routing_diag.json. No timing; a diagnostic. Needs the adapter hook (monkeypatch of adapter._moe, no source file changed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import moe_common as D
import torch
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, decode_selective
from regret_remasking.shadow_tap import ShadowTap

C = D.C
REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MASK = D.MASK


class Rec:
    on = False
    store = []


def main():
    C.gpu_guard()
    es = C.he164()[:N]
    model, tok = D.load()
    ad = model.cache_adapter
    orig = ad._moe

    def patched(l, h):
        if Rec.on:
            n, Dm = h.shape
            logits = torch.nn.functional.linear(h, ad._w(f"model.layers.{l}.mlp.gate.weight"))
            Rec.store.append((l, torch.topk(torch.softmax(logits, 1, dtype=torch.float32), ad.top_k, -1).indices.cpu()))
        return orig(l, h)
    ad._moe = patched
    per_layer = {l: {"blank": [0, 0], "prev_written": [0, 0], "other": [0, 0]} for l in range(ad.n_layers)}
    wrong = {}                                                    # call index within block -> [wrong writes, candidate writes]
    calls_total = 0

    class Tap(ShadowTap):
        def forward(self, tokens, rows, n_track, dec_local, mass_local, blank_mask, **kw):
            Rec.store, Rec.on = [], True
            logits, info = self._eng.forward(tokens, rows, n_track, dec_local, mass_local, blank_mask, **kw)
            real = Rec.store
            Rec.store = []
            T = self._eng.T
            allr = torch.arange(T, device=rows.device)
            src = rows[dec_local]
            dense_logits, _ = self._shadow.forward(tokens, allr, 0, src, src, blank_mask, fresh=True)
            dense = Rec.store
            Rec.on = False
            if len(real) == ad.n_layers and len(dense) == ad.n_layers and rows.numel() < T:
                rr = rows.cpu()
                is_blank = (tokens[0, rows] == MASK).cpu()
                for (l, a), (_, b) in zip(real, dense):
                    bb = b[rr]                                    # dense choices of the same rows
                    ov = (a.unsqueeze(2) == bb.unsqueeze(1)).any(2).float().mean(1)          # per row share of top-8 experts in common
                    for cls, msk in (("blank", is_blank), ("prev_written", ~is_blank)):
                        if msk.any():
                            per_layer[l][cls][0] += float(ov[msk].sum()); per_layer[l][cls][1] += int(msk.sum())
            from regret_remasking.shadow_tap import proposals
            ct, cp = proposals(logits, self.mask_id)
            dt, dp = proposals(dense_logits, self.mask_id)
            self.calls.append({"cached_tok": ct, "cached_p": cp, "dense_tok": dt, "dense_p": dp})
            return logits, info

    for e in es:
        ids = D.prompt_ids(tok, e)
        cfg = D.cfg_for(512, eos_truncation=False)
        T = ids.shape[1] + 512
        tap = Tap(SelectiveEngine(model, T, cfg), SelectiveEngine(model, T, cfg), MASK)
        res = decode_selective(model, ids, cfg, engine=tap)
        for k, c in enumerate(tap.calls):
            for t1, p1, t2 in zip(c["cached_tok"], c["cached_p"], c["dense_tok"]):
                if p1 >= 0.9:
                    w = wrong.setdefault(min(k, 40), [0, 0]); w[1] += 1; w[0] += int(t1 != t2)
        calls_total += len(tap.calls)
        print(e["example_id"], "calls", len(tap.calls), flush=True)
    out = {"n_prompts": N, "calls": calls_total,
           "expert_overlap_by_layer": {l: {c: (v[0] / v[1] if v[1] else None) for c, v in d.items()} for l, d in per_layer.items()},
           "wrong_write_rate_overall": sum(v[0] for v in wrong.values()) / max(1, sum(v[1] for v in wrong.values())),
           "candidate_writes": sum(v[1] for v in wrong.values()), "wrong_writes": sum(v[0] for v in wrong.values())}
    REC.mkdir(exist_ok=True)
    (REC / "moe_routing_diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
