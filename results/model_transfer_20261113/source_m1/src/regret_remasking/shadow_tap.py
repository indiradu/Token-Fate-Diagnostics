"""Shadow tap: record, at every call of a cached decode, the proposals of the cached call and of a DENSE pass over the same tokens (default off; a diagnostic, not part of any timed run).

``ShadowTap(engine, shadow_engine, mask_id)`` wraps a ``SelectiveEngine`` and is passed to ``decode_selective(..., engine=tap)``. After the real forward of each call, a second engine
(``shadow_engine``, its own cache) recomputes every row at every layer from the same ``tokens`` and returns the logits of the same decoding rows. The decode itself is unchanged (the shadow never
writes into the real engine's cache). ``calls[i]`` is call i: positions of the decoding rows, and for the cached and the dense pass the mask-suppressed argmax token and its probability (the
proposal the C-block word lock uses). Used by results/refresh_family_20261111 (stage G: does the agreement of proposals track staleness?).
"""
from __future__ import annotations

import torch


def proposals(logits: torch.Tensor, mask_id: int):
    """(argmax token, probability) per decoding row; the mask token is suppressed, float32 softmax (as the confidence word lock)."""
    lg = logits[0].float().clone()
    lg[:, mask_id] = float("-inf")
    conf, tok = torch.softmax(lg, dim=-1).max(dim=-1)
    return tok.tolist(), conf.tolist()


class ShadowTap:
    def __init__(self, engine, shadow_engine, mask_id: int) -> None:
        self._eng, self._shadow, self.mask_id = engine, shadow_engine, mask_id
        self.calls: list[dict] = []

    def __getattr__(self, name):                       # blocks, token_since, T, ... of the real engine
        return getattr(self._eng, name)

    @torch.no_grad()
    def forward(self, tokens, rows, n_track, dec_local, mass_local, blank_mask, **kw):
        logits, info = self._eng.forward(tokens, rows, n_track, dec_local, mass_local, blank_mask, **kw)
        pos = rows[dec_local]
        all_rows = torch.arange(self._eng.T, device=rows.device)
        dense_logits, _ = self._shadow.forward(tokens, all_rows, 0, pos, pos, blank_mask, fresh=True)
        ct, cp = proposals(logits, self.mask_id)
        dt, dp = proposals(dense_logits, self.mask_id)
        self.calls.append({"pos": pos.tolist(), "cached_tok": ct, "cached_p": cp, "dense_tok": dt, "dense_p": dp})
        return logits, info
