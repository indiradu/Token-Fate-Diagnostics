"""Learn2PD word lock (Bao et al., ICLR 2026, arXiv 2509.25188; official code github.com/ims-kdks/Learning-to-Parallel-Decoding, llada/generate.py at commit 06bfd7f).

WORD LOCK only (which blanks of the current 32-block are written in a call). As in the official code:
  * the filter is a 2-layer MLP  Linear(32,32) -> ReLU -> Linear(32,32)  (2,112 parameters) whose INPUT is the confidence of all 32 slots of the current block
    (already-written slots included) and whose output is one logit per slot;
  * a still-blank slot is written iff sigmoid(logit) > tau (strict; tau is applied to the SIGMOID, not to the raw logit; the paper text is ambiguous);
  * the vanilla top-k schedule stays on: in the official runs steps == gen_length, so EVERY call also writes the single most confident blank of the block, whether or
    not the filter accepted it (``always_top1``; ties resolved to the leftmost blank).
Plugs into the dense X-block path of ``paper_decode`` (config.block_rule) and the selective-cache engine (SelectiveConfig.lock == "rule") through ``needs_block_vector``:
the decoder then hands in the confidence of EVERY slot of the block (real values, never imputed; the selective-cache engine computes the whole block's rows for it).
Counters in ``self.stats`` (reset by ``reset_stats``): calls, words (written), filter_words (accepted by the filter), empty (filter accepted no blank: only the forced
top-1 was written), top1_added (the most confident blank was written although the filter rejected it).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

NEEDS_BLOCK_VECTOR = True


class Learn2PDRule:
    kind = "l2pd"
    needs_block_vector = NEEDS_BLOCK_VECTOR

    def __init__(self, state: dict[str, torch.Tensor], tau: float, name: str = "", always_top1: bool = True, block_length: int = 32) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0,1]")
        want = {"linear1.weight": (block_length, block_length), "linear1.bias": (block_length,), "linear2.weight": (block_length, block_length), "linear2.bias": (block_length,)}
        for k, shape in want.items():
            if k not in state or tuple(state[k].shape) != shape:
                raise ValueError(f"filter state_dict needs {k} of shape {shape}")
        self.state = {k: state[k].detach().float().cpu().clone() for k in want}
        self.tau, self.name, self.always_top1, self.block_length = float(tau), name, bool(always_top1), int(block_length)
        self._dev: dict[str, torch.Tensor] | None = None
        self._dev_key = None
        self.last_empty = False
        self.reset_stats()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_file(cls, path: str | Path, tau: float, name: str = "", **kw) -> "Learn2PDRule":
        """Load a state_dict saved by torch.save (weights_only: no pickled code is executed)."""
        return cls(torch.load(str(path), map_location="cpu", weights_only=True), tau, name=name, **kw)

    def sha256(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.state):
            h.update(k.encode())
            h.update(self.state[k].numpy().tobytes())
        return h.hexdigest()

    @property
    def n_parameters(self) -> int:
        return sum(int(v.numel()) for v in self.state.values())

    # ------------------------------------------------------------------ filter
    def _weights(self, device: torch.device) -> dict[str, torch.Tensor]:
        if self._dev is None or self._dev_key != device:
            self._dev = {k: v.to(device) for k, v in self.state.items()}
            self._dev_key = device
        return self._dev

    @torch.no_grad()
    def probabilities(self, slot_conf: torch.Tensor) -> torch.Tensor:
        """sigmoid(filter(confidence of the 32 block slots)); slot_conf: (32,) float, block order."""
        if slot_conf.shape[-1] != self.block_length:
            raise ValueError(f"expected {self.block_length} slot confidences, got {tuple(slot_conf.shape)}")
        w = self._weights(slot_conf.device)
        h = torch.relu(slot_conf.float() @ w["linear1.weight"].T + w["linear1.bias"])
        return torch.sigmoid(h @ w["linear2.weight"].T + w["linear2.bias"])

    # ------------------------------------------------------------------ the word lock
    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "words": 0, "filter_words": 0, "empty": 0, "top1_added": 0}

    @torch.no_grad()
    def accept_mask(self, slot_conf: torch.Tensor, blank_slots: torch.Tensor) -> torch.Tensor:
        """slot_conf (32,) confidences of every block slot; blank_slots (n,) ascending slot indices (0..31) still blank. Returns a bool mask over the blanks: written this call."""
        prob = self.probabilities(slot_conf)
        accept = prob.index_select(0, blank_slots) > self.tau
        conf_blank = slot_conf.index_select(0, blank_slots).float()
        top1 = torch.zeros_like(accept)
        top1[conf_blank.argmax()] = True
        keep = accept | top1 if self.always_top1 else accept
        if not self.always_top1 and not bool(keep.any()):
            keep = top1                          # not the official behaviour (kept for the 'filter only' control); never write nothing
        st = self.stats
        st["calls"] += 1
        st["filter_words"] += int(accept.sum())
        st["words"] += int(keep.sum())
        self.last_empty = not bool(accept.any())
        st["empty"] += int(self.last_empty)
        st["top1_added"] += int(bool((top1 & ~accept).any()))
        return keep
