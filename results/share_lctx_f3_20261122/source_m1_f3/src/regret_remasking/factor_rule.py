"""Fast-dLLM's dynamic "factor" rule as a decoder selection (arm ``F-block`` of ``paper_decode``).

Sort the block's blank confidences descending, c_1 >= c_2 >= ... >= c_m (ties keep the earlier position first). The n-th word is
admissible iff (n + 1) * (1 - c_n) <= factor, i.e. c_n >= 1 - factor / (n + 1); word 1 is always admissible. Write the leading run of
admissible words (stop at the first failure), always at least one.

Deviation from the release (``get_transfer_index_dynamic`` in Fast-dLLM v1 llada/generate.py): the release finds ``top_i`` = index of the
first failing word and then does ``if top_i == 0 or top_i == len - 1: top_i += 1``. The ``== len - 1`` branch fires when every word passes (the
+1 is then correct) but also when *only the last-ranked* word fails, in which case the release writes that failing word as well. This clean
rule writes ``len - 1`` words there. Our confidence is the mask-suppressed proposal probability supplied by the caller.
"""
from __future__ import annotations

import math

import torch


def factor_select(confidence: torch.Tensor, factor: float) -> tuple[torch.Tensor, int]:
    """Return (indices into ``confidence`` of the words to write, sorted ascending, and their count k >= 1).

    ``confidence`` is a non-empty 1-D tensor of the block's blank confidences in position order.
    """
    if confidence.ndim != 1 or confidence.numel() == 0:
        raise ValueError("factor_select expects a non-empty 1-D confidence tensor")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    conf = confidence.double()
    order = torch.sort(conf, descending=True, stable=True).indices
    ranked = conf.index_select(0, order)
    rank = torch.arange(1, ranked.numel() + 1, dtype=torch.float64, device=conf.device)
    admissible = (rank + 1.0) * (1.0 - ranked) <= factor
    admissible[0] = True
    failures = torch.nonzero(~admissible, as_tuple=False).flatten()
    k = int(failures[0]) if failures.numel() else int(ranked.numel())
    return order[:k].sort().values, k
