"""LESS (released-code rule) as a decoder arm of ``paper_decode``.

``LessSelectionState.select`` needs full-length ``[1, sequence, vocab]`` probabilities. In the timed decoder only the current
block's blank rows have posteriors, so ``LessBlockSelector`` runs the identical admission math on that subset
(``select_at``) and keeps the released rule's per-block history reset. Equivalence with ``LessSelectionState.select`` is pinned in
``tests/test_less_arm.py``.
"""
from __future__ import annotations

import torch

from regret_remasking.less_sampling import LessConfig, LessSelectionState, coarse_topk_js


class _SubsetLessState(LessSelectionState):
    def select_at(
        self, positions: torch.Tensor, position_probs: torch.Tensor, default_k: int
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Same decision as ``select`` for the blank slots ``positions`` (sorted absolute indices) with posteriors ``[n, vocab]``.

        Returns (committed positions, accepted mask over ``positions``, used_fallback).
        """
        if position_probs.ndim != 2 or position_probs.shape[0] != positions.numel() or positions.numel() == 0:
            raise ValueError("LESS select_at expects non-empty positions with matching [n, vocab] posteriors")
        if position_probs.shape[-1] < self.config.top_k:
            raise ValueError("LESS top_k exceeds vocabulary size")
        current_topk_probs, current_topk_ids = torch.topk(position_probs, k=self.config.top_k, dim=-1)
        current_topk_probs = current_topk_probs.float()
        winners = current_topk_ids[:, 0]
        confidence = current_topk_probs[:, 0]

        winner_history = self.previous_winners[0, positions]
        valid_history = (winner_history >= 0).all(dim=-1)
        persistent = valid_history & winner_history.eq(winners.unsqueeze(-1)).all(dim=-1)
        previous_ids = self.previous_topk_ids[0, positions]
        previous_probs = self.previous_topk_probs[0, positions]
        has_previous = previous_probs.sum(dim=-1) > 0.0
        drift = coarse_topk_js(current_topk_ids, current_topk_probs, previous_ids, previous_probs)
        drift = torch.where(has_previous, drift, torch.ones_like(drift))
        accepted = persistent & confidence.ge(self.config.confidence_threshold) & drift.le(self.config.jsd_threshold)

        if self.config.history_length > 1:
            self.previous_winners[0, positions] = torch.cat([winner_history[:, 1:], winners.unsqueeze(-1)], dim=-1)
        else:
            self.previous_winners[0, positions, 0] = winners
        self.previous_topk_ids[0, positions] = current_topk_ids
        self.previous_topk_probs[0, positions] = current_topk_probs

        if bool(accepted.any()):
            return positions[accepted], accepted, False
        take = max(1, min(int(default_k), int(positions.numel())))
        _, fallback_indices = torch.topk(confidence, k=take)
        return positions[fallback_indices].sort().values, accepted, True


class LessBlockSelector:
    """One decode's LESS state; histories reset whenever the active block changes."""

    def __init__(self, sequence_length: int, device: torch.device, config: LessConfig | None = None):
        self.config = config or LessConfig()
        self._state = _SubsetLessState(sequence_length, device, self.config)
        self._block_left: int | None = None

    def select(
        self, logp: torch.Tensor, eligible: torch.Tensor, left: int, right: int, default_k: int = 1
    ) -> tuple[torch.Tensor, bool]:
        """``logp`` is ``[1, len(eligible), vocab]`` log-posteriors of the block's blank slots. Returns (positions, used_fallback)."""
        if self._block_left != left:
            self._state.reset_range(left, right)
            self._block_left = left
        selected, _, used_fallback = self._state.select_at(eligible, logp[0].exp(), default_k)
        return selected, used_fallback
