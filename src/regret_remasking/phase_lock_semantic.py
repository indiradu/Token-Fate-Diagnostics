"""Future-stability semantic proposals for PhaseLock Phase 1.

This module deliberately decides token identity only.  Representation freezing
and row removal remain downstream Phase-2 and Phase-3 decisions.
"""

from __future__ import annotations

import math

import torch


def select_future_stability_commits(
    probability: torch.Tensor,
    confidence: torch.Tensor,
    allowed: torch.Tensor,
    *,
    probability_threshold: float,
    confidence_threshold: float,
    maximum_fraction: float,
    fallback: bool = True,
    fallback_score: str = "probability",
) -> torch.Tensor:
    """Select Phase-1 commits from a learned future-stability score.

    Candidates must pass both the learned stability threshold and an optional
    language-model confidence veto.  A per-update fraction cap prevents a
    single optimistic controller call from committing an entire block.  The
    TraceLock-style fallback commits the highest-stability allowed token when
    no candidate passes, guaranteeing progress without granting any Phase-2
    or Phase-3 permission.
    """
    if not (
        probability.shape == confidence.shape == allowed.shape
        and probability.ndim == 2
        and allowed.dtype == torch.bool
    ):
        raise ValueError(
            "probability, confidence, and allowed must align as [batch, rows]"
        )
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability threshold must be in (0, 1)")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    if not 0.0 < maximum_fraction <= 1.0:
        raise ValueError("maximum fraction must be in (0, 1]")
    if fallback_score not in {"probability", "confidence", "joint"}:
        raise ValueError("fallback score must be probability, confidence, or joint")

    selected = torch.zeros_like(allowed)
    eligible = (
        allowed
        & (probability > probability_threshold)
        & (confidence >= confidence_threshold)
    )
    for batch_index in range(allowed.shape[0]):
        available_positions = torch.nonzero(
            allowed[batch_index], as_tuple=False
        ).flatten()
        if available_positions.numel() == 0:
            continue
        cap = max(
            1,
            int(
                math.ceil(
                    maximum_fraction * float(available_positions.numel())
                )
            ),
        )
        candidate_positions = torch.nonzero(
            eligible[batch_index], as_tuple=False
        ).flatten()
        if candidate_positions.numel() > 0:
            count = min(cap, int(candidate_positions.numel()))
            scores = probability[batch_index].index_select(
                0, candidate_positions
            )
            chosen = candidate_positions.index_select(
                0, torch.topk(scores, k=count).indices
            )
            selected[batch_index, chosen] = True
        elif fallback:
            if fallback_score == "probability":
                fallback_values = probability[batch_index]
            elif fallback_score == "confidence":
                fallback_values = confidence[batch_index]
            else:
                fallback_values = probability[batch_index] * confidence[batch_index]
            scores = fallback_values.index_select(0, available_positions)
            chosen = available_positions[int(torch.argmax(scores).item())]
            selected[batch_index, chosen] = True
    return selected


def update_token_identity_patience(
    previous_count: torch.Tensor,
    current_token: torch.Tensor,
    previous_token: torch.Tensor | None,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Count consecutive updates with the same top-token identity."""
    if not (
        previous_count.shape == current_token.shape == allowed.shape
        and previous_count.ndim == 2
        and allowed.dtype == torch.bool
    ):
        raise ValueError(
            "previous_count, current_token, and allowed must align as [batch, rows]"
        )
    if previous_token is not None and previous_token.shape != current_token.shape:
        raise ValueError("previous_token must match current_token")

    updated = torch.zeros_like(previous_count)
    if previous_token is None:
        updated[allowed] = 1
        return updated
    repeated = allowed & (current_token == previous_token)
    updated[allowed] = 1
    updated[repeated] = previous_count[repeated] + 1
    return updated


def select_sure_semantic_commits(
    sure_candidate: torch.Tensor,
    confidence: torch.Tensor,
    token_patience: torch.Tensor,
    allowed: torch.Tensor,
    *,
    confidence_threshold: float,
    minimum_token_patience: int,
    maximum_fraction: float = 1.0,
    fallback: bool = True,
) -> torch.Tensor:
    """Turn SureLock posterior-stability proposals into Phase-1 commits.

    SureLock's KL/uncertainty rule supplies ``sure_candidate``.  This function
    adds semantic safeguards that the original representation/compute-locking
    use case did not need: an absolute confidence veto, optional top-token
    persistence, and a per-update cap.  The confidence fallback guarantees
    progress but grants no Phase-2 or Phase-3 permission.
    """
    if not (
        sure_candidate.shape
        == confidence.shape
        == token_patience.shape
        == allowed.shape
        and sure_candidate.ndim == 2
        and sure_candidate.dtype == torch.bool
        and allowed.dtype == torch.bool
    ):
        raise ValueError(
            "Sure candidates, confidence, patience, and allowed must align"
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    if minimum_token_patience < 1:
        raise ValueError("minimum token patience must be positive")
    if not 0.0 < maximum_fraction <= 1.0:
        raise ValueError("maximum fraction must be in (0, 1]")

    eligible = (
        sure_candidate
        & allowed
        & (confidence >= confidence_threshold)
        & (token_patience >= minimum_token_patience)
    )
    selected = torch.zeros_like(allowed)
    for batch_index in range(allowed.shape[0]):
        available = torch.nonzero(allowed[batch_index], as_tuple=False).flatten()
        if available.numel() == 0:
            continue
        cap = max(
            1,
            int(math.ceil(maximum_fraction * float(available.numel()))),
        )
        candidates = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
        if candidates.numel() > 0:
            count = min(cap, int(candidates.numel()))
            scores = confidence[batch_index].index_select(0, candidates)
            chosen = candidates.index_select(0, torch.topk(scores, k=count).indices)
            selected[batch_index, chosen] = True
        elif fallback:
            scores = confidence[batch_index].index_select(0, available)
            chosen = available[int(torch.argmax(scores).item())]
            selected[batch_index, chosen] = True
    return selected


def scalar_posterior_stability_candidates(
    confidence: torch.Tensor,
    previous_confidence: torch.Tensor | None,
    allowed: torch.Tensor,
    *,
    maximum_confidence_delta: float,
    confidence_percentile: float = 20.0,
) -> torch.Tensor:
    """Approximate Sure-style posterior stability without storing the vocabulary.

    A position must be among the most confident ``confidence_percentile``
    percent of the current decision domain and its scalar top-token confidence
    must have changed by at most ``maximum_confidence_delta``.  Top-token
    identity persistence is deliberately handled separately so this cheap
    proposal can be ablated independently.
    """
    if confidence.shape != allowed.shape or confidence.ndim != 2:
        raise ValueError("confidence and allowed must align as [batch, rows]")
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must be boolean")
    if previous_confidence is not None and previous_confidence.shape != confidence.shape:
        raise ValueError("previous confidence must match current confidence")
    if maximum_confidence_delta < 0.0:
        raise ValueError("maximum confidence delta must be non-negative")
    if not 0.0 < confidence_percentile <= 100.0:
        raise ValueError("confidence percentile must be in (0, 100]")
    if previous_confidence is None:
        return torch.zeros_like(allowed)

    uncertainty = 1.0 - confidence
    confidence_delta = (confidence - previous_confidence).abs()
    selected = torch.zeros_like(allowed)
    quantile = confidence_percentile / 100.0
    for batch_index in range(allowed.shape[0]):
        values = uncertainty[batch_index][allowed[batch_index]]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.float(), quantile)
        selected[batch_index] = (
            allowed[batch_index]
            & (uncertainty[batch_index] <= threshold)
            & (confidence_delta[batch_index] <= maximum_confidence_delta)
        )
    return selected
