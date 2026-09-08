"""Decision rules for the three PhaseLock intervention layers.

The module is intentionally model-agnostic. It only turns online posterior,
trace, and representation measurements into auditable priorities or decisions;
the decoder owns the actual irreversible state transition and compute backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# A gate is looser when it accepts larger one-step representation drift and
# demands fewer consecutive confirmations.  Minimum age is kept separate so
# every preset observes at least one post-commit update before freezing.
REPRESENTATION_GATE_PRESETS: dict[str, dict[str, float | int]] = {
    "strict": {"threshold": 0.005, "patience": 4, "min_age": 1},
    "default": {"threshold": 0.02, "patience": 2, "min_age": 1},
    "loose": {"threshold": 0.05, "patience": 1, "min_age": 1},
    "very_loose": {"threshold": 0.10, "patience": 1, "min_age": 1},
}


def representation_gate_preset(name: str) -> dict[str, float | int]:
    if name not in REPRESENTATION_GATE_PRESETS:
        raise ValueError(f"unknown representation gate preset: {name}")
    return dict(REPRESENTATION_GATE_PRESETS[name])


@dataclass(frozen=True)
class PhaseLockPolicy:
    semantic: str = "confidence"
    representation: str = "drift"
    compute: str = "row_sparse"
    semantic_lock_fraction: float = 0.06
    beta_kl: float = 1.0
    beta_fate: float = 2.0
    representation_threshold: float = 0.02
    representation_patience: int = 2
    representation_min_age: int = 1


def cosine_distance(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    current = torch.nn.functional.normalize(current.float(), dim=-1)
    previous = torch.nn.functional.normalize(previous.float(), dim=-1)
    return (1.0 - (current * previous).sum(dim=-1)).clamp_min(0.0)


def semantic_priority(
    policy: str,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    entropy: torch.Tensor,
    kl: torch.Tensor,
    runlength: torch.Tensor,
    top1_flip: torch.Tensor,
    predicted_regret: torch.Tensor | None = None,
    beta_kl: float = 1.0,
    beta_fate: float = 2.0,
) -> torch.Tensor:
    """Return larger-is-safer priority for token identity commitment."""
    if policy == "confidence":
        return confidence
    if policy in {"posterior", "surelock"}:
        return confidence * torch.exp(-beta_kl * kl)
    if policy == "margin":
        return margin
    if policy == "persistence":
        return confidence * (1.0 + torch.log1p(runlength))
    if policy == "consensus":
        return confidence * torch.exp(-beta_kl * kl) * (1.0 + 0.25 * runlength) * (1.0 - 0.5 * top1_flip)
    if policy in {"token_fate", "trace", "fate"}:
        if predicted_regret is None:
            raise ValueError(f"{policy} requires predicted_regret")
        return confidence * torch.exp(-beta_fate * predicted_regret)
    raise ValueError(f"unknown semantic policy: {policy}")


def representation_ready(
    policy: str,
    drift: torch.Tensor,
    confidence: torch.Tensor,
    kl: torch.Tensor,
    runlength: torch.Tensor,
    age: torch.Tensor,
    below_threshold_count: torch.Tensor,
    threshold: float,
    patience: int,
    min_age: int,
) -> torch.Tensor:
    """Return positions whose representation can be frozen this step.

    The default rule requires low hidden-state drift for consecutive online
    observations. Posterior and confidence gates are available as controls;
    they do not replace the representation measurement.
    """
    if policy == "always":
        return age >= min_age
    if policy == "confidence":
        return (confidence >= threshold) & (age >= min_age)
    if policy in {"posterior", "surelock"}:
        return (kl <= threshold) & (age >= min_age)
    if policy in {"drift", "representation"}:
        return (drift <= threshold) & (below_threshold_count >= patience) & (age >= min_age)
    if policy == "drift_posterior":
        return (drift <= threshold) & (kl <= threshold) & (below_threshold_count >= patience) & (age >= min_age)
    if policy == "drift_confidence":
        return (drift <= threshold) & (confidence >= 0.5) & (below_threshold_count >= patience) & (age >= min_age)
    raise ValueError(f"unknown representation policy: {policy}")


def compute_mode_name(policy: str) -> str:
    """Normalize the explicit compute intervention names used in ledgers."""
    aliases = {
        "none": "dense",
        "dense": "dense",
        "kv": "kv_cache",
        "kv_cache": "kv_cache",
        "row": "row_sparse",
        "row_sparse": "row_sparse",
    }
    if policy not in aliases:
        raise ValueError(f"unknown compute policy: {policy}")
    return aliases[policy]
