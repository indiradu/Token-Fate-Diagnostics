"""Faithful TraceLock controller components for a LLaDA transfer baseline.

The paper specifies the contextual controller but does not release its LLaDA
code or projection checkpoint.  This module implements the specified
controller and keeps the backbone-specific projection choice explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


STATE_PROMPT = 0
STATE_LOCKED = 1
STATE_GEN = 2
STATE_EOT = 3


@dataclass(frozen=True)
class TraceLockConfig:
    projected_hidden_dim: int = 128
    controller_dim: int = 384
    encoder_layers: int = 3
    attention_heads: int = 8
    ffn_dim: int = 768
    dropout: float = 0.1
    max_sequence_length: int = 768
    mistake_weight: float = 4.0
    threshold_regularizer: float = 1e-3
    deployment_probability: float = 0.95

    @property
    def input_dim(self) -> int:
        # Three snapshots plus two adjacent late-layer differences.
        return 5 * self.projected_hidden_dim

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TraceLockController(nn.Module):
    """Sequence-conditioned future-stability controller from TraceLock."""

    def __init__(self, config: TraceLockConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.controller_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length + 1, config.controller_dim
        )
        self.threshold_token = nn.Parameter(torch.zeros(1, 1, config.controller_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.controller_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.controller_dim),
        )
        self.token_head = nn.Linear(config.controller_dim, 1)
        self.threshold_head = nn.Linear(config.controller_dim, 1)
        nn.init.normal_(self.threshold_token, mean=0.0, std=0.02)

    @staticmethod
    def trace_state_features(projected_hidden: torch.Tensor) -> torch.Tensor:
        """Build ``[h1,h2,h3,h2-h1,h3-h2]`` features."""
        if projected_hidden.ndim != 4 or projected_hidden.shape[2] != 3:
            raise ValueError(
                "projected_hidden must have shape [batch, sequence, 3, dimension]"
            )
        h1, h2, h3 = projected_hidden.unbind(dim=2)
        return torch.cat([h1, h2, h3, h2 - h1, h3 - h2], dim=-1)

    def forward(
        self,
        projected_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if attention_mask.shape != projected_hidden.shape[:2]:
            raise ValueError("attention_mask and projected_hidden sequence shapes differ")
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        batch_size, sequence_length = attention_mask.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds TraceLock positional embedding limit")
        features = self.trace_state_features(projected_hidden)
        tokens = self.input_projection(features)
        positions = torch.arange(
            1, sequence_length + 1, device=tokens.device, dtype=torch.long
        )
        tokens = tokens + self.position_embedding(positions).unsqueeze(0)
        threshold = self.threshold_token.expand(batch_size, -1, -1)
        threshold = threshold + self.position_embedding.weight[0].view(1, 1, -1)
        encoded = self.encoder(
            torch.cat([threshold, tokens], dim=1),
            src_key_padding_mask=torch.cat(
                [
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
                    ~attention_mask,
                ],
                dim=1,
            ),
        )
        raw_logits = self.token_head(encoded[:, 1:]).squeeze(-1)
        threshold_logit = self.threshold_head(encoded[:, 0]).squeeze(-1)
        margin = raw_logits - threshold_logit.unsqueeze(-1)
        return {
            "raw_logits": raw_logits,
            "threshold_logit": threshold_logit,
            "margin": margin,
            "probability": torch.sigmoid(margin),
        }


def pack_left_padded_controller_batch(
    projected_hidden: torch.Tensor,
    prompt_attention: torch.Tensor,
    active_generation: torch.Tensor,
    *,
    generation_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strip per-sample left padding and right-pad TraceLock controller inputs.

    The language model may use left padding for batched prompts, while the
    saved TraceLock training samples are laid out as ``prompt | generation``.
    This function makes those positional coordinate systems identical without
    changing the language-model batch.
    """
    if projected_hidden.ndim != 4 or projected_hidden.shape[2] != 3:
        raise ValueError("projected_hidden must be [batch, sequence, 3, dimension]")
    if prompt_attention.ndim != 2:
        raise ValueError("prompt_attention must be [batch, prompt_sequence]")
    if active_generation.shape != projected_hidden.shape[:2]:
        raise ValueError("active_generation must align with projected_hidden")
    if projected_hidden.shape[0] != prompt_attention.shape[0]:
        raise ValueError("prompt and hidden batch sizes differ")
    prompt_width = int(prompt_attention.shape[1])
    if projected_hidden.shape[1] != prompt_width + generation_length:
        raise ValueError("hidden sequence does not match prompt plus generation")
    if generation_length <= 0:
        raise ValueError("generation_length must be positive")

    prompt_lengths = prompt_attention.sum(dim=1).to(dtype=torch.long)
    packed_hidden = torch.zeros_like(projected_hidden)
    packed_attention = torch.zeros_like(active_generation, dtype=torch.bool)
    packed_active = torch.zeros_like(active_generation, dtype=torch.bool)
    for batch_index in range(projected_hidden.shape[0]):
        actual_prompt = int(prompt_lengths[batch_index].item())
        if actual_prompt <= 0 or actual_prompt > prompt_width:
            raise ValueError("every sample must contain at least one prompt token")
        prompt_start = prompt_width - actual_prompt
        packed_hidden[batch_index, :actual_prompt] = projected_hidden[
            batch_index, prompt_start:prompt_width
        ]
        packed_hidden[
            batch_index, actual_prompt : actual_prompt + generation_length
        ] = projected_hidden[batch_index, prompt_width:]
        packed_attention[
            batch_index, : actual_prompt + generation_length
        ] = True
        packed_active[
            batch_index, actual_prompt : actual_prompt + generation_length
        ] = active_generation[batch_index, prompt_width:]
    return packed_hidden, packed_attention, packed_active, prompt_lengths


def tracelock_loss(
    margin: torch.Tensor,
    threshold_logit: torch.Tensor,
    labels: torch.Tensor,
    active_generation: torch.Tensor,
    *,
    mistake_weight: float = 4.0,
    threshold_regularizer: float = 1e-3,
) -> torch.Tensor:
    """Weighted dynamic-threshold objective from TraceLock Section 3.2."""
    if not (
        margin.shape == labels.shape == active_generation.shape
        and active_generation.dtype == torch.bool
    ):
        raise ValueError("margin, labels, and active_generation must align")
    if not active_generation.any():
        raise ValueError("TraceLock loss requires at least one active token")
    labels = labels.float()
    base = F.binary_cross_entropy_with_logits(margin, labels, reduction="none")
    predictions = margin > 0.0
    mistakes = predictions != labels.bool()
    weights = torch.where(
        mistakes,
        torch.full_like(base, float(mistake_weight)),
        torch.ones_like(base),
    )
    token_loss = (base * weights)[active_generation].mean()
    return token_loss + float(threshold_regularizer) * threshold_logit.square().mean()


def soft_crop_window(
    active_generation: torch.Tensor,
    *,
    prompt_length: int,
    window_width: int,
) -> tuple[int, int]:
    """Return TraceLock's two-pointer local decision window for batch size one."""
    if active_generation.ndim != 1 or active_generation.dtype != torch.bool:
        raise ValueError("active_generation must be a one-dimensional bool tensor")
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    positions = torch.nonzero(
        active_generation[prompt_length:], as_tuple=False
    ).flatten()
    if positions.numel() == 0:
        return prompt_length, prompt_length
    start = prompt_length + int(positions[0].item())
    return start, min(active_generation.numel(), start + window_width)


def select_tracelock_commits(
    probability: torch.Tensor,
    active_generation: torch.Tensor,
    *,
    prompt_length: int,
    window_width: int,
    threshold: float = 0.95,
) -> torch.Tensor:
    """Apply the paper's 0.95 soft-crop rule with top-score fallback."""
    if probability.shape != active_generation.shape or probability.ndim != 1:
        raise ValueError("probability and active_generation must be aligned vectors")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    start, end = soft_crop_window(
        active_generation,
        prompt_length=prompt_length,
        window_width=window_width,
    )
    selected = torch.zeros_like(active_generation)
    window = active_generation.clone()
    window[:start] = False
    window[end:] = False
    if not window.any():
        return selected
    selected = window & (probability > threshold)
    if not selected.any():
        masked_scores = torch.where(
            window, probability, torch.full_like(probability, -torch.inf)
        )
        selected[int(torch.argmax(masked_scores).item())] = True
    return selected
