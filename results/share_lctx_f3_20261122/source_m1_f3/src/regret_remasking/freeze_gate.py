"""Learned freeze gate: a TraceLock-style controller that decides *when to freeze a written word's representation*.

TraceLock reads the last three hidden states of every position and decides which blank to write. This gate reads the same
hidden states (through the same fitted projector) and decides which already-written answer word to freeze. It is a
logistic regression over [832-d projector features, scalar signals]; the scalars are the ones a fixed gate such as
SureLock can see plus the call-to-call change of the feature vector.

Feature order is part of the checkpoint contract and must match training exactly:
  trace (832) then SCALARS = [rc_feat, log10_vkl, log10_1_minus_echo, entropy, top1_match, log1p_age].

A gate checkpoint trained on answer rows never touches prompt rows unless the decoder is given it as the prompt gate. Candidates need
age >= `min_age` calls since the word was written (stage 1 trained on age >= 2, which is a one-call handicap against a fixed rule that
may freeze at age 1). A prompt row's age is step + 1 (it was "written" before call 0).

Optional extra scalars (results/gate_q_20261026), appended after the six above in this order when the checkpoint lists them:
  progress     written answer slots / gen_length at the start of the call (EOS/EOT slots count as written)
  is_question  constant column: 1.0 for a gate scoring prompt rows, 0.0 for answer rows (joint gates only; the decoder never passes it,
               it is fixed per gate instance through `row_type`)
Checkpoints that list only the six original scalars behave exactly as before.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

SCALARS = ["rc_feat", "log10_vkl", "log10_1_minus_echo", "entropy", "top1_match", "log1p_age"]
EXTRA_SCALARS = ["progress", "is_question"]


def check_scalars(names: list[str]) -> list[str]:
    """A checkpoint's scalar list must be the six originals optionally followed by progress, then is_question (in that order)."""
    names = list(names)
    allowed = [SCALARS, SCALARS + EXTRA_SCALARS[:1], SCALARS + EXTRA_SCALARS]
    if names not in allowed:
        raise ValueError(f"gate checkpoint scalar order {names} is not one of {allowed}")
    return names


VKL_FLOOR, ECHO_FLOOR = 1e-12, 1e-9


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LearnedFreezeGate:
    def __init__(self, projector: Any, mean: torch.Tensor, scale: torch.Tensor, coef: torch.Tensor, intercept: float,
                 *, threshold: float = 0.9, min_age: int = 2, sha256: str | None = None,
                 scalars: list[str] | None = None, row_type: float | None = None) -> None:
        self.projector = projector
        self.mean, self.scale, self.coef = mean.float(), scale.float().clamp_min(1e-6), coef.float()
        self.intercept = float(intercept)
        self.threshold, self.min_age, self.sha256 = float(threshold), int(min_age), sha256
        self.scalars = check_scalars(SCALARS if scalars is None else scalars)
        if ("is_question" in self.scalars) != (row_type is not None):
            raise ValueError("row_type (0 = answer rows, 1 = question rows) is required exactly when the gate has an is_question scalar")
        self.row_type = None if row_type is None else float(row_type)
        self.feature_dim = int(self.coef.numel()) - len(self.scalars)
        self._prev: torch.Tensor | None = None
        self._has_prev: torch.Tensor | None = None

    @classmethod
    def load(cls, path: str | Path, projector: Any, device: str | torch.device = "cpu", **kwargs: Any) -> "LearnedFreezeGate":
        payload = torch.load(path, map_location=device, weights_only=True)
        names = check_scalars(payload["scalars"])
        gate = cls(projector, payload["mean"].to(device), payload["scale"].to(device), payload["coef"].to(device),
                   float(payload["intercept"]), sha256=file_sha256(path), scalars=names, **kwargs)
        gate.mean, gate.scale, gate.coef = gate.mean.to(device), gate.scale.to(device), gate.coef.to(device)
        return gate

    def reset(self, length: int, device: torch.device) -> None:
        self._prev = torch.zeros((length, self.feature_dim), dtype=torch.float32, device=device)
        self._has_prev = torch.zeros(length, dtype=torch.bool, device=device)

    @torch.no_grad()
    def observe(self, hidden3: Any, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the given positions, return (features, rc_feat) and remember the features for the next call.

        rc_feat is NaN where the position had no features at the previous call."""
        feats = self.projector.encode_hidden(tuple(h.index_select(1, positions) for h in hidden3))[0].float()
        assert self._prev is not None and self._has_prev is not None, "call reset() before observe()"
        prev = self._prev.index_select(0, positions)
        have = self._has_prev.index_select(0, positions)
        rc = (feats - prev).norm(dim=-1) / feats.norm(dim=-1).clamp_min(1e-6)
        rc = torch.where(have, rc, torch.full_like(rc, float("nan")))
        self._prev[positions] = feats
        self._has_prev[positions] = True
        return feats, rc

    @torch.no_grad()
    def score(self, feats: torch.Tensor, rc_feat: torch.Tensor, vkl: torch.Tensor, echo_p: torch.Tensor,
              entropy: torch.Tensor, top1_match: torch.Tensor, age: torch.Tensor, *, return_logit: bool = False,
              progress: torch.Tensor | None = None):
        """Probability that each row is settled. Rows with a missing signal (NaN) score 0.

        With return_logit=True returns (probability, pre-sigmoid logit); rows with a missing signal get logit -inf."""
        columns = [
            rc_feat,
            torch.log10(vkl.clamp_min(VKL_FLOOR)),
            torch.log10((1.0 - echo_p).clamp_min(ECHO_FLOOR)),
            entropy, top1_match.float(), torch.log1p(age.float()),
        ]
        if "progress" in self.scalars:
            if progress is None:
                raise ValueError("this gate needs the progress scalar")
            columns.append(progress.float())
        if "is_question" in self.scalars:
            columns.append(torch.full_like(rc_feat, self.row_type))
        scalars = torch.stack(columns, dim=1)
        x = torch.cat([feats, scalars], dim=1)
        z = (x - self.mean) / self.scale
        logit = z @ self.coef + self.intercept
        p = torch.sigmoid(logit)
        finite = torch.isfinite(x).all(dim=1)
        p = torch.where(finite, p, torch.zeros_like(p))
        if return_logit:
            return p, torch.where(finite, logit, torch.full_like(logit, -torch.inf))
        return p

    def decide(self, scores: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        return (scores >= self.threshold) & (age >= self.min_age)


PACING_MODES = ("none", "intersect", "rank")


def paced_decision(ok: torch.Tensor, mode: str, *, conf_pass: torch.Tensor | None = None, logit: torch.Tensor | None = None,
                   age: torch.Tensor | None = None, min_age: int = 2, top_percent: float = 20.0) -> torch.Tensor:
    """Confidence pacing for the learned gate's answer-row decision (all tensors are over the answer-row candidates of one call).

    ok         the gate's own decision (p >= theta and age >= min_age)
    none       ok unchanged
    intersect  ok AND conf_pass, where conf_pass is SureLock's per-call test (uncertainty <= its 20th percentile over the
               fixed rule's candidate pool); the caller computes conf_pass over that pool
    rank       ok AND the row's logit is in the top `top_percent` percent of the pool = rows with age >= min_age and a finite logit:
               logit >= torch.quantile(pool logits, 1 - top_percent/100) (linear interpolation, ties admitted). An empty pool
               freezes nothing; a pool of one row freezes it iff ok.
    """
    if mode == "none":
        return ok
    if mode == "intersect":
        if conf_pass is None:
            raise ValueError("intersect pacing needs conf_pass")
        return ok & conf_pass
    if mode == "rank":
        if logit is None or age is None:
            raise ValueError("rank pacing needs logit and age")
        pool = (age >= min_age) & torch.isfinite(logit)
        if not bool(pool.any()):
            return torch.zeros_like(ok)
        cutoff = torch.quantile(logit[pool].float(), 1.0 - top_percent / 100.0)
        return ok & pool & (logit >= cutoff)
    raise ValueError(f"unknown pacing mode {mode}")
