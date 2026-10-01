"""Separate reference-cache and hidden-state commitment decoders.

The original 256-token budget is shared across methods. Cached-reference arms
remove compute rows; token-commitment arms leave all representations live.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID
from regret_remasking.locking_policy import _as_device, _prepare_prompt, _decode_response, _sync_if_cuda

PAPER_ARMS = {
    "B0", "B-entropy", "S-no-lock", "S-dense", "S-sparse", "S-window",
    "W-confidence", "W-entropy", "T-fixed", "T-adaptive", "C-adaptive", "K-adaptive",
    "C-block",
}
REFERENCE_ARMS = {"S-no-lock", "S-dense", "S-sparse", "S-window"}
TRACE_ARMS = {"T-fixed", "T-adaptive"}
TOKEN_KL_ARMS = {"K-adaptive"}
WINDOW_ARMS = {"W-confidence", "W-entropy", "S-window", "T-fixed", "T-adaptive", "C-adaptive", "K-adaptive"}


@dataclass(frozen=True)
class PaperDecodeConfig:
    arm: str = "B0"
    gen_length: int = 256
    block_length: int = 32
    window_length: int = 32
    epsilon: float = 0.005
    confidence_percentile: float = 100.0
    threshold: float = 0.95
    freeze_scope: str = "generated"
    allow_newly_unmasked: bool = False
    kl_negative_tolerance: float = 2e-6
    kl_space: str = "token"
    representation_temperature: float = 1.0
    mask_id: int = MASK_ID
    runtime_cls: Any | None = None


@dataclass
class PaperDecodeResult:
    text: str
    token_ids: list[int]
    nfe: int
    forward_calls: int
    latency_s: float
    prompt_length: int
    final_mask_count: int
    immutable_token_violations: int
    frozen_count: int
    generated_row_updates_avoided: int
    total_query_rows: int
    total_projection_rows: int
    total_ffn_rows: int
    total_output_head_rows: int
    attention_layout: str
    events: list[dict[str, Any]] = field(default_factory=list)
    mask_proposals_rejected: int = 0
    raw_kl_min: float | None = None
    raw_kl_negative_count: int = 0
    raw_kl_exact_zero_count: int = 0
    fallback_count: int = 0
    adaptive_accept_count: int = 0
    frozen_prompt_count: int = 0
    controller_calls: int = 0
    gate_time_s: float = 0.0
    scorer_time_s: float = 0.0
    recording_mode: str = "timed_minimal"
    generated_row_updates_avoided_basis: str = "reference_freeze_coverage"
    policy_subtimers: str = "diagnostic_wall_clock"
    row_count_basis: str = "per_layer_equivalent; output_head_count_is_actual"
    kl_space: str = "token"
    representation_temperature: float = 1.0
    representation_layer: str | None = None
    representation_feature_dim: int | None = None


def _validate(config: PaperDecodeConfig) -> None:
    if config.arm not in PAPER_ARMS:
        raise ValueError(f"unknown paper arm: {config.arm}")
    if config.gen_length < 1 or config.block_length < 1 or config.window_length < 1:
        raise ValueError("generation/block/window lengths must be positive")
    if config.gen_length % config.block_length:
        raise ValueError("generation length must be divisible by teacher/fixed block length")
    if config.freeze_scope not in {"generated", "all_unmasked"}:
        raise ValueError("freeze_scope must be generated or all_unmasked")
    if not math.isfinite(config.epsilon) or config.epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not 0 <= config.confidence_percentile <= 100:
        raise ValueError("confidence_percentile must lie in [0,100]")
    if not 0 <= config.threshold <= 1:
        raise ValueError("threshold must lie in [0,1]")
    if config.kl_space not in {"token", "representation"}:
        raise ValueError("kl_space must be token or representation")
    if config.arm in TOKEN_KL_ARMS and config.kl_space != "token":
        raise ValueError("token-KL commitment arms only support token kl_space")
    if not math.isfinite(config.representation_temperature) or config.representation_temperature <= 0:
        raise ValueError("representation_temperature must be finite and positive")


def _stability_logp(token_logp: torch.Tensor, hidden: torch.Tensor | None, config: PaperDecodeConfig) -> torch.Tensor:
    """Choose the online gate's probability space independently of token selection."""
    if config.kl_space == "token":
        return token_logp
    if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != token_logp.shape[:2]:
        raise ValueError("representation KL requires aligned pre-vocabulary hidden states")
    result = F.log_softmax(hidden.float() / config.representation_temperature, dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite representation distribution")
    return result


def _argmax_stable(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # Positions are sorted; torch.argmax therefore resolves a tie by position.
    return positions[scores.argmax().reshape(1)]


def _states(x: torch.Tensor, prompt_length: int, mask_id: int) -> torch.Tensor:
    states = torch.where(x == mask_id, 2, 1).to(torch.uint8)
    states[:, :prompt_length] = 0
    states[(x == EOS_ID) | (x == EOT_ID)] = 3
    states[:, :prompt_length] = 0
    return states


def _last_hidden(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = getattr(output, "hidden_states", None)
    if hidden is None or len(hidden) != 33:
        raise ValueError("expected native LLaDA hidden tuple length 33 (32 layers plus final norm)")
    # Zero-based tuple indices: one-based block30/block31 outputs and final-normalized block32.
    return hidden[30], hidden[31], hidden[32]


@torch.no_grad()
def decode_paper_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    example_id: str,
    config: PaperDecodeConfig,
    *,
    controller: torch.nn.Module | None = None,
    projector: torch.nn.Module | None = None,
    collect_events: bool = False,
    step_observer: Callable[[dict[str, Any]], None] | None = None,
) -> PaperDecodeResult:
    _validate(config)
    if config.arm in TRACE_ARMS and (controller is None or projector is None):
        raise ValueError("hidden commitment requires a trained controller and projector")
    model.eval()
    if controller is not None:
        controller.eval()
    if projector is not None:
        projector.eval()
    device = _as_device(model)
    _sync_if_cuda(device)
    started = time.perf_counter()
    prompt_ids, prompt_attention = _prepare_prompt(tokenizer, prompt, device)
    if prompt_ids.shape[0] != 1:
        raise ValueError("paper study supports batch size one")
    prompt_length = int(prompt_ids.shape[1])
    length = prompt_length + config.gen_length
    x = torch.full((1, length), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt_ids
    attention = torch.cat([prompt_attention, torch.ones((1, config.gen_length), device=device, dtype=prompt_attention.dtype)], dim=1)
    committed = torch.full_like(x, -1)
    frozen: set[int] = set()
    all_positions = torch.arange(length, device=device)
    prev_gate_logp: torch.Tensor | None = None
    prev_positions: torch.Tensor | None = None
    prev_decision_logp: torch.Tensor | None = None
    prev_decision_positions: torch.Tensor | None = None
    events: list[dict[str, Any]] = []
    nfe = query_rows = projection_rows = ffn_rows = head_rows = avoided = 0
    mask_rejected = neg_kl = zero_kl = fallbacks = adaptive_count = controller_calls = 0
    min_kl: float | None = None
    scorer_seconds = gate_seconds = 0.0
    representation_feature_dim: int | None = None
    runtime = None
    if config.arm in REFERENCE_ARMS:
        if config.runtime_cls is None:
            from regret_remasking.paper_reference import PaperReferenceRuntime
            runtime_cls = PaperReferenceRuntime
        else:
            runtime_cls = config.runtime_cls
        runtime = runtime_cls(model, mode="dense" if config.arm == "S-dense" else "sparse", total_len=length, capture_diagnostics=False)
    try:
        if runtime is not None:
            runtime.install()
        for step in range(config.gen_length):
            remaining = torch.nonzero(x[0, prompt_length:] == config.mask_id, as_tuple=False).flatten() + prompt_length
            if remaining.numel() == 0:
                break
            first = int(remaining[0])
            if config.arm in WINDOW_ARMS:
                left, right = first, min(length, first + config.window_length)
            else:
                block = (first - prompt_length) // config.block_length
                left = prompt_length + block * config.block_length
                right = left + config.block_length
            eligible = remaining[(remaining >= left) & (remaining < right)]
            frozen_generated = sum(p >= prompt_length for p in frozen)
            avoided += frozen_generated
            needs_hidden = config.arm in TRACE_ARMS or step_observer is not None
            if runtime is not None and hasattr(runtime, "begin_forward"):
                runtime.begin_forward()
            kwargs: dict[str, Any] = {"attention_mask": attention}
            if needs_hidden:
                kwargs["output_hidden_states"] = True
            output = model(x, **kwargs)
            nfe += 1
            logits = output.logits if hasattr(output, "logits") else output[0]
            positions = runtime.output_positions if runtime is not None else all_positions
            if logits.ndim != 3 or logits.shape[:2] != (1, positions.numel()):
                raise ValueError("output logits do not match runtime output positions")
            active_count = length - len(frozen)
            if runtime is not None:
                work = runtime.last_work_counts
                layer_count = len(runtime.blocks)
                expected_rows = active_count if config.arm in {"S-sparse", "S-no-lock", "S-window"} else length
                for name in ["attention_query_rows", "projection_rows", "ffn_rows"]:
                    if work[name] != expected_rows * layer_count:
                        raise AssertionError(f"actual {name} differs from the intended backend work")
                if work["head_rows"] != expected_rows:
                    raise AssertionError("actual output-head rows differ from intended work")
                query_rows += work["attention_query_rows"] // layer_count
                projection_rows += work["projection_rows"] // layer_count
                ffn_rows += work["ffn_rows"] // layer_count
                head_rows += work["head_rows"]
            else:
                query_rows += length
                projection_rows += length
                ffn_rows += length
                head_rows += length

            # Reference gates observe active positions. Other policies only need
            # the current decision window, unless an observer requests full traces.
            if runtime is not None:
                observed = positions if config.freeze_scope == "all_unmasked" else positions[positions >= prompt_length]
            elif step_observer is not None:
                observed = all_positions
            else:
                observed = eligible
            gather = torch.searchsorted(positions, observed)
            current_logits = logits.index_select(1, gather).float()
            logp = F.log_softmax(current_logits, dim=-1)
            if not bool(torch.isfinite(logp).all()):
                raise ValueError("nonfinite model posterior")
            proposal_logits = current_logits.clone()
            if 0 <= config.mask_id < current_logits.shape[-1]:
                proposal_logits[..., config.mask_id] = -torch.inf
            proposed = proposal_logits.argmax(dim=-1)
            confidence = logp.gather(-1, proposed.unsqueeze(-1)).squeeze(-1).exp()
            current_mask = x.index_select(1, observed) == config.mask_id
            mask_rejected += int(((current_logits.argmax(-1) == config.mask_id) & current_mask).sum())
            eligible_index = torch.searchsorted(observed, eligible)
            eligible_conf = confidence[0].index_select(0, eligible_index)
            eligible_tokens = proposed[0].index_select(0, eligible_index)
            gate_logp = None
            if runtime is not None:
                hidden = None
                if config.kl_space == "representation":
                    captured = getattr(runtime, "last_output_hidden", None)
                    if captured is None or captured.ndim != 3 or captured.shape[:2] != logits.shape[:2]:
                        raise ValueError("runtime did not capture aligned pre-vocabulary hidden states")
                    if not torch.equal(runtime.last_output_positions, positions):
                        raise ValueError("representation capture position mismatch")
                    hidden = captured.index_select(1, gather)
                    representation_feature_dim = int(hidden.shape[-1])
                gate_logp = _stability_logp(logp, hidden, config)
            raw_kl = kl = None
            if runtime is not None and prev_gate_logp is not None:
                assert gate_logp is not None
                if prev_positions is None:
                    raise AssertionError("missing stability history positions")
                old_index = torch.searchsorted(prev_positions, observed)
                if not torch.equal(prev_positions.index_select(0, old_index), observed):
                    raise AssertionError("active stability history lost position alignment")
                raw_kl = (gate_logp.exp() * (gate_logp - prev_gate_logp.index_select(1, old_index))).sum(-1)
                value = float(raw_kl.min())
                if not math.isfinite(value) or value < -config.kl_negative_tolerance:
                    raise ValueError(f"invalid KL residual {value}")
                min_kl = value if min_kl is None else min(min_kl, value)
                neg_kl += int((raw_kl < 0).sum())
                zero_kl += int((raw_kl == 0).sum())
                kl = raw_kl.clamp_min(0)

            decision_kl: torch.Tensor | None = None
            decision_had_history: torch.Tensor | None = None
            if config.arm in TOKEN_KL_ARMS:
                decision_had_history = torch.zeros(eligible.shape, dtype=torch.bool, device=device)
                if prev_decision_logp is not None:
                    if prev_decision_positions is None:
                        raise AssertionError("missing decision stability history positions")
                    old_index = torch.searchsorted(prev_decision_positions, eligible)
                    in_bounds = old_index < prev_decision_positions.numel()
                    matched = torch.zeros_like(in_bounds)
                    in_bounds_indices = torch.nonzero(in_bounds, as_tuple=False).flatten()
                    if in_bounds_indices.numel():
                        bounded_old = old_index.index_select(0, in_bounds_indices)
                        bounded_positions = prev_decision_positions.index_select(0, bounded_old)
                        bounded_eligible = eligible.index_select(0, in_bounds_indices)
                        matched.index_put_((in_bounds_indices,), bounded_positions == bounded_eligible)
                    matched_indices = torch.nonzero(matched, as_tuple=False).flatten()
                    if matched_indices.numel():
                        current_overlap = logp.index_select(1, eligible_index.index_select(0, matched_indices))
                        previous_overlap = prev_decision_logp.index_select(1, old_index.index_select(0, matched_indices))
                        raw_decision_kl = (current_overlap.exp() * (current_overlap - previous_overlap)).sum(-1)
                        value = float(raw_decision_kl.min())
                        if not math.isfinite(value) or value < -config.kl_negative_tolerance:
                            raise ValueError(f"invalid token KL residual {value}")
                        min_kl = value if min_kl is None else min(min_kl, value)
                        neg_kl += int((raw_decision_kl < 0).sum())
                        zero_kl += int((raw_decision_kl == 0).sum())
                        decision_kl = torch.full(eligible.shape, torch.inf, dtype=raw_decision_kl.dtype, device=device)
                        decision_kl.index_put_((matched_indices,), raw_decision_kl.clamp_min(0)[0])
                    decision_had_history = matched

            trace_scores = None
            if config.arm in TRACE_ARMS:
                score_start = time.perf_counter()
                hidden = tuple(h[:, :right] for h in _last_hidden(output))
                features = projector.encode_hidden(hidden)  # type: ignore[union-attr]
                if features.shape[-1] != 832:
                    raise ValueError("primary hidden controller requires exactly 832 features")
                valid = torch.ones(features.shape[:2], device=device, dtype=torch.bool)
                scored = controller(features, valid_mask=valid, position_ids=all_positions[:right].unsqueeze(0))  # type: ignore[misc]
                trace_scores = torch.sigmoid(scored["margins"])[0].index_select(0, eligible)
                if not bool(torch.isfinite(trace_scores).all()):
                    raise ValueError("nonfinite learned commitment probabilities")
                scorer_seconds += time.perf_counter() - score_start
                controller_calls += 1

            fallback = False
            if config.arm == "T-adaptive":
                assert trace_scores is not None
                accept = trace_scores > config.threshold
                selected = eligible[accept]
                if selected.numel() == 0:
                    selected = _argmax_stable(trace_scores, eligible)
                    fallback = True
            elif config.arm in {"C-adaptive", "C-block"}:
                # C-block is Fast-dLLM-style threshold decoding inside the fixed block.
                selected = eligible[eligible_conf >= config.threshold]
                if selected.numel() == 0:
                    selected = _argmax_stable(eligible_conf, eligible)
                    fallback = True
            elif config.arm == "K-adaptive":
                assert decision_had_history is not None
                if decision_kl is None:
                    selected = _argmax_stable(eligible_conf, eligible)
                    fallback = True
                else:
                    selected = eligible[decision_had_history & (decision_kl <= config.epsilon)]
                    if selected.numel() == 0:
                        selected = _argmax_stable(eligible_conf, eligible)
                        fallback = True
            elif config.arm == "T-fixed":
                assert trace_scores is not None
                selected = _argmax_stable(trace_scores, eligible)
            elif config.arm in {"B-entropy", "W-entropy"}:
                eligible_logp = logp[0].index_select(0, eligible_index)
                entropy = -(eligible_logp.exp() * eligible_logp).sum(-1)
                selected = _argmax_stable(-entropy, eligible)
            else:
                selected = _argmax_stable(eligible_conf, eligible)

            if step_observer is not None:
                step_observer({
                    "example_id": example_id, "step": step, "prompt_length": prompt_length,
                    "gen_length": config.gen_length, "block_start": left, "block_end": right,
                    "input_ids": x.detach().clone(), "states": _states(x, prompt_length, config.mask_id),
                    "proposal_positions": observed, "proposal_tokens": proposed.detach(),
                    "confidence": confidence.detach(), "last_hidden": _last_hidden(output),
                    "selected_positions": selected.detach(),
                })
            token_index = torch.searchsorted(eligible, selected)
            chosen_tokens = eligible_tokens.index_select(0, token_index)
            if bool((committed[0, selected] >= 0).any()):
                raise AssertionError("selected an already committed token")
            x[0, selected] = chosen_tokens
            committed[0, selected] = chosen_tokens
            if config.arm in {"T-adaptive", "C-adaptive", "C-block"}:
                adaptive_count += int(selected.numel())
            if config.arm == "K-adaptive":
                adaptive_count += int(selected.numel())
            fallbacks += int(fallback)
            if collect_events:
                for pos, token in zip(selected.tolist(), chosen_tokens.tolist()):
                    event = {"type": "token_commit", "global_step": step, "position": pos,
                             "relative_position": pos - prompt_length, "token": token, "fallback": fallback}
                    if config.arm == "K-adaptive":
                        assert decision_had_history is not None
                        selected_index = int(torch.searchsorted(eligible, torch.tensor(pos, device=device)).item())
                        event["had_immediate_history"] = bool(decision_had_history[selected_index])
                        event["kl"] = (
                            None if decision_kl is None or not bool(torch.isfinite(decision_kl[selected_index]))
                            else float(decision_kl[selected_index])
                        )
                    events.append(event)
            if runtime is not None and kl is not None and config.arm != "S-no-lock":
                gate_start = time.perf_counter()
                # Primary snapshots must have been computed with the revealed token
                # already present. The paper-order ablation admits just-revealed rows.
                revealed = (x[0].index_select(0, observed) != config.mask_id) if config.allow_newly_unmasked else ~current_mask[0]
                revealed &= (x[0].index_select(0, observed) != EOS_ID) & (x[0].index_select(0, observed) != EOT_ID)
                candidate = torch.nonzero(revealed, as_tuple=False).flatten()
                if candidate.numel():
                    passes = kl[0].index_select(0, candidate) <= config.epsilon
                    if config.confidence_percentile < 100.0:
                        uncertainty = 1 - confidence[0].index_select(0, candidate)
                        cutoff = torch.quantile(uncertainty, config.confidence_percentile / 100.0)
                        passes &= uncertainty <= cutoff
                    accepted = candidate[passes]
                    to_freeze = observed.index_select(0, accepted).tolist()
                    if to_freeze:
                        runtime.commit(to_freeze)
                        frozen.update(to_freeze)
                        if collect_events:
                            for idx, pos in zip(accepted.tolist(), to_freeze):
                                events.append({"type": "reference_freeze", "global_step": step, "position": pos,
                                               "relative_position": pos-prompt_length, "kl": float(kl[0, idx]),
                                               "kl_space": config.kl_space,
                                               "representation_temperature": config.representation_temperature,
                                               "effective_from_step": step+1, "input_was_masked": bool(current_mask[0, idx])})
                gate_seconds += time.perf_counter() - gate_start
            if runtime is not None:
                assert gate_logp is not None
                prev_gate_logp, prev_positions = gate_logp.detach(), observed
            if config.arm in TOKEN_KL_ARMS:
                prev_decision_logp = logp.detach() if observed.data_ptr() == eligible.data_ptr() else logp.index_select(1, eligible_index).detach()
                prev_decision_positions = eligible.detach()
            del output, logits, current_logits, proposal_logits
    finally:
        if runtime is not None:
            runtime.remove()
    violation = int(((committed >= 0) & (committed != x)).sum())
    if not torch.equal(x[:, :prompt_length], prompt_ids):
        raise AssertionError("prompt token identity changed")
    masks = int((x[:, prompt_length:] == config.mask_id).sum())
    if masks or violation:
        raise ValueError(f"decode incomplete: masks={masks}, immutable_violations={violation}")
    tokens = x[0, prompt_length:].tolist()
    text = _decode_response(tokenizer, tokens)
    _sync_if_cuda(device)
    elapsed = time.perf_counter()-started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid decode latency")
    return PaperDecodeResult(
        text=text, token_ids=tokens, nfe=nfe, forward_calls=nfe, latency_s=elapsed,
        prompt_length=prompt_length, final_mask_count=masks, immutable_token_violations=violation,
        frozen_count=sum(p>=prompt_length for p in frozen), frozen_prompt_count=sum(p<prompt_length for p in frozen),
        generated_row_updates_avoided=avoided, total_query_rows=query_rows, total_projection_rows=projection_rows,
        total_ffn_rows=ffn_rows,total_output_head_rows=head_rows,
        attention_layout="compact" if config.arm in {"S-no-lock","S-sparse","S-window"} else "full",
        events=events, mask_proposals_rejected=mask_rejected,raw_kl_min=min_kl,
        raw_kl_negative_count=neg_kl,raw_kl_exact_zero_count=zero_kl,
        fallback_count=fallbacks,adaptive_accept_count=adaptive_count,controller_calls=controller_calls,
        gate_time_s=gate_seconds,scorer_time_s=scorer_seconds,
        recording_mode="observer" if step_observer is not None else ("events" if collect_events else "timed_minimal"),
        generated_row_updates_avoided_basis="reference_freeze_coverage" if runtime is not None else "none",
        kl_space=config.kl_space, representation_temperature=config.representation_temperature,
        representation_layer="final_normalized_pre_vocab" if representation_feature_dim is not None else None,
        representation_feature_dim=representation_feature_dim,
    )
