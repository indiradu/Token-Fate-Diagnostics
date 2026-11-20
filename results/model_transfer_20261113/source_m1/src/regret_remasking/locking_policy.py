from __future__ import annotations

import math
import time
import inspect
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from regret_remasking import FEATURE_NAMES
from regret_remasking.features import entropy_from_log_probs, local_window_average
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID, model_device


LOCKING_ARMS = {"B0", "R-dense", "R-sparse", "T-fixed", "C-adaptive", "T-adaptive"}


@dataclass
class LockingPolicyConfig:
    arm: str = "B0"
    gen_length: int = 256
    block_length: int = 32
    epsilon: float = 0.0
    threshold: float = 0.95
    kl_negative_tolerance: float = 1e-6
    cpv_window: int = 4
    mask_id: int = MASK_ID
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    exclude_eos_eot_from_freeze: bool = True
    reject_mask_proposals: bool = True
    reference_runtime_cls: Any | None = None


@dataclass
class LockingPolicyResult:
    text: str
    token_ids: list[int]
    trace_rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    nfe: int
    forward_calls: int
    latency_s: float
    prompt_length: int
    final_mask_count: int
    immutable_token_violations: int
    frozen_count: int
    generated_row_updates_avoided: int
    generated_row_updates_avoided_basis: str
    total_query_rows: int
    total_projection_rows: int
    total_ffn_rows: int
    attention_layout: str
    gate_time_s: float
    scorer_time_s: float
    mask_proposals_rejected: int = 0
    raw_kl_min: float | None = None
    raw_kl_negative_count: int = 0
    raw_kl_exact_zero_count: int = 0
    adaptive_accept_count: int = 0
    fallback_count: int = 0
    recording_mode: str = "timed_minimal"
    policy_subtimers: str = "diagnostic_wall_clock"


def _as_device(model: torch.nn.Module) -> torch.device:
    try:
        return model_device(model)
    except StopIteration:
        return torch.device("cpu")


def _prepare_prompt(tokenizer: Any, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    text = prompt
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    if getattr(tokenizer, "padding_side", None) not in {None, "left"}:
        tokenizer.padding_side = "left"
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    return input_ids.to(device), attention_mask.to(device)


def _decode_response(tokenizer: Any, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode([token_ids], skip_special_tokens=True)[0].strip()
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
    return " ".join(str(token_id) for token_id in token_ids)


def _forward_logits(model: torch.nn.Module, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(x, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out[0]
    if logits.ndim != 3:
        raise ValueError(f"model logits must be [batch, length, vocab], got {tuple(logits.shape)}")
    if logits.shape[:2] != x.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not match ids shape {tuple(x.shape)}")
    return logits


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _raw_kl(
    log_probs: torch.Tensor,
    prev_log_probs: torch.Tensor,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    kl = _raw_kl_unclamped(log_probs, prev_log_probs)
    if not bool(torch.isfinite(kl).all()):
        raise ValueError("non-finite KL value from model logits")
    min_kl = torch.amin(kl)
    if bool((min_kl < -float(tolerance)).detach().cpu()):
        raise ValueError(f"materially negative KL value {float(min_kl.detach().cpu())}")
    return kl.clamp_min(0.0)


def _raw_kl_unclamped(log_probs: torch.Tensor, prev_log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    return (probs * (log_probs - prev_log_probs.float())).sum(dim=-1)


def _validate_kl(
    kl: torch.Tensor,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    if not bool(torch.isfinite(kl).all()):
        raise ValueError("non-finite KL value from model logits")
    min_kl = torch.amin(kl)
    if bool((min_kl < -float(tolerance)).detach().cpu()):
        raise ValueError(f"materially negative KL value {float(min_kl.detach().cpu())}")
    return kl.clamp_min(0.0)


def _top_features(
    log_probs: torch.Tensor,
    x0: torch.Tensor,
    mask_index: torch.Tensor,
    prev_top1: torch.Tensor,
    runlength: torch.Tensor,
    t_frac: float,
    cpv_window: int,
    kl: torch.Tensor | None,
    names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    requested = set(FEATURE_NAMES if names is None else names)
    unknown = requested.difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature names: {sorted(unknown)}")
    feature_map: dict[str, torch.Tensor] = {}
    top_probs: torch.Tensor | None = None
    if requested.intersection({"confidence", "margin"}):
        probs = log_probs.exp()
        top_probs, _ = probs.topk(k=2, dim=-1)
    if "confidence" in requested:
        if top_probs is None:
            raise AssertionError("top probabilities were not computed")
        feature_map["confidence"] = top_probs[:, :, 0]
    if "entropy" in requested:
        feature_map["entropy"] = entropy_from_log_probs(log_probs)
    if "margin" in requested:
        if top_probs is None:
            raise AssertionError("top probabilities were not computed")
        feature_map["margin"] = top_probs[:, :, 0] - top_probs[:, :, 1]
    if "kl" in requested or "context_volatility" in requested:
        if kl is None:
            kl = torch.zeros(log_probs.shape[:-1], dtype=torch.float32, device=log_probs.device)
        feature_map["kl"] = kl
    if "top1_flip" in requested:
        feature_map["top1_flip"] = ((prev_top1 >= 0) & (prev_top1 != x0)).float()
    current_runlength = torch.where(prev_top1 == x0, runlength + 1.0, torch.ones_like(runlength))
    if "runlength" in requested:
        feature_map["runlength"] = current_runlength
    if "t_frac" in requested:
        feature_map["t_frac"] = torch.full(log_probs.shape[:-1], float(t_frac), dtype=torch.float32, device=log_probs.device)
    if "local_mask_ratio" in requested:
        feature_map["local_mask_ratio"] = local_window_average(mask_index.float(), cpv_window)
    if "context_volatility" in requested:
        if kl is None:
            kl = torch.zeros(log_probs.shape[:-1], dtype=torch.float32, device=log_probs.device)
        feature_map["context_volatility"] = local_window_average(kl, cpv_window)
    return feature_map


def _feature_names_for_step(config: LockingPolicyConfig, scorer: Any | None, collect_trace: bool) -> set[str]:
    names: set[str] = set()
    if collect_trace:
        names.update(FEATURE_NAMES)
    if config.arm == "C-adaptive":
        if scorer is None:
            raise ValueError("C-adaptive requires a stability scorer")
        names.update(getattr(scorer, "features", ["confidence"]))
    elif config.arm in {"T-fixed", "T-adaptive"}:
        names.update(FEATURE_NAMES)
    if config.arm in {"R-dense", "R-sparse"}:
        names.add("kl")
    return names


def _score_rows(
    scorer: Any,
    feature_map: dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=positions.device), 0.0
    features = list(getattr(scorer, "features", FEATURE_NAMES))
    missing = [name for name in features if name not in feature_map]
    if missing:
        raise ValueError(f"scorer requested unknown features: {missing}")
    start = time.perf_counter()
    rows_tensor = torch.stack([feature_map[name][0].index_select(0, positions) for name in features], dim=1)
    rows = rows_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    scores = np.asarray(scorer.predict_stability(rows), dtype=np.float32)
    elapsed = time.perf_counter() - start
    if scores.shape != (int(positions.numel()),):
        raise ValueError(f"scorer returned shape {scores.shape}; expected {(int(positions.numel()),)}")
    if not np.isfinite(scores).all():
        raise ValueError("scorer returned non-finite stability scores")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("scorer stability probabilities must be in [0, 1]")
    return torch.as_tensor(scores, dtype=torch.float32, device=positions.device), elapsed


def _make_runtime(
    model: torch.nn.Module,
    config: LockingPolicyConfig,
    total_len: int,
) -> Any | None:
    if config.arm not in {"R-dense", "R-sparse"}:
        return None
    runtime_cls = config.reference_runtime_cls or getattr(model, "reference_runtime_cls", None)
    if runtime_cls is None:
        from regret_remasking.locking_reference import ReferenceRuntime

        runtime_cls = cast(Any, ReferenceRuntime)
    mode = "dense" if config.arm == "R-dense" else "sparse"
    try:
        signature = inspect.signature(runtime_cls)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "capture_diagnostics" in signature.parameters:
        return runtime_cls(
            model,
            mode=mode,
            total_len=total_len,
            capture_diagnostics=False,
        )
    return runtime_cls(model, mode=mode, total_len=total_len)


def _attention_layout(runtime: Any | None, config: LockingPolicyConfig) -> str:
    if config.arm != "R-sparse":
        return "full"
    if runtime is None:
        return "compact"
    layout = getattr(runtime, "attention_layout", "compact")
    if layout not in {"compact", "full"}:
        raise ValueError(f"unsupported sparse attention layout: {layout!r}")
    return str(layout)


def _select_positions(
    config: LockingPolicyConfig,
    allowed_positions: torch.Tensor,
    confidence: torch.Tensor,
    stability_scores: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if allowed_positions.numel() == 0:
        return allowed_positions, False
    if config.arm in {"B0", "R-dense", "R-sparse"}:
        raw_scores = confidence[0].index_select(0, allowed_positions)
    else:
        if stability_scores is None:
            raise ValueError(f"{config.arm} requires stability scores")
        raw_scores = stability_scores

    local_order = torch.argsort(raw_scores, descending=True, stable=True)
    if config.arm in {"C-adaptive", "T-adaptive"}:
        accepted = allowed_positions[raw_scores >= config.threshold]
        if accepted.numel() > 0:
            return torch.sort(accepted).values, False
        return allowed_positions[local_order[:1]], True
    return allowed_positions[local_order[:1]], False


@torch.no_grad()
def decode_locking_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    example_id: str,
    config: LockingPolicyConfig,
    scorer: Any | None = None,
    collect_trace: bool = False,
    collect_events: bool = True,
) -> LockingPolicyResult:
    if config.arm not in LOCKING_ARMS:
        raise ValueError(f"unknown locking arm: {config.arm}")
    if config.gen_length <= 0 or config.block_length <= 0:
        raise ValueError("gen_length and block_length must be positive")
    if config.gen_length % config.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    if not math.isfinite(config.epsilon) or config.epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if not math.isfinite(config.kl_negative_tolerance) or config.kl_negative_tolerance < 0.0:
        raise ValueError("kl_negative_tolerance must be finite and non-negative")
    if not math.isfinite(config.threshold) or not 0.0 <= config.threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    if config.arm in {"T-fixed", "C-adaptive", "T-adaptive"} and scorer is None:
        raise ValueError(f"{config.arm} requires a stability scorer")

    device = _as_device(model)
    _sync_if_cuda(device)
    start_time = time.perf_counter()
    input_ids, attention_mask = _prepare_prompt(tokenizer, prompt, device)
    if input_ids.shape[0] != 1:
        raise ValueError("decode_locking_policy expects a single prompt")
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + config.gen_length
    x = torch.full((1, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )

    runtime = _make_runtime(model, config, total_len)
    if runtime is not None:
        runtime.install()
    attention_layout = _attention_layout(runtime, config)
    feature_names = _feature_names_for_step(config, scorer, collect_trace)
    needs_kl = bool(feature_names.intersection({"kl", "context_volatility"})) or config.arm in {"R-dense", "R-sparse"}

    prev_log_probs: torch.Tensor | None = None
    prev_top1 = torch.full((1, total_len), -1, dtype=torch.long, device=device)
    runlength = torch.zeros((1, total_len), dtype=torch.float32, device=device)
    committed_step = torch.full((1, total_len), -1, dtype=torch.long, device=device)
    committed_tokens = torch.full((1, total_len), -1, dtype=torch.long, device=device)
    frozen_mask = torch.zeros((1, total_len), dtype=torch.bool, device=device)
    trace_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    nfe = 0
    gate_time_s = 0.0
    scorer_time_s = 0.0
    generated_row_updates_avoided = 0
    total_query_rows = 0
    total_projection_rows = 0
    total_ffn_rows = 0
    immutable_token_violations = 0
    mask_proposals_rejected = 0
    raw_kl_min: float | None = None
    raw_kl_negative_count = 0
    raw_kl_exact_zero_count = 0
    adaptive_accept_count = 0
    fallback_count = 0

    try:
        for step in range(config.gen_length):
            masks_remaining = int((x[:, prompt_len:] == config.mask_id).sum().detach().cpu())
            if masks_remaining == 0:
                break
            block_idx = min(config.gen_length // config.block_length - 1, (config.gen_length - masks_remaining) // config.block_length)
            block_start = prompt_len + block_idx * config.block_length
            block_end = prompt_len + (block_idx + 1) * config.block_length

            logits = _forward_logits(model, x, attention_mask)
            nfe += 1
            frozen_count_now = int(frozen_mask.sum().detach().cpu())
            active_rows_now = int(total_len - frozen_count_now)
            if config.arm == "R-sparse":
                if attention_layout == "full":
                    total_query_rows += total_len
                else:
                    total_query_rows += active_rows_now
                total_projection_rows += active_rows_now
                total_ffn_rows += active_rows_now
            else:
                total_query_rows += total_len
                total_projection_rows += total_len
                total_ffn_rows += total_len
            generated_row_updates_avoided += frozen_count_now

            raw_log_probs = F.log_softmax(logits.float(), dim=-1)
            if not bool(torch.isfinite(raw_log_probs).all()):
                raise ValueError("model produced non-finite log probabilities")
            raw_kl_for_step: torch.Tensor | None = None
            kl_for_step: torch.Tensor | None = None
            if needs_kl and prev_log_probs is not None:
                raw_kl_for_step = _raw_kl_unclamped(raw_log_probs, prev_log_probs)
                kl_for_step = _validate_kl(raw_kl_for_step, tolerance=config.kl_negative_tolerance)
                step_min_tensor = raw_kl_for_step.min()
                step_min = float(step_min_tensor.detach().cpu())
                raw_kl_min = step_min if raw_kl_min is None else min(raw_kl_min, step_min)
                raw_kl_negative_count += int((raw_kl_for_step < 0.0).sum().detach().cpu())
                raw_kl_exact_zero_count += int((raw_kl_for_step == 0.0).sum().detach().cpu())
            logits_for_choice = logits.clone()
            if config.logits_eos_inf:
                logits_for_choice[:, :, EOS_ID] = -torch.inf
            if config.confidence_eos_eot_inf:
                logits_for_choice[:, :, EOS_ID] = -torch.inf
                logits_for_choice[:, :, EOT_ID] = -torch.inf
            if config.reject_mask_proposals and 0 <= config.mask_id < logits_for_choice.shape[-1]:
                logits_for_choice[:, :, config.mask_id] = -torch.inf
            mask_index = x == config.mask_id
            x0 = torch.argmax(logits_for_choice, dim=-1)
            proposal_probability = raw_log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1).exp()
            if config.reject_mask_proposals:
                mask_proposals_rejected += int(((torch.argmax(logits, dim=-1) == config.mask_id) & mask_index).sum().detach().cpu())
            current_runlength = torch.where(prev_top1 == x0, runlength + 1.0, torch.ones_like(runlength))
            feature_map = _top_features(
                log_probs=raw_log_probs,
                x0=x0,
                mask_index=mask_index,
                prev_top1=prev_top1,
                runlength=runlength,
                t_frac=(step + 1) / config.gen_length,
                cpv_window=config.cpv_window,
                kl=kl_for_step,
                names=feature_names,
            )
            if "confidence" not in feature_map:
                feature_map["confidence"] = proposal_probability
            if "runlength" not in feature_map:
                feature_map["runlength"] = current_runlength

            block_positions = torch.arange(block_start, block_end, dtype=torch.long, device=device)
            allowed_positions = block_positions[mask_index[0, block_start:block_end]]
            stability_scores: torch.Tensor | None = None
            if config.arm in {"T-fixed", "C-adaptive", "T-adaptive"}:
                stability_scores, elapsed = _score_rows(scorer, feature_map, allowed_positions)
                scorer_time_s += elapsed

            selected_positions, fallback = _select_positions(
                config=config,
                allowed_positions=allowed_positions,
                confidence=proposal_probability,
                stability_scores=stability_scores,
            )

            selected_position_set: set[int] = set()
            if collect_trace or collect_events:
                selected_position_set = {int(pos) for pos in selected_positions.detach().cpu().tolist()}

            if collect_trace:
                score_by_pos = {}
                if config.arm in {"T-fixed", "C-adaptive", "T-adaptive"} and stability_scores is not None:
                    score_by_pos = dict(
                        zip(
                            [int(pos) for pos in allowed_positions.detach().cpu().tolist()],
                            [float(score) for score in stability_scores.detach().cpu().tolist()],
                        )
                    )
                for pos in [int(pos) for pos in allowed_positions.detach().cpu().tolist()]:
                    row: dict[str, Any] = {
                        "example_id": example_id,
                        "global_step": step,
                        "block": block_idx,
                        "position": pos,
                        "relative_position": pos - prompt_len,
                        "top_token": int(x0[0, pos].detach().cpu()),
                        "score": float(score_by_pos.get(pos, feature_map["confidence"][0, pos].detach().cpu())),
                        "selected": pos in selected_position_set,
                        "fallback": fallback and pos in selected_position_set,
                        "mask_token_id": config.mask_id,
                        "arm": config.arm,
                    }
                    for name in FEATURE_NAMES:
                        row[name] = float(feature_map[name][0, pos].detach().cpu())
                    trace_rows.append(row)

            if selected_positions.numel() > 0:
                selected_tokens = x0[0].index_select(0, selected_positions)
                prior_tokens = committed_tokens[0].index_select(0, selected_positions)
                prior_committed = prior_tokens >= 0
                immutable_token_violations += int(((prior_committed) & (prior_tokens != selected_tokens)).sum().detach().cpu())
                x[0, selected_positions] = selected_tokens
                new_commit = committed_step[0].index_select(0, selected_positions) < 0
                if bool(new_commit.any().detach().cpu()):
                    new_positions = selected_positions[new_commit]
                    committed_step[0, new_positions] = step
                    committed_tokens[0, new_positions] = x0[0].index_select(0, new_positions)
                adaptive_accept_count += int(selected_positions.numel()) if config.arm in {"C-adaptive", "T-adaptive"} else 0
                fallback_count += int(fallback)
                if collect_events:
                    event_positions = [int(pos) for pos in selected_positions.detach().cpu().tolist()]
                    event_tokens = [int(token) for token in selected_tokens.detach().cpu().tolist()]
                    for pos, token in zip(event_positions, event_tokens):
                        events.append(
                            {
                                "type": "token_commit",
                                "arm": config.arm,
                                "global_step": step,
                                "position": pos,
                                "relative_position": pos - prompt_len,
                                "token": token,
                                "fallback": fallback,
                            }
                        )

            if config.arm in {"R-dense", "R-sparse"} and prev_log_probs is not None:
                gate_start = time.perf_counter()
                if kl_for_step is None:
                    raise RuntimeError("representation policy is missing KL for freeze gate")
                kl = kl_for_step
                eligible = (committed_step >= 0) & (step >= committed_step + 2) & ~frozen_mask
                eligible[:, :prompt_len] = False
                if config.exclude_eos_eot_from_freeze:
                    eligible &= (x != EOS_ID) & (x != EOT_ID)
                to_freeze_tensor = torch.nonzero(eligible & (kl <= config.epsilon), as_tuple=False)[:, 1]
                if to_freeze_tensor.numel() > 0:
                    to_freeze = [int(pos) for pos in to_freeze_tensor.detach().cpu().tolist()]
                    if runtime is None:
                        raise RuntimeError("representation policy has no reference backend")
                    runtime.commit(to_freeze)
                    frozen_mask[0, to_freeze_tensor] = True
                    if collect_events:
                        for pos in to_freeze:
                            events.append(
                                {
                                    "type": "reference_freeze",
                                    "arm": config.arm,
                                    "global_step": step,
                                    "position": pos,
                                    "relative_position": pos - prompt_len,
                                    "kl": float(kl[0, pos].detach().cpu()),
                                    "effective_from_step": step + 1,
                                }
                            )
                gate_time_s += time.perf_counter() - gate_start

            runlength = feature_map["runlength"].detach()
            prev_top1 = x0.detach()
            prev_log_probs = raw_log_probs.detach() if needs_kl else None
    finally:
        if runtime is not None:
            runtime.remove()

    final_committed = committed_tokens >= 0
    immutable_token_violations += int(((final_committed) & (x != committed_tokens)).sum().detach().cpu())
    final_tokens_tensor = x[:, prompt_len:].detach().cpu()
    final_tokens = [int(v) for v in final_tokens_tensor[0].tolist()]
    for row in trace_rows:
        rel_pos = int(row["relative_position"])
        final_token = final_tokens[rel_pos]
        row["final_token"] = final_token
        row["stable"] = int(int(row["top_token"]) == final_token)
        row["trace_regret"] = int(int(row["top_token"]) != final_token)

    text = _decode_response(tokenizer, final_tokens)
    _sync_if_cuda(device)
    latency_s = time.perf_counter() - start_time
    final_mask_count = int((x[:, prompt_len:] == config.mask_id).sum().detach().cpu())
    if final_mask_count:
        raise RuntimeError(f"decode ended with {final_mask_count} response masks")
    if not math.isfinite(latency_s):
        raise RuntimeError("non-finite decoder latency")
    if collect_trace and collect_events:
        recording_mode = "trace_and_events"
    elif collect_trace:
        recording_mode = "trace"
    elif collect_events:
        recording_mode = "events"
    else:
        recording_mode = "timed_minimal"
    return LockingPolicyResult(
        text=text,
        token_ids=final_tokens,
        trace_rows=trace_rows,
        events=events,
        nfe=nfe,
        forward_calls=nfe,
        latency_s=latency_s,
        prompt_length=prompt_len,
        final_mask_count=final_mask_count,
        immutable_token_violations=immutable_token_violations,
        frozen_count=int(frozen_mask.sum().detach().cpu()),
        generated_row_updates_avoided=generated_row_updates_avoided,
        generated_row_updates_avoided_basis=(
            "hypothetical_dense_reference_rows"
            if config.arm == "R-dense"
            else "sparse_projection_and_ffn_rows"
            if config.arm == "R-sparse"
            else "none"
        ),
        total_query_rows=total_query_rows,
        total_projection_rows=total_projection_rows,
        total_ffn_rows=total_ffn_rows,
        attention_layout=attention_layout,
        gate_time_s=gate_time_s,
        scorer_time_s=scorer_time_s,
        mask_proposals_rejected=mask_proposals_rejected,
        raw_kl_min=raw_kl_min,
        raw_kl_negative_count=raw_kl_negative_count,
        raw_kl_exact_zero_count=raw_kl_exact_zero_count,
        adaptive_accept_count=adaptive_accept_count,
        fallback_count=fallback_count,
        recording_mode=recording_mode,
    )
