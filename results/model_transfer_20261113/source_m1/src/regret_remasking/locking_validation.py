"""Validation harness for dense and sparse reference-locking backends."""

from __future__ import annotations

from typing import Any

import torch

from regret_remasking.llada_trace import infer_mask_token_id
from regret_remasking.locking_reference import ReferenceRuntime


def _as_tensor_output(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", output)
    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must be a tensor or expose .logits")
    return logits


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _tokenize_prompt(tokenizer: Any, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer([prompt], return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _mask_token_id(tokenizer: Any) -> int:
    try:
        return int(infer_mask_token_id(tokenizer))
    except Exception:
        value = getattr(tokenizer, "mask_token_id", None)
        if value is None:
            raise
        return int(value)


def _forward(model: torch.nn.Module, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    try:
        return _as_tensor_output(model(x, attention_mask=attention_mask))
    except TypeError:
        return _as_tensor_output(model(x))


def _make_inputs(
    tokenizer: Any,
    prompt: str,
    gen_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if gen_length < 3:
        raise ValueError("gen_length must be at least 3 for one locked and two active generated positions")
    prompt_ids, prompt_mask = _tokenize_prompt(tokenizer, prompt, device)
    mask_id = _mask_token_id(tokenizer)
    generated = torch.full((1, gen_length), mask_id, dtype=prompt_ids.dtype, device=device)
    x = torch.cat([prompt_ids, generated], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(generated)], dim=1)
    return x, attention_mask, prompt_ids.shape[1]


def _fill_generated(x: torch.Tensor, prompt_len: int, token_base: int = 11) -> torch.Tensor:
    filled = x.clone()
    for offset in range(x.shape[1] - prompt_len):
        filled[0, prompt_len + offset] = token_base + offset
    return filled


def _validate_positive_finite(name: str, value: float) -> None:
    tensor = torch.tensor(float(value))
    if not bool(torch.isfinite(tensor)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _perturb_observation(
    x: torch.Tensor,
    prompt_len: int,
    step: int,
    locked_positions: set[int],
) -> torch.Tensor:
    perturbed = x.clone()
    for offset in range(1, x.shape[1] - prompt_len):
        pos = prompt_len + offset
        if pos in locked_positions:
            continue
        perturbed[0, pos] = perturbed[0, pos] + step + offset
    return perturbed


def _generated_lock_positions(prompt_len: int, gen_length: int, lock_offset: int, lock_count: int) -> list[int]:
    if lock_offset < 0 or lock_offset >= gen_length:
        raise ValueError("lock_offset must identify a generated position")
    if lock_count < 1:
        raise ValueError("lock_count must be positive")
    if lock_offset + lock_count > gen_length:
        raise ValueError("lock_count must fit within the generated window")
    if gen_length - lock_count < 2:
        raise ValueError("validation requires at least two active generated positions")
    return list(range(prompt_len + lock_offset, prompt_len + lock_offset + lock_count))


def _diff_stats(a: torch.Tensor, b: torch.Tensor, positions: torch.Tensor | None = None) -> dict[str, Any]:
    if positions is not None:
        a = a.index_select(1, positions)
        b = b.index_select(1, positions)
    diff = (a.float() - b.float()).detach()
    denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-12)
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(diff.square())).item()) if diff.numel() else 0.0
    max_rel = float((diff.abs() / denom).max().item()) if diff.numel() else 0.0
    argmax_changes = int((a.argmax(dim=-1) != b.argmax(dim=-1)).sum().item()) if a.ndim >= 3 else 0
    return {"max_abs": max_abs, "rms": rms, "max_rel": max_rel, "argmax_changes": argmax_changes}


def _kv_errors(left: ReferenceRuntime, right: ReferenceRuntime) -> dict[str, Any]:
    per_layer: dict[str, dict[str, float]] = {}
    max_abs = 0.0
    rms_values: list[float] = []
    for layer in sorted(set(left.last_k) & set(right.last_k)):
        k = _diff_stats(left.last_k[layer], right.last_k[layer])
        v = _diff_stats(left.last_v[layer], right.last_v[layer])
        layer_max = max(k["max_abs"], v["max_abs"])
        max_abs = max(max_abs, layer_max)
        rms_values.extend([k["rms"], v["rms"]])
        per_layer[str(layer)] = {
            "k_max_abs": k["max_abs"],
            "k_rms": k["rms"],
            "k_max_rel": k["max_rel"],
            "v_max_abs": v["max_abs"],
            "v_rms": v["rms"],
            "v_max_rel": v["max_rel"],
        }
    return {"max_abs": max_abs, "rms": max(rms_values) if rms_values else 0.0, "per_layer": per_layer}


def _snapshot_constancy(runtime: ReferenceRuntime) -> dict[str, float]:
    max_abs = 0.0
    rms_values: list[float] = []
    locked = runtime.locked[0]
    positions = torch.nonzero(locked, as_tuple=False).flatten()
    for layer in runtime.frozen_inputs:
        checks = [
            runtime.last_inputs[layer][:, positions, :] - runtime.frozen_inputs[layer][:, positions, :],
            runtime.last_outputs[layer][:, positions, :] - runtime.frozen_outputs[layer][:, positions, :],
        ]
        if runtime.mode == "sparse":
            checks.extend(
                [
                    runtime.last_k[layer][:, :, positions, :] - runtime.frozen_k[layer][:, :, positions, :],
                    runtime.last_v[layer][:, :, positions, :] - runtime.frozen_v[layer][:, :, positions, :],
                ]
            )
        for diff in checks:
            diff = diff.float()
            max_abs = max(max_abs, float(diff.abs().max().item()))
            rms_values.append(float(torch.sqrt(torch.mean(diff.square())).item()))
    return {"max_abs": max_abs, "rms": max(rms_values) if rms_values else 0.0}


def _tolerances(
    no_lock: dict[str, Any],
    fresh_reference_control: dict[str, Any],
    multiplier: float,
    min_abs: float,
    min_rel: float,
) -> dict[str, Any]:
    logit_abs_source = max(
        float(no_lock["dense_vs_sparse"]["max_abs"]),
        float(fresh_reference_control["active_logits"]["max_abs"]),
        float(fresh_reference_control["active_generated_logits"]["max_abs"]),
    )
    logit_rel_source = max(
        float(no_lock["dense_vs_sparse"]["max_rel"]),
        float(fresh_reference_control["active_logits"]["max_rel"]),
        float(fresh_reference_control["active_generated_logits"]["max_rel"]),
    )
    kv_abs_source = max(float(no_lock["kv"]["max_abs"]), float(fresh_reference_control["kv"]["max_abs"]))
    logit_abs = max(min_abs, logit_abs_source * multiplier)
    logit_rel = max(min_rel, logit_rel_source * multiplier)
    kv_abs = max(min_abs, kv_abs_source * multiplier)
    return {
        "formula": "max(minimum, max(no_lock_error, fresh_reference_control_error) * multiplier)",
        "multiplier": multiplier,
        "minimum_abs": min_abs,
        "minimum_rel": min_rel,
        "logit_abs_source": logit_abs_source,
        "logit_rel_source": logit_rel_source,
        "kv_abs_source": kv_abs_source,
        "logit_abs": logit_abs,
        "logit_rel": logit_rel,
        "kv_abs": kv_abs,
    }


def _passes(stats: dict[str, Any], tolerances: dict[str, Any]) -> bool:
    return (
        stats["active_logits"]["max_abs"] <= tolerances["logit_abs"]
        and stats["active_logits"]["max_rel"] <= tolerances["logit_rel"]
        and stats["active_generated_logits"]["max_abs"] <= tolerances["logit_abs"]
        and stats["active_generated_logits"]["max_rel"] <= tolerances["logit_rel"]
        and stats["kv"]["max_abs"] <= tolerances["kv_abs"]
        and stats["snapshot_constancy"]["dense"]["max_abs"] <= tolerances["logit_abs"]
        and stats["snapshot_constancy"]["sparse"]["max_abs"] <= tolerances["kv_abs"]
    )


def _strict_logits_pass(stats: dict[str, Any], strict_logit_abs_tolerance: float) -> bool:
    return (
        stats["active_logits"]["max_abs"] <= strict_logit_abs_tolerance
        and stats["active_generated_logits"]["max_abs"] <= strict_logit_abs_tolerance
        and stats["active_logits"]["argmax_changes"] == 0
        and stats["active_generated_logits"]["argmax_changes"] == 0
    )


def _strict_status(
    no_lock: dict[str, Any],
    fresh_reference_control: dict[str, Any],
    freeze_trial: dict[str, Any],
    cleanup: dict[str, Any],
    strict_logit_abs_tolerance: float,
    strict_kv_abs_tolerance: float,
) -> tuple[bool, str]:
    cleanup_ok = cleanup["post_remove_matches_baseline"]["max_abs"] <= strict_logit_abs_tolerance
    backend_equivalent = (
        no_lock["dense_vs_sparse"]["max_abs"] <= strict_logit_abs_tolerance
        and no_lock["dense_vs_sparse"]["argmax_changes"] == 0
        and no_lock["kv"]["max_abs"] <= strict_kv_abs_tolerance
        and _strict_logits_pass(fresh_reference_control, strict_logit_abs_tolerance)
        and fresh_reference_control["kv"]["max_abs"] <= strict_kv_abs_tolerance
        and _strict_logits_pass(freeze_trial, strict_logit_abs_tolerance)
        and freeze_trial["kv"]["max_abs"] <= strict_kv_abs_tolerance
        and cleanup_ok
    )
    if backend_equivalent:
        return True, ""
    reason = (
        "strict equivalence failed: "
        f"fresh max_abs={fresh_reference_control['active_logits']['max_abs']:.6g}, "
        f"fresh argmax={fresh_reference_control['active_logits']['argmax_changes']}, "
        f"fresh generated argmax={fresh_reference_control['active_generated_logits']['argmax_changes']}, "
        f"fresh kv={fresh_reference_control['kv']['max_abs']:.6g}, "
        f"stale max_abs={freeze_trial['active_logits']['max_abs']:.6g}, "
        f"stale argmax={freeze_trial['active_logits']['argmax_changes']}, "
        f"stale generated argmax={freeze_trial['active_generated_logits']['argmax_changes']}, "
        f"stale kv={freeze_trial['kv']['max_abs']:.6g}, "
        f"cleanup max_abs={cleanup['post_remove_matches_baseline']['max_abs']:.6g}"
    )
    return False, reason


@torch.no_grad()
def validate_backends(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    gen_length: int = 8,
    *,
    lock_offset: int = 0,
    lock_count: int = 1,
    tolerance_multiplier: float = 16.0,
    min_abs_tolerance: float = 1e-5,
    min_rel_tolerance: float = 1e-4,
    strict_logit_abs_tolerance: float = 1e-5,
    strict_kv_abs_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare dense and sparse locking backends on fixed synthetic states.

    The function performs no free decoding and no output-token replay.  It uses
    prompt-plus-mask inputs, creates representation locks from a fixed committed
    state, runs a fresh same-state control, then runs two fixed observation
    states that keep locked token IDs unchanged while changing active tokens.
    """

    _validate_positive_finite("strict_logit_abs_tolerance", strict_logit_abs_tolerance)
    _validate_positive_finite("strict_kv_abs_tolerance", strict_kv_abs_tolerance)
    model.eval()
    device = _model_device(model)
    x_masked, attention_mask, prompt_len = _make_inputs(tokenizer, prompt, gen_length, device)
    lock_positions = _generated_lock_positions(prompt_len, gen_length, lock_offset, lock_count)
    x_commit = _fill_generated(x_masked, prompt_len)
    locked = set(lock_positions)
    x_obs1 = _perturb_observation(x_commit, prompt_len, step=1, locked_positions=locked)
    x_obs2 = _perturb_observation(x_commit, prompt_len, step=2, locked_positions=locked)
    active_positions = torch.tensor(
        [pos for pos in range(x_commit.shape[1]) if pos not in locked],
        dtype=torch.long,
        device=device,
    )
    active_generated_positions = torch.tensor(
        [pos for pos in range(prompt_len, x_commit.shape[1]) if pos not in locked],
        dtype=torch.long,
        device=device,
    )

    cleanup: dict[str, Any] = {"dense_removed": False, "sparse_removed": False}
    baseline_logits = _forward(model, x_commit, attention_mask)

    dense = ReferenceRuntime(model, mode="dense", total_len=x_commit.shape[1])
    dense.install()
    try:
        dense_no_lock = _forward(model, x_commit, attention_mask)
        dense_no_lock_kv = {layer: tensor.clone() for layer, tensor in dense.last_k.items()}
        dense_no_lock_v = {layer: tensor.clone() for layer, tensor in dense.last_v.items()}
    finally:
        dense.remove()
        cleanup["dense_removed"] = True

    sparse = ReferenceRuntime(model, mode="sparse", total_len=x_commit.shape[1])
    sparse.install()
    try:
        sparse_no_lock = _forward(model, x_commit, attention_mask)
        sparse_no_lock_kv = {layer: tensor.clone() for layer, tensor in sparse.last_k.items()}
        sparse_no_lock_v = {layer: tensor.clone() for layer, tensor in sparse.last_v.items()}
    finally:
        sparse.remove()
        cleanup["sparse_removed"] = True

    dense = ReferenceRuntime(model, mode="dense", total_len=x_commit.shape[1])
    dense.install()
    cleanup["dense_removed"] = False
    try:
        _forward(model, x_commit, attention_mask)
        dense.commit(lock_positions)
        dense_frozen_inputs = {layer: tensor.clone() for layer, tensor in dense.frozen_inputs.items()}
        dense_fresh_logits = _forward(model, x_commit, attention_mask)
        dense_fresh_snapshot = _snapshot_constancy(dense)
        dense_fresh_k = {layer: tensor.clone() for layer, tensor in dense.last_k.items()}
        dense_fresh_v = {layer: tensor.clone() for layer, tensor in dense.last_v.items()}
        _forward(model, x_obs1, attention_mask)
        dense_locked_logits = _forward(model, x_obs2, attention_mask)
        dense_snapshot = _snapshot_constancy(dense)
        dense_locked_k = {layer: tensor.clone() for layer, tensor in dense.last_k.items()}
        dense_locked_v = {layer: tensor.clone() for layer, tensor in dense.last_v.items()}
        dense_snapshot_changed = max(
            (
                float((dense.frozen_inputs[layer] - dense_frozen_inputs[layer]).abs().max().item())
                for layer in dense_frozen_inputs
            ),
            default=0.0,
        )
    finally:
        dense.remove()
        cleanup["dense_removed"] = True

    sparse = ReferenceRuntime(model, mode="sparse", total_len=x_commit.shape[1])
    sparse.install()
    cleanup["sparse_removed"] = False
    try:
        _forward(model, x_commit, attention_mask)
        sparse.commit(lock_positions)
        sparse_frozen_k = {layer: tensor.clone() for layer, tensor in sparse.frozen_k.items()}
        sparse_fresh_logits = _forward(model, x_commit, attention_mask)
        sparse_fresh_snapshot = _snapshot_constancy(sparse)
        sparse_fresh_k = {layer: tensor.clone() for layer, tensor in sparse.last_k.items()}
        sparse_fresh_v = {layer: tensor.clone() for layer, tensor in sparse.last_v.items()}
        _forward(model, x_obs1, attention_mask)
        sparse_locked_logits = _forward(model, x_obs2, attention_mask)
        sparse_snapshot = _snapshot_constancy(sparse)
        sparse_locked_k = {layer: tensor.clone() for layer, tensor in sparse.last_k.items()}
        sparse_locked_v = {layer: tensor.clone() for layer, tensor in sparse.last_v.items()}
        sparse_snapshot_changed = max(
            (
                float((sparse.frozen_k[layer] - sparse_frozen_k[layer]).abs().max().item())
                for layer in sparse_frozen_k
            ),
            default=0.0,
        )
    finally:
        sparse.remove()
        cleanup["sparse_removed"] = True

    no_lock = {
        "baseline_vs_dense": _diff_stats(baseline_logits, dense_no_lock),
        "baseline_vs_sparse": _diff_stats(baseline_logits, sparse_no_lock),
        "dense_vs_sparse": _diff_stats(dense_no_lock, sparse_no_lock),
        "kv": _kv_errors(
            type("NoLockDense", (), {"last_k": dense_no_lock_kv, "last_v": dense_no_lock_v})(),
            type("NoLockSparse", (), {"last_k": sparse_no_lock_kv, "last_v": sparse_no_lock_v})(),
        ),
    }
    fresh_reference_control = {
        "active_logits": _diff_stats(dense_fresh_logits, sparse_fresh_logits, active_positions),
        "active_generated_logits": _diff_stats(dense_fresh_logits, sparse_fresh_logits, active_generated_positions),
        "kv": _kv_errors(
            type("FreshDense", (), {"last_k": dense_fresh_k, "last_v": dense_fresh_v})(),
            type("FreshSparse", (), {"last_k": sparse_fresh_k, "last_v": sparse_fresh_v})(),
        ),
        "snapshot_constancy": {"dense": dense_fresh_snapshot, "sparse": sparse_fresh_snapshot},
    }
    tolerances = _tolerances(
        no_lock,
        fresh_reference_control,
        tolerance_multiplier,
        min_abs_tolerance,
        min_rel_tolerance,
    )
    active_logits = _diff_stats(dense_locked_logits, sparse_locked_logits, active_positions)
    active_generated_logits = _diff_stats(dense_locked_logits, sparse_locked_logits, active_generated_positions)
    kv = _kv_errors(
        type("LockedDense", (), {"last_k": dense_locked_k, "last_v": dense_locked_v})(),
        type("LockedSparse", (), {"last_k": sparse_locked_k, "last_v": sparse_locked_v})(),
    )
    freeze_trial = {
        "active_logits": active_logits,
        "active_generated_logits": active_generated_logits,
        "kv": kv,
        "snapshot_constancy": {"dense": dense_snapshot, "sparse": sparse_snapshot},
        "frozen_snapshot_mutation_max_abs": {
            "dense_inputs": dense_snapshot_changed,
            "sparse_k": sparse_snapshot_changed,
        },
    }
    within_measured_envelope = _passes(fresh_reference_control, tolerances) and _passes(freeze_trial, tolerances)
    cleanup_logits = _forward(model, x_commit, attention_mask)
    cleanup["post_remove_matches_baseline"] = _diff_stats(baseline_logits, cleanup_logits)
    backend_equivalent, review_reason = _strict_status(
        no_lock,
        fresh_reference_control,
        freeze_trial,
        cleanup,
        strict_logit_abs_tolerance,
        strict_kv_abs_tolerance,
    )
    return {
        "schema_version": 1,
        "prompt": prompt,
        "gen_length": gen_length,
        "prompt_length": prompt_len,
        "lock_position": lock_positions[0],
        "lock_positions": lock_positions,
        "lock_offset": lock_offset,
        "lock_count": lock_count,
        "active_generated_count": int(active_generated_positions.numel()),
        "observations_after_commit": 2,
        "no_output_token_replay": True,
        "no_lock": no_lock,
        "fresh_reference_control": fresh_reference_control,
        "tolerances": tolerances,
        "strict_tolerances": {
            "logit_abs": strict_logit_abs_tolerance,
            "kv_abs": strict_kv_abs_tolerance,
            "argmax_changes": 0,
            "interpretation": "strict backend-equivalence gate; measured-envelope diagnostics do not relax this gate",
        },
        "freeze_trial": freeze_trial,
        "cleanup": cleanup,
        "within_measured_envelope": bool(within_measured_envelope),
        "completed": True,
        "backend_equivalent": bool(backend_equivalent),
        "requires_review": not bool(backend_equivalent),
        "review_reason": review_reason,
        "passed": bool(backend_equivalent),
    }


@torch.no_grad()
def validate_perturbation_control(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    gen_length: int = 8,
    *,
    lock_offset: int = 0,
    lock_count: int = 1,
    perturbation: float = 10.0,
) -> dict[str, Any]:
    """Toy-friendly positive control: perturb frozen V and check downstream logits."""

    model.eval()
    device = _model_device(model)
    x_masked, attention_mask, prompt_len = _make_inputs(tokenizer, prompt, gen_length, device)
    x_commit = _fill_generated(x_masked, prompt_len)
    lock_positions = _generated_lock_positions(prompt_len, gen_length, lock_offset, lock_count)
    locked = set(lock_positions)
    sparse = ReferenceRuntime(model, mode="sparse", total_len=x_commit.shape[1])
    sparse.install()
    try:
        _forward(model, x_commit, attention_mask)
        sparse.commit(lock_positions)
        normal = _forward(model, x_commit, attention_mask)
        for layer in sparse.frozen_v:
            sparse.frozen_v[layer][:, :, lock_positions, :] += perturbation
        changed = _forward(model, x_commit, attention_mask)
        active_positions = torch.tensor(
            [pos for pos in range(x_commit.shape[1]) if pos not in locked],
            dtype=torch.long,
            device=device,
        )
        stats = _diff_stats(normal, changed, active_positions)
        return {"changed": stats["max_abs"] > 0.0, "active_logits": stats}
    finally:
        sparse.remove()
