"""Measured runner for the paper locking/commitment comparison."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from regret_remasking.data import Example, build_prompt, score_generation_with_status
from regret_remasking.llada_trace import infer_mask_token_id
from regret_remasking.locking_experiment import load_local_llada
from regret_remasking.locking_manifest import prompt_content_hash
from regret_remasking.paper_decode import PAPER_ARMS, TRACE_ARMS, TOKEN_KL_ARMS, PaperDecodeConfig, decode_paper_policy
from regret_remasking.paper_manifests import prompt_hash as paper_prompt_hash

POLICY_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
REPRESENTATION_KL_LAYER = "final_normalized_pre_vocab"
DEFAULT_ARMS = (
    "B0",
    "B-entropy",
    "S-no-lock",
    "S-dense",
    "S-sparse",
    "S-window",
    "W-confidence",
    "W-entropy",
    "T-fixed",
    "T-adaptive",
    "C-adaptive",
    "K-adaptive",
)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(root: Path) -> dict[str, Any]:
    files = sorted(p for folder in ("src", "scripts") for p in (root / folder).rglob("*.py"))
    hashes = {str(path.relative_to(root)): file_hash(path) for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        revision = None
    return {
        "git_revision": revision,
        "files": hashes,
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def runtime_environment(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "accelerate", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "host": platform.node(),
        "packages": versions,
        "device": device,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
    }


def checkpoint_provenance(checkpoint: str | Path) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    result: dict[str, Any] = {"path": str(root)}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        result["checkpoint_index_sha256"] = file_hash(index)
    else:
        result["checkpoint_index_sha256"] = None
    result["model_source_sha256"] = {
        name: file_hash(root / name)
        for name in ("config.json", "configuration_llada.py", "modeling_llada.py")
        if (root / name).is_file()
    }
    return result


def _write_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["dataset"]),
        str(record["example_id"]),
        str(record["policy_id"]),
        int(record["repeat"]),
    )


def _unique_attempt_dir(output: Path) -> None:
    config_path = output / "resolved_config.json"
    records_path = output / "records.jsonl"
    if not config_path.exists() and (records_path.exists() or (output / "completion.json").exists()):
        raise FileExistsError("orphaned paper-run artifacts require a new output directory")


def _resolve_prompt(entry: Mapping[str, Any]) -> str:
    prompt = entry.get("prompt_text")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("manifest prompt_text must be a nonempty string when present")
        return prompt
    return build_prompt(str(entry["question"]), str(entry["dataset"]))


def _example_from_entry(entry: Mapping[str, Any], *, dataset: str | None = None) -> Example:
    return Example(
        example_id=str(entry["example_id"]),
        question=str(entry["question"]),
        gold_answer=str(entry["gold_answer"]),
        dataset=str(dataset or entry["dataset"]),
        metadata=dict(entry["metadata"]),
    )


def score_paper_generation(generation: str, entry: Mapping[str, Any]) -> tuple[bool | None, str]:
    dataset = str(entry["dataset"]).lower()
    if dataset in {"qa", "qa_collection", "alpaca", "instruction"}:
        return None, "metadata_only"
    example = _example_from_entry(entry)
    if dataset in {"kodcode", "code"}:
        metadata = dict(entry.get("metadata", {}))
        if "test" not in metadata or "entry_point" not in metadata:
            return None, "missing_code_tests"
        mapped = Example(example.example_id, example.question, example.gold_answer, "humaneval", metadata)
        return score_generation_with_status(generation, mapped)
    return score_generation_with_status(generation, example)


def _require_nonempty_text(record: Mapping[str, Any], field: str, path: Path, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number} missing nonempty {field}")
    return value


def _normalize_manifest_record(path: Path, line_number: int, record: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "record_id",
        "source_id",
        "dataset",
        "example_id",
        "question",
        "gold_answer",
        "metadata",
        "split",
        "seed",
        "source_path",
        "source_revision",
    ):
        if field not in record:
            raise ValueError(f"{path}:{line_number} manifest record missing required field {field!r}")
    if not isinstance(record["metadata"], dict):
        raise ValueError(f"{path}:{line_number} metadata must be an object")
    normalized = dict(record)
    dataset = _require_nonempty_text(normalized, "dataset", path, line_number)
    question = _require_nonempty_text(normalized, "question", path, line_number)
    split = _require_nonempty_text(normalized, "split", path, line_number)
    _require_nonempty_text(normalized, "source_id", path, line_number)
    source_path_text = _require_nonempty_text(normalized, "source_path", path, line_number)
    _require_nonempty_text(normalized, "source_revision", path, line_number)
    if "prompt_text" in normalized:
        prompt = _require_nonempty_text(normalized, "prompt_text", path, line_number)
        if "prompt_hash" not in normalized:
            raise ValueError(f"{path}:{line_number} paper manifest missing prompt_hash")
        if normalized["prompt_hash"] != paper_prompt_hash(prompt):
            raise ValueError(f"{path}:{line_number} prompt_hash mismatch")
    else:
        expected = prompt_content_hash(question, dataset)
        if normalized.get("content_hash") != expected:
            raise ValueError(f"{path}:{line_number} content_hash mismatch")
        normalized["prompt_text"] = build_prompt(question, dataset)
        normalized["prompt_hash"] = expected
    content_hash = normalized.get("chat_prompt_hash") or normalized.get("prompt_hash") or normalized.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError(f"{path}:{line_number} manifest record has no prompt/content hash")
    normalized["content_hash"] = content_hash
    normalized.setdefault("family", normalized.get("dataset_name", dataset))
    normalized.setdefault("dataset_name", dataset)
    normalized.setdefault("source_sha256", None)
    if normalized["source_sha256"] is not None:
        if not isinstance(normalized["source_sha256"], str) or not normalized["source_sha256"]:
            raise ValueError(f"{path}:{line_number} source_sha256 must be nonempty when present")
        source_path = Path(source_path_text)
        if source_path.exists() and file_hash(source_path) != normalized["source_sha256"]:
            raise ValueError(f"{path}:{line_number} source_sha256 mismatch")
    if not split.strip():
        raise ValueError(f"{path}:{line_number} split must be nonempty")
    return normalized


def load_paper_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = _normalize_manifest_record(manifest_path, line_number, json.loads(line))
            record_id = str(record["record_id"])
            source_id = str(record["source_id"])
            if record_id in seen_record_ids:
                raise ValueError(f"{manifest_path}:{line_number} duplicate record_id {record_id!r}")
            if source_id in seen_source_ids:
                raise ValueError(f"{manifest_path}:{line_number} duplicate source_id {source_id!r}")
            seen_record_ids.add(record_id)
            seen_source_ids.add(source_id)
            records.append(record)
    return records


def _finite_probability(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite probability")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def make_paper_policies(
    arms: Sequence[str] = DEFAULT_ARMS,
    epsilons: Sequence[float] = (0.0, 0.0005, 0.005),
    thresholds: Sequence[float] = (0.95,),
    *,
    gen_length: int = 256,
    block_length: int = 32,
    window_length: int = 32,
    freeze_scope: str = "generated",
    confidence_percentile: float = 100.0,
    kl_space: str = "token",
    representation_temperature: float = 1.0,
    controller_checkpoint: str | Path | None = None,
    projector_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    if not arms or len(set(arms)) != len(arms) or any(arm not in PAPER_ARMS for arm in arms):
        raise ValueError(f"arms must be a nonempty unique subset of {sorted(PAPER_ARMS)}")
    if gen_length < 1 or block_length < 1 or window_length < 1 or gen_length % block_length:
        raise ValueError("positive generation/window lengths and divisible block length are required")
    if freeze_scope not in {"generated", "all_unmasked"}:
        raise ValueError("freeze_scope must be generated or all_unmasked")
    if kl_space not in {"token", "representation"}:
        raise ValueError("kl_space must be token or representation")
    representation_temperature = _finite_positive(representation_temperature, "representation_temperature")
    confidence_percentile = _finite_probability(confidence_percentile / 100.0, "confidence_percentile/100") * 100.0
    epsilon_values = [_finite_nonnegative(value, "epsilon") for value in epsilons]
    threshold_values = [_finite_probability(value, "threshold") for value in thresholds]
    if len(set(epsilon_values)) != len(epsilon_values) or len(set(threshold_values)) != len(threshold_values):
        raise ValueError("duplicate operating points")
    if not epsilon_values or not threshold_values:
        raise ValueError("epsilons and thresholds must be nonempty")
    learned_path = Path(controller_checkpoint).resolve() if controller_checkpoint else None
    projector_path = Path(projector_checkpoint).resolve() if projector_checkpoint else None
    if any(arm in TRACE_ARMS for arm in arms) and (learned_path is None or projector_path is None):
        raise ValueError("T-fixed/T-adaptive require controller and projector checkpoints")

    policies: list[dict[str, Any]] = []
    for arm in arms:
        values: list[float | None]
        if arm in {"S-sparse", "S-window", "K-adaptive"}:
            values = list(epsilon_values)
        elif arm == "S-dense":
            values = [0.005 if 0.005 in epsilon_values else epsilon_values[-1]]
        elif arm in {"T-adaptive", "C-adaptive", "C-block"}:
            values = list(threshold_values)
        else:
            values = [None]
        for value in values:
            policy: dict[str, Any] = {
                "schema_version": POLICY_SCHEMA_VERSION,
                "arm": arm,
                "epsilon": None,
                "threshold": None,
                "gen_length": int(gen_length),
                "block_length": int(block_length),
                "window_length": int(window_length),
                "freeze_scope": freeze_scope,
                "confidence_percentile": float(confidence_percentile),
                "allow_newly_unmasked": False,
                "kl_space": kl_space,
                "representation_temperature": float(representation_temperature),
                "representation_layer": REPRESENTATION_KL_LAYER if kl_space == "representation" and arm.startswith("S-") else None,
                "controller_checkpoint": None,
                "controller_checkpoint_sha256": None,
                "projector_checkpoint": None,
                "projector_checkpoint_sha256": None,
            }
            policy_id = arm
            if arm in TOKEN_KL_ARMS:
                if kl_space != "token":
                    raise ValueError("K-adaptive only supports token kl_space")
                policy["confidence_percentile"] = 100.0
                policy["representation_layer"] = None
            if arm.startswith("S-") and value is not None:
                policy["epsilon"] = float(value)
                if arm in {"S-sparse", "S-window"}:
                    policy_id += f"-eps-{float(value):g}"
                    if arm == "S-window" and confidence_percentile != 100.0:
                        policy_id += f"-cp-{float(confidence_percentile):g}"
            if arm in TOKEN_KL_ARMS and value is not None:
                policy["epsilon"] = float(value)
                policy["threshold"] = None
                policy_id += f"-eps-{float(value):g}"
            if arm.startswith("S-") and kl_space == "representation":
                policy_id += "-rep-kl"
            if arm in {"T-adaptive", "C-adaptive", "C-block"} and value is not None:
                policy["threshold"] = float(value)
                policy_id += f"-tau-{float(value):g}"
            elif arm == "T-fixed":
                policy["threshold"] = threshold_values[0]
            if arm in TRACE_ARMS:
                assert learned_path is not None and projector_path is not None
                policy["controller_checkpoint"] = str(learned_path)
                policy["controller_checkpoint_sha256"] = file_hash(learned_path)
                policy["projector_checkpoint"] = str(projector_path)
                policy["projector_checkpoint_sha256"] = file_hash(projector_path)
            policy["policy_id"] = policy_id
            policies.append(policy)
    if len({p["policy_id"] for p in policies}) != len(policies):
        raise ValueError("policy IDs must be unique")
    return policies


def _validate_policy(policy: Mapping[str, Any], *, gen_length: int) -> None:
    required = {
        "policy_id",
        "arm",
        "epsilon",
        "threshold",
        "window_length",
        "block_length",
        "freeze_scope",
        "confidence_percentile",
        "allow_newly_unmasked",
        "kl_space",
        "representation_temperature",
        "representation_layer",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"policy missing required fields: {missing}")
    if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
        raise ValueError("policy_id must be a nonempty string")
    arm = str(policy["arm"])
    if arm not in PAPER_ARMS:
        raise ValueError(f"unknown policy arm: {arm}")
    if policy.get("gen_length", gen_length) != gen_length:
        raise ValueError("policy gen_length differs from runner gen_length")
    if int(policy["block_length"]) < 1 or int(policy["window_length"]) < 1:
        raise ValueError("policy block/window lengths must be positive")
    if policy["freeze_scope"] not in {"generated", "all_unmasked"}:
        raise ValueError("policy freeze_scope must be generated or all_unmasked")
    if not isinstance(policy["allow_newly_unmasked"], bool):
        raise ValueError("policy allow_newly_unmasked must be a boolean")
    if policy["kl_space"] not in {"token", "representation"}:
        raise ValueError("policy kl_space must be token or representation")
    if arm in TOKEN_KL_ARMS and policy["kl_space"] != "token":
        raise ValueError("K-adaptive only supports token kl_space")
    _finite_positive(policy["representation_temperature"], "representation_temperature")
    expected_layer = REPRESENTATION_KL_LAYER if policy["kl_space"] == "representation" and arm.startswith("S-") else None
    if policy.get("representation_layer") != expected_layer:
        raise ValueError("policy representation_layer does not match kl_space")
    _finite_probability(float(policy["confidence_percentile"]) / 100.0, "confidence_percentile/100")
    if arm.startswith("S-") or arm in TOKEN_KL_ARMS:
        _finite_nonnegative(policy.get("epsilon", 0.0) or 0.0, "epsilon")
    if arm in {"T-adaptive", "C-adaptive", "C-block"}:
        _finite_probability(policy.get("threshold"), "threshold")
    if arm in TRACE_ARMS:
        for key in ("controller_checkpoint", "projector_checkpoint"):
            if not policy.get(key):
                raise ValueError(f"{arm} requires {key}")
            expected = policy.get(f"{key}_sha256")
            if expected and file_hash(policy[key]) != expected:
                raise ValueError(f"{key} hash differs from frozen policy")
        _validate_controller_projector_pair(policy)


def _validate_controller_projector_pair(policy: Mapping[str, Any]) -> None:
    projector_sha256 = file_hash(policy["projector_checkpoint"])
    payload = torch.load(policy["controller_checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("controller checkpoint payload must be a mapping")
    declared = payload.get("projection_sha256")
    if declared != projector_sha256:
        raise ValueError("controller checkpoint projection_sha256 does not match projector checkpoint")


def _load_learned_artifacts(
    policies: Sequence[Mapping[str, Any]],
    *,
    device: str,
    controller_loader: Callable[[str | Path, str], Any] | None,
    projector_loader: Callable[[str | Path, str], Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if controller_loader is None:
        from regret_remasking.paper_trace import PaperTraceController

        controller_loader = lambda path, dev: PaperTraceController.load(path, device=dev)
    if projector_loader is None:
        from regret_remasking.paper_trace import HiddenProjector

        projector_loader = lambda path, dev: HiddenProjector.load(path, device=dev)
    controllers: dict[str, Any] = {}
    projectors: dict[str, Any] = {}
    for policy in policies:
        if policy["arm"] not in TRACE_ARMS:
            continue
        controller_path = str(Path(str(policy["controller_checkpoint"])).resolve())
        projector_path = str(Path(str(policy["projector_checkpoint"])).resolve())
        if controller_path not in controllers:
            controllers[controller_path] = controller_loader(controller_path, device)
        if projector_path not in projectors:
            projectors[projector_path] = projector_loader(projector_path, device)
    return controllers, projectors


def _policy_config(policy: Mapping[str, Any], mask_id: int, default_gen_length: int) -> PaperDecodeConfig:
    threshold = policy.get("threshold")
    return PaperDecodeConfig(
        arm=str(policy["arm"]),
        gen_length=int(policy.get("gen_length", default_gen_length)),
        block_length=int(policy["block_length"]),
        window_length=int(policy["window_length"]),
        epsilon=float(policy.get("epsilon") or 0.0),
        confidence_percentile=float(policy["confidence_percentile"]),
        threshold=float(threshold if threshold is not None else 0.95),
        freeze_scope=str(policy["freeze_scope"]),
        allow_newly_unmasked=bool(policy["allow_newly_unmasked"]),
        kl_space=str(policy["kl_space"]),
        representation_temperature=float(policy["representation_temperature"]),
        mask_id=int(mask_id),
    )


def run_paper_experiment(
    manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    policies: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str | Path,
    gen_length: int = 256,
    repeats: int = 3,
    device: str = "cuda",
    dtype: str = "bf16",
    seed: int = 23,
    limit: int | None = None,
    collect_events: bool = False,
    warmup: bool = True,
    resume: bool = False,
    purpose: str = "evaluation",
    model: Any = None,
    tokenizer: Any = None,
    decoder: Callable[..., Any] | None = None,
    controller_loader: Callable[[str | Path, str], Any] | None = None,
    projector_loader: Callable[[str | Path, str], Any] | None = None,
) -> dict[str, Any]:
    if repeats < 1 or gen_length < 1:
        raise ValueError("positive repeats and generation length are required")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    mutable_policies = [dict(policy) for policy in policies]
    for normalized_policy in mutable_policies:
        normalized_policy.setdefault("allow_newly_unmasked", False)
        normalized_policy.setdefault("kl_space", "token")
        normalized_policy.setdefault("representation_temperature", 1.0)
        normalized_policy.setdefault("representation_layer", REPRESENTATION_KL_LAYER if normalized_policy["kl_space"] == "representation" and str(normalized_policy.get("arm", "")).startswith("S-") else None)
    policies = mutable_policies
    if not policies or len({p.get("policy_id") for p in policies}) != len(policies):
        raise ValueError("policies must be nonempty with unique policy IDs")
    for policy in policies:
        _validate_policy(policy, gen_length=gen_length)
    manifests = [Path(path).resolve() for path in manifest_paths]
    entries = [entry for path in manifests for entry in load_paper_manifest(path)]
    if limit is not None:
        entries = entries[:limit]
    if not entries or len({(entry["dataset"], entry["example_id"]) for entry in entries}) != len(entries):
        raise ValueError("empty or duplicate example manifest")
    if len({_resolve_prompt(entry) for entry in entries}) != len(entries):
        raise ValueError("duplicate prompt text in run")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _unique_attempt_dir(output)
    source_root = Path(__file__).resolve().parents[2]
    checkpoint = Path(checkpoint).resolve()
    config: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "examples": [
            {
                "example_id": entry["example_id"],
                "dataset": entry["dataset"],
                "split": entry["split"],
                "source_id": entry["source_id"],
                "content_hash": entry["content_hash"],
                "prompt_hash": entry.get("prompt_hash"),
                "chat_prompt_hash": entry.get("chat_prompt_hash"),
            }
            for entry in entries
        ],
        "manifest_sha256": {str(path): file_hash(path) for path in manifests},
        "policies": policies,
        "repeats": int(repeats),
        "gen_length": int(gen_length),
        "device": device,
        "dtype": dtype,
        "seed": int(seed),
        "checkpoint": checkpoint_provenance(checkpoint),
        "source": source_provenance(source_root),
        "environment": runtime_environment(device),
        "collect_events": bool(collect_events),
        "warmup": bool(warmup),
        "timing": "timed decode only; no observer; excludes model load, scoring and artifact writes",
        "quality_repeat": 1,
        "purpose": purpose,
    }
    config_path = output / "resolved_config.json"
    records_path = output / "records.jsonl"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"run already exists; use --resume with unchanged inputs: {output}")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume configuration/provenance differs from saved run")
    else:
        if records_path.exists() or (output / "events.jsonl").exists():
            raise FileExistsError("orphaned paper-run artifacts require a new output directory")
        _write_json_atomic(config_path, config)

    expected = {
        (entry["dataset"], entry["example_id"], policy["policy_id"], repeat)
        for entry in entries
        for policy in policies
        for repeat in range(1, repeats + 1)
    }
    complete: set[tuple[str, str, str, int]] = set()
    if records_path.exists():
        for row in (json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()):
            key = _record_key(row)
            if key not in expected or key in complete or row.get("status") != "ok":
                raise ValueError("cannot resume duplicate, failed, or unexpected paper records")
            if row.get("final_mask_count") != 0 or row.get("immutable_token_violations") != 0:
                raise ValueError("cannot resume invalid terminal records")
            if len(row.get("token_ids", [])) != gen_length or row.get("nfe") != row.get("forward_calls"):
                raise ValueError("cannot resume inconsistent completion records")
            complete.add(key)
    if len(complete) == len(expected):
        summary = {"complete": True, "records": len(complete), "output_dir": str(output)}
        _write_json_atomic(output / "completion.json", summary)
        return summary

    if model is None or tokenizer is None:
        model, tokenizer = load_local_llada(checkpoint, device, dtype)
    mask_id = infer_mask_token_id(tokenizer)
    controllers, projectors = _load_learned_artifacts(
        policies,
        device=device,
        controller_loader=controller_loader,
        projector_loader=projector_loader,
    )
    decoder = decoder or decode_paper_policy

    def decode(entry: Mapping[str, Any], policy: Mapping[str, Any], events: bool) -> Any:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = _policy_config(policy, mask_id, gen_length)
        controller = projector = None
        if policy["arm"] in TRACE_ARMS:
            controller = controllers[str(Path(str(policy["controller_checkpoint"])).resolve())]
            projector = projectors[str(Path(str(policy["projector_checkpoint"])).resolve())]
        return decoder(
            model,
            tokenizer,
            _resolve_prompt(entry),
            str(entry["example_id"]),
            cfg,
            controller=controller,
            projector=projector,
            collect_events=events,
            step_observer=None,
        )

    if warmup:
        for policy in policies:
            print(f"[paper] warmup {policy['policy_id']}", flush=True)
            decode(entries[0], policy, False)
    for index, entry in enumerate(entries):
        for repeat in range(1, repeats + 1):
            rotation = (index + repeat - 1) % len(policies)
            ordered = policies[rotation:] + policies[:rotation]
            for policy in ordered:
                record = {
                    "dataset": entry["dataset"],
                    "example_id": entry["example_id"],
                    "policy_id": policy["policy_id"],
                    "arm": policy["arm"],
                    "repeat": repeat,
                    "epsilon": policy.get("epsilon"),
                    "threshold": policy.get("threshold"),
                    "window_length": policy.get("window_length"),
                    "block_length": policy.get("block_length"),
                    "freeze_scope": policy.get("freeze_scope"),
                    "confidence_percentile": policy.get("confidence_percentile"),
                    "allow_newly_unmasked": policy.get("allow_newly_unmasked"),
                    "kl_space": policy.get("kl_space"),
                    "representation_temperature": policy.get("representation_temperature"),
                    "representation_layer": None,
                    "representation_feature_dim": None,
                    "content_hash": entry["content_hash"],
                    "gen_length": gen_length,
                    "source_sha256": config["source"]["sha256"],
                    "config_sha256": file_hash(config_path),
                    "checkpoint_index_sha256": config["checkpoint"]["checkpoint_index_sha256"],
                    "controller_checkpoint_sha256": policy.get("controller_checkpoint_sha256"),
                    "projector_checkpoint_sha256": policy.get("projector_checkpoint_sha256"),
                }
                if _record_key(record) in complete:
                    continue
                print(
                    f"[paper] {index + 1}/{len(entries)} {entry['example_id']} "
                    f"{policy['policy_id']} repeat={repeat}",
                    flush=True,
                )
                try:
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    result = decode(entry, policy, collect_events)
                    if hasattr(result, "kl_space") and getattr(result, "kl_space") != policy["kl_space"]:
                        raise ValueError("decoder result kl_space differs from declared policy")
                    if hasattr(result, "representation_temperature") and float(getattr(result, "representation_temperature")) != float(policy["representation_temperature"]):
                        raise ValueError("decoder result representation_temperature differs from declared policy")
                    for name in (
                        "nfe",
                        "forward_calls",
                        "latency_s",
                        "prompt_length",
                        "final_mask_count",
                        "immutable_token_violations",
                        "frozen_count",
                        "frozen_prompt_count",
                        "generated_row_updates_avoided",
                        "total_query_rows",
                        "total_projection_rows",
                        "total_ffn_rows",
                        "total_output_head_rows",
                        "gate_time_s",
                        "scorer_time_s",
                        "controller_calls",
                    ):
                        if hasattr(result, name):
                            record[name] = getattr(result, name)
                    for name in (
                        "raw_kl_min",
                        "raw_kl_negative_count",
                        "raw_kl_exact_zero_count",
                        "adaptive_accept_count",
                        "fallback_count",
                        "mask_proposals_rejected",
                        "generated_row_updates_avoided_basis",
                        "attention_layout",
                        "recording_mode",
                        "policy_subtimers",
                        "row_count_basis",
                        "representation_layer",
                        "representation_feature_dim",
                    ):
                        if hasattr(result, name):
                            record[name] = getattr(result, name)
                    record.update(token_ids=list(result.token_ids), generation=result.text)
                    if (
                        len(record["token_ids"]) != gen_length
                        or mask_id in record["token_ids"]
                        or record.get("final_mask_count") != 0
                        or record.get("immutable_token_violations") != 0
                        or record.get("nfe") != record.get("forward_calls")
                        or not 0 < int(record["nfe"]) <= gen_length
                        or not math.isfinite(float(record["latency_s"]))
                        or float(record["latency_s"]) <= 0
                    ):
                        raise ValueError("decoder violated completion/forward/timing invariants")
                    correct, status = score_paper_generation(result.text, entry)
                    record.update(correct=correct, scoring_status=status, status="ok")
                    record["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
                    if collect_events:
                        _append_record(
                            output / "events.jsonl",
                            {
                                "dataset": entry["dataset"],
                                "example_id": entry["example_id"],
                                "policy_id": policy["policy_id"],
                                "repeat": repeat,
                                "events": result.events,
                            },
                        )
                    _append_record(records_path, record)
                    complete.add(_record_key(record))
                except Exception as error:
                    record.update(status="error", error=f"{type(error).__name__}: {error}")
                    _append_record(records_path, record)
                    raise

    summary = {
        "complete": True,
        "records": len(complete),
        "examples": len(entries),
        "policies": len(policies),
        "repeats": repeats,
        "output_dir": str(output),
    }
    if len(complete) != len(expected):
        raise AssertionError("paper runner finished without all expected records")
    _write_json_atomic(output / "completion.json", summary)
    return summary


def save_policy_config(path: str | Path, policies: Sequence[Mapping[str, Any]]) -> None:
    _write_json_atomic(Path(path), [dict(policy) for policy in policies])


__all__ = [
    "DEFAULT_ARMS",
    "make_paper_policies",
    "run_paper_experiment",
    "save_policy_config",
    "score_paper_generation",
]
