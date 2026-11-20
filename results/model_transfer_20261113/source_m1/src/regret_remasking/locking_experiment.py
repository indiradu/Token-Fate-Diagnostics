"""Reproducible orchestration for the two separate locking experiments."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch

from regret_remasking import FEATURE_NAMES
from regret_remasking.data import build_prompt, score_generation_with_status
from regret_remasking.llada_trace import ensure_transformers_tied_weight_compat, infer_mask_token_id


ARMS = ("B0", "R-dense", "R-sparse", "T-fixed", "C-adaptive", "T-adaptive")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(root: Path) -> dict[str, Any]:
    files = sorted(p for folder in ("src", "scripts") for p in (root / folder).rglob("*.py"))
    hashes = {str(p.relative_to(root)): file_hash(p) for p in files}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_revision": revision, "files": hashes,
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def make_policies(
    arms: list[str], epsilons: list[float], thresholds: list[float],
    trace_checkpoint: str | None = None, confidence_checkpoint: str | None = None,
) -> list[dict[str, Any]]:
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError(f"arms must be a nonempty unique subset of {ARMS}")
    if any(not math.isfinite(x) or x < 0 for x in epsilons) or not epsilons:
        raise ValueError("epsilons must be finite and nonnegative")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in thresholds) or not thresholds:
        raise ValueError("thresholds must be finite probabilities")
    if len(set(epsilons)) != len(epsilons) or len(set(thresholds)) != len(thresholds):
        raise ValueError("duplicate operating points")
    policies = []
    for arm in arms:
        checkpoint = confidence_checkpoint if arm == "C-adaptive" else trace_checkpoint
        if arm in {"T-fixed", "T-adaptive", "C-adaptive"} and not checkpoint:
            raise ValueError(f"{arm} requires its calibrated checkpoint")
        values: list[Any] = [None]
        if arm.startswith("R-"):
            values = list(epsilons)
        elif arm.endswith("adaptive"):
            values = list(thresholds)
        for value in values:
            policy: dict[str, Any] = {"arm": arm, "epsilon": None, "threshold": None}
            policy_id = arm
            if arm.startswith("R-"):
                policy["epsilon"] = value
                policy_id += f"-eps-{value:g}"
            elif arm.endswith("adaptive"):
                policy["threshold"] = value
                policy_id += f"-tau-{value:g}"
            if arm in {"T-fixed", "T-adaptive", "C-adaptive"}:
                path = Path(str(checkpoint)).resolve()
                policy.update(checkpoint=str(path), checkpoint_sha256=file_hash(path))
            policy["policy_id"] = policy_id
            policies.append(policy)
    return policies


def load_local_llada(checkpoint: str | Path, device: str, dtype: str) -> tuple[Any, Any]:
    """Dispatch an existing local checkpoint from meta tensors without CPU duplication."""
    # Transformers' local-source hashing follows Hub symlinks into blobs/, where
    # sibling relative imports do not live. Materialize only the small source
    # files; checkpoint tensors remain in the original immutable snapshot.
    checkpoint = Path(checkpoint).resolve()
    module_cache = Path(os.environ.setdefault(
        "HF_MODULES_CACHE", str(Path(__file__).resolve().parents[2] / ".omx" / "model_modules")))
    source_names = ("config.json", "configuration_llada.py", "modeling_llada.py")
    source_key = hashlib.sha256("".join(file_hash(checkpoint / name) for name in source_names).encode()).hexdigest()[:16]
    module_source = module_cache / f"llada_source_{source_key}"
    module_source.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        destination = module_source / name
        content = (checkpoint / name).read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"materialized model source was modified: {destination}")
        if not destination.exists():
            destination.write_bytes(content)
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"expected an existing sharded LLaDA checkpoint: {checkpoint}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype}")
    ensure_transformers_tied_weight_compat()
    config = AutoConfig.from_pretrained(str(module_source), trust_remote_code=True, local_files_only=True)
    config.use_cache = False
    config.return_dict = True
    model_class = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM", str(module_source), local_files_only=True,
    )
    with init_empty_weights():
        model = model_class(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(checkpoint), device_map={"": device}, dtype=dtype_map[dtype],
    )
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("checkpoint dispatch left uninitialized meta parameters")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id == infer_mask_token_id(tokenizer):
        raise ValueError("padding token must differ from the diffusion mask")
    return model, tokenizer


def runtime_environment(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "accelerate", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "host": platform.node(), "packages": versions,
            "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return record["dataset"], record["example_id"], record["policy_id"], record["repeat"]


def _aggregate_traces(output: Path, records_path: Path) -> None:
    saved = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    aggregate = output / "traces.csv.tmp"
    with aggregate.open("w", newline="") as handle:
        writer = None
        fields = None
        for row in sorted(saved, key=_record_key):
            trace_file = output / row["trace_file"]
            if file_hash(trace_file) != row["trace_sha256"]:
                raise ValueError("trace artifact changed before aggregation")
            with trace_file.open(newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    fields = reader.fieldnames
                    if fields is None:
                        raise ValueError("trace artifact has no CSV header")
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                if reader.fieldnames != fields:
                    raise ValueError("trace schema changed within one run")
                writer.writerows(reader)
    aggregate.replace(output / "traces.csv")


def run_experiment(
    manifest_paths: list[str | Path], output_dir: str | Path, policies: list[dict[str, Any]],
    *, checkpoint: str | Path, gen_length: int = 256, block_length: int = 32,
    repeats: int = 3, device: str = "cuda", dtype: str = "bf16", seed: int = 23,
    collect_trace: bool = False, collect_events: bool = False, limit: int | None = None,
    warmup: bool = True, resume: bool = False, model: Any = None, tokenizer: Any = None,
    purpose: str = "evaluation",
) -> dict[str, Any]:
    from regret_remasking.locking_manifest import load_manifest, manifest_example
    from regret_remasking.locking_policy import LockingPolicyConfig, decode_locking_policy
    from regret_remasking.trace_commitment import TraceCommitmentScorer

    if repeats < 1 or gen_length < 1 or block_length < 1 or gen_length % block_length:
        raise ValueError("positive repeats and generation length divisible by block length are required")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not policies or len({p["policy_id"] for p in policies}) != len(policies):
        raise ValueError("policies must have unique IDs")
    if any(p["arm"] not in ARMS for p in policies):
        raise ValueError("unknown policy arm")
    for policy in policies:
        if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
            raise ValueError("policy IDs must be nonempty strings")
        if policy["arm"].startswith("R-"):
            epsilon = policy.get("epsilon")
            if epsilon is None or not math.isfinite(epsilon) or epsilon < 0:
                raise ValueError("representation policy must declare a finite nonnegative epsilon")
        if policy["arm"].endswith("adaptive"):
            threshold = policy.get("threshold")
            if threshold is None or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("adaptive policy must declare a probability threshold")
        if policy["arm"] in {"T-fixed", "T-adaptive", "C-adaptive"}:
            if not policy.get("checkpoint") or file_hash(policy["checkpoint"]) != policy.get("checkpoint_sha256"):
                raise ValueError("policy checkpoint hash differs from the frozen policy")
    if collect_trace and (repeats != 1 or any(p["arm"] != "B0" for p in policies)):
        raise ValueError("training trace collection requires B0 alone with one repeat")
    manifests = [Path(p).resolve() for p in manifest_paths]
    entries = [entry for path in manifests for entry in load_manifest(path)]
    if limit is not None:
        entries = entries[:limit]
    if not entries or len({(e["dataset"], e["example_id"]) for e in entries}) != len(entries):
        raise ValueError("empty or duplicate example manifest")
    if len({e["content_hash"] for e in entries}) != len(entries):
        raise ValueError("duplicate prompt content in run")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    source_root = Path(__file__).resolve().parents[2]
    config: dict[str, Any] = {
        "schema_version": 1, "examples": [{"example_id": e["example_id"], "dataset": e["dataset"]} for e in entries],
        "manifest_sha256": {str(p): file_hash(p) for p in manifests}, "policies": policies,
        "repeats": repeats, "gen_length": gen_length, "block_length": block_length,
        "device": device, "dtype": dtype, "seed": seed, "checkpoint": str(checkpoint),
        "checkpoint_index_sha256": file_hash(checkpoint / "model.safetensors.index.json"),
        "model_source_sha256": {name: file_hash(checkpoint / name)
                                for name in ("config.json", "configuration_llada.py", "modeling_llada.py")
                                if (checkpoint / name).is_file()},
        "source": source_provenance(source_root), "environment": runtime_environment(device),
        "collect_trace": collect_trace, "collect_events": collect_events, "warmup": warmup,
        "instrumented": collect_trace or collect_events,
        "timing": "decode with controller overhead; excludes model load, scoring and artifact writes",
        "quality_repeat": 1,
        "purpose": purpose,
    }
    config_path = output / "resolved_config.json"
    records_path = output / "records.jsonl"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"run already exists; use --resume with unchanged inputs: {output}")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume configuration/provenance differs from saved run")
    else:
        if records_path.exists() or (output / "traces.csv").exists():
            raise FileExistsError("orphaned run artifacts require a new output directory")
        _write_json(config_path, config)
    complete: set[tuple[str, str, str, int]] = set()
    existing = []
    if records_path.exists():
        existing = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        expected = {(e["dataset"], e["example_id"], p["policy_id"], r)
                    for e in entries for p in policies for r in range(1, repeats + 1)}
        for row in existing:
            key = _record_key(row)
            if key not in expected or key in complete or row.get("status") != "ok":
                raise ValueError("cannot resume duplicate, failed, or unexpected records; preserve this run")
            if row.get("final_mask_count") != 0 or row.get("immutable_token_violations") != 0:
                raise ValueError("cannot resume invalid terminal records")
            if len(row.get("token_ids", [])) != gen_length or row.get("nfe") != row.get("forward_calls"):
                raise ValueError("cannot resume inconsistent completion records")
            if collect_trace and (not row.get("trace_file") or
                    file_hash(output / row["trace_file"]) != row.get("trace_sha256")):
                raise ValueError("cannot resume missing or modified trace artifact")
            complete.add(key)
    if len(complete) == len(entries) * len(policies) * repeats:
        if collect_trace:
            _aggregate_traces(output, records_path)
        _write_json(output / "completion.json", {"complete": True, "records": len(complete)})
        return {"complete": True, "records": len(complete), "output_dir": str(output)}
    scorers = {p["policy_id"]: TraceCommitmentScorer.load(p["checkpoint"])
               for p in policies if p.get("checkpoint")}
    for policy in policies:
        if policy["policy_id"] in scorers:
            expected_features = ["confidence"] if policy["arm"] == "C-adaptive" else FEATURE_NAMES
            if scorers[policy["policy_id"]].features != expected_features:
                raise ValueError(f"checkpoint feature schema does not match {policy['arm']}")
    if model is None or tokenizer is None:
        model, tokenizer = load_local_llada(checkpoint, device, dtype)
    mask_id = infer_mask_token_id(tokenizer)

    def decode(entry: dict[str, Any], policy: dict[str, Any], traces: bool, events: bool):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = LockingPolicyConfig(
            arm=policy["arm"], gen_length=gen_length, block_length=block_length,
            epsilon=policy.get("epsilon") or 0.0,
            threshold=policy["threshold"] if policy.get("threshold") is not None else 0.95,
            mask_id=mask_id,
        )
        return decode_locking_policy(
            model, tokenizer, build_prompt(entry["question"], entry["dataset"]), entry["example_id"],
            cfg, scorer=scorers.get(policy["policy_id"]), collect_trace=traces, collect_events=events,
        )

    if warmup:
        for policy in policies:
            print(f"[locking] warmup {policy['policy_id']}", flush=True)
            decode(entries[0], policy, False, False)
    for index, entry in enumerate(entries):
        for repeat in range(1, repeats + 1):
            rotation = (index + repeat - 1) % len(policies)
            ordered = policies[rotation:] + policies[:rotation]
            for policy in ordered:
                record = {"dataset": entry["dataset"], "example_id": entry["example_id"],
                          "policy_id": policy["policy_id"], "arm": policy["arm"], "repeat": repeat,
                          "epsilon": policy.get("epsilon"), "threshold": policy.get("threshold"),
                          "content_hash": entry["content_hash"], "gen_length": gen_length,
                          "source_sha256": config["source"]["sha256"],
                          "checkpoint_sha256": policy.get("checkpoint_sha256"),
                          "config_sha256": file_hash(config_path)}
                if _record_key(record) in complete:
                    continue
                print(f"[locking] {index+1}/{len(entries)} {entry['example_id']} {policy['policy_id']} repeat={repeat}", flush=True)
                try:
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    result = decode(entry, policy, collect_trace, collect_events)
                    for name in ("nfe", "forward_calls", "latency_s", "prompt_length", "final_mask_count",
                                 "immutable_token_violations", "frozen_count", "generated_row_updates_avoided",
                                 "total_query_rows", "gate_time_s", "scorer_time_s"):
                        record[name] = getattr(result, name)
                    for name in ("raw_kl_min", "raw_kl_negative_count", "raw_kl_exact_zero_count",
                                 "adaptive_accept_count", "fallback_count", "mask_proposals_rejected",
                                 "generated_row_updates_avoided_basis", "attention_layout",
                                 "total_projection_rows", "total_ffn_rows", "recording_mode",
                                 "policy_subtimers"):
                        if hasattr(result, name):
                            record[name] = getattr(result, name)
                    record.update(token_ids=list(result.token_ids), generation=result.text)
                    if (len(record["token_ids"]) != gen_length or mask_id in record["token_ids"]
                            or record["final_mask_count"] or record["immutable_token_violations"]
                            or record["nfe"] != record["forward_calls"] or not 0 < record["nfe"] <= gen_length
                            or not math.isfinite(record["latency_s"]) or record["latency_s"] <= 0):
                        raise ValueError("decoder violated completion/forward/timing invariants")
                    correct, status = score_generation_with_status(result.text, manifest_example(entry))
                    record.update(correct=bool(correct), scoring_status=status, status="ok")
                    record["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
                    if collect_trace:
                        rows = [dict(row, dataset=entry["dataset"], content_hash=entry["content_hash"],
                                     split=entry["split"], source_sha256=config["source"]["sha256"],
                                     config_sha256=record["config_sha256"], gen_length=gen_length,
                                     block_length=block_length, model_revision=checkpoint.name,
                                     feature_schema=json.dumps(FEATURE_NAMES)) for row in result.trace_rows]
                        if not rows:
                            raise ValueError("B0 trace collection produced no rows")
                        fields = list(rows[0])
                        trace_name = hashlib.sha256(
                            json.dumps([entry["dataset"], entry["example_id"]]).encode()).hexdigest()
                        trace_path = output / "traces" / f"{trace_name}.csv"
                        trace_path.parent.mkdir(exist_ok=True)
                        temporary = trace_path.with_suffix(".csv.tmp")
                        with temporary.open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
                        temporary.replace(trace_path)
                        record.update(trace_file=str(trace_path.relative_to(output)),
                                      trace_sha256=file_hash(trace_path))
                    if collect_events:
                        _append_record(output / "events.jsonl", {"example_id": entry["example_id"],
                            "dataset": entry["dataset"], "policy_id": policy["policy_id"],
                            "repeat": repeat, "events": result.events})
                    _append_record(records_path, record)
                    complete.add(_record_key(record))
                except Exception as error:
                    record.update(status="error", error=f"{type(error).__name__}: {error}")
                    _append_record(records_path, record)
                    raise
    if collect_trace:
        _aggregate_traces(output, records_path)
    summary = {"complete": True, "records": len(complete), "examples": len(entries),
               "policies": len(policies), "repeats": repeats, "output_dir": str(output)}
    _write_json(output / "completion.json", summary)
    return summary
