"""Train-only projection bootstrap and contextual hidden-trace collection."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from regret_remasking.code_eval import assemble_humaneval_program
from regret_remasking.locking_experiment import load_local_llada
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.paper_experiment import file_hash, source_provenance, _resolve_prompt, score_paper_generation
from regret_remasking.paper_trace import HiddenProjector, fit_projector, save_paper_trace_example, validate_example_dir


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["record_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("manifest must have unique nonempty record IDs")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def reserve_storage(ledger_path: str | Path, key: str, amount: int, max_gib: float) -> None:
    """Atomically reserve trace bytes across concurrent collection workers."""
    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(ledger.read_text()) if ledger.exists() else {"bytes": 0, "reservations": {}}
        if key in state["reservations"]:
            if state["reservations"][key] != amount:
                raise ValueError("trace storage reservation changed")
            return
        if amount < 0 or state["bytes"] + amount > int(max_gib * 1024**3):
            raise RuntimeError("declared trace-storage cap would be exceeded")
        state["reservations"][key] = amount
        state["bytes"] += amount
        write_json(ledger, state)


def verify_trace_files(directory: str | Path, projection_hash: str) -> None:
    target = Path(directory)
    hashes = json.loads((target / "hashes.json").read_text())
    for name, expected in hashes.items():
        if file_hash(target / name) != expected:
            raise ValueError(f"hidden trace file hash mismatch: {target / name}")
    validate_example_dir(target, projection_hash)


def example_seed(record_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{record_id}".encode()).digest()[:8], "little")


def retained_steps(gen_length: int, fraction: float, rng: random.Random) -> list[int]:
    if gen_length < 2 or not 0 < fraction <= 1:
        raise ValueError("retention requires generation>=2 and fraction in (0,1]")
    count = min(gen_length, max(2, math.ceil(gen_length * fraction)))
    return sorted([0, gen_length - 1, *rng.sample(range(1, gen_length - 1), count - 2)])


def teacher_config(entry: Mapping[str, Any]) -> PaperDecodeConfig:
    teacher = entry["teacher_config"]
    if int(teacher["steps"]) != int(teacher["gen_length"]):
        raise ValueError("teacher must use one token per actual model call")
    return PaperDecodeConfig(arm="B-entropy", gen_length=int(teacher["gen_length"]), block_length=int(teacher["block_length"]))


def trace_quality_filter(text: str, entry: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Cheap, predeclared filtering; correctness is NOT the agreement label."""
    reasons: list[str] = []
    if len(text.strip()) < 8:
        reasons.append("empty_or_too_short")
    words = text.split()
    if len(words) >= 40:
        grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        if len(set(grams)) / len(grams) < 0.15:
            reasons.append("severely_repetitive")
    if entry.get("family") == "code" or entry.get("dataset") in {"kodcode", "humaneval"}:
        meta = entry.get("metadata", {})
        if not meta.get("entry_point") or not meta.get("test"):
            reasons.append("missing_code_validation_metadata")
        else:
            program = assemble_humaneval_program(text, str(entry["question"]), str(meta["test"]), str(meta["entry_point"]))
            try:
                tree = ast.parse(program)
                funcs = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if meta["entry_point"] not in funcs:
                    reasons.append("missing_generated_entrypoint")
            except (SyntaxError, ValueError, MemoryError):
                reasons.append("invalid_code_syntax")
    return not reasons, reasons


class HiddenTraceObserver:
    def __init__(self, selected: list[int], projector: HiddenProjector) -> None:
        self.selected = set(selected)
        self.projector = projector
        self.features: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.proposals: list[np.ndarray] = []
        self.inputs: list[np.ndarray] = []
        self.steps: list[int] = []
        self.prompt_length: int | None = None

    def __call__(self, frame: dict[str, Any]) -> None:
        if frame["step"] not in self.selected:
            return
        features = self.projector.encode_hidden(frame["last_hidden"])[0].cpu().numpy().astype(np.float16)
        if not bool(np.isfinite(features).all()):
            raise ValueError("projected hidden features are nonfinite after storage rounding")
        if frame["proposal_positions"].tolist() != list(range(features.shape[0])):
            raise ValueError("teacher trace must include all contextual positions")
        self.features.append(features)
        self.states.append(frame["states"][0].cpu().numpy().copy())
        self.proposals.append(frame["proposal_tokens"][0].cpu().numpy().astype(np.int32, copy=True))
        self.inputs.append(frame["input_ids"][0].cpu().numpy().astype(np.int32, copy=True))
        self.steps.append(int(frame["step"]))
        self.prompt_length = int(frame["prompt_length"])

    def arrays(self, final_token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.prompt_length is None or not self.features:
            raise ValueError("teacher observer did not record any frames")
        if set(self.steps) != self.selected:
            raise ValueError("teacher did not reach every preselected observation step")
        proposals = np.stack(self.proposals)
        inputs = np.stack(self.inputs)
        final = inputs[0].copy()
        final[self.prompt_length:] = np.asarray(final_token_ids, dtype=np.int32)
        labels = (proposals == final[None, :]).astype(np.uint8)
        return np.stack(self.features), np.stack(self.states), labels, proposals, inputs


def _infrastructure_error(status: str) -> bool:
    return status in {"sandbox_error", "sandbox_unavailable", "execution_unavailable", "missing_code_tests"}


def collect_hidden_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    projector_checkpoint: str | Path,
    *,
    device: str = "cuda",
    seed: int = 23,
    model: Any = None,
    tokenizer: Any = None,
    storage_ledger: str | Path | None = None,
    max_storage_gib: float = 600,
) -> dict[str, Any]:
    entries = read_rows(manifest)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    projection_hash = file_hash(projector_checkpoint)
    config: dict[str, Any] = {"manifest": str(Path(manifest).resolve()), "manifest_sha256": file_hash(manifest),
              "checkpoint": str(Path(checkpoint).resolve()), "projection_sha256": projection_hash,
              "source": source_provenance(root), "seed": seed,
              "storage_ledger": str(Path(storage_ledger).resolve()) if storage_ledger else None,
              "max_storage_gib": max_storage_gib,
              "quality_filter": "min8chars; fourgram diversity>=.15 for>=40words; code syntax+entrypoint",
              "executable_check": "diagnostic; failing tests do not define the trace-agreement label"}
    cfg_path = output / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError("hidden collection configuration changed; use a new attempt")
    write_json(cfg_path, config)
    if (output / "collection.json").exists():
        saved = json.loads((output / "collection.json").read_text())
        if saved.get("complete"):
            for row in saved["examples"]:
                if row["status"] == "accepted":
                    verify_trace_files(row["example_dir"], projection_hash)
            return saved
    if model is None or tokenizer is None:
        model, tokenizer = load_local_llada(checkpoint, device, "bf16")
    projector = HiddenProjector.load(projector_checkpoint, device=device)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for number, entry in enumerate(entries):
        ident = hashlib.sha256(entry["record_id"].encode()).hexdigest()[:24]
        status_path = output / "statuses" / f"{ident}.json"
        if status_path.exists():
            row = json.loads(status_path.read_text())
            if row["record_id"] != entry["record_id"]:
                raise ValueError("collection identity changed")
            if row["status"] == "accepted":
                verify_trace_files(row["example_dir"], projection_hash)
            rows.append(row)
            continue
        cfg = teacher_config(entry)
        selected = retained_steps(cfg.gen_length, float(entry["teacher_config"].get("retained_step_fraction", .3)), random.Random(example_seed(entry["record_id"], seed)))
        observer = HiddenTraceObserver(selected, projector)
        print(f"[hidden] {number+1}/{len(entries)} {entry['record_id']}", flush=True)
        result = decode_paper_policy(model, tokenizer, _resolve_prompt(entry), entry["example_id"], cfg, step_observer=observer)
        if result.nfe != cfg.gen_length or result.final_mask_count or result.immutable_token_violations:
            raise ValueError("teacher completion invariant failed")
        good, reasons = trace_quality_filter(result.text, entry)
        correct, scoring_status = score_paper_generation(result.text, entry)
        if _infrastructure_error(scoring_status):
            raise RuntimeError(f"teacher scoring infrastructure failure: {scoring_status}")
        row = {"record_id": entry["record_id"], "example_id": entry["example_id"],
               "source_id": entry["source_id"], "split": entry["split"], "family": entry["family"],
               "status": "accepted" if good else "filtered", "reason": reasons,
               "prompt_hash": entry["prompt_hash"], "generation": result.text, "token_ids": result.token_ids,
               "correct": correct, "scoring_status": scoring_status, "nfe": result.nfe,
               "source_sha256": config["source"]["sha256"], "projection_sha256": projection_hash}
        if good:
            features, states, labels, proposals, inputs = observer.arrays(result.token_ids)
            target = output / "examples" / ident
            if storage_ledger is not None:
                size = sum(a.nbytes for a in [features, states, labels, proposals, inputs]) + 65536
                reserve_storage(storage_ledger, str(target), size, max_storage_gib)
            # Replacement candidates remain members of their original training pool.
            split = "validation" if entry["split"].startswith("validation") else "fit"
            metadata = {"example_id": entry["example_id"], "record_id": entry["record_id"],
                        "source_id": entry["source_id"], "split": split, "source_manifest_split": entry["split"],
                        "family": entry["family"], "prompt_hash": entry["prompt_hash"],
                        "content_hash": entry.get("chat_prompt_hash", entry["prompt_hash"]),
                        "prompt_length": result.prompt_length, "gen_length": cfg.gen_length,
                        "selected_steps": observer.steps, "selected_ids": [f"{entry['example_id']}:{s}" for s in observer.steps],
                        "source": config["source"], "projection_sha256": projection_hash,
                        "teacher_config": entry["teacher_config"], "nonfinite_feature_count": 0}
            save_paper_trace_example(target, features, states, labels, metadata)
            np.save(target / "proposal_tokens.npy", proposals)
            np.save(target / "input_ids.npy", inputs)
            write_json(target / "terminal.json", {"token_ids": result.token_ids, "generation": result.text, "correct": correct, "scoring_status": scoring_status})
            files = {p.name: file_hash(p) for p in target.iterdir() if p.is_file()}
            write_json(target / "hashes.json", files)
            row.update(example_dir=str(target), features_bytes=sum(p.stat().st_size for p in target.iterdir() if p.is_file()), retained_frames=len(observer.steps))
        write_json(status_path, row)
        rows.append(row)
        del observer
    payload = {"complete": True, "processed_count": len(rows), "accepted_count": sum(x["status"] == "accepted" for x in rows),
               "examples": rows, "elapsed_s": time.perf_counter()-start,
               "manifest_sha256": config["manifest_sha256"], "projection_sha256": projection_hash,
               "source_sha256": config["source"]["sha256"]}
    write_json(output / "collection.json", payload)
    return payload


def bootstrap_projector(
    fit_manifest: str | Path,
    bootstrap_ids: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    seed: int = 42,
    fit_steps: int = 1000,
    samples_per_example: int = 512,
) -> dict[str, Any]:
    entries = read_rows(fit_manifest)
    requested = json.loads(Path(bootstrap_ids).read_text())["record_ids"]
    by_id = {x["record_id"]: x for x in entries}
    if len(requested) != len(set(requested)) or not set(requested).issubset(by_id):
        raise ValueError("projection bootstrap IDs must be unique fit candidates")
    selected_entries = [by_id[x] for x in requested]
    if any(x["split"] != "fit" for x in selected_entries):
        raise ValueError("projection fit cannot see validation or evaluation")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    source = source_provenance(Path(__file__).resolve().parents[2])
    frozen_config = {"fit_manifest_sha256": file_hash(fit_manifest), "bootstrap_ids_sha256": file_hash(bootstrap_ids),
                     "checkpoint": str(Path(checkpoint).resolve()), "source_sha256": source["sha256"],
                     "fit_steps": fit_steps, "samples_per_example": samples_per_example, "seed": seed}
    config_path = output / "bootstrap_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != frozen_config:
        raise ValueError("projector bootstrap configuration changed; use a new attempt")
    write_json(config_path, frozen_config)
    if (output / "success.json").exists():
        saved = json.loads((output / "success.json").read_text())
        if saved["projection_sha256"] != file_hash(output / "projector.pt") or saved["reservoir_sha256"] != file_hash(output / "reservoir.pt"):
            raise ValueError("published projector artifacts changed")
        return saved
    model, tokenizer = load_local_llada(checkpoint, device, "bf16")
    all_hidden: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for number, entry in enumerate(selected_entries):
        cfg = teacher_config(entry)
        rng = random.Random(example_seed(entry["record_id"], seed))
        capture_steps = set(retained_steps(cfg.gen_length, min(1.0, 16 / cfg.gen_length), rng))
        per_step = math.ceil(samples_per_example / len(capture_steps))
        pieces: list[torch.Tensor] = []
        def observe(frame):
            if frame["step"] not in capture_steps:
                return
            n = int(frame["input_ids"].shape[1])
            positions = torch.tensor(rng.sample(range(n), min(per_step, n)), device=frame["input_ids"].device)
            # Store original BF16 activations; delta construction is FP32 in both fit/inference.
            values = torch.stack([h[0].index_select(0, positions) for h in frame["last_hidden"]], dim=1)
            pieces.append(values.detach().cpu())
        print(f"[projection] {number+1}/{len(selected_entries)} {entry['record_id']}", flush=True)
        result = decode_paper_policy(model, tokenizer, _resolve_prompt(entry), entry["example_id"], cfg, step_observer=observe)
        hidden = torch.cat(pieces)[:samples_per_example]
        if hidden.shape[0] != samples_per_example:
            raise ValueError("not enough bootstrap hidden rows")
        all_hidden.append(hidden)
        good, reasons = trace_quality_filter(result.text, entry)
        diagnostics.append({"record_id": entry["record_id"], "family": entry["family"], "filter_would_accept": good,
                            "filter_reasons": reasons, "generation": result.text, "nfe": result.nfe})
        write_json(output / "bootstrap_progress.json", {"completed": number+1, "requested": len(selected_entries), "diagnostics": diagnostics})
    if source_provenance(Path(__file__).resolve().parents[2])["sha256"] != source["sha256"]:
        raise ValueError("bootstrap source changed during execution")
    reservoir = {"hidden": torch.cat(all_hidden), "hidden_count": 33, "hidden_indices": [30,31,32],
                 "hidden_semantics": ["raw_block_30","raw_block_31","final_normalized_block_32"],
                 "split": "fit", "example_ids": [x["example_id"] for x in selected_entries],
                 "record_ids": requested, "source_sha256": source["sha256"],
                 "fit_manifest_sha256": file_hash(fit_manifest), "bootstrap_ids_sha256": file_hash(bootstrap_ids)}
    reservoir_path = output / "reservoir.pt"; torch.save(reservoir, reservoir_path)
    del model, all_hidden
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    projector_path = output / "projector.pt"
    fit_projector(reservoir, projector_path, device=device, steps=fit_steps, seed=seed)
    payload = {"complete": True, "fit_candidate_examples": len(requested), "reservoir_rows": int(reservoir["hidden"].shape[0]),
               "projector_checkpoint": str(projector_path), "projection_sha256": file_hash(projector_path),
               "reservoir_sha256": file_hash(reservoir_path), "source_sha256": source["sha256"],
               "fit_manifest_sha256": file_hash(fit_manifest), "bootstrap_ids_sha256": file_hash(bootstrap_ids),
               "hidden_count": 33, "hidden_indices": [30,31,32], "fit_steps": fit_steps, "diagnostics": diagnostics}
    write_json(output / "success.json", payload)
    return payload
