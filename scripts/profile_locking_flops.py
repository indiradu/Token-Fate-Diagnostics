"""Separate, untimed FLOP replay of frozen locking experiments.

Count algorithmic matrix operations, including fused attention/encoder calls,
without replacing operators or attaching hooks inside Transformer layers.
Elementwise arithmetic, KL, cache I/O, and kernel padding are not GEMM FLOPs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def operation_flops(name: str, args: tuple[Any, ...]) -> int:
    """Two FLOPs per multiply-add; fused calls counted once, not decomposed."""
    if name == "aten.mm.default":
        a, b = args[:2]
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]
    if name == "aten.addmm.default":
        a, b = args[1:3]
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]
    if name == "aten.bmm.default":
        a, b = args[:2]
        return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
    if name == "aten.baddbmm.default":
        return operation_flops("aten.bmm.default", args[1:3])
    if name in {
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_cudnn_attention.default",
    }:
        q, k, v = args[:3]
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("SDPA count requires [batch,heads,sequence,width]")
        return 2 * q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * (q.shape[3] + v.shape[3])
    if name == "aten._transformer_encoder_layer_fwd.default":
        src, width = args[:2]
        if src.ndim != 3 or src.is_nested:
            raise ValueError("only dense batch-first fused encoders are supported")
        batch, length, _ = src.shape
        ff_width = args[14].shape[0]
        return batch * (8 * length * width**2 + 4 * length**2 * width + 4 * length * width * ff_width)
    if name == "aten._native_multi_head_attention.default":
        q, k, v, width = args[:4]
        if q.shape != k.shape or k.shape != v.shape or q.ndim != 3:
            raise ValueError("only dense self-attention native MHA is supported")
        batch, length, _ = q.shape
        return batch * (8 * length * width**2 + 4 * length**2 * width)
    if any(part in name for part in ("attention", "transformer_encoder", "matmul", "addmm", "baddbmm", "scaled_mm")):
        raise ValueError(f"unaccounted matrix/attention operation: {name}")
    return 0


class MatrixFlopObserver(TorchDispatchMode):
    """Observe actual dispatched shapes while leaving execution unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.flops: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.uncounted: Counter[str] = Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        name = str(func)
        count = operation_flops(name, args)
        result = func(*args, **(kwargs or {}))
        if count:
            self.flops[name] += int(count)
            self.calls[name] += 1
        else:
            self.uncounted[name] += 1
        return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--policy-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--dispatch-cases", type=int, default=2)
    parser.add_argument("--parity-cases", type=int, default=2)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use a new output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(out / "model_modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    from regret_remasking.locking_experiment import load_local_llada
    from regret_remasking.llada_trace import infer_mask_token_id
    from regret_remasking.paper_experiment import (
        _policy_config, _resolve_prompt, load_paper_manifest, source_provenance,
    )
    from regret_remasking.paper_decode import decode_paper_policy
    from regret_remasking.paper_trace import HiddenProjector, PaperTraceController
    from locking_flop_accounting import (
        backbone_flops, controller_flops, load_model_config,
        ProjectorConfig as FlopProjectorConfig, ControllerConfig as FlopControllerConfig,
    )

    model_config = load_model_config(args.checkpoint / "config.json")
    policies = [p for p in json.loads(args.policies.read_text()) if p["policy_id"] in args.policy_ids]
    if {p["policy_id"] for p in policies} != set(args.policy_ids):
        raise ValueError("requested policy not in frozen configuration")
    groups: dict[str, list[dict[str, Any]]] = {}
    for manifest in args.manifests:
        for entry in load_paper_manifest(manifest):
            groups.setdefault(entry["dataset"], []).append(entry)
    if args.limit_per_dataset is not None:
        groups = {name: rows[:args.limit_per_dataset] for name, rows in groups.items()}
    # Put one case from each dataset first for numerical validation.
    entries = [rows[0] for rows in groups.values()] + [row for rows in groups.values() for row in rows[1:]]
    if len(entries) != args.expected_cases or len({(e['dataset'], e['example_id']) for e in entries}) != len(entries):
        raise ValueError("profile sample coverage differs from requested panel")
    references = {}
    if args.reference_records:
        for line in args.reference_records.read_text().splitlines():
            row = json.loads(line)
            if row["repeat"] == 1:
                references[(row["dataset"], row["example_id"], row["policy_id"])] = row
    provenance = {
        "source": source_provenance(args.source_root.resolve()),
        "policies_sha256": sha256(args.policies),
        "manifests": {str(p.resolve()): sha256(p) for p in args.manifests},
        "model_config_sha256": sha256(args.checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(args.checkpoint / "model.safetensors.index.json"),
        "script_sha256": sha256(Path(__file__)),
        "accounting_script_sha256": sha256(Path(__file__).with_name("locking_flop_accounting.py")),
        "convention": "Algorithmic GEMM FLOPs; 2 per multiply-add. Elementwise/KL/cache I/O excluded.",
        "purpose": "FLOP-only profiling; instrumentation latency is not benchmark timing",
        "expected_cases": len(entries), "policy_ids": args.policy_ids,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "seed": 23, "dtype": "bf16", "generation_length": 256,
        "reference_records_sha256": sha256(args.reference_records) if args.reference_records else None,
    }
    write_json(out / "provenance.json", provenance)
    model, tokenizer = load_local_llada(args.checkpoint, "cuda", "bf16")
    learned: dict[str, Any] = {}
    for policy in policies:
        for key, cls in [("controller_checkpoint", PaperTraceController), ("projector_checkpoint", HiddenProjector)]:
            if policy.get(key):
                path = Path(policy[key])
                if sha256(path) != policy[f"{key}_sha256"]:
                    raise ValueError(f"learned checkpoint changed: {path}")
                learned.setdefault(str(path), cls.load(path, "cuda"))
    mask_id = infer_mask_token_id(tokenizer)
    checked_fields = ["token_ids", "nfe", "forward_calls", "prompt_length", "total_query_rows",
                      "total_projection_rows", "total_ffn_rows", "total_output_head_rows", "controller_calls"]
    record_count = 0
    for case_index, entry in enumerate(entries):
        for policy in policies:
            print(f"[flops] {case_index + 1}/{len(entries)} {entry['example_id']} {policy['policy_id']}", flush=True)
            controller = learned.get(str(policy.get("controller_checkpoint")))
            projector = learned.get(str(policy.get("projector_checkpoint")))
            cfg = _policy_config(policy, mask_id, 256)

            def decode() -> Any:
                random.seed(23)
                np.random.seed(23)
                torch.manual_seed(23)
                return decode_paper_policy(model, tokenizer, _resolve_prompt(entry), entry["example_id"], cfg,
                                           controller=controller, projector=projector,
                                           collect_events=False, step_observer=None)

            reference = references.get((entry["dataset"], entry["example_id"], policy["policy_id"]))
            if args.reference_records and reference is None:
                raise ValueError("missing requested reference record")
            if case_index < args.parity_cases and reference is None:
                reference = asdict(decode())
            lengths: list[int] = []
            handle = None
            if controller is not None:
                def capture_length(module: Any, inputs: Any) -> None:
                    lengths.append(int(inputs[0].shape[1]))
                handle = controller.register_forward_pre_hook(capture_length)
            observer = MatrixFlopObserver() if case_index < args.dispatch_cases else None
            try:
                with observer if observer is not None else nullcontext():
                    result = decode()
            finally:
                if handle is not None:
                    handle.remove()
            result_data = asdict(result)
            if reference is not None:
                for field in checked_fields:
                    if result_data[field] != reference[field]:
                        raise ValueError(f"profile replay changed {field}: {entry['example_id']} {policy['policy_id']}")
            if result.final_mask_count or result.immutable_token_violations or len(result.token_ids) != 256:
                raise ValueError("invalid profiled terminal output")
            if len(lengths) != result.controller_calls:
                raise ValueError("controller shape capture incomplete")
            row = {
                **{k: result_data[k] for k in checked_fields + ["row_count_basis", "frozen_count", "final_mask_count", "immutable_token_violations"]},
                "dataset": entry["dataset"], "example_id": entry["example_id"], "policy_id": policy["policy_id"],
                "arm": policy["arm"], "gen_length": 256, "status": "ok", "repeat": 1,
                "controller_prefix_lengths": lengths,
                "source_sha256": provenance["source"]["sha256"],
                "controller_checkpoint_sha256": policy.get("controller_checkpoint_sha256"),
                "projector_checkpoint_sha256": policy.get("projector_checkpoint_sha256"),
                "parity_checked": reference is not None,
                "parity_source": "saved record" if args.reference_records else "separate uninstrumented decode",
            }
            row["backbone_counts"] = backbone_flops(row, model_config)
            if controller is not None:
                if projector is None:
                    raise ValueError("controller requires its frozen projector")
                row["controller_config"] = asdict(controller.config)
                row["projector_config"] = asdict(projector.config)
                pconfig, cconfig = projector.config, controller.config
                row["learned_counts"] = controller_flops(
                    lengths,
                    FlopProjectorConfig(pconfig.hidden_dim, pconfig.state_hidden_dim, pconfig.state_dim,
                                        pconfig.delta_hidden_dim, pconfig.delta_dim),
                    FlopControllerConfig(cconfig.model_dim, cconfig.layers, cconfig.ff_dim),
                )
            else:
                row["learned_counts"] = None
            if observer is not None:
                row["dispatch_flops"] = dict(observer.flops)
                row["dispatch_calls"] = dict(observer.calls)
                row["uncounted_operations"] = dict(observer.uncounted)
                row["dispatch_total_flops"] = sum(observer.flops.values())
                expected = row["backbone_counts"]["total_with_lm_head_flops"]
                if row["learned_counts"] is not None:
                    expected += row["learned_counts"]["controller_total_flops"]
                if expected != row["dispatch_total_flops"]:
                    write_json(out / "dispatch_mismatch.json", row)
                    raise ValueError(f"shape count {row['dispatch_total_flops']} != analytical count {expected}")
                row["dispatch_matches_formula"] = True
            with (out / "profiles.jsonl").open("a") as handle_out:
                handle_out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle_out.flush()
            record_count += 1
            write_json(out / "progress.json", {"complete": False, "cases_started": case_index + 1,
                                                "records": record_count, "expected_records": len(entries) * len(policies)})
    write_json(out / "completion.json", {"complete": True, "cases": len(entries), "records": record_count,
                                         "profiles_sha256": sha256(out / "profiles.jsonl")})
    write_json(out / "progress.json", {"complete": True, "cases_started": len(entries),
                                      "records": record_count, "expected_records": len(entries) * len(policies)})


if __name__ == "__main__":
    main()
