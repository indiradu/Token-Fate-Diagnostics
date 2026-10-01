"""Summarize a complete TraceLock FLOP-only replay against fixed-step work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from locking_flop_accounting import (
    ControllerConfig, ProjectorConfig, backbone_flops, controller_flops, load_model_config,
)


def summarize(profile_dir: Path, model_config_path: Path, output_dir: Path) -> dict:
    completion = json.loads((profile_dir / "completion.json").read_text())
    provenance = json.loads((profile_dir / "provenance.json").read_text())
    if len(provenance["policy_ids"]) != 1:
        raise ValueError("FLOP-only E2 summary requires exactly one policy; do not pool thresholds")
    path = profile_dir / "profiles.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if completion.get("complete") is not True or digest != completion["profiles_sha256"]:
        raise ValueError("profile completion or checksum mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = set()
    for manifest, sha in provenance["manifests"].items():
        p = Path(manifest)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError("manifest changed")
        for line in p.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                for policy in provenance["policy_ids"]:
                    expected.add((entry["dataset"], entry["example_id"], policy))
    actual = [(r["dataset"], r["example_id"], r["policy_id"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected or completion["records"] != len(rows):
        raise ValueError("incomplete or duplicate profile panel")
    model_config = load_model_config(model_config_path)
    verified = []
    percase = []
    for row in rows:
        if row["arm"] != "T-adaptive" or row["source_sha256"] != provenance["source"]["sha256"]:
            raise ValueError("unexpected policy/source")
        if row["final_mask_count"] or row["immutable_token_violations"]:
            raise ValueError("invalid terminal profile")
        length = row["prompt_length"] + row["gen_length"]
        if row["nfe"] != row["forward_calls"] or not 0 < row["nfe"] <= row["gen_length"]:
            raise ValueError("invalid forward count")
        for key in ["total_query_rows", "total_projection_rows", "total_ffn_rows", "total_output_head_rows"]:
            if row[key] != length * row["nfe"]:
                raise ValueError("token locking must keep full backbone rows")
        prefixes = row["controller_prefix_lengths"]
        if len(prefixes) != row["nfe"] or len(prefixes) != row["controller_calls"]:
            raise ValueError("missing controller observations")
        if prefixes != sorted(prefixes) or any(m < row["prompt_length"] or m > length for m in prefixes):
            raise ValueError("invalid controller context geometry")
        pc, cc = row["projector_config"], row["controller_config"]
        b = backbone_flops(row, model_config)
        c = controller_flops(prefixes,
            ProjectorConfig(pc["hidden_dim"], pc["state_hidden_dim"], pc["state_dim"], pc["delta_hidden_dim"], pc["delta_dim"]),
            ControllerConfig(cc["model_dim"], cc["layers"], cc["ff_dim"]))
        if b != row["backbone_counts"] or c != row["learned_counts"]:
            raise ValueError("saved formulas differ from recomputation")
        total = b["total_with_lm_head_flops"] + c["controller_total_flops"]
        if "dispatch_total_flops" in row:
            if row["dispatch_total_flops"] != total or row.get("parity_checked") is not True:
                raise ValueError("operator formula or trajectory parity validation failed")
            verified.append({"dataset": row["dataset"], "example_id": row["example_id"]})
        # This is an explicitly analytical fixed-quota baseline, not an invented run record.
        base_record = dict(row)
        for key in ["total_query_rows", "total_projection_rows", "total_ffn_rows", "total_output_head_rows"]:
            base_record[key] = length * row["gen_length"]
        base = backbone_flops(base_record, model_config)
        percase.append({
            "dataset": row["dataset"], "example_id": row["example_id"], "policy_id": row["policy_id"],
            "model_calls": row["nfe"], "baseline_model_calls": row["gen_length"],
            "backbone_flops": b["backbone_flops"], "lm_head_flops": b["lm_head_flops"],
            "projector_controller_flops": c["controller_total_flops"], "extended_gemm_flops": total,
            "baseline_backbone_flops": base["backbone_flops"],
            "baseline_extended_gemm_flops": base["total_with_lm_head_flops"],
        })
    datasets = sorted({row["dataset"] for row in percase})
    if {v["dataset"] for v in verified} != set(datasets):
        raise ValueError("each dataset needs at least one operator-count and parity check")
    aggregates = []
    for dataset in datasets + ["overall"]:
        selected = [r for r in percase if dataset == "overall" or r["dataset"] == dataset]
        count = len(selected)
        sums = {k: sum(r[k] for r in selected) for k in selected[0] if k not in {"dataset", "example_id", "policy_id"}}
        aggregates.append({
            "dataset": dataset, "cases": count,
            "baseline_backbone_tflops_per_example": sums["baseline_backbone_flops"] / count / 1e12,
            "backbone_tflops_per_example": sums["backbone_flops"] / count / 1e12,
            "backbone_reduction_pct": 100 * (1 - sums["backbone_flops"] / sums["baseline_backbone_flops"]),
            "baseline_extended_tflops_per_example": sums["baseline_extended_gemm_flops"] / count / 1e12,
            "extended_tflops_per_example": sums["extended_gemm_flops"] / count / 1e12,
            "extended_reduction_pct": 100 * (1 - sums["extended_gemm_flops"] / sums["baseline_extended_gemm_flops"]),
            "projector_controller_tflops_per_example": sums["projector_controller_flops"] / count / 1e12,
            "mean_model_calls": sums["model_calls"] / count,
            "baseline_model_calls": sums["baseline_model_calls"] / count,
            "correctness": None, "measured_speedup_pct": None,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("percase_flops.csv", percase), ("aggregate_flops.csv", aggregates)]:
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "complete": True, "cases": len(percase), "purpose": "FLOP-only; original E2 quality/runtime study remains separate",
        "baseline": "Analytical fixed-quota 256 full-row forward calls on each identical prompt; no learned controller",
        "convention": "2 FLOPs/MAC; backbone follows SureLock Sec2.4; extended adds vocabulary head and projector/controller GEMMs",
        "excluded": ["elementwise/norm/activation/RoPE/softmax/KL", "cache/memory I/O", "kernel padding", "model load", "training"],
        "profile_sha256": digest, "dispatch_and_parity_verified_cases": verified,
        "aggregate": aggregates,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.profile_dir, args.model_config, args.output_dir), indent=2))
