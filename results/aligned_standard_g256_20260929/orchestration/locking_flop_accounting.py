#!/usr/bin/env python3
"""Algorithmic FLOP accounting for locking experiments.

The accounting follows the convention used by compute-locking papers:
one multiply-accumulate is counted as two FLOPs.  The backbone total covers
Q/K/V/O projections, QK/AV attention matmuls, and the gated MLP.  Vocabulary
head and learned controller/projector GEMMs are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROW_KEYS = (
    "total_query_rows",
    "total_projection_rows",
    "total_ffn_rows",
    "total_output_head_rows",
)
CASE_KEYS = ("dataset", "example_id")
TFLOP = 1_000_000_000_000.0


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * (self.hidden_size // self.num_attention_heads)


@dataclass(frozen=True)
class ProjectorConfig:
    hidden_size: int
    state_hidden: int
    state_dim: int
    delta_hidden: int
    delta_dim: int


@dataclass(frozen=True)
class ControllerConfig:
    model_dim: int
    num_layers: int
    ffn_dim: int
    moving_window: int = 32
    input_feature_dim: int = 832


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def _config_value(raw: dict[str, Any], names: tuple[str, ...], label: str) -> int:
    for name in names:
        if name in raw and raw[name] is not None:
            return _positive_int(raw[name], label)
    raise ValueError(f"model config is missing {label}; tried {', '.join(names)}")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load LLaDA/HF-style model config and validate independent dimensions."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = ModelConfig(
        hidden_size=_config_value(raw, ("hidden_size", "d_model"), "hidden size"),
        num_hidden_layers=_config_value(raw, ("num_hidden_layers", "n_layers"), "layer count"),
        num_attention_heads=_config_value(raw, ("num_attention_heads", "n_heads"), "attention heads"),
        num_key_value_heads=_config_value(raw, ("num_key_value_heads", "n_kv_heads"), "kv heads"),
        intermediate_size=_config_value(raw, ("intermediate_size", "mlp_hidden_size"), "ffn hidden size"),
        vocab_size=_config_value(raw, ("vocab_size",), "vocab size"),
    )
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    return cfg


def backbone_flops(record: dict[str, Any], config: ModelConfig) -> dict[str, int]:
    """Compute backbone/head FLOPs from one saved decode record.

    Required row counters are actual recorded work.  Query/projection/FFN rows
    must use the per-layer-equivalent basis saved by the sparse runner; output
    head rows are actual rows and are reported outside the backbone total.
    """
    if record.get("status") != "ok":
        raise ValueError(f"record status is not ok: {record.get('status')!r}")
    basis = str(record.get("row_count_basis", ""))
    if "per_layer_equivalent" not in basis or "output_head_count_is_actual" not in basis:
        raise ValueError(f"unsupported row_count_basis: {basis!r}")
    if any(key not in record for key in ROW_KEYS):
        missing = [key for key in ROW_KEYS if key not in record]
        raise ValueError(f"record missing row counters: {missing}")

    sum_queries = _nonnegative_int(record["total_query_rows"], "total_query_rows")
    sum_projection = _nonnegative_int(record["total_projection_rows"], "total_projection_rows")
    sum_ff = _nonnegative_int(record["total_ffn_rows"], "total_ffn_rows")
    sum_head = _nonnegative_int(record["total_output_head_rows"], "total_output_head_rows")
    prompt_length = _positive_int(record.get("prompt_length"), "prompt_length")
    gen_length = _positive_int(record.get("gen_length"), "gen_length")

    d = config.hidden_size
    layers = config.num_hidden_layers
    kv_dim = config.kv_hidden_size
    seq_len = prompt_length + gen_length

    attention = 4 * layers * d * seq_len * sum_queries
    projections = layers * (4 * d * d + 4 * d * kv_dim) * sum_projection
    ffn = 6 * layers * d * config.intermediate_size * sum_ff
    lm_head = 2 * d * config.vocab_size * sum_head
    backbone = attention + projections + ffn
    return {
        "attention_flops": int(attention),
        "projection_flops": int(projections),
        "ffn_flops": int(ffn),
        "backbone_flops": int(backbone),
        "lm_head_flops": int(lm_head),
        "total_with_lm_head_flops": int(backbone + lm_head),
        "sequence_length": int(seq_len),
    }


def controller_flops(
    prefix_lengths: Iterable[int],
    projector_config: ProjectorConfig,
    controller_config: ControllerConfig,
) -> dict[str, int]:
    """Compute learned projector/controller GEMM FLOPs for prefix replays."""
    lengths = [_positive_int(m, "prefix length") for m in prefix_lengths]
    h = _positive_int(projector_config.hidden_size, "projector hidden_size")
    state_hidden = _positive_int(projector_config.state_hidden, "projector state_hidden")
    state_dim = _positive_int(projector_config.state_dim, "projector state_dim")
    delta_hidden = _positive_int(projector_config.delta_hidden, "projector delta_hidden")
    delta_dim = _positive_int(projector_config.delta_dim, "projector delta_dim")
    d_c = _positive_int(controller_config.model_dim, "controller model_dim")
    ff_c = _positive_int(controller_config.ffn_dim, "controller ffn_dim")
    layers = _positive_int(controller_config.num_layers, "controller num_layers")
    input_feature_dim = _positive_int(controller_config.input_feature_dim, "controller input_feature_dim")

    per_prefix_projector = (
        3 * (2 * h * state_hidden + 2 * state_hidden * state_dim)
        + 2 * (2 * h * delta_hidden + 2 * delta_hidden * delta_dim)
    )
    projector = per_prefix_projector * sum(lengths)
    controller_input = sum(2 * m * input_feature_dim * d_c for m in lengths)
    transformer = 0
    heads = 0
    for m in lengths:
        s = m + 1
        transformer += layers * (8 * s * d_c * d_c + 4 * s * s * d_c + 4 * s * d_c * ff_c)
        heads += 2 * d_c * (m + 1)
    total = projector + controller_input + transformer + heads
    return {
        "controller_calls": len(lengths),
        "controller_prefix_tokens": int(sum(lengths)),
        "projector_flops": int(projector),
        "controller_input_flops": int(controller_input),
        "controller_transformer_flops": int(transformer),
        "controller_token_threshold_head_flops": int(heads),
        "controller_total_flops": int(total),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _profile_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["example_id"]), str(row["policy_id"]))


def _case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["example_id"]), str(row["policy_id"]))


def _baseline_case_key(row: dict[str, Any], baseline_id: str) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["example_id"]), baseline_id)


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def _format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6g}"


def _read_config_dataclass(path: str | None, cls: type) -> Any | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return cls(**raw)


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_case_key(record)].append(record)
    return grouped


def _select_repeat1_and_check(grouped: dict[tuple[str, str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected = []
    for key, rows in sorted(grouped.items()):
        rows_by_repeat: dict[Any, dict[str, Any]] = {}
        for row in rows:
            repeat = row.get("repeat")
            if repeat in rows_by_repeat:
                raise ValueError(f"duplicate repeat {repeat!r} for {key}")
            rows_by_repeat[repeat] = row
        if 1 not in rows_by_repeat:
            raise ValueError(f"missing repeat 1 for {key}")
        first_work = tuple(rows_by_repeat[1].get(k) for k in ROW_KEYS)
        for repeat, row in sorted(rows_by_repeat.items()):
            if tuple(row.get(k) for k in ROW_KEYS) != first_work:
                raise ValueError(f"work counters vary across repeats for {key}; repeat {repeat}")
        selected.append(rows_by_repeat[1])
    return selected


def _complete_policy_ids(selected: list[dict[str, Any]], policy_ids: list[str] | None) -> list[str]:
    present = sorted({str(row["policy_id"]) for row in selected})
    if not present:
        raise ValueError("records contain no complete repeat-1 rows")
    if policy_ids:
        missing = sorted(set(policy_ids) - set(present))
        if missing:
            raise ValueError(f"requested policies missing from records: {missing}")
        return policy_ids
    return present


def _validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("records file is empty")
    bad_status = [row for row in records if row.get("status") != "ok"]
    if bad_status:
        first = bad_status[0]
        raise ValueError(
            "records contain non-ok status rows; first "
            f"{_case_key(first)} repeat={first.get('repeat')!r} status={first.get('status')!r}"
        )
    for row in records:
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"record correctness must be boolean for {_case_key(row)} repeat={row.get('repeat')!r}")


def _load_profile_map(path: Path) -> dict[tuple[str, str, str], list[int]]:
    profile_map: dict[tuple[str, str, str], list[int]] = {}
    for row in _load_jsonl(path):
        key = _profile_key(row)
        if key in profile_map:
            raise ValueError(f"duplicate controller profile for {key}")
        lengths = row.get("controller_prefix_lengths")
        if not isinstance(lengths, list):
            raise ValueError(f"profile {key} missing controller_prefix_lengths list")
        profile_map[key] = lengths
    return profile_map


def _controller_work_for_record(
    row: dict[str, Any],
    profile_status: str,
    profile_map: dict[tuple[str, str, str], list[int]],
    projector_config: ProjectorConfig | None,
    controller_config: ControllerConfig | None,
) -> tuple[dict[str, int] | None, bool, str]:
    expected_calls = _nonnegative_int(row.get("controller_calls", 0), "controller_calls")
    if expected_calls == 0:
        return {"controller_total_flops": 0}, False, "known_zero"
    if profile_status != "joined":
        return None, True, profile_status
    key = _profile_key(row)
    lengths = profile_map.get(key)
    if lengths is None:
        return None, True, "missing_profile"
    if len(lengths) != expected_calls:
        return None, True, f"profile_length_mismatch:{len(lengths)}!={expected_calls}"
    return controller_flops(lengths, projector_config, controller_config), False, "profiled"  # type: ignore[arg-type]


def analyze_records(
    records_path: str | Path,
    model_config: ModelConfig,
    output_dir: str | Path,
    *,
    baseline_id: str = "B0",
    purpose: str = "locking-flop-accounting",
    policy_ids: list[str] | None = None,
    expected_cases: int | None = None,
    profiles_path: str | Path | None = None,
    projector_config: ProjectorConfig | None = None,
    controller_config: ControllerConfig | None = None,
) -> dict[str, Any]:
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(records_path)
    _validate_records(records)
    grouped_all = _group_records(records)
    selected = _select_repeat1_and_check(grouped_all)
    policies = _complete_policy_ids(selected, policy_ids)
    if baseline_id not in policies:
        raise ValueError(f"baseline_id {baseline_id!r} must be included in analyzed policies")
    selected = [row for row in selected if str(row["policy_id"]) in set(policies)]
    grouped_selected = _group_records(selected)

    baseline_cases = {
        (str(row["dataset"]), str(row["example_id"]))
        for row in selected
        if str(row["policy_id"]) == baseline_id
    }
    complete_case_pairs = []
    for dataset, example_id in sorted(baseline_cases):
        if all((dataset, example_id, policy) in grouped_selected for policy in policies):
            complete_case_pairs.append((dataset, example_id))
    if not complete_case_pairs:
        raise ValueError(f"no complete paired cases found for baseline {baseline_id!r} and policies {policies}")
    if expected_cases is not None and len(complete_case_pairs) != expected_cases:
        raise ValueError(f"expected {expected_cases} complete paired cases, found {len(complete_case_pairs)}")

    profile_map: dict[tuple[str, str, str], list[int]] = {}
    profile_status = "not_provided"
    if profiles_path is not None:
        if projector_config is None or controller_config is None:
            profile_status = "profiles_provided_without_shapes"
        else:
            profile_map = _load_profile_map(Path(profiles_path))
            profile_status = "joined"

    latency_by_case_policy: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        if row.get("status") == "ok" and "latency_s" in row:
            latency_by_case_policy[_case_key(row)].append(float(row["latency_s"]))

    percase_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    datasets = sorted({d for d, _ in complete_case_pairs})
    dataset_keys = datasets + ["overall"]
    for dataset_name in dataset_keys:
        for policy in policies:
            aggregate[(dataset_name, policy)] = {
                "dataset": dataset_name,
                "policy_id": policy,
                "cases": 0,
                "correct": 0,
                "latency_median_sum_s": 0.0,
                "backbone_flops": 0,
                "lm_head_flops": 0,
                "total_with_lm_head_flops": 0,
                "controller_total_flops": 0,
                "controller_profile_expected_cases": 0,
                "controller_profile_covered_cases": 0,
                "controller_profile_pending_cases": 0,
            }

    for dataset_name in datasets:
        case_filter = [(d, e) for d, e in complete_case_pairs if d == dataset_name]
        for dataset, example_id in case_filter:
            baseline_row = grouped_selected[(dataset, example_id, baseline_id)][0]
            baseline_work = backbone_flops(baseline_row, model_config)
            baseline_latency = _median(latency_by_case_policy[_baseline_case_key(baseline_row, baseline_id)])
            for policy in policies:
                row = grouped_selected[(dataset, example_id, policy)][0]
                work = backbone_flops(row, model_config)
                latency = _median(latency_by_case_policy[(dataset, example_id, policy)])
                if math.isnan(latency) or math.isnan(baseline_latency):
                    raise ValueError(f"missing latency for {(dataset, example_id, policy)}")
                controller_work, controller_pending, controller_profile_case_status = _controller_work_for_record(
                    row,
                    profile_status,
                    profile_map,
                    projector_config,
                    controller_config,
                )
                percase: dict[str, Any] = {
                    "dataset": dataset,
                    "example_id": example_id,
                    "policy_id": policy,
                    "repeat_for_quality_and_flops": 1,
                    "repeats_for_latency": len(latency_by_case_policy[(dataset, example_id, policy)]),
                    "correct": bool(row.get("correct")),
                    "latency_median_s": latency,
                    "baseline_latency_median_s": baseline_latency,
                    "speedup_pct_vs_baseline": 100.0 * (baseline_latency / latency - 1.0),
                    "backbone_tflops": work["backbone_flops"] / TFLOP,
                    "lm_head_tflops": work["lm_head_flops"] / TFLOP,
                    "total_with_lm_head_tflops": work["total_with_lm_head_flops"] / TFLOP,
                    "baseline_backbone_tflops": baseline_work["backbone_flops"] / TFLOP,
                    "backbone_reduction_pct_vs_baseline": 100.0
                    * (1.0 - work["backbone_flops"] / baseline_work["backbone_flops"]),
                    "total_with_lm_head_reduction_pct_vs_baseline": 100.0
                    * (1.0 - work["total_with_lm_head_flops"] / baseline_work["total_with_lm_head_flops"]),
                    "controller_calls": _nonnegative_int(row.get("controller_calls", 0), "controller_calls"),
                    "controller_profile_status": controller_profile_case_status,
                    "controller_total_tflops": ""
                    if controller_work is None
                    else controller_work["controller_total_flops"] / TFLOP,
                }
                percase_rows.append(percase)
                for agg_dataset in (dataset, "overall"):
                    agg = aggregate[(agg_dataset, policy)]
                    agg["cases"] += 1
                    agg["correct"] += int(bool(row.get("correct")))
                    agg["latency_median_sum_s"] += latency
                    agg["backbone_flops"] += work["backbone_flops"]
                    agg["lm_head_flops"] += work["lm_head_flops"]
                    agg["total_with_lm_head_flops"] += work["total_with_lm_head_flops"]
                    if percase["controller_calls"] > 0:
                        agg["controller_profile_expected_cases"] += 1
                        if controller_pending:
                            agg["controller_profile_pending_cases"] += 1
                        else:
                            agg["controller_profile_covered_cases"] += 1
                    if controller_work is not None:
                        agg["controller_total_flops"] += controller_work["controller_total_flops"]

    aggregate_rows: list[dict[str, Any]] = []
    for dataset_name in dataset_keys:
        baseline = aggregate[(dataset_name, baseline_id)]
        for policy in policies:
            agg = aggregate[(dataset_name, policy)]
            cases = int(agg["cases"])
            if cases == 0:
                continue
            controller_total = agg["controller_total_flops"]
            controller_pending_cases = int(agg["controller_profile_pending_cases"])
            row = {
                "dataset": dataset_name,
                "policy_id": policy,
                "cases": cases,
                "correct": int(agg["correct"]),
                "accuracy_pct": 100.0 * int(agg["correct"]) / cases,
                "seconds_per_example": agg["latency_median_sum_s"] / cases,
                "speedup_pct": 100.0 * (baseline["latency_median_sum_s"] / agg["latency_median_sum_s"] - 1.0),
                "backbone_tflops_per_example": agg["backbone_flops"] / cases / TFLOP,
                "backbone_ratio_vs_baseline": agg["backbone_flops"] / baseline["backbone_flops"],
                "backbone_reduction_pct": 100.0 * (1.0 - agg["backbone_flops"] / baseline["backbone_flops"]),
                "lm_head_tflops_per_example": agg["lm_head_flops"] / cases / TFLOP,
                "total_with_lm_head_tflops_per_example": agg["total_with_lm_head_flops"] / cases / TFLOP,
                "total_with_lm_head_ratio_vs_baseline": agg["total_with_lm_head_flops"]
                / baseline["total_with_lm_head_flops"],
                "total_with_lm_head_reduction_pct": 100.0
                * (1.0 - agg["total_with_lm_head_flops"] / baseline["total_with_lm_head_flops"]),
                "controller_profile_expected_cases": int(agg["controller_profile_expected_cases"]),
                "controller_profile_covered_cases": int(agg["controller_profile_covered_cases"]),
                "controller_profile_pending_cases": controller_pending_cases,
                "controller_total_tflops_per_example": ""
                if controller_pending_cases
                else controller_total / cases / TFLOP,
            }
            aggregate_rows.append(row)

    _write_csv(output_dir / "percase_flops.csv", percase_rows)
    _write_csv(output_dir / "aggregate_flops.csv", aggregate_rows)
    summary = {
        "purpose": purpose,
        "records_path": str(records_path),
        "records_sha256": _hash_file(records_path),
        "baseline_id": baseline_id,
        "policy_ids": policies,
        "complete_paired_cases": len(complete_case_pairs),
        "expected_cases": expected_cases,
        "repeat_for_quality_and_flops": 1,
        "latency": "median over available ok repeats per dataset/example/policy; aggregate speedup uses paired time sums",
        "flop_convention": "1 MAC = 2 FLOPs; backbone excludes LM vocabulary head, KL, elementwise ops, cache I/O, and controller",
        "model_config": model_config.__dict__,
        "controller_profile_status": profile_status,
        "aggregate": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary, aggregate_rows)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "policy_id",
        "cases",
        "correct",
        "accuracy_pct",
        "speedup_pct",
        "backbone_tflops_per_example",
        "backbone_reduction_pct",
        "total_with_lm_head_tflops_per_example",
        "total_with_lm_head_reduction_pct",
        "controller_profile_expected_cases",
        "controller_profile_covered_cases",
        "controller_profile_pending_cases",
        "controller_total_tflops_per_example",
    ]
    lines = [
        "# Locking FLOP accounting",
        "",
        f"Purpose: {summary['purpose']}",
        "",
        "Convention: 1 MAC = 2 FLOPs. Backbone includes attention projections, QK/AV attention matmuls, "
        "and gated FFN only. LM head and controller/projector GEMMs are separate.",
        "",
        f"Complete paired cases: {summary['complete_paired_cases']}",
        f"Controller profile status: {summary['controller_profile_status']}",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_format_cell(row.get(h, "")) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze saved locking records with algorithmic FLOP accounting.")
    parser.add_argument("--records", required=True, help="JSONL records from a completed locking run")
    parser.add_argument("--model-config", required=True, help="Model config JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-id", default="B0")
    parser.add_argument("--purpose", default="locking-flop-accounting")
    parser.add_argument("--policy-ids", nargs="*", help="Policies to include; default is all policies in records")
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--profiles", help="Optional JSONL with controller_prefix_lengths keyed by dataset/example_id/policy_id")
    parser.add_argument("--projector-config", help="JSON for ProjectorConfig fields")
    parser.add_argument("--controller-config", help="JSON for ControllerConfig fields")
    args = parser.parse_args(argv)

    projector_config = _read_config_dataclass(args.projector_config, ProjectorConfig)
    controller_config = _read_config_dataclass(args.controller_config, ControllerConfig)
    analyze_records(
        args.records,
        load_model_config(args.model_config),
        args.output_dir,
        baseline_id=args.baseline_id,
        purpose=args.purpose,
        policy_ids=args.policy_ids,
        expected_cases=args.expected_cases,
        profiles_path=args.profiles,
        projector_config=projector_config,
        controller_config=controller_config,
    )


if __name__ == "__main__":
    main()
