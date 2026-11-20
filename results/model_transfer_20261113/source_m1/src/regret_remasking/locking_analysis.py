from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from regret_remasking.data import generated_answers_equivalent


INFRASTRUCTURE_SCORING_FAILURES = {"sandbox_unavailable", "sandbox_error"}

REQUIRED_OK_FIELDS = {
    "example_id",
    "dataset",
    "policy_id",
    "arm",
    "epsilon",
    "threshold",
    "repeat",
    "nfe",
    "forward_calls",
    "latency_s",
    "token_ids",
    "generation",
    "correct",
    "scoring_status",
    "final_mask_count",
    "immutable_token_violations",
    "prompt_length",
    "gen_length",
    "total_query_rows",
    "generated_row_updates_avoided",
    "frozen_count",
    "gate_time_s",
    "scorer_time_s",
    "status",
}


@dataclass(frozen=True)
class ExampleKey:
    dataset: str
    example_id: str


@dataclass(frozen=True)
class RecordKey:
    dataset: str
    example_id: str
    policy_id: str
    repeat: int


@dataclass(frozen=True)
class Comparison:
    dataset: str
    kind: str
    baseline_policy_id: str
    candidate_policy_id: str

    @property
    def comparison_id(self) -> str:
        return (
            f"{self.dataset}:{self.kind}:"
            f"{self.baseline_policy_id}:vs:{self.candidate_policy_id}"
        )


def analyze_run(
    run_dir: str | Path,
    output_dir: str | Path,
    bootstrap_reps: int = 10000,
    seed: int = 23,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = _load_json(run_path / "resolved_config.json")
    examples, policies, repeats, gen_length = _validate_config(config)
    records = _load_and_validate_records(
        run_path / "records.jsonl", examples, policies, repeats, gen_length
    )
    comparisons = _build_comparisons(examples, policies)
    if not comparisons:
        raise ValueError("no valid policy comparisons found")

    metric_rows: list[dict[str, Any]] = []
    comparison_summaries: list[dict[str, Any]] = []
    for comparison in comparisons:
        paired_rows = _paired_rows(comparison, examples, records, repeats)
        metrics = _comparison_metrics(paired_rows)
        intervals = _bootstrap_intervals(
            paired_rows,
            bootstrap_reps=bootstrap_reps,
            seed=_comparison_seed(seed, comparison.comparison_id),
        )
        counts = _comparison_counts(paired_rows)
        comparison_summaries.append(
            {
                "comparison_id": comparison.comparison_id,
                "dataset": comparison.dataset,
                "comparison_kind": comparison.kind,
                "baseline_policy_id": comparison.baseline_policy_id,
                "candidate_policy_id": comparison.candidate_policy_id,
                "counts": counts,
                "metrics": metrics,
                "intervals": intervals,
            }
        )
        for metric_name, point in metrics.items():
            interval = intervals.get(metric_name, {})
            metric_rows.append(
                {
                    "dataset": comparison.dataset,
                    "comparison_id": comparison.comparison_id,
                    "comparison_kind": comparison.kind,
                    "baseline_policy_id": comparison.baseline_policy_id,
                    "candidate_policy_id": comparison.candidate_policy_id,
                    "metric": metric_name,
                    "point": point,
                    "ci_low": interval.get("ci_low"),
                    "ci_high": interval.get("ci_high"),
                    "bootstrap_reps": bootstrap_reps,
                    "seed": seed,
                    **counts,
                }
            )

    _write_metrics_csv(output_path / "metrics.csv", metric_rows)
    summary = {
        "source_run": str(run_path),
        "examples": len(examples),
        "policies": len(policies),
        "repeats": repeats,
        "gen_length": gen_length,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_unit": "dataset/example_id paired clusters; repeats are not resampled independently",
        "scoring": (
            "offline analysis only; runner-provided correctness is used and generated code is never executed"
        ),
        "infrastructure_scoring_failures": sorted(INFRASTRUCTURE_SCORING_FAILURES),
        "comparisons": comparison_summaries,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int, int]:
    for key in ("examples", "policies", "repeats", "gen_length"):
        if key not in config:
            raise ValueError(f"resolved_config.json missing {key!r}")
    examples = config["examples"]
    policies_list = config["policies"]
    repeats = config["repeats"]
    gen_length = config["gen_length"]
    if not isinstance(examples, list) or not examples:
        raise ValueError("resolved_config.json examples must be a non-empty list")
    if not isinstance(policies_list, list) or not policies_list:
        raise ValueError("resolved_config.json policies must be a non-empty list")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("resolved_config.json repeats must be a positive integer")
    if not isinstance(gen_length, int) or gen_length < 1:
        raise ValueError("resolved_config.json gen_length must be a positive integer")

    seen_examples: set[ExampleKey] = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("each example must be an object")
        if "example_id" not in example or "dataset" not in example:
            raise ValueError("each example must include example_id and dataset")
        example_key = ExampleKey(str(example["dataset"]), str(example["example_id"]))
        if example_key in seen_examples:
            raise ValueError(f"duplicate example key in config: {example_key.dataset}/{example_key.example_id}")
        seen_examples.add(example_key)

    policies: dict[str, dict[str, Any]] = {}
    for policy in policies_list:
        if not isinstance(policy, dict):
            raise ValueError("each policy must be an object")
        if "policy_id" not in policy or "arm" not in policy:
            raise ValueError("each policy must include policy_id and arm")
        policy_id = str(policy["policy_id"])
        if policy_id in policies:
            raise ValueError(f"duplicate policy_id in config: {policy_id}")
        policies[policy_id] = policy
    return examples, policies, repeats, gen_length


def _load_and_validate_records(
    path: Path,
    examples: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    repeats: int,
    gen_length: int,
) -> dict[RecordKey, dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"missing required file: {path}")
    expected = {
        RecordKey(str(example["dataset"]), str(example["example_id"]), policy_id, repeat)
        for example in examples
        for policy_id in policies
        for repeat in range(1, repeats + 1)
    }
    records: dict[RecordKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = _record_key(record, line_number)
            if key in records:
                raise ValueError(
                    "duplicate record for "
                    f"{key.dataset}/{key.example_id}/{key.policy_id}/repeat-{key.repeat}"
                )
            if key not in expected:
                raise ValueError(
                    "unexpected record for "
                    f"{key.dataset}/{key.example_id}/{key.policy_id}/repeat-{key.repeat}"
                )
            _validate_ok_record(record, policies[key.policy_id], gen_length, line_number)
            records[key] = record
    missing = sorted(
        expected - set(records),
        key=lambda k: (k.dataset, k.example_id, k.policy_id, k.repeat),
    )
    if missing:
        first = missing[0]
        raise ValueError(
            "missing record for "
            f"{first.dataset}/{first.example_id}/{first.policy_id}/repeat-{first.repeat}"
        )
    return records


def _record_key(record: dict[str, Any], line_number: int) -> RecordKey:
    for key in ("dataset", "example_id", "policy_id", "repeat"):
        if key not in record:
            raise ValueError(f"records.jsonl line {line_number} missing {key!r}")
    repeat = record["repeat"]
    if not isinstance(repeat, int):
        raise ValueError(f"records.jsonl line {line_number} repeat must be an integer")
    return RecordKey(
        dataset=str(record["dataset"]),
        example_id=str(record["example_id"]),
        policy_id=str(record["policy_id"]),
        repeat=repeat,
    )


def _validate_ok_record(
    record: dict[str, Any],
    policy: dict[str, Any],
    gen_length: int,
    line_number: int,
) -> None:
    status = record.get("status")
    if status != "ok":
        message = record.get("error", "no error message")
        raise ValueError(f"records.jsonl line {line_number} is not a complete decode: {message}")
    missing = sorted(REQUIRED_OK_FIELDS - set(record))
    if missing:
        raise ValueError(f"records.jsonl line {line_number} missing required fields: {missing}")
    if str(record["arm"]) != str(policy["arm"]):
        raise ValueError(f"records.jsonl line {line_number} arm does not match config policy")
    for config_key in ("epsilon", "threshold"):
        if config_key in policy and record.get(config_key) != policy.get(config_key):
            raise ValueError(f"records.jsonl line {line_number} {config_key} does not match config")
    if record["gen_length"] != gen_length:
        raise ValueError(f"records.jsonl line {line_number} gen_length does not match config")
    token_ids = record["token_ids"]
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != gen_length
        or any(not isinstance(token, int) or token < 0 for token in token_ids)
    ):
        raise ValueError(f"records.jsonl line {line_number} token_ids must be full non-negative ids")
    if record["final_mask_count"] != 0:
        raise ValueError(f"records.jsonl line {line_number} has final masked tokens")
    if record["immutable_token_violations"] != 0:
        raise ValueError(f"records.jsonl line {line_number} has immutable token violations")
    if record["nfe"] != record["forward_calls"]:
        raise ValueError(f"records.jsonl line {line_number} nfe must equal forward_calls")
    if not isinstance(record["correct"], bool):
        raise ValueError(f"records.jsonl line {line_number} correct must be boolean")
    for key in ("nfe", "forward_calls", "prompt_length", "gen_length", "total_query_rows", "generated_row_updates_avoided", "frozen_count"):
        if not isinstance(record[key], int) or record[key] < 0:
            raise ValueError(f"records.jsonl line {line_number} {key} must be a non-negative integer")
    for key in ("total_projection_rows", "total_ffn_rows"):
        if key in record and (not isinstance(record[key], int) or record[key] < 0):
            raise ValueError(f"records.jsonl line {line_number} {key} must be a non-negative integer")
    if record["nfe"] > gen_length:
        raise ValueError(f"records.jsonl line {line_number} nfe exceeds gen_length")
    for key in ("latency_s", "gate_time_s", "scorer_time_s"):
        if not _finite_number(record[key]) or float(record[key]) < 0:
            raise ValueError(f"records.jsonl line {line_number} {key} must be finite")
    if float(record["latency_s"]) <= 0:
        raise ValueError(f"records.jsonl line {line_number} latency_s must be positive")
    if not isinstance(record["generation"], str):
        raise ValueError(f"records.jsonl line {line_number} generation must be a string")
    if not isinstance(record["scoring_status"], str) or not record["scoring_status"]:
        raise ValueError(f"records.jsonl line {line_number} scoring_status must be a non-empty string")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _build_comparisons(
    examples: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> list[Comparison]:
    datasets = sorted({str(example["dataset"]) for example in examples})
    baseline = _baseline_policy_id(policies)
    comparisons: list[Comparison] = []
    seen: set[Comparison] = set()
    if baseline is not None:
        for dataset in datasets:
            for policy_id in policies:
                if policy_id != baseline:
                    comparison = Comparison(dataset, "policy_vs_B0", baseline, policy_id)
                    comparisons.append(comparison)
                    seen.add(comparison)

    for dense, sparse in _matched_role_pairs(policies, "r_dense", "r_sparse", require_metadata=("epsilon",)):
        for dataset in datasets:
            comparison = Comparison(dataset, "R_dense_vs_R_sparse", dense, sparse)
            if comparison not in seen:
                comparisons.append(comparison)
                seen.add(comparison)

    for t_policy, c_policy in _matched_role_pairs(
        policies, "t_adaptive", "c_adaptive", prefer_metadata=("operating_point",)
    ):
        for dataset in datasets:
            comparison = Comparison(dataset, "T_adaptive_vs_C_adaptive", t_policy, c_policy)
            if comparison not in seen:
                comparisons.append(comparison)
                seen.add(comparison)
    return comparisons


def _baseline_policy_id(policies: dict[str, dict[str, Any]]) -> str | None:
    for policy_id, policy in policies.items():
        if _normal(policy_id) == "b0" or _normal(str(policy.get("arm", ""))) in {"b0", "baseline"}:
            return policy_id
    return None


def _matched_role_pairs(
    policies: dict[str, dict[str, Any]],
    left_role: str,
    right_role: str,
    require_metadata: tuple[str, ...] = (),
    prefer_metadata: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    left = [(policy_id, policy) for policy_id, policy in policies.items() if _policy_role(policy_id, policy) == left_role]
    right = [(policy_id, policy) for policy_id, policy in policies.items() if _policy_role(policy_id, policy) == right_role]
    pairs: list[tuple[str, str]] = []
    for left_id, left_policy in left:
        for right_id, right_policy in right:
            if any(left_policy.get(key) is None or left_policy.get(key) != right_policy.get(key) for key in require_metadata):
                continue
            if prefer_metadata and any(key in left_policy or key in right_policy for key in prefer_metadata):
                if all(
                    left_policy.get(key) is not None and left_policy.get(key) == right_policy.get(key)
                    for key in prefer_metadata
                ):
                    pairs.append((left_id, right_id))
                continue
            if require_metadata or _metadata_signature(left_policy, right_policy) is not None:
                pairs.append((left_id, right_id))
    return pairs


def _policy_role(policy_id: str, policy: dict[str, Any]) -> str | None:
    label = _normal(" ".join([policy_id, str(policy.get("arm", "")), str(policy.get("name", ""))]))
    if "r_dense" in label:
        return "r_dense"
    if "r_sparse" in label:
        return "r_sparse"
    if "c_adaptive" in label:
        return "c_adaptive"
    if "t_adaptive" in label:
        return "t_adaptive"
    return None


def _normal(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _metadata_signature(left: dict[str, Any], right: dict[str, Any]) -> tuple[tuple[str, Any], ...] | None:
    ignored = {"policy_id", "arm", "name", "description"}
    common = sorted((set(left) & set(right)) - ignored)
    comparable = []
    for key in common:
        left_value = left[key]
        right_value = right[key]
        if not isinstance(left_value, (str, int, float, bool, type(None))):
            continue
        if not isinstance(right_value, (str, int, float, bool, type(None))):
            continue
        if left_value != right_value:
            return None
        if left_value is not None:
            comparable.append((key, left_value))
    return tuple(comparable) if comparable else None


def _paired_rows(
    comparison: Comparison,
    examples: list[dict[str, Any]],
    records: dict[RecordKey, dict[str, Any]],
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        dataset = str(example["dataset"])
        if dataset != comparison.dataset:
            continue
        example_id = str(example["example_id"])
        baseline_repeats = [
            records[RecordKey(dataset, example_id, comparison.baseline_policy_id, repeat)]
            for repeat in range(1, repeats + 1)
        ]
        candidate_repeats = [
            records[RecordKey(dataset, example_id, comparison.candidate_policy_id, repeat)]
            for repeat in range(1, repeats + 1)
        ]
        baseline = baseline_repeats[0]
        candidate = candidate_repeats[0]
        scorable = not (
            baseline["scoring_status"] in INFRASTRUCTURE_SCORING_FAILURES
            or candidate["scoring_status"] in INFRASTRUCTURE_SCORING_FAILURES
        )
        answer_value_changed = not generated_answers_equivalent(
            baseline["generation"], candidate["generation"], dataset
        )
        rows.append(
            {
                "dataset": dataset,
                "example_id": example_id,
                "baseline_policy_id": comparison.baseline_policy_id,
                "candidate_policy_id": comparison.candidate_policy_id,
                "baseline_correct": bool(baseline["correct"]),
                "candidate_correct": bool(candidate["correct"]),
                "baseline_scoring_status": baseline["scoring_status"],
                "candidate_scoring_status": candidate["scoring_status"],
                "paired_scorable": scorable,
                "exact_sequence_changed": baseline["token_ids"] != candidate["token_ids"],
                "answer_value_changed": answer_value_changed,
                "baseline_nfe": int(baseline["nfe"]),
                "candidate_nfe": int(candidate["nfe"]),
                "baseline_forward_calls": int(baseline["forward_calls"]),
                "candidate_forward_calls": int(candidate["forward_calls"]),
                "gen_length": int(candidate["gen_length"]),
                "baseline_latency_median_s": statistics.median(
                    float(record["latency_s"]) for record in baseline_repeats
                ),
                "candidate_latency_median_s": statistics.median(
                    float(record["latency_s"]) for record in candidate_repeats
                ),
                "baseline_total_query_rows": int(baseline["total_query_rows"]),
                "candidate_total_query_rows": int(candidate["total_query_rows"]),
                "baseline_total_projection_rows": _row_counter(baseline, "total_projection_rows", "total_query_rows"),
                "candidate_total_projection_rows": _row_counter(candidate, "total_projection_rows", "total_query_rows"),
                "baseline_total_ffn_rows": _row_counter(baseline, "total_ffn_rows", "total_query_rows"),
                "candidate_total_ffn_rows": _row_counter(candidate, "total_ffn_rows", "total_query_rows"),
                "candidate_generated_row_updates_avoided": int(
                    candidate["generated_row_updates_avoided"]
                ),
                "candidate_frozen_count": int(candidate["frozen_count"]),
                "baseline_nondeterministic": _nondeterministic(baseline_repeats),
                "candidate_nondeterministic": _nondeterministic(candidate_repeats),
            }
        )
    return rows


def _row_counter(record: dict[str, Any], key: str, fallback_key: str) -> int:
    value = record.get(key)
    if value is None:
        return int(record[fallback_key])
    return int(value)


def _nondeterministic(records: list[dict[str, Any]]) -> bool:
    first = records[0]
    return any(
        record["token_ids"] != first["token_ids"]
        or record["generation"] != first["generation"]
        or record["correct"] != first["correct"]
        or record["scoring_status"] != first["scoring_status"]
        for record in records[1:]
    )


def _comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    scorable = [row for row in rows if row["paired_scorable"]]
    wall_clock_speedup = _ratio(
        _sum(row["baseline_latency_median_s"] for row in rows),
        _sum(row["candidate_latency_median_s"] for row in rows),
    )
    return {
        "examples": float(len(rows)),
        "quality_denominator": float(len(scorable)),
        "excluded_infrastructure_pairs": float(len(rows) - len(scorable)),
        "exact_sequence_change_rate": _mean(row["exact_sequence_changed"] for row in rows),
        "answer_value_change_rate": _mean(row["answer_value_changed"] for row in rows),
        "baseline_accuracy": _mean(row["baseline_correct"] for row in scorable),
        "candidate_accuracy": _mean(row["candidate_correct"] for row in scorable),
        "accuracy_delta": _mean(
            float(row["candidate_correct"]) - float(row["baseline_correct"]) for row in scorable
        ),
        "correctness_change_rate": _mean(
            row["baseline_correct"] != row["candidate_correct"] for row in scorable
        ),
        "correct_to_wrong_rate": _mean(
            row["baseline_correct"] and not row["candidate_correct"] for row in scorable
        ),
        "wrong_to_correct_rate": _mean(
            (not row["baseline_correct"]) and row["candidate_correct"] for row in scorable
        ),
        "baseline_median_latency_sum_s": _sum(row["baseline_latency_median_s"] for row in rows),
        "candidate_median_latency_sum_s": _sum(row["candidate_latency_median_s"] for row in rows),
        "wall_clock_speedup": wall_clock_speedup,
        "wall_clock_speedup_percent": (
            None if wall_clock_speedup is None else (wall_clock_speedup - 1.0) * 100.0
        ),
        "nfe_ratio": _ratio(
            _sum(row["candidate_nfe"] for row in rows),
            _sum(row["baseline_nfe"] for row in rows),
        ),
        "forward_call_ratio": _ratio(
            _sum(row["candidate_forward_calls"] for row in rows),
            _sum(row["baseline_forward_calls"] for row in rows),
        ),
        "generated_row_updates_avoided_fraction": _ratio(
            _sum(row["candidate_generated_row_updates_avoided"] for row in rows),
            _sum(row["gen_length"] * row["baseline_forward_calls"] for row in rows),
        ),
        "query_row_updates_avoided_fraction": _ratio(
            _sum(row["baseline_total_query_rows"] - row["candidate_total_query_rows"] for row in rows),
            _sum(row["baseline_total_query_rows"] for row in rows),
        ),
        "projection_row_updates_avoided_fraction": _ratio(
            _sum(row["baseline_total_projection_rows"] - row["candidate_total_projection_rows"] for row in rows),
            _sum(row["baseline_total_projection_rows"] for row in rows),
        ),
        "ffn_row_updates_avoided_fraction": _ratio(
            _sum(row["baseline_total_ffn_rows"] - row["candidate_total_ffn_rows"] for row in rows),
            _sum(row["baseline_total_ffn_rows"] for row in rows),
        ),
        "mean_candidate_frozen_count": _mean(row["candidate_frozen_count"] for row in rows),
        "nondeterministic_example_rate": _mean(
            row["baseline_nondeterministic"] or row["candidate_nondeterministic"]
            for row in rows
        ),
    }


def _comparison_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "paired_examples": len(rows),
        "quality_denominator": sum(1 for row in rows if row["paired_scorable"]),
        "excluded_infrastructure_pairs": sum(1 for row in rows if not row["paired_scorable"]),
        "baseline_infra_failures": sum(
            1 for row in rows if row["baseline_scoring_status"] in INFRASTRUCTURE_SCORING_FAILURES
        ),
        "candidate_infra_failures": sum(
            1 for row in rows if row["candidate_scoring_status"] in INFRASTRUCTURE_SCORING_FAILURES
        ),
        "baseline_timeouts": sum(1 for row in rows if row["baseline_scoring_status"] == "timeout"),
        "candidate_timeouts": sum(1 for row in rows if row["candidate_scoring_status"] == "timeout"),
        "nondeterministic_examples": sum(
            1
            for row in rows
            if row["baseline_nondeterministic"] or row["candidate_nondeterministic"]
        ),
    }


def _bootstrap_intervals(
    rows: list[dict[str, Any]],
    bootstrap_reps: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    if bootstrap_reps < 1:
        return {}
    rng = random.Random(seed)
    metric_names = list(_comparison_metrics(rows))
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(bootstrap_reps):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        sample_metrics = _comparison_metrics(sample)
        for name, value in sample_metrics.items():
            if value is not None:
                values[name].append(float(value))
    return {
        name: {"ci_low": _percentile(sample, 0.025), "ci_high": _percentile(sample, 0.975)}
        for name, sample in values.items()
    }


def _comparison_seed(seed: int, comparison_id: str) -> int:
    return seed + sum(ord(ch) for ch in comparison_id)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _sum(values: Iterable[Any]) -> float:
    return float(sum(float(value) for value in values))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    low_index = math.floor(position)
    high_index = math.ceil(position)
    if low_index == high_index:
        return ordered[low_index]
    low = ordered[low_index]
    high = ordered[high_index]
    return low + (high - low) * (position - low_index)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "comparison_id",
        "comparison_kind",
        "baseline_policy_id",
        "candidate_policy_id",
        "metric",
        "point",
        "ci_low",
        "ci_high",
        "bootstrap_reps",
        "seed",
        "paired_examples",
        "quality_denominator",
        "excluded_infrastructure_pairs",
        "baseline_infra_failures",
        "candidate_infra_failures",
        "baseline_timeouts",
        "candidate_timeouts",
        "nondeterministic_examples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Locking Policy Paired Analysis",
        "",
        f"- Source run: `{summary['source_run']}`",
        f"- Examples: {summary['examples']}",
        f"- Policies: {summary['policies']}",
        f"- Repeats: {summary['repeats']}",
        f"- Bootstrap reps: {summary['bootstrap_reps']}",
        f"- Bootstrap unit: {summary['bootstrap_unit']}",
        "",
        "| dataset | comparison | quality n | accuracy delta | speedup % | sequence change | answer change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in summary["comparisons"]:
        metrics = comparison["metrics"]
        lines.append(
            "| {dataset} | {comparison_id} | {quality_n} | {delta} | {speedup} | {sequence} | {answer} |".format(
                dataset=comparison["dataset"],
                comparison_id=comparison["comparison_id"],
                quality_n=comparison["counts"]["quality_denominator"],
                delta=_format_metric(metrics["accuracy_delta"]),
                speedup=_format_metric(metrics["wall_clock_speedup_percent"]),
                sequence=_format_metric(metrics["exact_sequence_change_rate"]),
                answer=_format_metric(metrics["answer_value_change_rate"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4g}"
