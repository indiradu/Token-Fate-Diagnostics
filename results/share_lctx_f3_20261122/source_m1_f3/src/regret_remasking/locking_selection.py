from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from regret_remasking.locking_analysis import _load_and_validate_records, _validate_config
from regret_remasking.locking_manifest import load_manifest, sha256_file


DEFAULT_NFE_TARGETS = (128.0, 64.0)
REFERENCE_WORK_TARGETS = (0.25, 0.50)


def select_operating_points(
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    nfe_targets: Iterable[float] = DEFAULT_NFE_TARGETS,
) -> dict[str, Any]:
    runs = [_load_selection_run(Path(run_dir)) for run_dir in run_dirs]
    if not runs:
        raise ValueError("at least one selection run directory is required")
    _validate_sweep_compatibility(runs)

    policies_by_id = _policies_by_id(runs)
    selected_policies: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "selection": "development-only operating-point selection; correctness is never used",
        "input_run_dirs": [str(run["run_dir"]) for run in runs],
        "input_file_hashes": _input_file_hashes(runs),
        "panel": {
            "examples": len(runs[0]["examples"]),
            "gen_length": runs[0]["gen_length"],
            "block_length": runs[0]["block_length"],
            "checkpoint": runs[0]["config"].get("checkpoint"),
            "checkpoint_index_sha256": runs[0]["config"].get("checkpoint_index_sha256"),
            "source_sha256": runs[0]["config"].get("source", {}).get("sha256"),
        },
        "targets": {
            "reference_work_fractions": list(REFERENCE_WORK_TARGETS),
            "trace_nfe": [float(target) for target in nfe_targets],
        },
    }

    baseline = _find_first_policy(policies_by_id.values(), "B0")
    if baseline is not None:
        selected_policies.append(_copy_policy(baseline))

    e1_summary, dense_policies = _select_representation_policies(runs, policies_by_id)
    selected_policies.extend(dense_policies)
    summary["representation_lock"] = e1_summary

    trace_summary, trace_policies = _select_trace_policies(
        runs, policies_by_id, [float(target) for target in nfe_targets]
    )
    selected_policies.extend(trace_policies)
    summary["trace_lock"] = trace_summary

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_frozen_json(output / "policies.json", selected_policies)
    _write_frozen_json(output / "selection_summary.json", summary)
    return summary


def _load_selection_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.json"
    records_path = run_dir / "records.jsonl"
    if not config_path.exists() or not records_path.exists():
        raise ValueError(f"{run_dir} must contain resolved_config.json and records.jsonl")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    examples, policies, repeats, gen_length = _validate_config(config)
    _validate_selection_manifests(run_dir, config, examples)
    records = _load_and_validate_records(records_path, examples, policies, repeats, gen_length)
    return {
        "run_dir": run_dir.resolve(),
        "config_path": config_path.resolve(),
        "records_path": records_path.resolve(),
        "config": config,
        "examples": examples,
        "policies": policies,
        "repeats": repeats,
        "gen_length": gen_length,
        "block_length": config.get("block_length"),
        "records": records,
    }


def _validate_selection_manifests(
    run_dir: Path,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
) -> None:
    manifest_hashes = config.get("manifest_sha256")
    if not isinstance(manifest_hashes, dict) or not manifest_hashes:
        raise ValueError(f"{run_dir} missing manifest_sha256 provenance")
    manifest_entries: list[dict[str, Any]] = []
    for manifest_name, saved_hash in manifest_hashes.items():
        manifest_path = Path(manifest_name)
        if not manifest_path.exists():
            raise ValueError(f"selection manifest not found: {manifest_path}")
        actual_hash = sha256_file(manifest_path)
        if actual_hash != saved_hash:
            raise ValueError(f"manifest hash drift for {manifest_path}")
        manifest_entries.extend(load_manifest(manifest_path, expected_split="selection"))
    manifest_examples = [
        {"example_id": entry["example_id"], "dataset": entry["dataset"]}
        for entry in manifest_entries
    ]
    config_examples = [
        {"example_id": example["example_id"], "dataset": example["dataset"]}
        for example in examples
    ]
    if manifest_examples != config_examples:
        raise ValueError(
            "selection run examples must exactly cover the saved selection manifests; "
            "hidden limits or reordered panels are not allowed"
        )


def _validate_sweep_compatibility(runs: list[dict[str, Any]]) -> None:
    reference = runs[0]
    reference_config = reference["config"]
    expected_examples = reference["examples"]
    expected_gen_length = reference["gen_length"]
    expected_block_length = reference["block_length"]
    expected_checkpoint = reference_config.get("checkpoint")
    expected_checkpoint_hash = reference_config.get("checkpoint_index_sha256")
    expected_source_hash = reference_config.get("source", {}).get("sha256")
    for run in runs[1:]:
        config = run["config"]
        if run["examples"] != expected_examples:
            raise ValueError("selection sweeps use different example panels")
        if run["gen_length"] != expected_gen_length or run["block_length"] != expected_block_length:
            raise ValueError("selection sweeps use different generation or block lengths")
        if (
            config.get("checkpoint") != expected_checkpoint
            or config.get("checkpoint_index_sha256") != expected_checkpoint_hash
        ):
            raise ValueError("selection sweeps use different checkpoint provenance")
        if config.get("source", {}).get("sha256") != expected_source_hash:
            raise ValueError("selection sweeps use different source provenance")


def _policies_by_id(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for run in runs:
        for policy_id, policy in run["policies"].items():
            existing = policies.get(policy_id)
            if existing is not None and existing != policy:
                raise ValueError(f"policy_id {policy_id!r} has conflicting definitions across sweeps")
            policies[policy_id] = policy
    return policies


def _select_representation_policies(
    runs: list[dict[str, Any]],
    policies_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dense_scores = _r_dense_work_scores(runs)
    if not dense_scores:
        raise ValueError("selection requires at least one R-dense policy")
    epsilon_by_policy = {
        policy_id: float(policies_by_id[policy_id]["epsilon"])
        for policy_id in dense_scores
        if policies_by_id[policy_id].get("epsilon") is not None
    }
    zero_policy_ids = [
        policy_id for policy_id, epsilon in epsilon_by_policy.items() if epsilon == 0.0
    ]
    if not zero_policy_ids:
        raise ValueError("selection requires an R-dense epsilon 0 policy")

    selected_dense_ids = [_tie_smallest_epsilon(zero_policy_ids, epsilon_by_policy)]
    positive_ids = [
        policy_id for policy_id, epsilon in epsilon_by_policy.items() if epsilon > 0.0
    ]
    target_entries: list[dict[str, Any]] = []
    used_eps = {0.0}
    for target in REFERENCE_WORK_TARGETS:
        candidates = [
            policy_id
            for policy_id in positive_ids
            if epsilon_by_policy[policy_id] not in used_eps
        ]
        if not candidates:
            target_entries.append(
                {"target": target, "selected_policy_id": None, "actual": None, "matched": False}
            )
            continue
        selected = min(
            candidates,
            key=lambda policy_id: (
                abs(dense_scores[policy_id] - target),
                epsilon_by_policy[policy_id],
            ),
        )
        used_eps.add(epsilon_by_policy[selected])
        selected_dense_ids.append(selected)
        target_entries.append(
            {
                "target": target,
                "selected_policy_id": selected,
                "epsilon": epsilon_by_policy[selected],
                "actual": dense_scores[selected],
                "matched": True,
            }
        )

    selected_policies: list[dict[str, Any]] = []
    selected_entries = []
    for dense_id in selected_dense_ids:
        dense_policy = _copy_policy(policies_by_id[dense_id])
        selected_policies.append(dense_policy)
        sparse_policy = _paired_sparse_policy(dense_policy)
        selected_policies.append(sparse_policy)
        selected_entries.append(
            {
                "dense_policy_id": dense_policy["policy_id"],
                "sparse_policy_id": sparse_policy["policy_id"],
                "epsilon": dense_policy.get("epsilon"),
                "actual_reference_work_fraction": dense_scores[dense_id],
            }
        )
    return {
        "scoring": "mean generated_row_updates_avoided / gen_length^2 by example",
        "targets": target_entries,
        "selected": selected_entries,
    }, selected_policies


def _r_dense_work_scores(runs: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        gen_length = int(run["gen_length"])
        for key, record in run["records"].items():
            policy = run["policies"][key.policy_id]
            if _normal(policy.get("arm")) != "r_dense":
                continue
            example_key = (key.dataset, key.example_id)
            totals[key.policy_id][example_key].append(
                float(record["generated_row_updates_avoided"]) / float(gen_length * gen_length)
            )
    return {
        policy_id: _mean(_mean(values) for values in by_example.values())
        for policy_id, by_example in totals.items()
    }


def _paired_sparse_policy(dense_policy: dict[str, Any]) -> dict[str, Any]:
    epsilon = dense_policy.get("epsilon")
    if epsilon is None:
        raise ValueError("dense policy must declare epsilon before pairing its sparse arm")
    policy = {
        "policy_id": f"R-sparse-eps-{float(epsilon):g}",
        "arm": "R-sparse",
        "epsilon": epsilon,
        "threshold": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
    }
    for key, value in dense_policy.items():
        if key.startswith("checkpoint"):
            policy[key] = None
        elif key not in policy and key not in {"policy_id", "arm"}:
            policy[key] = value
    return policy


def _select_trace_policies(
    runs: list[dict[str, Any]],
    policies_by_id: dict[str, dict[str, Any]],
    nfe_targets: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t_adaptive = _adaptive_nfe_scores(runs, "T-adaptive")
    c_adaptive = _adaptive_nfe_scores(runs, "C-adaptive")
    selected: list[dict[str, Any]] = []
    summary_targets: list[dict[str, Any]] = []

    t_fixed = _find_first_policy(policies_by_id.values(), "T-fixed")
    if t_fixed is not None:
        selected.append(_copy_policy(t_fixed))

    if not t_adaptive:
        raise ValueError("selection requires a T-adaptive trace grid")
    if not c_adaptive:
        raise ValueError("selection requires a C-adaptive confidence grid")

    used_t_ids: set[str] = set()
    for index, target in enumerate(nfe_targets, start=1):
        candidates = [policy_id for policy_id in t_adaptive if policy_id not in used_t_ids]
        if not candidates:
            summary_targets.append(
                {
                    "target_nfe": target,
                    "operating_point": f"trace-op-{index}",
                    "t_policy_id": None,
                    "c_policy_id": None,
                    "matched_within_5pct": False,
                }
            )
            continue
        t_policy_id = min(
            candidates,
            key=lambda policy_id: (
                abs(t_adaptive[policy_id] - target),
                -float(policies_by_id[policy_id].get("threshold") or 0.0),
            ),
        )
        used_t_ids.add(t_policy_id)
        c_policy_id = min(
            c_adaptive,
            key=lambda policy_id: (
                abs(c_adaptive[policy_id] - t_adaptive[t_policy_id]),
                -float(policies_by_id[policy_id].get("threshold") or 0.0),
            ),
        )
        within = _within_five_percent(c_adaptive[c_policy_id], t_adaptive[t_policy_id])
        operating_point = f"trace-op-{index}"
        t_policy = _copy_policy(policies_by_id[t_policy_id])
        c_policy = _copy_policy(policies_by_id[c_policy_id])
        selected_c_policy_id = f"{c_policy_id}-{operating_point}"
        t_policy["operating_point"] = operating_point
        c_policy["source_policy_id"] = c_policy_id
        c_policy["policy_id"] = selected_c_policy_id
        c_policy["operating_point"] = operating_point
        selected.extend([t_policy, c_policy])
        summary_targets.append(
            {
                "target_nfe": target,
                "operating_point": operating_point,
                "t_policy_id": t_policy_id,
                "t_threshold": t_policy.get("threshold"),
                "t_mean_nfe": t_adaptive[t_policy_id],
                "c_policy_id": selected_c_policy_id,
                "c_source_policy_id": c_policy_id,
                "c_threshold": c_policy.get("threshold"),
                "c_mean_nfe": c_adaptive[c_policy_id],
                "matched_within_5pct": within,
            }
        )
    return {
        "scoring": "mean NFE only; correctness is ignored",
        "targets": summary_targets,
    }, selected


def _adaptive_nfe_scores(runs: list[dict[str, Any]], arm: str) -> dict[str, float]:
    by_policy: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for key, record in run["records"].items():
            policy = run["policies"][key.policy_id]
            if _normal(policy.get("arm")) != _normal(arm):
                continue
            by_policy[key.policy_id][(key.dataset, key.example_id)].append(float(record["nfe"]))
    return {
        policy_id: _mean(_mean(values) for values in by_example.values())
        for policy_id, by_example in by_policy.items()
    }


def _within_five_percent(candidate: float, reference: float) -> bool:
    if reference == 0:
        return candidate == 0
    return abs(candidate - reference) / abs(reference) <= 0.05


def _find_first_policy(policies: Iterable[dict[str, Any]], arm: str) -> dict[str, Any] | None:
    for policy in policies:
        if _normal(policy.get("arm")) == _normal(arm):
            return policy
    return None


def _tie_smallest_epsilon(policy_ids: list[str], epsilon_by_policy: dict[str, float]) -> str:
    return min(policy_ids, key=lambda policy_id: (epsilon_by_policy[policy_id], policy_id))


def _copy_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(policy, sort_keys=True))


def _normal(value: Any) -> str:
    return str(value).lower().replace("-", "_").replace(" ", "_")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot average an empty sequence")
    return sum(items) / len(items)


def _input_file_hashes(runs: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for run in runs:
        output[str(run["config_path"])] = sha256_file(run["config_path"])
        output[str(run["records_path"])] = sha256_file(run["records_path"])
        for manifest_path in run["config"]["manifest_sha256"]:
            output[str(Path(manifest_path).resolve())] = sha256_file(manifest_path)
    return output


def _write_frozen_json(path: Path, data: Any) -> str:
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"refusing to overwrite nonidentical frozen output: {path}")
    else:
        path.write_text(text, encoding="utf-8")
    return digest
