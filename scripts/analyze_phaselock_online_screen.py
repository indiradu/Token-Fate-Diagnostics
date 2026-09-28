#!/usr/bin/env python3
"""Summarize matched live runs without inflating sample size with repeats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def summarize(records: list[dict], *, draws: int = 10000, seed: int = 271) -> dict:
    records = [r for r in records if r["repeat"] >= 0]
    methods = sorted({r["method"] for r in records})
    if not {"dense", "surelock", "tracelock", "h2"} <= set(methods):
        raise ValueError("required comparison arms are missing")
    timing = {}
    accuracy = {}
    nfe = {}
    controller = {}
    for row in records:
        method = row["method"]
        batch = (row["dataset"], row["batch_id"])
        timing.setdefault(method, {}).setdefault(batch, []).append(row["online_latency_s"])
        nfe.setdefault(method, {}).setdefault(batch, []).append(row["nfe"])
        if "controller_latency_s" in row:
            controller.setdefault(method, []).append(row["controller_latency_s"])
        for eid, correct in zip(row["example_ids"], row["correct"]):
            # First measured run is the declared single-sample prediction.
            accuracy.setdefault(method, {}).setdefault((row["dataset"], eid), int(correct))
    keys = sorted(accuracy["dense"])
    batches = sorted(timing["dense"])
    for method in methods:
        if sorted(accuracy[method]) != keys or sorted(timing[method]) != batches:
            raise ValueError(f"unpaired or incomplete run: {method}")
        for batch in batches:
            if len(timing[method][batch]) != len(timing["dense"][batch]):
                raise ValueError(f"unbalanced repeat count: {method}, {batch}")
    rng = np.random.default_rng(seed)
    # Stratify examples/batches by dataset. Repeated timings are collapsed first.
    def stratified_indices(items):
        groups = [[i for i, k in enumerate(items) if k[0] == d] for d in sorted({k[0] for k in items})]
        return np.concatenate([rng.choice(group, size=(draws, len(group)), replace=True) for group in groups], axis=1)
    example_draws = stratified_indices(keys)
    batch_draws = stratified_indices(batches)
    results = {}
    for method in methods:
        scores = np.array([accuracy[method][k] for k in keys])
        seconds = np.array([np.median(timing[method][b]) for b in batches])
        entry = {
            "examples": len(keys), "accuracy_screening": float(scores.mean()),
            "per_dataset_accuracy_screening": {d: float(np.mean([accuracy[method][k] for k in keys if k[0] == d])) for d in sorted({k[0] for k in keys})},
            "sum_median_batch_seconds": float(seconds.sum()),
            "mean_batch_nfe": float(np.mean([np.median(nfe[method][b]) for b in batches])),
            "mean_reported_controller_seconds": float(np.mean(controller[method])) if method in controller else None,
            "deterministic": all(r.get("deterministic_vs_first", True) for r in records if r["method"] == method),
            "comparisons": {},
        }
        for reference in ("surelock", "tracelock", "h2"):
            ref_scores = np.array([accuracy[reference][k] for k in keys])
            ref_seconds = np.array([np.median(timing[reference][b]) for b in batches])
            difference = scores - ref_scores
            gain_draws = 1. - seconds[batch_draws].sum(axis=1) / ref_seconds[batch_draws].sum(axis=1)
            entry["comparisons"][reference] = {
                "accuracy_difference_pp": float(difference.mean() * 100),
                "exploratory_bootstrap_accuracy_difference_ci95_pp": (np.quantile(difference[example_draws].mean(axis=1), [.025, .975]) * 100).tolist(),
                "regressions": int(((scores == 0) & (ref_scores == 1)).sum()),
                "improvements": int(((scores == 1) & (ref_scores == 0)).sum()),
                "runtime_reduction": float(1. - seconds.sum() / ref_seconds.sum()),
                "speedup": float(ref_seconds.sum() / seconds.sum()),
                "exploratory_bootstrap_runtime_reduction_ci95": np.quantile(gain_draws, [.025, .975]).tolist(),
            }
        results[method] = entry
    return {"stage": "development_only", "quality_metric_warning": "MATH500 uses legacy numeric proxy pending symbolic re-scoring. Small-pilot bootstrap intervals are descriptive, not a noninferiority test (zero observed disagreements can give degenerate intervals).", "selection": None,
            "selection_note": "No final winner selected on reused small development data; train/calibrate/freeze and confirm independently.", "methods": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = []
    configs = []
    for directory in args.run_dirs:
        if not (directory / "completion.json").exists():
            raise ValueError(f"run has not completed: {directory}")
        configs.append(json.loads((directory / "run_config.json").read_text()))
        records.extend(json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines())
    for key in ("gen_length", "steps", "block_length", "batch_size", "checkpoint_sha256", "projection_sha256", "policies", "gpu", "source_sha256", "model_name", "model_config_commit", "torch", "transformers", "host"):
        if any(c[key] != configs[0][key] for c in configs[1:]):
            raise ValueError(f"mismatched run configs: {key}")
    result = summarize(records)
    symbolic = all("Math-Verify 0.9.0" in c.get("scoring", "") for c in configs)
    if symbolic:
        result["quality_metric_warning"] = "MATH500 re-scored with pinned symbolic grader; GSM8K/HumanEval unchanged. Tiny reused-development sample; bootstrap intervals are descriptive, not noninferiority certification."
    result["run_dirs"] = [str(p) for p in args.run_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    label = "Development accuracy" if symbolic else "Accuracy proxy"
    lines = ["# H2 live controller screen", "", result["quality_metric_warning"], "",
             f"| Method | {label} | Mean NFE | vs SureLock speedup | vs TraceLock speedup |", "|---|---:|---:|---:|---:|"]
    for name, row in result["methods"].items():
        lines.append(f"| {name} | {row['accuracy_screening']:.1%} | {row['mean_batch_nfe']:.2f} | {row['comparisons']['surelock']['speedup']:.3f}x | {row['comparisons']['tracelock']['speedup']:.3f}x |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
