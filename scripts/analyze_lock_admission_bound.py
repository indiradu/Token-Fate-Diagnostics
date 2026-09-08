#!/usr/bin/env python3
"""Phase 0: can ANY lock-admission rule achieve representation settlement?

CPU-only, no new decode. Prior compute-lock methods (SureLock, LESS, TACG)
lock an *already unmasked* position once its posterior looks stable, so their
lock step s' is generally later than LLaDA's transfer step. Premise A as
currently measured baselines drift at the transfer step, which overstates the
case against them. This script asks the predicate-free upper-limit question:
across all possible lock steps, what is the least post-lock representation
movement any admission rule could achieve?

Method. `drift_time_rows.csv` stores d_t = ||h_t - b|| / ||b|| against the
post-commit base b. For two observed steps s' and t the reverse triangle
inequality gives a LOWER bound on post-lock movement in the same units:

    ||h_t - h_s'|| / ||b||  >=  | d_t - d_s' |

so LB(s') = max_{t > s'} |d_t - d_s'| lower-bounds the movement a reference
frozen at s' would fail to see. Locking at the final observed step makes LB
vanish while saving nothing, so the bound is reported against a savings
budget f = (t_last - s') / (t_last - b): the fraction of the token's
post-commit horizon that stays frozen. oracle(f) = min{ LB(s') : f(s') >= f }
is then the best any predicate could do at that budget, and it is a prefix
minimum because f decreases as s' advances.

Direction of evidence. This bound can only CONVICT, never acquit. A large
oracle(f) proves movement continues past every admissible lock and no
predicate can prevent it. A small oracle(f) proves nothing -- the true
movement may be far larger than |d_t - d_s'|. Establishing settlement needs
the consecutive-step upper bound, which requires the Phase 1 re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EOS_ID = 126081
EOT_ID = 126348

# Budget grid: fraction of the post-commit horizon that remains frozen.
BUDGETS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
# Drift thresholds reused from the premise-A reporting in results/README.md.
THRESHOLDS = [0.1, 0.25, 0.5, 1.0]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def oracle_curve(steps: np.ndarray, drifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix-minimum lower bound on post-lock movement, and its budget grid.

    steps/drifts include the base step at index 0 with drift 0.
    Returns (budgets, prefix_min_LB) over candidate locks 0..n-2; the final
    step is never a candidate because it leaves no horizon to freeze.
    """
    n = steps.size
    if n < 2:
        return np.empty(0), np.empty(0)
    # Suffix extrema over strictly later steps.
    suffix_max = np.empty(n)
    suffix_min = np.empty(n)
    suffix_max[-1] = -np.inf
    suffix_min[-1] = np.inf
    for i in range(n - 2, -1, -1):
        suffix_max[i] = max(drifts[i + 1], suffix_max[i + 1])
        suffix_min[i] = min(drifts[i + 1], suffix_min[i + 1])
    candidates = np.arange(n - 1)
    lb = np.maximum(suffix_max[candidates] - drifts[candidates], drifts[candidates] - suffix_min[candidates])
    span = steps[-1] - steps[0]
    budgets = (steps[-1] - steps[candidates]) / span if span > 0 else np.ones(candidates.size)
    return budgets, np.minimum.accumulate(lb)


def token_bounds(time_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (example, position, layer) with oracle(f) at each budget."""
    records: list[dict[str, Any]] = []
    grouped = time_rows.sort_values("global_step").groupby(["example_id", "position", "layer"], sort=False)
    for (example_id, position, layer), grp in grouped:
        base_step = int(grp["base_step"].iloc[0])
        steps = np.concatenate([[base_step], grp["global_step"].to_numpy()])
        drifts = np.concatenate([[0.0], grp["relative_l2_drift"].to_numpy()])
        budgets, prefix_lb = oracle_curve(steps, drifts)
        if budgets.size == 0:
            continue
        row: dict[str, Any] = {
            "example_id": example_id,
            "position": int(position),
            "layer": int(layer),
            "observed_steps": int(grp.shape[0]),
            "max_drift_from_commit": float(np.max(drifts)),
        }
        for f in BUDGETS:
            eligible = np.nonzero(budgets >= f)[0]
            # budgets is non-increasing, so eligible is a prefix; take its last index.
            row[f"oracle_lb_f{f}"] = float(prefix_lb[eligible[-1]]) if eligible.size else np.nan
        records.append(row)
    return pd.DataFrame(records)


def cluster_bootstrap(
    values_by_example: dict[str, np.ndarray], reps: int, seed: int, stat: str = "median"
) -> tuple[float, float, float]:
    example_ids = list(values_by_example)
    fn = np.median if stat == "median" else np.mean
    pooled = np.concatenate([values_by_example[e] for e in example_ids])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(fn(pooled))
    if len(example_ids) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for rep in range(reps):
        sampled = rng.choice(len(example_ids), size=len(example_ids), replace=True)
        draw = np.concatenate([values_by_example[example_ids[i]] for i in sampled])
        draw = draw[np.isfinite(draw)]
        stats[rep] = fn(draw) if draw.size else np.nan
    low, high = np.nanpercentile(stats, [2.5, 97.5])
    return point, float(low), float(high)


def summarize(bounds: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer in sorted(bounds["layer"].unique()) + ["mean"]:
        if layer == "mean":
            sub = bounds.groupby(["example_id", "position"], as_index=False).mean(numeric_only=True)
        else:
            sub = bounds[bounds["layer"] == layer]
        for f in BUDGETS:
            col = f"oracle_lb_f{f}"
            by_example = {e: g[col].to_numpy(dtype=float) for e, g in sub.groupby("example_id")}
            median, low, high = cluster_bootstrap(by_example, reps, seed, "median")
            values = sub[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row = {
                "layer": layer,
                "frozen_horizon_fraction": f,
                "tokens": int(values.size),
                "median_oracle_lb": median,
                "ci_low": low,
                "ci_high": high,
            }
            for thr in THRESHOLDS:
                row[f"frac_gt_{thr}"] = float(np.mean(values > thr)) if values.size else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_run(run_dir: Path, exclude_eos: bool) -> pd.DataFrame:
    time_rows = pd.read_csv(run_dir / "drift_time_rows.csv")
    if time_rows.empty:
        raise ValueError(
            f"{run_dir}/drift_time_rows.csv is empty -- the run used --skip-drift-time-rows, "
            "so it carries no trajectories and cannot support the Phase 0 bound."
        )
    if exclude_eos:
        drift_path = run_dir / "committed_token_drift.csv"
        if not drift_path.exists():
            raise FileNotFoundError(f"need {drift_path} to identify EOS/EOT tokens")
        tokens = pd.read_csv(drift_path)[["example_id", "position", "selected_token", "final_token"]]
        before = time_rows.shape[0]
        time_rows = time_rows.merge(tokens, on=["example_id", "position"], how="inner")
        padding = time_rows["selected_token"].isin([EOS_ID, EOT_ID]) | time_rows["final_token"].isin(
            [EOS_ID, EOT_ID]
        )
        time_rows = time_rows[~padding]
        print(
            f"[phase0] {run_dir.name}: {before} rows -> {time_rows.shape[0]} after EOS/EOT exclusion",
            flush=True,
        )
    return time_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predicate-free lower bound on post-lock representation movement (Phase 0)."
    )
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--keep-eos-eot",
        action="store_true",
        help="Keep EOS/EOT padding. Off by default: padding passes stability predicates trivially "
        "while drifting ~2x body tokens, so including it manufactures the convicting result.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_summaries = []
    all_bounds = []
    for spec in args.run:
        label, _, run_dir = spec.partition("=")
        if not run_dir:
            raise ValueError(f"--run must be LABEL=RUN_DIR, got: {spec}")
        time_rows = load_run(Path(run_dir), exclude_eos=not args.keep_eos_eot)
        bounds = token_bounds(time_rows)
        bounds["run"] = label
        all_bounds.append(bounds)
        summary = summarize(bounds, args.bootstrap_reps, args.seed)
        summary.insert(0, "run", label)
        per_run_summaries.append(summary)
        print(f"[phase0] {label}: {bounds.shape[0]} (token, layer) trajectories", flush=True)

    bounds_all = pd.concat(all_bounds, ignore_index=True)
    summary_all = pd.concat(per_run_summaries, ignore_index=True)
    pooled = summarize(bounds_all, args.bootstrap_reps, args.seed)
    pooled.insert(0, "run", "POOLED")
    summary_all = pd.concat([summary_all, pooled], ignore_index=True)

    bounds_all.to_csv(output_dir / "token_oracle_bounds.csv", index=False)
    summary_all.to_csv(output_dir / "oracle_bound_summary.csv", index=False)

    mean_pooled = pooled[pooled["layer"] == "mean"]
    summary_json = {
        "runs": args.run,
        "eos_eot_excluded": not args.keep_eos_eot,
        "bootstrap_reps": args.bootstrap_reps,
        "token_layer_trajectories": int(bounds_all.shape[0]),
        "pooled_layer_mean": mean_pooled.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    lines = [
        "# Phase 0: predicate-free bound on post-lock representation movement",
        "",
        "Lower bound `oracle(f) = min over lock steps s' with frozen-horizon >= f of",
        "max_{t>s'} |d_t - d_s'|`, from the reverse triangle inequality on the drift",
        "trajectories already on disk. No new decode. `f = 1.0` is locking at commit",
        "(reproduces premise A); smaller `f` delays the lock and saves less compute.",
        "",
        "**This bound can only convict, not acquit.** A large value proves movement",
        "continues past every admissible lock, so no admission predicate -- SureLock-,",
        "LESS-, or TraceLock-style -- can deliver settlement at that savings budget. A",
        "small value proves nothing; settlement needs the Phase 1 upper bound.",
        "",
        f"EOS/EOT padding excluded: {not args.keep_eos_eot}.",
        "",
        "## Pooled, mean over layers",
        "",
        table_text(mean_pooled.drop(columns=["run", "layer"])),
        "",
        "## Per run and layer",
        "",
        table_text(summary_all),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase0] wrote {output_dir}/summary.md", flush=True)


if __name__ == "__main__":
    main()
