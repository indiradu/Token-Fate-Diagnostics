#!/usr/bin/env python3
"""Phase 1: do prior work's lock-admission rules satisfy the PhaseLock premises?

CPU-only. Reads a settlement-fate run's `post_commit_posterior.csv` and
`drift_time_rows.csv` (both require the post-commit logging added 2026-09-01)
and asks, for each published admission rule, whether the representation is
still moving *after that rule says it is safe to lock*.

Why this is not premise A. Premise A baselines drift at LLaDA's transfer step.
SureLock, LESS, and TACG lock an *already unmasked* position once its posterior
looks stable, so their lock step s' is generally later than the transfer step
and they are entitled to the settling that happens in between. Evaluating them
at the transfer step would be a strawman.

Bounds. In units of ||b|| (the same units as `relative_l2_drift`), with lock at
observed index i and last observed index T:

    lower:  max_{j>i} | d_j - d_i |                      (reverse triangle)
    upper:  sum_{j>i} step_l2_drift_j * row_norm_{j-1} / base_norm   (triangle)

Both are also reported renormalized to the frozen reference's own scale,
||h_i|| / ||b|| = row_norm_i / base_norm, which is the quantity a cache
staleness budget would actually be written against. The lower bound convicts
(movement provably continued); the upper bound acquits (movement provably
bounded). Reporting only one of them would let the conclusion be chosen.

Common compute axis. Rules are not compared at equal threshold -- thresholds
are not commensurable across families -- but at equal *savings*: the fraction
of post-commit token-steps left frozen, which is the row-update proxy used by
`simulate_compute_utility.py`. Differences at matched savings are differences
in *which* tokens a rule picks, not in how aggressive it is.

Threshold discipline. With `--train-run`, each family's threshold is chosen on
the training run to hit a savings target and then frozen for the evaluation
run. Sweeping thresholds on the reported run would tune a critique of someone
else's method against its own outcome. Where no same-regime training run
exists, `--self-split-train-frac` holds out a fraction of the run's own
examples instead; transferring gen-64 thresholds to a gen-256 run is not
equivalent, because the KL and JSD distributions shift with generation length.
`savings_drift_vs_train` in `frozen_threshold_eval.csv` reports how far the
realized savings moved from the target a threshold was picked for -- a large
drift means the rules are no longer being compared at matched savings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

EOS_ID = 126081
EOT_ID = 126348

SAVINGS_TARGETS = [0.5, 0.4, 0.3, 0.2, 0.1]
DRIFT_THRESHOLDS = [0.1, 0.25, 0.5]
# LESS-style rules require top-1 to agree with the locked identity for this
# many consecutive observed steps before admitting the lock.
DEFAULT_PERSISTENCE = 2


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


# --------------------------------------------------------------------------
# Admission rules. Each returns a boolean array over a token's observed steps:
# True where the rule would admit the lock at that step. The lock is taken at
# the FIRST True and is irreversible, which is what makes it a compute saving.
# --------------------------------------------------------------------------


def rule_earliest(post: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """Lock at the earliest step any rule here *can* lock.

    NOT lock-at-commit. Post-commit posteriors are only logged from
    `base_step + 1`, because `base_step` is where the reference row is
    captured, so no rule in this file can fire at the commit step itself. That
    one step of delay matters a lot: on the gen-64 eval run, max drift from the
    commit-step reference (premise A) is median 0.442 with 91.6% above 0.25,
    while locking one step later gives 0.266 and 55.0%. True lock-at-commit is
    premise A / Phase 0 at `f = 1.0`; this row is the floor of *this* sweep.
    """
    fires = np.zeros(post["global_step"].size, dtype=bool)
    fires[0] = True
    return fires


def rule_surelock_kl(post: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """SureLock-style: adjacent-step posterior KL below tau."""
    return post["post_kl"] <= tau


def rule_surelock_kl_conf(post: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """SureLock-style with its optional confidence gate."""
    return (post["post_kl"] <= tau) & (post["post_confidence"] >= 0.9)


def rule_less_jsd_persistence(post: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """LESS-style: inter-step JSD below tau AND top-1 persistence.

    Persistence is measured against the locked identity: `raw_top1` is the
    argmax before the decode's where-overwrite, so agreement means the model
    still independently prefers the committed token.
    """
    stable = post["post_jsd"] <= tau
    agree = post["raw_top1_matches_commit"].astype(bool)
    run = np.zeros(agree.size, dtype=int)
    count = 0
    for i, ok in enumerate(agree):
        count = count + 1 if ok else 0
        run[i] = count
    return stable & (run >= DEFAULT_PERSISTENCE)


def rule_confidence(post: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """Confidence-only control: the cheap signal every method beats or ties."""
    return post["post_confidence"] >= tau


RULES: dict[str, tuple[Callable[[dict[str, np.ndarray], float], np.ndarray], str, bool]] = {
    # name: (fn, threshold source column, threshold is an upper bound)
    "earliest_logged_lock": (rule_earliest, "", True),
    "surelock_kl": (rule_surelock_kl, "post_kl", True),
    "surelock_kl_conf": (rule_surelock_kl_conf, "post_kl", True),
    "less_jsd_persist": (rule_less_jsd_persistence, "post_jsd", True),
    "confidence_only": (rule_confidence, "post_confidence", False),
}


def split_examples(
    post: pd.DataFrame, time_rows: pd.DataFrame, train_frac: float
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition a run's examples into (threshold-fitting, reporting) halves.

    Split by sorted example id so it is deterministic and reproducible, and by
    example rather than by row so no trajectory straddles the boundary.
    """
    ids = sorted(post["example_id"].unique())
    n_train = max(1, int(round(len(ids) * train_frac)))
    train_ids = set(ids[:n_train])
    eval_ids = set(ids[n_train:])
    if not eval_ids:
        raise ValueError(f"--self-split-train-frac leaves no reporting examples ({len(ids)} total)")
    print(f"[phase1] self-split: {len(train_ids)} fit / {len(eval_ids)} reported examples", flush=True)
    return (
        (post[post["example_id"].isin(train_ids)], time_rows[time_rows["example_id"].isin(train_ids)]),
        (post[post["example_id"].isin(eval_ids)], time_rows[time_rows["example_id"].isin(eval_ids)]),
    )


def premise_a_reference(run_dir: Path, exclude_eos: bool) -> dict[str, Any] | None:
    """Max drift from the commit-step reference: true lock-at-commit.

    Included because no rule in this sweep can fire at the commit step, so
    without this row the sweep's floor looks like lock-at-commit and understates
    it substantially.
    """
    path = Path(run_dir) / "committed_token_drift.csv"
    if not path.exists():
        return None
    drift = pd.read_csv(path)
    if exclude_eos:
        drift = drift[
            ~drift["selected_token"].isin([EOS_ID, EOT_ID])
            & ~drift["final_token"].isin([EOS_ID, EOT_ID])
        ]
    values = drift["max_drift_mean"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    row = {"rule": "premise_A_lock_at_commit", "tokens": int(values.size),
           "median_lower": float(np.median(values))}
    for thr in DRIFT_THRESHOLDS:
        row[f"frac_lower_gt_{thr}"] = float(np.mean(values > thr))
    return row


def load_run(run_dir: Path, exclude_eos: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    post_path = run_dir / "post_commit_posterior.csv"
    time_path = run_dir / "drift_time_rows.csv"
    if not post_path.exists():
        raise FileNotFoundError(
            f"{post_path} missing. This run predates the post-commit posterior logging; "
            "re-run run_settlement_fate_audit.py without --skip-drift-time-rows."
        )
    post = pd.read_csv(post_path)
    time_rows = pd.read_csv(time_path)
    for col in ("step_l2_drift", "row_norm", "base_norm"):
        if col not in time_rows.columns:
            raise ValueError(
                f"{time_path} lacks '{col}'. The upper bound cannot be computed without it; "
                "this run predates the Phase 1 schema."
            )
    if exclude_eos:
        padding = post["committed_token"].isin([EOS_ID, EOT_ID])
        padded = set(map(tuple, post.loc[padding, ["example_id", "position"]].drop_duplicates().to_numpy()))
        if padded:
            keys_post = list(map(tuple, post[["example_id", "position"]].to_numpy()))
            post = post[[k not in padded for k in keys_post]]
            keys_time = list(map(tuple, time_rows[["example_id", "position"]].to_numpy()))
            time_rows = time_rows[[k not in padded for k in keys_time]]
        print(f"[phase1] {run_dir.name}: dropped {len(padded)} EOS/EOT tokens", flush=True)
    return post, time_rows


def token_posteriors(post: pd.DataFrame) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Per-token posterior arrays, ordered by step."""
    out: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    cols = ["global_step", "post_kl", "post_jsd", "post_confidence", "raw_top1_matches_commit"]
    for (example_id, position), grp in post.sort_values("global_step").groupby(["example_id", "position"]):
        out[(example_id, int(position))] = {c: grp[c].to_numpy() for c in cols}
    return out


def token_layer_geometry(time_rows: pd.DataFrame) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Per (token, layer) precomputation so any lock step is an O(1) lookup."""
    out: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    grouped = time_rows.sort_values("global_step").groupby(["example_id", "position", "layer"])
    for (example_id, position, layer), grp in grouped:
        steps = grp["global_step"].to_numpy()
        d = grp["relative_l2_drift"].to_numpy()
        step_drift = grp["step_l2_drift"].to_numpy()
        row_norm = grp["row_norm"].to_numpy()
        base_norm = float(grp["base_norm"].iloc[0])
        n = steps.size
        suffix_max = np.empty(n)
        suffix_min = np.empty(n)
        suffix_max[-1] = -np.inf
        suffix_min[-1] = np.inf
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(d[i + 1], suffix_max[i + 1])
            suffix_min[i] = min(d[i + 1], suffix_min[i + 1])
        # Absolute step movement in ||b|| units: ||h_j - h_{j-1}|| / ||b||.
        # step_l2_drift is normalized by ||h_{j-1}||, i.e. row_norm[j-1].
        prev_norm = np.concatenate([[base_norm], row_norm[:-1]])
        abs_step = step_drift * prev_norm / max(base_norm, 1e-6)
        out[(example_id, int(position), int(layer))] = {
            "steps": steps,
            "d": d,
            "suffix_max": suffix_max,
            "suffix_min": suffix_min,
            "cum_abs_step": np.concatenate([[0.0], np.cumsum(abs_step)]),
            "rel_row_norm": row_norm / max(base_norm, 1e-6),
        }
    return out


def evaluate_rule(
    posteriors: dict[tuple[str, int], dict[str, np.ndarray]],
    geometry: dict[tuple[str, int, int], dict[str, np.ndarray]],
    layers: list[int],
    fn: Callable[[dict[str, np.ndarray], float], np.ndarray],
    tau: float,
) -> pd.DataFrame:
    """Per (token, layer) post-lock drift bounds under one rule and threshold."""
    records: list[dict[str, Any]] = []
    for (example_id, position), post in posteriors.items():
        fires = fn(post, tau)
        hit = np.nonzero(fires)[0]
        locked = hit.size > 0
        lock_idx = int(hit[0]) if locked else -1
        n_steps = post["global_step"].size
        for layer in layers:
            geo = geometry.get((example_id, position, layer))
            if geo is None:
                continue
            total_steps = geo["steps"].size
            row: dict[str, Any] = {
                "example_id": example_id,
                "position": position,
                "layer": layer,
                "locked": int(locked),
                "observed_steps": total_steps,
                # Steps left frozen by an irreversible lock at lock_idx.
                "frozen_steps": (total_steps - 1 - lock_idx) if locked else 0,
            }
            if locked and lock_idx < total_steps - 1:
                i = lock_idx
                d_i = geo["d"][i]
                lower = max(geo["suffix_max"][i] - d_i, d_i - geo["suffix_min"][i])
                upper = geo["cum_abs_step"][-1] - geo["cum_abs_step"][i + 1]
                scale = max(geo["rel_row_norm"][i], 1e-6)
                row["lower_bound_base_units"] = float(lower)
                row["upper_bound_base_units"] = float(upper)
                row["lower_bound_vs_reference"] = float(lower / scale)
                row["upper_bound_vs_reference"] = float(upper / scale)
            else:
                # Locked only at the final step (saves nothing) or never.
                for key in (
                    "lower_bound_base_units",
                    "upper_bound_base_units",
                    "lower_bound_vs_reference",
                    "upper_bound_vs_reference",
                ):
                    row[key] = np.nan
            records.append(row)
    return pd.DataFrame(records)


def cluster_bootstrap_median(
    frame: pd.DataFrame, column: str, reps: int, seed: int
) -> tuple[float, float, float]:
    by_example = {e: g[column].to_numpy(dtype=float) for e, g in frame.groupby("example_id")}
    ids = list(by_example)
    pooled = np.concatenate([by_example[e] for e in ids]) if ids else np.empty(0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(pooled))
    if len(ids) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.full(reps, np.nan)
    for rep in range(reps):
        sampled = rng.choice(len(ids), size=len(ids), replace=True)
        draw = np.concatenate([by_example[ids[i]] for i in sampled])
        draw = draw[np.isfinite(draw)]
        if draw.size:
            stats[rep] = np.median(draw)
    low, high = np.nanpercentile(stats, [2.5, 97.5])
    return point, float(low), float(high)


def summarize_rule(
    per_token: pd.DataFrame, rule: str, tau: float, reps: int, seed: int
) -> dict[str, Any]:
    # Layer-mean per token, matching the max_drift_mean convention elsewhere.
    mean_by_token = per_token.groupby(["example_id", "position"], as_index=False).mean(numeric_only=True)
    frozen = float(mean_by_token["frozen_steps"].sum())
    horizon = float(mean_by_token["observed_steps"].sum())
    row: dict[str, Any] = {
        "rule": rule,
        "tau": tau,
        "tokens": int(mean_by_token.shape[0]),
        "lock_rate": float(mean_by_token["locked"].mean()),
        "savings_frac_token_steps": frozen / horizon if horizon > 0 else np.nan,
    }
    for col, label in (
        ("lower_bound_vs_reference", "lower"),
        ("upper_bound_vs_reference", "upper"),
    ):
        median, low, high = cluster_bootstrap_median(mean_by_token, col, reps, seed)
        row[f"median_{label}"] = median
        row[f"median_{label}_ci_low"] = low
        row[f"median_{label}_ci_high"] = high
    values = mean_by_token["lower_bound_vs_reference"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    for thr in DRIFT_THRESHOLDS:
        row[f"frac_lower_gt_{thr}"] = float(np.mean(values > thr)) if values.size else np.nan
    return row


def threshold_grid(post: pd.DataFrame, column: str, upper: bool) -> list[float]:
    if not column:
        return [0.0]
    values = post[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0]
    quantiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    if not upper:
        quantiles = [1 - q for q in quantiles]
    return sorted({float(np.quantile(values, q)) for q in quantiles})


def sweep(post: pd.DataFrame, time_rows: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Summarize every (rule, threshold) pair. Per-token frames are not retained:
    at gen 256 they are ~1.8M rows across the sweep and nothing consumes them."""
    posteriors = token_posteriors(post)
    geometry = token_layer_geometry(time_rows)
    layers = sorted(time_rows["layer"].unique().tolist())
    rows: list[dict[str, Any]] = []
    for rule, (fn, column, upper) in RULES.items():
        for tau in threshold_grid(post, column, upper):
            per_token = evaluate_rule(posteriors, geometry, layers, fn, tau)
            if per_token.empty:
                continue
            rows.append(summarize_rule(per_token, rule, tau, reps, seed))
        print(f"[phase1] swept {rule}", flush=True)
    return pd.DataFrame(rows)


def pick_frozen_thresholds(train_sweep: pd.DataFrame, target: float) -> pd.DataFrame:
    """Per rule, the threshold whose train savings is closest to the target."""
    picks = []
    for rule, grp in train_sweep.groupby("rule"):
        eligible = grp.dropna(subset=["savings_frac_token_steps"])
        if eligible.empty:
            continue
        idx = (eligible["savings_frac_token_steps"] - target).abs().idxmin()
        chosen = eligible.loc[idx]
        picks.append(
            {
                "rule": rule,
                "savings_target": target,
                "tau": float(chosen["tau"]),
                "train_savings": float(chosen["savings_frac_token_steps"]),
                "train_lock_rate": float(chosen["lock_rate"]),
            }
        )
    return pd.DataFrame(picks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit prior work's lock-admission rules against the PhaseLock premises."
    )
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR to report on; repeatable")
    parser.add_argument(
        "--train-run",
        default=None,
        help="LABEL=RUN_DIR used only to freeze each rule's threshold at a savings target. "
        "Omit to report the raw sweep without frozen thresholds.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--self-split-train-frac",
        type=float,
        default=0.0,
        help="Freeze thresholds on a held-out fraction of THIS run's examples instead of a "
        "separate --train-run. Examples are split by sorted id, so the reported half is "
        "disjoint from the fitted half. Use when no same-regime training run exists (e.g. the "
        "level-5 gen-256 run): transferring gen-64 thresholds across regimes is not equivalent, "
        "because the KL/JSD distributions shift with generation length.",
    )
    parser.add_argument("--keep-eos-eot", action="store_true")
    args = parser.parse_args()
    if args.train_run and args.self_split_train_frac > 0:
        raise ValueError("--train-run and --self-split-train-frac are mutually exclusive")
    if not 0.0 <= args.self_split_train_frac < 1.0:
        raise ValueError("--self-split-train-frac must be in [0, 1)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_eos = not args.keep_eos_eot

    frozen: pd.DataFrame | None = None
    if args.train_run:
        label, _, run_dir = args.train_run.partition("=")
        post, time_rows = load_run(Path(run_dir), exclude_eos)
        train_sweep = sweep(post, time_rows, args.bootstrap_reps, args.seed)
        train_sweep.insert(0, "run", label)
        train_sweep.to_csv(output_dir / "train_sweep.csv", index=False)
        frozen = pd.concat(
            [pick_frozen_thresholds(train_sweep, t) for t in SAVINGS_TARGETS], ignore_index=True
        )
        frozen.to_csv(output_dir / "frozen_thresholds.csv", index=False)
        print(f"[phase1] froze thresholds on {label}", flush=True)

    sweeps = []
    frozen_evals = []
    reference_rows: list[dict[str, Any]] = []
    for spec in args.run:
        label, _, run_dir = spec.partition("=")
        reference = premise_a_reference(Path(run_dir), exclude_eos)
        if reference is not None:
            reference["run"] = label
            reference_rows.append(reference)
        post, time_rows = load_run(Path(run_dir), exclude_eos)
        if args.self_split_train_frac > 0:
            (tr_post, tr_time), (post, time_rows) = split_examples(
                post, time_rows, args.self_split_train_frac
            )
            tr_sweep = sweep(tr_post, tr_time, args.bootstrap_reps, args.seed)
            tr_sweep.insert(0, "run", f"{label}:selfsplit_fit")
            tr_sweep.to_csv(output_dir / f"train_sweep_{label}.csv", index=False)
            frozen = pd.concat(
                [pick_frozen_thresholds(tr_sweep, t) for t in SAVINGS_TARGETS], ignore_index=True
            )
            frozen.to_csv(output_dir / f"frozen_thresholds_{label}.csv", index=False)
        run_sweep = sweep(post, time_rows, args.bootstrap_reps, args.seed)
        run_sweep.insert(0, "run", label)
        sweeps.append(run_sweep)

        if frozen is not None:
            posteriors = token_posteriors(post)
            geometry = token_layer_geometry(time_rows)
            layers = sorted(time_rows["layer"].unique().tolist())
            for _, pick in frozen.iterrows():
                fn = RULES[pick["rule"]][0]
                per_token = evaluate_rule(posteriors, geometry, layers, fn, float(pick["tau"]))
                if per_token.empty:
                    continue
                row = summarize_rule(
                    per_token, pick["rule"], float(pick["tau"]), args.bootstrap_reps, args.seed
                )
                row["run"] = label
                row["savings_target"] = float(pick["savings_target"])
                row["train_savings"] = float(pick["train_savings"])
                # A large drift means the threshold did not transfer: the signal
                # distribution differs between the fitting and reporting sets,
                # so the rules are no longer being compared at matched savings.
                row["savings_drift_vs_train"] = row["savings_frac_token_steps"] - float(
                    pick["train_savings"]
                )
                frozen_evals.append(row)
            print(f"[phase1] evaluated frozen thresholds on {label}", flush=True)

    sweep_all = pd.concat(sweeps, ignore_index=True)
    sweep_all.to_csv(output_dir / "predicate_sweep.csv", index=False)
    frozen_frame = pd.DataFrame(frozen_evals)
    if not frozen_frame.empty:
        cols = [
            "run",
            "savings_target",
            "rule",
            "tau",
            "train_savings",
            "savings_frac_token_steps",
            "savings_drift_vs_train",
            "lock_rate",
        ] + [c for c in frozen_frame.columns if c.startswith("median_") or c.startswith("frac_")]
        frozen_frame = frozen_frame[cols].sort_values(["run", "savings_target", "rule"])
        frozen_frame.to_csv(output_dir / "frozen_threshold_eval.csv", index=False)

    summary = {
        "runs": args.run,
        "train_run": args.train_run,
        "eos_eot_excluded": exclude_eos,
        "bootstrap_reps": args.bootstrap_reps,
        "rules": list(RULES),
        "persistence_steps": DEFAULT_PERSISTENCE,
        "savings_targets": SAVINGS_TARGETS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 1: do prior lock-admission rules abide by the PhaseLock premises?",
        "",
        "Each rule is evaluated at its own lock step s' -- not at LLaDA's transfer",
        "step -- so it is credited with the settling that happens before it commits",
        "to a cache. Rules are compared at matched **savings** (fraction of",
        "post-commit token-steps left frozen), because thresholds are not",
        "commensurable across families.",
        "",
        "`median_lower` / `median_upper` bracket post-lock representation movement,",
        "renormalized to the frozen reference's own scale. The **lower** bound",
        "convicts (movement provably continued past the lock); the **upper** bound",
        "acquits (movement provably bounded). Read both.",
        "",
        f"EOS/EOT padding excluded: {exclude_eos}. Persistence requirement: "
        f"{DEFAULT_PERSISTENCE} consecutive steps.",
        "",
    ]
    if not frozen_frame.empty:
        lines += [
            "## Frozen thresholds (chosen on the train run, applied here)",
            "",
            table_text(frozen_frame),
            "",
        ]
    if reference_rows:
        lines += [
            "## Reference: true lock-at-commit (premise A)",
            "",
            "No rule in the sweep can fire at the commit step -- posteriors are",
            "logged from `base_step + 1` onward -- so the sweep's floor",
            "(`earliest_logged_lock`) is already one step delayed and understates",
            "lock-at-commit. This row is max drift from the commit-step reference.",
            "",
            table_text(pd.DataFrame(reference_rows)),
            "",
        ]
    lines += ["## Full threshold sweep", "", table_text(sweep_all)]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase1] wrote {output_dir}/summary.md", flush=True)


if __name__ == "__main__":
    main()
