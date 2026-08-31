#!/usr/bin/env python3
"""Phase-6 compute-utility proxy: what does a drift-safety gate cost in compute?

CPU-only. Reads a settlement-fate run's drift_time_rows.csv and simulates
freeze-to-base policies: a committed token's row is frozen to its cached
post-commit (base) representation once drift-from-base has stayed below tau
for k consecutive observed steps. Tokens whose drift never settles below tau
are never frozen (conservative by construction). Compares against the
freeze-at-commit upper bound and reports the token-step row updates avoided
— the SureLock-style algorithmic-FLOP proxy for row-wise Q/attention-output/
FFN compute (frozen rows still serve cached K/V to active positions).

Caveats stated in the output: this is a proxy that decides whether systems
work is justified. FLOPs are not wall-clock; freeze-vs-remove output
equivalence (A2 vs A3) is untested; the harm panel is exploratory because
measured harm labels come from immediate-freeze interventions on a biased
(top-drift + matched control) sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TAUS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
KS = [1, 2, 4]


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def cluster_bootstrap_mean(values_by_example: dict[str, np.ndarray], reps: int, seed: int) -> tuple[float, float, float]:
    example_ids = list(values_by_example)
    pooled = np.concatenate([values_by_example[e] for e in example_ids])
    point = float(pooled.mean())
    if len(example_ids) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for rep in range(reps):
        sampled = rng.choice(len(example_ids), size=len(example_ids), replace=True)
        means[rep] = float(np.concatenate([values_by_example[example_ids[i]] for i in sampled]).mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return point, float(low), float(high)


def load_drift_trajectories(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    time_rows = pd.read_csv(run_dir / "drift_time_rows.csv")
    config = json.loads((run_dir / "run_config.json").read_text())
    # Mean over hooked layers per (example, position, step), matching how
    # max_drift_mean aggregates layers elsewhere in the repo.
    traj = (
        time_rows.groupby(["example_id", "position", "relative_position", "selected_step", "base_step", "global_step"])[
            "relative_l2_drift"
        ]
        .mean()
        .reset_index()
        .sort_values(["example_id", "position", "global_step"], kind="mergesort")
    )
    return traj, config


def settle_step(drifts: np.ndarray, steps: np.ndarray, tau: float, k: int) -> int | None:
    """First observed step from which drift-from-base stays < tau for k consecutive observations."""
    below = drifts < tau
    run = 0
    for idx in range(len(below)):
        run = run + 1 if below[idx] else 0
        if run >= k:
            return int(steps[idx - k + 1])
    return None


def simulate(
    traj: pd.DataFrame,
    config: dict[str, Any],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps_total = int(config["steps"])
    gen_length = int(config["gen_length"])
    policy_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    grouped = {
        key: group for key, group in traj.groupby(["example_id", "position"], sort=False)
    }
    prompt_len_by_example: dict[str, int] = {}
    for (example_id, position), group in grouped.items():
        prompt_len_by_example.setdefault(
            str(example_id), int(group["position"].iloc[0]) - int(group["relative_position"].iloc[0])
        )

    for tau in TAUS:
        for k in KS:
            per_example_avoided: dict[str, float] = {}
            per_example_delay: dict[str, list[float]] = {}
            per_example_frozen: dict[str, list[float]] = {}
            for (example_id, position), group in grouped.items():
                example_id = str(example_id)
                drifts = group["relative_l2_drift"].to_numpy(dtype=float)
                obs_steps = group["global_step"].to_numpy(dtype=int)
                selected = int(group["selected_step"].iloc[0])
                base = int(group["base_step"].iloc[0])
                t_settle = settle_step(drifts, obs_steps, tau, k)
                frozen = t_settle is not None
                # Row updates avoided by this token under the drift gate: every
                # step from settlement to the end of the decode.
                avoided = float(steps_total - t_settle) if frozen else 0.0
                upper = float(steps_total - base)  # freeze-at-commit bound
                delay = float(t_settle - base) if frozen else float(steps_total - base)
                per_example_avoided[example_id] = per_example_avoided.get(example_id, 0.0) + avoided
                per_example_delay.setdefault(example_id, []).append(delay)
                per_example_frozen.setdefault(example_id, []).append(1.0 if frozen else 0.0)
                if tau == TAUS[0] and k == KS[0]:
                    pass
                token_rows.append(
                    {
                        "tau": tau,
                        "k": k,
                        "example_id": example_id,
                        "position": int(position),
                        "selected_step": selected,
                        "base_step": base,
                        "settle_step": t_settle if frozen else -1,
                        "frozen": frozen,
                        "settle_delay_steps": delay,
                        "row_updates_avoided": avoided,
                        "row_updates_upper_bound": upper,
                    }
                )

            # Compute fractions per example, then aggregate with cluster bootstrap.
            frac_gen_only: dict[str, np.ndarray] = {}
            frac_with_prompt: dict[str, np.ndarray] = {}
            for example_id, avoided in per_example_avoided.items():
                prompt_len = prompt_len_by_example[example_id]
                total_rows = steps_total * (prompt_len + gen_length)
                frac_gen_only[example_id] = np.asarray([avoided / total_rows])
                frac_with_prompt[example_id] = np.asarray([(avoided + steps_total * prompt_len) / total_rows])
            delay_by_example = {e: np.asarray(v) for e, v in per_example_delay.items()}
            frozen_by_example = {e: np.asarray(v) for e, v in per_example_frozen.items()}

            frac_point, frac_lo, frac_hi = cluster_bootstrap_mean(frac_gen_only, reps, seed)
            fracp_point, fracp_lo, fracp_hi = cluster_bootstrap_mean(frac_with_prompt, reps, seed)
            delay_point, delay_lo, delay_hi = cluster_bootstrap_mean(delay_by_example, reps, seed)
            frozen_point, _, _ = cluster_bootstrap_mean(frozen_by_example, reps, seed)
            policy_rows.append(
                {
                    "policy": f"drift_gate tau={tau} k={k}",
                    "tau": tau,
                    "k": k,
                    "frozen_token_fraction": frozen_point,
                    "mean_settle_delay_steps": delay_point,
                    "settle_delay_ci": f"[{delay_lo:.2f}, {delay_hi:.2f}]",
                    "row_update_fraction_avoided_gen_rows": frac_point,
                    "gen_rows_ci": f"[{frac_lo:.4f}, {frac_hi:.4f}]",
                    "row_update_fraction_avoided_incl_prompt_freeze": fracp_point,
                    "incl_prompt_ci": f"[{fracp_lo:.4f}, {fracp_hi:.4f}]",
                }
            )

    # Freeze-at-commit upper bound and prompt-only reference.
    upper_gen: dict[str, np.ndarray] = {}
    upper_prompt: dict[str, np.ndarray] = {}
    for (example_id, position), group in grouped.items():
        example_id = str(example_id)
        base = int(group["base_step"].iloc[0])
        upper_gen[example_id] = upper_gen.get(example_id, np.asarray([0.0])) + (steps_total - base)
    for example_id in upper_gen:
        prompt_len = prompt_len_by_example[example_id]
        total_rows = steps_total * (prompt_len + gen_length)
        gen_avoided = float(upper_gen[example_id][0])
        upper_gen[example_id] = np.asarray([gen_avoided / total_rows])
        upper_prompt[example_id] = np.asarray([(gen_avoided + steps_total * prompt_len) / total_rows])
    up_point, up_lo, up_hi = cluster_bootstrap_mean(upper_gen, reps, seed)
    upp_point, upp_lo, upp_hi = cluster_bootstrap_mean(upper_prompt, reps, seed)
    policy_rows.insert(
        0,
        {
            "policy": "freeze_at_commit (upper bound, no drift gate)",
            "tau": np.nan,
            "k": np.nan,
            "frozen_token_fraction": 1.0,
            "mean_settle_delay_steps": 0.0,
            "settle_delay_ci": "",
            "row_update_fraction_avoided_gen_rows": up_point,
            "gen_rows_ci": f"[{up_lo:.4f}, {up_hi:.4f}]",
            "row_update_fraction_avoided_incl_prompt_freeze": upp_point,
            "incl_prompt_ci": f"[{upp_lo:.4f}, {upp_hi:.4f}]",
        },
    )
    return pd.DataFrame(policy_rows), pd.DataFrame(token_rows)


def exploratory_harm_panel(run_dir: Path) -> pd.DataFrame:
    """Measured immediate-freeze harm by max-drift bin. Exploratory only: the
    intervention sample is top-drift targets plus matched controls, not a
    random draw, and all interventions froze at base_step."""
    path = run_dir / "freeze_intervention_results.csv"
    if not path.exists():
        return pd.DataFrame()
    interventions = pd.read_csv(path)
    if interventions.empty:
        return pd.DataFrame()
    bins = [0.0, 0.3, 0.5, 0.75, 1.0, np.inf]
    interventions = interventions.copy()
    interventions["max_drift_bin"] = pd.cut(interventions["max_drift_mean"], bins=bins)
    panel = (
        interventions.groupby("max_drift_bin", observed=True)
        .agg(
            interventions=("example_id", "size"),
            non_target_change_rate=("non_target_token_changed", "mean"),
            normalized_answer_change_rate=("normalized_answer_changed", "mean"),
        )
        .reset_index()
    )
    panel["max_drift_bin"] = panel["max_drift_bin"].astype(str)
    return panel


def semantic_timing_panel(timing_run: Path, reps: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Three-way timing: earliest semantic-stable step vs commit step vs drift
    settlement, per token, from a commit-mode run that has both the proposal
    ledger and drift trajectories for the same examples."""
    ledger = pd.read_csv(timing_run / "proposal_ledger.csv")
    drift = pd.read_csv(timing_run / "committed_token_drift.csv")
    traj, config = load_drift_trajectories(timing_run)
    commits = drift[["example_id", "position", "selected_step", "final_token"]]
    merged = ledger.merge(commits, on=["example_id", "position"], how="inner")
    merged = merged[merged["global_step"] <= merged["selected_step"]]
    merged["stable"] = merged["top_token"] == merged["final_token"]
    stable_first = (
        merged[merged["stable"]].groupby(["example_id", "position"])["global_step"].min().rename("first_stable_step")
    )
    rows = commits.set_index(["example_id", "position"]).join(stable_first).reset_index()

    settle = (
        traj.groupby(["example_id", "position"])
        .apply(
            lambda g: settle_step(
                g["relative_l2_drift"].to_numpy(dtype=float), g["global_step"].to_numpy(dtype=int), 0.3, 2
            ),
            include_groups=False,
        )
        .rename("drift_settle_step_tau0.3_k2")
    )
    rows = rows.set_index(["example_id", "position"]).join(settle).reset_index()
    valid = rows.dropna(subset=["first_stable_step"]).copy()
    valid["semantic_lead_steps"] = valid["selected_step"] - valid["first_stable_step"]
    settled = valid.dropna(subset=["drift_settle_step_tau0.3_k2"]).copy()
    settled["drift_lag_steps"] = settled["drift_settle_step_tau0.3_k2"] - settled["selected_step"]

    def _ci(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
        groups = {str(e): g[column].to_numpy(dtype=float) for e, g in frame.groupby("example_id")}
        return cluster_bootstrap_mean(groups, reps, seed)

    lead_point, lead_lo, lead_hi = _ci(valid, "semantic_lead_steps")
    lag_point, lag_lo, lag_hi = _ci(settled, "drift_lag_steps")
    summary = {
        "tokens": int(len(rows)),
        "tokens_with_semantic_stability_before_commit": int(len(valid)),
        "tokens_drift_settled_tau0.3_k2": int(settled.shape[0]),
        "mean_semantic_lead_steps": lead_point,
        "semantic_lead_ci": [lead_lo, lead_hi],
        "mean_drift_settle_lag_steps_after_commit": lag_point,
        "drift_lag_ci": [lag_lo, lag_hi],
        "note": "semantic stability arrives before commit; drift settlement arrives after — the gap is compute a drift-safe gate forfeits relative to a semantic-only gate.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate drift-gated freeze policies' compute utility (CPU proxy).")
    parser.add_argument("--run-dir", required=True, help="Settlement-fate run with drift_time_rows.csv")
    parser.add_argument("--timing-run", default=None, help="Commit-mode run with proposal_ledger.csv for the three-way timing panel")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    traj, config = load_drift_trajectories(run_dir)
    policies, tokens = simulate(traj, config, args.bootstrap_reps, args.seed)
    policies.to_csv(output_dir / "policy_sweep.csv", index=False)
    tokens.to_csv(output_dir / "token_settlement.csv", index=False)
    harm = exploratory_harm_panel(run_dir)
    if not harm.empty:
        harm.to_csv(output_dir / "harm_by_drift_bin.csv", index=False)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "examples": int(traj["example_id"].nunique()),
        "tokens": int(traj.groupby(["example_id", "position"]).ngroups),
        "steps": int(config["steps"]),
        "gen_length": int(config["gen_length"]),
        "bootstrap_unit": "example_id clusters",
        "caveats": [
            "Phase-6 proxy: token-step row updates avoided, not wall-clock.",
            "Freeze-to-base policies only; freeze-to-current-state is unquantifiable from this data.",
            "Freeze-vs-remove output equivalence (A2 vs A3 arms) untested.",
            "Harm panel is exploratory: labels come from immediate-freeze interventions on a top-drift-biased sample.",
            "No drift trajectories exist for the gen-256 runs (--skip-drift-time-rows); this analysis covers gen=64 only.",
        ],
    }
    lines = [
        "# Compute-Utility Proxy (drift-gated freeze simulation)",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Policy Sweep",
        table_text(policies),
    ]
    if args.timing_run:
        timing_rows, timing_summary = semantic_timing_panel(Path(args.timing_run), args.bootstrap_reps, args.seed)
        timing_rows.to_csv(output_dir / "semantic_vs_drift_timing.csv", index=False)
        summary["semantic_vs_drift_timing"] = timing_summary
        lines.extend(["", "## Semantic vs Commit vs Drift Settlement Timing", "```json", json.dumps(timing_summary, indent=2), "```"])
    if not harm.empty:
        lines.extend(["", "## Exploratory: measured immediate-freeze harm by max-drift bin", table_text(harm)])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(policies))
    if "semantic_vs_drift_timing" in summary:
        print(json.dumps(summary["semantic_vs_drift_timing"], indent=2))


if __name__ == "__main__":
    main()
