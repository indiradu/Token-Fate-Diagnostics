#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


LAYER_RE = re.compile(r"layer_(\d+)_(hidden(?:_dynamics)?(?:_plus_logits)?)$")


def probe_family(probe: str) -> tuple[int | None, str]:
    if probe.endswith("/logits"):
        return None, "trace/logits"
    tail = probe.split("/")[-1]
    match = LAYER_RE.match(tail)
    if not match:
        return None, tail
    return int(match.group(1)), match.group(2)


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize layerwise token-fate probe results.")
    parser.add_argument("probe_results_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.probe_results_csv)
    parsed = df["probe"].apply(probe_family)
    df["layer"] = [item[0] for item in parsed]
    df["family"] = [item[1] for item in parsed]

    rows = []
    for target, target_df in df.groupby("target", sort=False):
        logits = target_df[target_df["family"] == "trace/logits"].head(1)
        logits_auc = float(logits["auc"].iloc[0]) if len(logits) else None
        for family in ["hidden", "hidden_dynamics", "hidden_plus_logits", "hidden_dynamics_plus_logits"]:
            sub = target_df[target_df["family"] == family].dropna(subset=["auc"]).copy()
            if sub.empty:
                continue
            best = sub.sort_values("auc", ascending=False).iloc[0]
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "best_layer": int(best["layer"]),
                    "best_auc": float(best["auc"]),
                    "best_ap": float(best["average_precision"]),
                    "trace_auc": logits_auc,
                    "delta_vs_trace": float(best["auc"] - logits_auc) if logits_auc is not None else None,
                }
            )

    summary = pd.DataFrame(rows)
    layerwise = (
        df[df["family"].isin(["hidden", "hidden_dynamics"])]
        .dropna(subset=["layer", "auc"])
        [["target", "family", "layer", "auc", "average_precision", "eval_rows", "positive_rate"]]
        .sort_values(["target", "family", "layer"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output.with_suffix(".csv"), index=False)
    layerwise.to_csv(args.output.with_name(args.output.stem + "_layerwise.csv"), index=False)

    lines = ["# Layerwise Token-Fate Probe Summary", ""]
    lines.append("## Best Layer By Probe Family")
    lines.append(table_text(summary))
    lines.append("")
    lines.append("## Layerwise Hidden Probes")
    lines.append(table_text(layerwise))
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
