#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ["confidence", "confidence_kl", "regret", "regret_no_cpv"]


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["example_id"]] = row
    return rows


def load_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pairwise(rows: dict[str, dict[str, dict]], a: str, b: str) -> tuple[list[str], list[str]]:
    ids = sorted(set(rows.get(a, {})) & set(rows.get(b, {})))
    wins = [idx for idx in ids if rows[a][idx]["correct"] and not rows[b][idx]["correct"]]
    losses = [idx for idx in ids if not rows[a][idx]["correct"] and rows[b][idx]["correct"]]
    return wins, losses


def format_table(summary_rows: list[dict[str, str]]) -> list[str]:
    lines = ["| Method | Examples | Accuracy | Avg NFE | Avg Latency |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary_rows:
        lines.append(
            "| {method} | {examples} | {accuracy} | {avg_nfe} | {avg_latency_s:.3f}s |".format(
                method=row["method"],
                examples=row["examples"],
                accuracy=f"{float(row['accuracy']):.3f}",
                avg_nfe=f"{float(row['avg_nfe']):.1f}",
                avg_latency_s=float(row["avg_latency_s"]),
            )
        )
    return lines


def analyze_run(run_dir: Path) -> str:
    summary_rows = load_summary(run_dir / "summary.csv")
    generations = {
        method: load_jsonl(run_dir / f"generations_eval_{method}.jsonl")
        for method in METHODS
    }
    report_path = run_dir / "regret_training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    lines = [f"## {run_dir.name}", ""]
    if summary_rows:
        lines.extend(format_table(summary_rows))
    else:
        lines.append("No `summary.csv` found.")
    lines.append("")

    if report:
        lines.append("| Predictor | AUC | Brier | Trace-Regret Rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, item in report.items():
            lines.append(
                f"| {name} | {item['auc']:.4f} | {item['brier']:.4f} | {item['positive_rate']:.4f} |"
            )
        lines.append("")

    for a, b in [("regret", "confidence"), ("regret", "confidence_kl"), ("regret", "regret_no_cpv")]:
        if generations.get(a) and generations.get(b):
            wins, losses = pairwise(generations, a, b)
            lines.append(f"- `{a}` vs `{b}`: {len(wins)} wins, {len(losses)} losses.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize regret-remasking pilot result directories.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = "# Pilot Analysis\n\n" + "\n".join(analyze_run(run) for run in args.runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

