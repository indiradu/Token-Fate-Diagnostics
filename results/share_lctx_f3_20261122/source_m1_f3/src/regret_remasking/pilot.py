from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from regret_remasking.data import Example, build_prompt, load_examples, score_generation
from regret_remasking.llada_trace import DecodeConfig, decode_batch, load_llada
from regret_remasking.llada_trace import infer_mask_token_id
from regret_remasking.regret_model import RegretScorer, train_regret_predictors


def _chunks(items: list[Example], batch_size: int) -> list[list[Example]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_decode_set(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    config: DecodeConfig,
    output_jsonl: Path,
    batch_size: int,
    regret_scorer: RegretScorer | None = None,
    collect_trace: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    batches = _chunks(examples, batch_size)
    for batch_idx, batch in enumerate(batches, start=1):
        print(
            f"[{config.policy}] batch {batch_idx}/{len(batches)} "
            f"examples={','.join(ex.example_id for ex in batch)}",
            flush=True,
        )
        prompts = [build_prompt(ex.question, ex.dataset) for ex in batch]
        result = decode_batch(
            model,
            tokenizer,
            prompts,
            [ex.example_id for ex in batch],
            config=config,
            regret_scorer=regret_scorer,
            collect_trace=collect_trace,
        )
        for ex, prompt, text in zip(batch, prompts, result.texts):
            generation_rows.append(
                {
                    "example_id": ex.example_id,
                    "dataset": ex.dataset,
                    "prompt": prompt,
                    "gold_answer": ex.gold_answer,
                    "generation": text,
                    "correct": score_generation(text, ex),
                    "nfe": result.nfe,
                    "latency_s": result.latency_s / max(1, len(batch)),
                    "policy": config.policy,
                    "steps": config.steps,
                    "gen_length": config.gen_length,
                    "block_length": config.block_length,
                }
            )
        trace_rows.extend(result.trace_rows)
        _write_jsonl(output_jsonl, generation_rows)
        print(
            f"[{config.policy}] finished batch {batch_idx}/{len(batches)} "
            f"nfe={result.nfe} latency_s={result.latency_s:.2f}",
            flush=True,
        )
    return generation_rows, trace_rows


def summarize(generation_rows_by_method: dict[str, list[dict[str, Any]]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for method, generations in generation_rows_by_method.items():
        if not generations:
            continue
        correct = [bool(row["correct"]) for row in generations]
        rows.append(
            {
                "method": method,
                "examples": len(generations),
                "accuracy": sum(correct) / len(correct),
                "avg_nfe": sum(float(row["nfe"]) for row in generations) / len(generations),
                "avg_latency_s": sum(float(row["latency_s"]) for row in generations) / len(generations),
            }
        )
    df = pd.DataFrame(rows).sort_values("method")
    df.to_csv(output_dir / "summary.csv", index=False)
    lines = ["# Regret-Aware Remasking Pilot Summary", ""]
    if not df.empty:
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.append("```text")
            lines.append(df.to_string(index=False))
            lines.append("```")
    else:
        lines.append("No generation rows were produced.")
    lines.append("")
    lines.append("Interpretation gate: regret-aware decoding needs to beat confidence+KL at the same NFE to justify scaling.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the regret-aware remasking pilot.")
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math-500", "countdown"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-limit", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--jsonl-path", default=None, help="Optional local JSONL dataset for both train/eval.")
    parser.add_argument("--output-dir", default="results/regret_pilot")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--beta-regret", type=float, default=2.0)
    parser.add_argument("--cpv-window", type=int, default=4)
    parser.add_argument("--methods", default="confidence,confidence_kl,regret,regret_no_cpv")
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing regret model files.")
    parser.add_argument("--regret-model-path", default=None)
    parser.add_argument("--regret-model-no-cpv-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    train_examples: list[Example] = []
    if not args.skip_train:
        train_examples = load_examples(
            args.dataset,
            args.train_split,
            args.train_limit,
            offset=args.train_offset,
            jsonl_path=args.jsonl_path,
        )
    eval_examples = load_examples(
        args.dataset,
        args.eval_split,
        args.eval_limit,
        offset=args.eval_offset,
        jsonl_path=args.jsonl_path,
    )
    model, tokenizer = load_llada(args.model_name, device=args.device, dtype=args.dtype)
    mask_id = infer_mask_token_id(tokenizer)

    base_config = DecodeConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        beta_kl=args.beta_kl,
        beta_regret=args.beta_regret,
        cpv_window=args.cpv_window,
        mask_id=mask_id,
    )

    if not args.skip_train:
        trace_csv = output_dir / "train_traces.csv"
        train_generations_path = output_dir / "generations_train_confidence.jsonl"
        _, train_trace_rows = run_decode_set(
            model,
            tokenizer,
            train_examples,
            replace(base_config, policy="confidence"),
            train_generations_path,
            batch_size=args.batch_size,
            collect_trace=True,
        )
        pd.DataFrame(train_trace_rows).to_csv(trace_csv, index=False)
        train_regret_predictors(
            trace_csv,
            output_dir,
            max_rows=args.max_train_rows,
            seed=args.seed,
        )
        regret_model_path = output_dir / "regret_model.joblib"
        regret_model_no_cpv_path = output_dir / "regret_model_no_cpv.joblib"
    else:
        if not args.regret_model_path or not args.regret_model_no_cpv_path:
            raise ValueError("--skip-train requires --regret-model-path and --regret-model-no-cpv-path")
        regret_model_path = Path(args.regret_model_path)
        regret_model_no_cpv_path = Path(args.regret_model_no_cpv_path)

    scorers = {
        "regret": RegretScorer(regret_model_path),
        "regret_no_cpv": RegretScorer(regret_model_no_cpv_path),
    }
    generations_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        scorer = scorers.get(method)
        generations, _ = run_decode_set(
            model,
            tokenizer,
            eval_examples,
            replace(base_config, policy=method),
            output_dir / f"generations_eval_{method}.jsonl",
            batch_size=args.batch_size,
            regret_scorer=scorer,
            collect_trace=False,
        )
        generations_by_method[method] = generations

    summary = summarize(generations_by_method, output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
