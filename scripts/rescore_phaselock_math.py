#!/usr/bin/env python3
"""Non-destructively re-score saved online records with pinned Math-Verify."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from benchmark_phase_lock_fused_replay import _examples
from regret_remasking.paper_baselines import parse_example_indices


def symbolic_score(gold: str, prediction: str) -> dict:
    # Gold MATH-500 answers contain LaTeX without environment delimiters.
    gold_parsed = parse("$" + gold.strip().strip("$") + "$",
                        extraction_config=[LatexExtractionConfig()],
                        fallback_mode="no_fallback")
    if not gold_parsed:
        raise ValueError(f"Gold answer could not be parsed: {gold!r}")
    predicted = parse(prediction,
                      extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
                      fallback_mode="no_fallback")
    return {"correct": int(bool(predicted) and verify(gold_parsed, predicted)),
            "gold_parsed": [str(x) for x in gold_parsed],
            "prediction_parsed": [str(x) for x in predicted],
            "prediction_parse_failed": not bool(predicted)}


def self_test():
    tests = [
        (r"\frac{1}{2}", r"\boxed{0.5}", 1),
        (r"\frac{1}{2}", r"\boxed{\frac{7}{2}}", 0),
        (r"\sqrt{8}", r"\boxed{2\sqrt{2}}", 1),
        ("-2", r"\boxed{2}", 0),
        ("1", "No answer available.", 0),
        ("2", r"Reasoning contains 1. Final answer: \boxed{2}", 1),
    ]
    for gold, pred, expected in tests:
        result = symbolic_score(gold, pred)
        if result["correct"] != expected:
            raise AssertionError((gold, pred, expected, result))
    print(f"Symbolic grader self-test: {len(tests)} passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-math-indices")
    args = parser.parse_args()
    if importlib.metadata.version("math-verify") != "0.9.0":
        raise RuntimeError("Expected the pinned math-verify 0.9.0 grader")
    self_test()
    if args.validate_math_indices:
        examples = _examples("math500", "test", parse_example_indices(args.validate_math_indices))
        for example in examples:
            result = symbolic_score(example.gold_answer, "\\boxed{" + example.gold_answer + "}")
            if result["correct"] != 1:
                raise ValueError(f"Gold self-equivalence failed for {example.example_id}: {result}")
        print(f"Gold parsing/equivalence preflight: {len(examples)} examples passed", flush=True)
        return
    if args.self_test:
        return
    if not args.run_dirs or args.output_root is None:
        parser.error("provide --run-dirs and --output-root")
    for run_dir in args.run_dirs:
        if not (run_dir / "completion.json").exists():
            raise RuntimeError(f"Source run is not complete: {run_dir}")
        config = json.loads((run_dir / "run_config.json").read_text())
        rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
        details = []
        if config["dataset"] == "math500":
            examples = _examples("math500", "test", parse_example_indices(config["example_indices"]))
            gold = {e.example_id: e.gold_answer for e in examples}
            cache = {}
            for row in rows:
                row["legacy_numeric_correct"] = row["correct"]
                new_scores = []
                for eid, prediction in zip(row["example_ids"], row["generations"]):
                    key = (eid, prediction)
                    if key not in cache:
                        cache[key] = symbolic_score(gold[eid], prediction)
                        details.append({"example_id": eid, "prediction": prediction,
                                        "gold": gold[eid], **cache[key]})
                    new_scores.append(cache[key]["correct"])
                row["correct"] = new_scores
        target = args.output_root / run_dir.name
        target.mkdir(parents=True, exist_ok=False)
        config["scoring"] = "MATH500 Math-Verify 0.9.0 symbolic equivalence; GSM8K and HumanEval unchanged repository graders"
        config["grading_provenance"] = {
            "source_run": str(run_dir),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_records_sha256": hashlib.sha256((run_dir / "records.jsonl").read_bytes()).hexdigest(),
            "packages": {k: importlib.metadata.version(k) for k in ("math-verify", "latex2sympy2_extended", "antlr4-python3-runtime", "sympy")},
        }
        (target / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (target / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (target / "grading_details.json").write_text(json.dumps(details, indent=2) + "\n")
        (target / "completion.json").write_text(json.dumps({"status": "complete", "rescore_only": True}) + "\n")
        print(f"Re-scored {run_dir} -> {target}", flush=True)


if __name__ == "__main__":
    main()
