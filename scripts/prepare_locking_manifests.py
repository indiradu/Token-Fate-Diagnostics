from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regret_remasking.data import Example, load_examples
from regret_remasking.locking_manifest import (
    make_manifest_record,
    prompt_content_hash,
    sha256_file,
    validate_disjoint,
    write_frozen_jsonl,
)

DEFAULT_GSM8K_ARROW = (
    "/home/rui.gao/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/"
    "740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow"
)
MATH_PANEL = "results/fast_dllm_official_l5g256_54_20260908/example_results.csv"
CODE_PANEL = "results/fast_dllm_official_humaneval_g256_70_20260908/example_results.csv"


def read_panel_ids(path: Path, expected_dataset: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "example_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no example_id column")
        for row_number, row in enumerate(reader, start=2):
            example_id = str(row["example_id"])
            dataset = str(row.get("dataset", expected_dataset)).lower()
            if dataset != expected_dataset:
                raise ValueError(
                    f"{path}:{row_number} dataset {dataset!r}; expected {expected_dataset!r}"
                )
            if example_id in seen:
                raise ValueError(f"{path}:{row_number} duplicate panel id {example_id!r}")
            seen.add(example_id)
            ids.append(example_id)
    return ids


def load_jsonl_examples(path: Path, dataset: str) -> list[Example]:
    return load_examples(dataset, "test", limit=None, jsonl_path=str(path))


def examples_by_id(examples: list[Example], label: str) -> dict[str, Example]:
    by_id: dict[str, Example] = {}
    for example in examples:
        if example.example_id in by_id:
            raise ValueError(f"{label} has duplicate source id {example.example_id!r}")
        by_id[example.example_id] = example
    return by_id


def select_panel_examples(panel_path: Path, source_examples: list[Example], dataset: str) -> list[Example]:
    ids = read_panel_ids(panel_path, dataset)
    by_id = examples_by_id(source_examples, str(panel_path))
    missing = [example_id for example_id in ids if example_id not in by_id]
    if missing:
        raise ValueError(f"{panel_path} ids missing from local source: {missing[:10]}")
    return [by_id[example_id] for example_id in ids]


def load_gsm8k_arrow(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(f"GSM8K Arrow cache not found: {path}")
    from datasets import Dataset

    dataset = Dataset.from_file(str(path))
    examples: list[Example] = []
    seen_source_ids: set[str] = set()
    for index, row in enumerate(dataset):
        example = Example(
            example_id=f"gsm8k-train-{index}",
            question=str(row["question"]),
            gold_answer=str(row["answer"]),
            dataset="gsm8k",
            metadata={"source_index": index},
        )
        if example.example_id in seen_source_ids:
            raise ValueError(f"duplicate GSM8K source id {example.example_id!r}")
        seen_source_ids.add(example.example_id)
        examples.append(example)
    return examples


def deterministic_split(
    examples: list[Example],
    *,
    seed: int,
    fit_count: int,
    calibration_count: int,
    selection_count: int,
    forbidden_hashes: set[str],
    forbidden_source_ids: set[str],
) -> tuple[list[Example], list[Example], list[Example]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    pool = [
        example
        for example in shuffled
        if f"{example.dataset}:{example.example_id}" not in forbidden_source_ids
        and prompt_content_hash(example.question, example.dataset) not in forbidden_hashes
    ]
    required = fit_count + calibration_count + selection_count
    if len(pool) < required:
        raise ValueError(f"need {required} disjoint GSM8K examples, found {len(pool)}")
    fit_end = fit_count
    calibration_end = fit_end + calibration_count
    selection_end = calibration_end + selection_count
    return pool[:fit_end], pool[fit_end:calibration_end], pool[calibration_end:selection_end]


def make_records(
    examples: list[Example],
    *,
    split: str,
    seed: int,
    source_path: Path,
    source_revision: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        make_manifest_record(
            example,
            split=split,
            seed=seed,
            source_path=source_path,
            source_revision=source_revision,
            metadata=metadata,
        )
        for example in examples
    ]


def pick_outside_eval(source: list[Example], eval_records: list[dict[str, Any]], count: int) -> list[Example]:
    eval_source_ids = {str(record["source_id"]) for record in eval_records}
    eval_hashes = {str(record["content_hash"]) for record in eval_records}
    selected: list[Example] = []
    for example in source:
        source_id = f"{example.dataset}:{example.example_id}"
        if source_id in eval_source_ids:
            continue
        if prompt_content_hash(example.question, example.dataset) in eval_hashes:
            continue
        selected.append(example)
        if len(selected) == count:
            return selected
    raise ValueError(f"could not find {count} examples outside eval")


def manifest_summary_entry(path: Path, records: list[dict[str, Any]], text_hash: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": text_hash,
        "count": len(records),
        "dataset_counts": _count_by(records, "dataset"),
        "split_counts": _count_by(records, "split"),
    }


def _count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen manifests for locking experiments.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fit-count", type=int, default=400)
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--selection-count", type=int, default=100)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=None)
    parser.add_argument("--gsm8k-arrow", type=Path, default=Path(DEFAULT_GSM8K_ARROW))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    output_dir = args.output_dir.resolve()

    math_source_path = workspace_root / "data/math500_level5.jsonl"
    code_source_path = workspace_root / "data/humaneval.jsonl"
    math_panel_path = workspace_root / MATH_PANEL
    code_panel_path = workspace_root / CODE_PANEL
    for required_path in (math_source_path, code_source_path, math_panel_path, code_panel_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    math_source = load_jsonl_examples(math_source_path, "math500")
    code_source = load_jsonl_examples(code_source_path, "humaneval")
    eval_math_examples = select_panel_examples(math_panel_path, math_source, "math500")
    eval_code_examples = select_panel_examples(code_panel_path, code_source, "humaneval")

    eval_math = make_records(
        eval_math_examples,
        split="eval_math",
        seed=args.seed,
        source_path=math_source_path,
        source_revision=sha256_file(math_source_path),
        metadata={"panel_csv": str(math_panel_path.resolve())},
    )
    eval_code = make_records(
        eval_code_examples,
        split="eval_code",
        seed=args.seed,
        source_path=code_source_path,
        source_revision=sha256_file(code_source_path),
        metadata={"panel_csv": str(code_panel_path.resolve())},
    )

    if args.gsm8k_jsonl is not None:
        gsm8k_source_path = args.gsm8k_jsonl.resolve()
        gsm8k_source = load_jsonl_examples(gsm8k_source_path, "gsm8k")
    else:
        gsm8k_source_path = args.gsm8k_arrow.resolve()
        gsm8k_source = load_gsm8k_arrow(gsm8k_source_path)
    gsm8k_revision = sha256_file(gsm8k_source_path)

    eval_hashes = {record["content_hash"] for record in eval_math + eval_code}
    eval_source_ids = {record["source_id"] for record in eval_math + eval_code}
    fit_examples, calibration_examples, selection_examples = deterministic_split(
        gsm8k_source,
        seed=args.seed,
        fit_count=args.fit_count,
        calibration_count=args.calibration_count,
        selection_count=args.selection_count,
        forbidden_hashes=set(eval_hashes),
        forbidden_source_ids=set(eval_source_ids),
    )
    fit = make_records(
        fit_examples,
        split="fit",
        seed=args.seed,
        source_path=gsm8k_source_path,
        source_revision=gsm8k_revision,
    )
    calibration = make_records(
        calibration_examples,
        split="calibration",
        seed=args.seed,
        source_path=gsm8k_source_path,
        source_revision=gsm8k_revision,
    )
    selection = make_records(
        selection_examples,
        split="selection",
        seed=args.seed,
        source_path=gsm8k_source_path,
        source_revision=gsm8k_revision,
    )
    smoke = make_records(
        fit_examples[:16],
        split="smoke",
        seed=args.seed,
        source_path=gsm8k_source_path,
        source_revision=gsm8k_revision,
        metadata={"intentional_overlap_with": "fit", "overlap_exception": "gsm8k_smoke_subset"},
    )
    smoke_math = make_records(
        pick_outside_eval(math_source, eval_math, 8),
        split="smoke_math",
        seed=args.seed,
        source_path=math_source_path,
        source_revision=sha256_file(math_source_path),
    )
    smoke_code = make_records(
        pick_outside_eval(code_source, eval_code, 8),
        split="smoke_code",
        seed=args.seed,
        source_path=code_source_path,
        source_revision=sha256_file(code_source_path),
    )

    validate_disjoint(
        {
            "fit": fit,
            "calibration": calibration,
            "selection": selection,
            "eval_math": eval_math,
            "eval_code": eval_code,
            "smoke_math": smoke_math,
            "smoke_code": smoke_code,
        }
    )
    _validate_smoke_overlap(smoke, fit)

    outputs = {
        "fit": fit,
        "calibration": calibration,
        "selection": selection,
        "smoke": smoke,
        "eval_math": eval_math,
        "eval_code": eval_code,
        "smoke_math": smoke_math,
        "smoke_code": smoke_code,
    }
    manifest_hashes: dict[str, str] = {}
    manifest_files: dict[str, str] = {}
    for split, records in outputs.items():
        path = output_dir / f"{split}.jsonl"
        manifest_hashes[split] = write_frozen_jsonl(path, records)
        manifest_files[split] = str(path.resolve())

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "workspace_root": str(workspace_root),
        "counts": {split: len(records) for split, records in outputs.items()},
        "manifests": {
            split: manifest_summary_entry(Path(manifest_files[split]), records, manifest_hashes[split])
            for split, records in outputs.items()
        },
        "sources": {
            "gsm8k": {
                "path": str(gsm8k_source_path),
                "sha256": gsm8k_revision,
                "count": len(gsm8k_source),
            },
            "math500_level5": {
                "path": str(math_source_path.resolve()),
                "sha256": sha256_file(math_source_path),
                "count": len(math_source),
                "panel_csv": str(math_panel_path.resolve()),
                "panel_count": len(eval_math),
            },
            "humaneval": {
                "path": str(code_source_path.resolve()),
                "sha256": sha256_file(code_source_path),
                "count": len(code_source),
                "panel_csv": str(code_panel_path.resolve()),
                "panel_count": len(eval_code),
            },
        },
        "disjoint_validation": {
            "strict_splits": [
                "fit",
                "calibration",
                "selection",
                "eval_math",
                "eval_code",
                "smoke_math",
                "smoke_code",
            ],
            "smoke_overlap_exception": {
                "split": "smoke",
                "overlaps_with": "fit",
                "count": len(smoke),
                "reason": "16 GSM8K smoke examples are an intentional subset of development.",
            },
        },
    }
    write_frozen_text(output_dir / "manifest_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "counts": summary["counts"]}, indent=2))


def _validate_smoke_overlap(smoke: list[dict[str, Any]], fit: list[dict[str, Any]]) -> None:
    fit_source_ids = {record["source_id"] for record in fit}
    fit_hashes = {record["content_hash"] for record in fit}
    for record in smoke:
        if record["source_id"] not in fit_source_ids:
            raise ValueError(f"smoke record {record['source_id']!r} is not in fit")
        if record["content_hash"] not in fit_hashes:
            raise ValueError(f"smoke record {record['source_id']!r} hash is not in fit")


def write_frozen_text(path: Path, text: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"refusing to overwrite nonidentical frozen file: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare_locking_manifests: {exc}", file=sys.stderr)
        raise
