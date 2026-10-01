from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regret_remasking.paper_manifests import (
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    KODCODE_EXPECTED_SHA256,
    KODCODE_REVISION,
    PROJECTION_BOOTSTRAP_COUNT,
    PromptTokenizer,
    build_humaneval_decontaminator,
    copy_preserved_manifest,
    load_alpaca_parquet,
    load_gsm8k_arrow,
    load_humaneval_reference,
    load_kodcode_jsonl,
    projection_bootstrap_ids,
    select_fit_replacements,
    select_reserve_records,
    select_training_records,
    sha256_file,
    summary_entry,
    validate_paper_records,
    write_manifest_set,
)

DEFAULT_GSM8K_ARROW = Path(
    "/home/rui.gao/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/"
    "740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow"
)
DEFAULT_ALPACA_PARQUET = Path(
    "/home/rui.gao/.cache/huggingface/hub/datasets--tatsu-lab--alpaca/snapshots/"
    "dce01c9b08f87459cf36a430d809084718273017/data/"
    "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
)
DEFAULT_ALPACA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
DEFAULT_CAMPAIGN_ROOT = Path(
    "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/"
    "locking_paper_redesign_20260920_161812"
)
DEFAULT_KODCODE_JSONL = DEFAULT_CAMPAIGN_ROOT / "datasets/kodcode_train.jsonl"
DEFAULT_OLD_MANIFEST_DIR = Path(
    "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/"
    "locking_original_plan_20260917_204619/manifests"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CAMPAIGN_ROOT / "manifests"

PRESERVED_MANIFESTS = (
    "eval_math.jsonl",
    "eval_code.jsonl",
    "smoke_math.jsonl",
    "smoke_code.jsonl",
    "smoke.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper-redesign mixed data manifests.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-manifest-dir", type=Path, default=DEFAULT_OLD_MANIFEST_DIR)
    parser.add_argument("--gsm8k-arrow", type=Path, default=DEFAULT_GSM8K_ARROW)
    parser.add_argument("--alpaca-parquet", type=Path, default=DEFAULT_ALPACA_PARQUET)
    parser.add_argument("--alpaca-revision", default=DEFAULT_ALPACA_REVISION)
    parser.add_argument("--kodcode-jsonl", type=Path, default=DEFAULT_KODCODE_JSONL)
    parser.add_argument("--kodcode-revision", default=KODCODE_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-token-cap", type=int, default=256)
    parser.add_argument("--allow-kodcode-sha-mismatch", action="store_true")
    parser.add_argument(
        "--extend-reserves",
        action="store_true",
        help="Only add deterministic reserve-extension manifests; preserve existing frozen files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    output_dir = args.output_dir.resolve()
    old_manifest_dir = args.old_manifest_dir.resolve()
    humaneval_path = workspace_root / "data/humaneval.jsonl"

    required_paths = (
        args.gsm8k_arrow,
        args.alpaca_parquet,
        args.kodcode_jsonl,
        humaneval_path,
        old_manifest_dir / "calibration.jsonl",
        old_manifest_dir / "selection.jsonl",
        *(old_manifest_dir / name for name in PRESERVED_MANIFESTS),
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("required manifest input files are missing: " + ", ".join(missing))

    kodcode_sha = sha256_file(args.kodcode_jsonl)
    if kodcode_sha != KODCODE_EXPECTED_SHA256 and not args.allow_kodcode_sha_mismatch:
        raise ValueError(
            f"KodCode SHA mismatch for {args.kodcode_jsonl}: {kodcode_sha}; "
            f"expected {KODCODE_EXPECTED_SHA256}"
        )

    tokenizer = PromptTokenizer(model_id=args.model_id)
    decontaminator = build_humaneval_decontaminator(load_humaneval_reference(humaneval_path))

    pools = {
        "math": load_gsm8k_arrow(args.gsm8k_arrow.resolve()),
        "code": load_kodcode_jsonl(args.kodcode_jsonl.resolve(), revision=args.kodcode_revision),
        "qa": load_alpaca_parquet(args.alpaca_parquet.resolve(), revision=args.alpaca_revision),
    }
    if args.extend_reserves:
        summary = _extend_reserves(
            args=args,
            output_dir=output_dir,
            pools=pools,
            tokenizer=tokenizer,
            decontaminator=decontaminator,
            humaneval_path=humaneval_path,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    preserved_records = _load_jsonl_records(old_manifest_dir, PRESERVED_MANIFESTS)
    forbidden_all_source_ids = {str(record["source_id"]) for record in preserved_records}
    forbidden_prompt_hashes = {
        str(record.get("content_hash") or record.get("prompt_hash"))
        for record in preserved_records
        if record.get("content_hash") or record.get("prompt_hash")
    }
    old_cal_select_records = _load_jsonl_records(old_manifest_dir, ("calibration.jsonl", "selection.jsonl"))
    forbidden_fit_source_ids = {str(record["source_id"]) for record in old_cal_select_records}

    training, selection_info = select_training_records(
        pools,
        {
            "fit": {"math": 4000, "code": 2000, "qa": 2000},
            "validation": {"math": 200, "code": 100, "qa": 100},
            "selection": {"math": 100, "code": 100},
        },
        tokenizer=tokenizer,
        seed=args.seed,
        token_cap=args.train_token_cap,
        decontaminator=decontaminator,
        forbidden_fit_source_ids=forbidden_fit_source_ids,
        forbidden_all_source_ids=forbidden_all_source_ids,
        forbidden_prompt_hashes=forbidden_prompt_hashes,
    )
    validate_paper_records(training, tokenizer=tokenizer, train_token_cap=args.train_token_cap)

    consumed_training = [record for records in training.values() for record in records]
    replacements, replacement_info = select_fit_replacements(
        pools,
        training["fit"],
        consumed_training + list(preserved_records),
        tokenizer=tokenizer,
        seed=args.seed,
        token_cap=args.train_token_cap,
        decontaminator=decontaminator,
    )
    validate_paper_records(
        {"fit_replacement": replacements},
        tokenizer=tokenizer,
        train_token_cap=args.train_token_cap,
    )

    bootstrap_ids = projection_bootstrap_ids(
        training["fit"],
        count=PROJECTION_BOOTSTRAP_COUNT,
    )

    written = write_manifest_set(output_dir, training | {"fit_replacement": replacements})
    preserved = {}
    for name in PRESERVED_MANIFESTS:
        source = old_manifest_dir / name
        target = output_dir / name
        digest = copy_preserved_manifest(source, target)
        preserved[name.removesuffix(".jsonl")] = {
            "path": str(target.resolve()),
            "sha256": digest,
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "preserved_exact": True,
        }

    bootstrap_path = output_dir / "projection_bootstrap_ids.json"
    bootstrap_payload = {
        "count": len(bootstrap_ids),
        "source_split": "fit",
        "strategy": "round_robin_stratified_by_family",
        "record_ids": bootstrap_ids,
    }
    _write_frozen_json(bootstrap_path, bootstrap_payload)

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "model_id": args.model_id,
        "tokenizer_available": tokenizer.available,
        "train_token_cap_after_chat_template": args.train_token_cap,
        "output_dir": str(output_dir),
        "sources": {
            "gsm8k": _source_summary(args.gsm8k_arrow, pools["math"]),
            "kodcode": _source_summary(args.kodcode_jsonl, pools["code"], revision=args.kodcode_revision),
            "alpaca": _source_summary(args.alpaca_parquet, pools["qa"], revision=args.alpaca_revision),
            "humaneval_decontamination_reference": {
                "path": str(humaneval_path.resolve()),
                "count": len(load_humaneval_reference(humaneval_path)),
                "sha256": sha256_file(humaneval_path),
                "used_for_exclusion_only": True,
            },
        },
        "manifests": written,
        "preserved_manifests": preserved,
        "projection_bootstrap": {
            "path": str(bootstrap_path.resolve()),
            "sha256": sha256_file(bootstrap_path),
            "count": len(bootstrap_ids),
        },
        "filtering": {
            "selection": selection_info,
            "fit_replacement": replacement_info,
            "old_calibration_selection_source_ids_excluded_from_new_fit": len(
                forbidden_fit_source_ids
            ),
            "preserved_eval_smoke_source_ids_forbidden_from_training": len(
                forbidden_all_source_ids
            ),
            "decontamination": {
                "reference": "all 164 HumanEval rows",
                "methods": [
                    "normalized prompt exact match",
                    "normalized docstring exact match",
                    "prompt token 5-gram Jaccard >= 0.55",
                    "canonical_solution AST fingerprint exact match",
                    "test fingerprint exact match",
                ],
                "claim": "conservative lexical/structural exclusion, not semantic proof",
            },
        },
    }
    _write_frozen_json(output_dir / "manifest_summary.json", summary)
    print(json.dumps(_console_report(summary), indent=2, sort_keys=True))


def _extend_reserves(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    pools: dict[str, list[Any]],
    tokenizer: PromptTokenizer,
    decontaminator: Any,
    humaneval_path: Path,
) -> dict[str, Any]:
    protected_names = (
        "fit.jsonl",
        "validation.jsonl",
        "selection.jsonl",
        "fit_replacement.jsonl",
        "eval_math.jsonl",
        "eval_code.jsonl",
        "smoke_math.jsonl",
        "smoke_code.jsonl",
        "smoke.jsonl",
        "projection_bootstrap_ids.json",
    )
    missing = [str(output_dir / name) for name in protected_names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError("cannot extend reserves before base manifests exist: " + ", ".join(missing))

    base_records = _load_jsonl_records(
        output_dir,
        (
            "fit.jsonl",
            "validation.jsonl",
            "selection.jsonl",
            "fit_replacement.jsonl",
            "eval_math.jsonl",
            "eval_code.jsonl",
            "smoke_math.jsonl",
            "smoke_code.jsonl",
            "smoke.jsonl",
        ),
    )
    extension_targets = {
        "fit_replacement_extra": {"code": 1600},
        "validation_replacement": {"math": 100, "code": 100, "qa": 100},
    }
    extension, reserve_info = select_reserve_records(
        pools,
        extension_targets,
        base_records,
        tokenizer=tokenizer,
        seed=args.seed,
        token_cap=args.train_token_cap,
        decontaminator=decontaminator,
        strict_code_syntax=True,
    )
    validate_paper_records(extension, tokenizer=tokenizer, train_token_cap=args.train_token_cap)
    _validate_extension_disjoint(base_records, extension)

    written: dict[str, dict[str, Any]] = {}
    for split, records in extension.items():
        path = output_dir / f"{split}.jsonl"
        digest = _write_frozen_jsonl(path, records)
        written[split] = summary_entry(path, records, digest)

    protected_hashes = {name: sha256_file(output_dir / name) for name in protected_names}
    provenance = {
        "schema_version": 1,
        "mode": "extend_reserves",
        "seed": args.seed,
        "model_id": args.model_id,
        "tokenizer_available": tokenizer.available,
        "train_token_cap_after_chat_template": args.train_token_cap,
        "output_dir": str(output_dir),
        "manifests": written,
        "reserve_selection": reserve_info,
        "protected_inputs": {
            "files": protected_hashes,
            "record_count": len(base_records),
            "source_id_count": len({str(record["source_id"]) for record in base_records}),
            "prompt_hash_count": len(
                {
                    str(record.get("prompt_hash") or record.get("content_hash"))
                    for record in base_records
                    if record.get("prompt_hash") or record.get("content_hash")
                }
            ),
        },
        "sources": {
            "gsm8k": _source_summary(args.gsm8k_arrow, pools["math"]),
            "kodcode": _source_summary(args.kodcode_jsonl, pools["code"], revision=args.kodcode_revision),
            "alpaca": _source_summary(args.alpaca_parquet, pools["qa"], revision=args.alpaca_revision),
            "humaneval_decontamination_reference": {
                "path": str(humaneval_path.resolve()),
                "count": len(load_humaneval_reference(humaneval_path)),
                "sha256": sha256_file(humaneval_path),
                "used_for_exclusion_only": True,
            },
        },
        "decontamination": {
            "reference": "all 164 HumanEval rows",
            "methods": [
                "normalized prompt exact match",
                "normalized docstring exact match",
                "prompt token 5-gram Jaccard >= 0.55",
                "canonical_solution AST fingerprint exact match",
                "test fingerprint exact match",
                "code prompt/solution/test syntax parse for extension reserves",
            ],
            "claim": "conservative lexical/structural exclusion, not semantic proof",
        },
    }
    provenance_path = output_dir / "reserve_extension_provenance.json"
    _write_frozen_json(provenance_path, provenance)
    return {
        "output_dir": str(output_dir),
        "counts": {split: entry["count"] for split, entry in written.items()},
        "family_counts": {split: entry["family_counts"] for split, entry in written.items()},
        "manifest_sha256": {split: entry["sha256"] for split, entry in written.items()},
        "provenance": {
            "path": str(provenance_path.resolve()),
            "sha256": sha256_file(provenance_path),
        },
        "filter_counts": reserve_info["filter_counts"],
    }


def _load_jsonl_records(base_dir: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in names:
        with (base_dir / name).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _source_summary(path: Path, pool: list[Any], *, revision: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "revision": revision,
        "sha256": sha256_file(path),
        "available_count": len(pool),
    }


def _write_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite nonidentical frozen JSON: {path}")
    path.write_text(text, encoding="utf-8")


def _write_frozen_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite nonidentical frozen JSONL: {path}")
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def _validate_extension_disjoint(
    protected_records: list[dict[str, Any]],
    extension: dict[str, list[dict[str, Any]]],
) -> None:
    protected_sources = {str(record["source_id"]) for record in protected_records}
    protected_hashes = {
        str(record.get("prompt_hash") or record.get("content_hash"))
        for record in protected_records
        if record.get("prompt_hash") or record.get("content_hash")
    }
    seen_sources: set[str] = set()
    seen_hashes: set[str] = set()
    for split, records in extension.items():
        for record in records:
            source_id = str(record["source_id"])
            phash = str(record["prompt_hash"])
            if source_id in protected_sources:
                raise ValueError(f"{split} source_id overlaps protected manifests: {source_id}")
            if phash in protected_hashes:
                raise ValueError(f"{split} prompt_hash overlaps protected manifests: {phash}")
            if source_id in seen_sources:
                raise ValueError(f"{split} duplicate extension source_id: {source_id}")
            if phash in seen_hashes:
                raise ValueError(f"{split} duplicate extension prompt_hash: {phash}")
            seen_sources.add(source_id)
            seen_hashes.add(phash)


def _console_report(summary: dict[str, Any]) -> dict[str, Any]:
    manifests = summary["manifests"]
    return {
        "output_dir": summary["output_dir"],
        "tokenizer_available": summary["tokenizer_available"],
        "counts": {name: entry["count"] for name, entry in manifests.items()},
        "family_counts": {
            name: entry["family_counts"] for name, entry in manifests.items()
        },
        "filtering": summary["filtering"],
        "preserved": sorted(summary["preserved_manifests"]),
    }


if __name__ == "__main__":
    main()
