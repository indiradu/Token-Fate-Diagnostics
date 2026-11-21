from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from regret_remasking.data import Example, build_prompt

REQUIRED_MANIFEST_FIELDS = (
    "record_id",
    "source_id",
    "dataset",
    "example_id",
    "question",
    "gold_answer",
    "metadata",
    "split",
    "seed",
    "source_path",
    "source_revision",
    "content_hash",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt_for_hash(prompt: str) -> str:
    return "\n".join(prompt.splitlines()).strip()


def prompt_content_hash(question: str, dataset: str) -> str:
    prompt = normalize_prompt_for_hash(build_prompt(question, dataset))
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def make_manifest_record(
    example: Example,
    *,
    split: str,
    seed: int,
    source_path: str | Path,
    source_revision: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_id = f"{example.dataset}:{example.example_id}"
    merged_metadata = dict(example.metadata)
    if metadata:
        merged_metadata.update(dict(metadata))
    return {
        "record_id": f"{split}:{source_id}",
        "source_id": source_id,
        "dataset": example.dataset,
        "example_id": example.example_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "metadata": merged_metadata,
        "split": split,
        "seed": seed,
        "source_path": str(Path(source_path).resolve()),
        "source_revision": source_revision,
        "content_hash": prompt_content_hash(example.question, example.dataset),
    }


def manifest_example(record: Mapping[str, Any]) -> Example:
    _validate_record(record)
    return Example(
        example_id=str(record["example_id"]),
        question=str(record["question"]),
        gold_answer=str(record["gold_answer"]),
        dataset=str(record["dataset"]),
        metadata=dict(record["metadata"]),
    )


def load_manifest(path: str | Path, expected_split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_record(record)
            if expected_split is not None and record["split"] != expected_split:
                raise ValueError(
                    f"{path}:{line_number} has split {record['split']!r}; "
                    f"expected {expected_split!r}"
                )
            record_id = str(record["record_id"])
            source_id = str(record["source_id"])
            if record_id in seen_record_ids:
                raise ValueError(f"{path}:{line_number} duplicate record_id {record_id!r}")
            if source_id in seen_source_ids:
                raise ValueError(f"{path}:{line_number} duplicate source_id {source_id!r}")
            seen_record_ids.add(record_id)
            seen_source_ids.add(source_id)
            records.append(record)
    return records


def validate_disjoint(manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    owner_by_source_id: dict[str, str] = {}
    owner_by_content_hash: dict[str, str] = {}
    for split, records in manifests.items():
        local_source_ids: set[str] = set()
        local_hashes: set[str] = set()
        for record in records:
            _validate_record(record)
            source_id = str(record["source_id"])
            content_hash = str(record["content_hash"])
            if source_id in local_source_ids:
                raise ValueError(f"{split} has duplicate source_id {source_id!r}")
            if content_hash in local_hashes:
                raise ValueError(f"{split} has duplicate content_hash {content_hash!r}")
            local_source_ids.add(source_id)
            local_hashes.add(content_hash)

            previous_split = owner_by_source_id.get(source_id)
            if previous_split is not None:
                raise ValueError(
                    f"source_id {source_id!r} appears in both "
                    f"{previous_split!r} and {split!r}"
                )
            previous_hash_split = owner_by_content_hash.get(content_hash)
            if previous_hash_split is not None:
                raise ValueError(
                    f"content_hash {content_hash!r} appears in both "
                    f"{previous_hash_split!r} and {split!r}"
                )
            owner_by_source_id[source_id] = split
            owner_by_content_hash[content_hash] = split


def write_frozen_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> str:
    path = Path(path)
    text = "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"refusing to overwrite nonidentical frozen manifest: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in record]
    if missing:
        raise ValueError(f"manifest record missing required fields: {missing}")
    if not isinstance(record["metadata"], dict):
        raise ValueError("manifest record metadata must be an object")
    expected_hash = prompt_content_hash(str(record["question"]), str(record["dataset"]))
    if record["content_hash"] != expected_hash:
        raise ValueError(
            f"content_hash mismatch for {record.get('source_id', record.get('example_id'))!r}"
        )
