from __future__ import annotations

import json
from pathlib import Path

import pytest

from regret_remasking.data import Example
from regret_remasking.locking_manifest import (
    load_manifest,
    make_manifest_record,
    manifest_example,
    prompt_content_hash,
    sha256_file,
    validate_disjoint,
    write_frozen_jsonl,
)
from scripts.prepare_locking_manifests import (
    deterministic_split,
    read_panel_ids,
    select_panel_examples,
)


def test_manifest_round_trip_and_hash_validation(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"gsm8k-train-1","question":"What is 2+2?","answer":"#### 4"}\n')
    example = Example("gsm8k-train-1", "What is 2+2?", "#### 4", "gsm8k")
    record = make_manifest_record(
        example,
        split="fit",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    assert record["content_hash"] == prompt_content_hash(example.question, example.dataset)

    manifest_path = tmp_path / "fit.jsonl"
    first_hash = write_frozen_jsonl(manifest_path, [record])
    second_hash = write_frozen_jsonl(manifest_path, [record])
    assert first_hash == second_hash

    loaded = load_manifest(manifest_path, expected_split="fit")
    assert loaded == [record]
    round_trip = manifest_example(loaded[0])
    assert round_trip == example


def test_write_frozen_jsonl_refuses_nonidentical_overwrite(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    first = make_manifest_record(
        Example("id-1", "Question A", "Answer A", "math500"),
        split="eval_math",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    second = make_manifest_record(
        Example("id-2", "Question B", "Answer B", "math500"),
        split="eval_math",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    manifest_path = tmp_path / "eval_math.jsonl"
    write_frozen_jsonl(manifest_path, [first])
    with pytest.raises(FileExistsError):
        write_frozen_jsonl(manifest_path, [second])


def test_validate_disjoint_rejects_duplicate_hash_across_splits(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    first = make_manifest_record(
        Example("a", "Same prompt", "1", "gsm8k"),
        split="fit",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    second = make_manifest_record(
        Example("b", "Same prompt", "2", "gsm8k"),
        split="calibration",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    with pytest.raises(ValueError, match="content_hash"):
        validate_disjoint({"fit": [first], "calibration": [second]})


def test_eval_panel_reconstruction_requires_exact_ids_and_order(tmp_path: Path):
    panel = tmp_path / "panel.csv"
    panel.write_text("example_id,dataset,gold_answer\nmath500-l5-2,math500,2\nmath500-l5-1,math500,1\n")
    source = [
        Example("math500-l5-1", "Question one", "1", "math500"),
        Example("math500-l5-2", "Question two", "2", "math500"),
    ]
    assert read_panel_ids(panel, "math500") == ["math500-l5-2", "math500-l5-1"]
    selected = select_panel_examples(panel, source, "math500")
    assert [example.example_id for example in selected] == ["math500-l5-2", "math500-l5-1"]
    assert [example.question for example in selected] == ["Question two", "Question one"]

    missing_panel = tmp_path / "missing.csv"
    missing_panel.write_text("example_id,dataset\nmath500-l5-404,math500\n")
    with pytest.raises(ValueError, match="missing from local source"):
        select_panel_examples(missing_panel, source, "math500")


def test_deterministic_splits_are_idempotent_and_disjoint_from_forbidden_content():
    examples = [
        Example(f"gsm8k-train-{idx}", f"Question {idx}", f"#### {idx}", "gsm8k")
        for idx in range(20)
    ]
    forbidden = {prompt_content_hash("Question 3", "gsm8k")}
    split_a = deterministic_split(
        examples,
        seed=23,
        fit_count=4,
        calibration_count=3,
        selection_count=2,
        forbidden_hashes=forbidden,
        forbidden_source_ids={"gsm8k:gsm8k-train-5"},
    )
    split_b = deterministic_split(
        examples,
        seed=23,
        fit_count=4,
        calibration_count=3,
        selection_count=2,
        forbidden_hashes=forbidden,
        forbidden_source_ids={"gsm8k:gsm8k-train-5"},
    )
    assert [[example.example_id for example in split] for split in split_a] == [
        [example.example_id for example in split] for split in split_b
    ]
    selected_ids = {example.example_id for split in split_a for example in split}
    selected_hashes = {
        prompt_content_hash(example.question, example.dataset)
        for split in split_a
        for example in split
    }
    assert "gsm8k-train-5" not in selected_ids
    assert forbidden.isdisjoint(selected_hashes)


def test_load_manifest_rejects_duplicate_source_ids(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    record = make_manifest_record(
        Example("dup", "Question", "Answer", "math500"),
        split="eval_math",
        seed=23,
        source_path=source,
        source_revision=sha256_file(source),
    )
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(record, sort_keys=True) + "\n" + json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="duplicate record_id"):
        load_manifest(path)
