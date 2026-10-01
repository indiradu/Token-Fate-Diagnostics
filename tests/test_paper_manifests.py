from __future__ import annotations

import json
from pathlib import Path

import pytest

from regret_remasking.data import Example, build_prompt
from regret_remasking.paper_manifests import (
    Candidate,
    HumanEvalDecontaminator,
    PromptTokenizer,
    default_teacher_config,
    make_paper_record,
    projection_bootstrap_ids,
    select_fit_replacements,
    select_reserve_records,
    select_training_records,
    stable_json_hash,
    validate_paper_records,
    write_manifest_set,
)


class WordTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        suffix = "\nassistant:" if add_generation_prompt else ""
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + suffix

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def _candidate(
    idx: int,
    *,
    family: str = "math",
    dataset_name: str = "gsm8k",
    prompt_text: str | None = None,
) -> Candidate:
    question = prompt_text or f"Question {family} {idx}?"
    answer = f"Answer {idx}"
    example = Example(
        example_id=f"{dataset_name}-{idx}",
        question=question,
        gold_answer=answer,
        dataset=dataset_name,
        metadata={"source_index": idx},
    )
    return Candidate(
        example=example,
        family=family,
        dataset_name=dataset_name,
        source_path=Path("/tmp/source.jsonl"),
        source_revision="rev",
        source_sha256="sha",
        prompt_text=prompt_text or build_prompt(question, "gsm8k"),
        metadata={"answer_hash": stable_json_hash(answer)},
    )


def test_make_record_schema_and_teacher_config_are_deterministic():
    tokenizer = PromptTokenizer(WordTokenizer())
    candidate = _candidate(1)
    record = make_paper_record(candidate, split="fit", seed=23, tokenizer=tokenizer)

    assert record["dataset_name"] == "gsm8k"
    assert record["family"] == "math"
    assert record["prompt_text"].startswith("Solve the math problem")
    assert record["metadata"]["answer_hash"] == stable_json_hash("Answer 1")
    assert record["teacher_config"] == default_teacher_config(record["record_id"], seed=23)
    assert record["teacher_config"]["steps"] == record["teacher_config"]["gen_length"]
    assert record["teacher_config"]["block_divisor"] in {8, 16, 32}


def test_validate_rejects_train_prompt_over_token_cap():
    tokenizer = PromptTokenizer(WordTokenizer())
    long_prompt = " ".join(f"tok{i}" for i in range(300))
    record = make_paper_record(
        _candidate(1, prompt_text=long_prompt),
        split="fit",
        seed=23,
        tokenizer=tokenizer,
    )
    with pytest.raises(ValueError, match="chat tokens"):
        validate_paper_records({"fit": [record]}, tokenizer=tokenizer, train_token_cap=256)


def test_humaneval_decontamination_rejects_prompt_solution_and_test_matches():
    reference = [
        {
            "prompt": 'def add_one(x):\n    """Add one."""\n',
            "canonical_solution": "    return x + 1\n",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        }
    ]
    decontaminator = HumanEvalDecontaminator(reference)
    exact_prompt = _candidate(
        1,
        family="code",
        dataset_name="kodcode",
        prompt_text='def add_one(x):\n    """Add one."""\n',
    )
    exact_prompt.example.metadata["canonical_solution"] = "    return x + 2\n"
    exact_prompt.example.metadata["test"] = "def check(candidate): pass"
    assert decontaminator.reason(exact_prompt) == "humaneval_normalized_prompt"

    same_solution = _candidate(
        2,
        family="code",
        dataset_name="kodcode",
        prompt_text='def different(x):\n    """Different task."""\n',
    )
    same_solution.example.metadata["canonical_solution"] = "    return x + 1\n"
    same_solution.example.metadata["test"] = "def check(candidate): pass"
    assert decontaminator.reason(same_solution) == "humaneval_solution_ast"

    same_test = _candidate(
        3,
        family="code",
        dataset_name="kodcode",
        prompt_text='def third(x):\n    """Different task."""\n',
    )
    same_test.example.metadata["canonical_solution"] = "    return x\n"
    same_test.example.metadata["test"] = "def check(candidate):\n assert candidate(1)==2\n"
    assert decontaminator.reason(same_test) == "humaneval_test_fingerprint"


def test_select_training_records_hits_requested_counts_and_excludes_old_fit_ids():
    tokenizer = PromptTokenizer(WordTokenizer())
    pools = {
        "math": [_candidate(i, family="math", dataset_name="gsm8k") for i in range(20)],
        "code": [_candidate(i, family="code", dataset_name="kodcode") for i in range(20)],
        "qa": [_candidate(i, family="qa", dataset_name="alpaca") for i in range(20)],
    }
    selected, info = select_training_records(
        pools,
        {
            "fit": {"math": 4, "code": 2, "qa": 2},
            "validation": {"math": 2, "code": 1, "qa": 1},
            "selection": {"math": 1, "code": 1},
        },
        tokenizer=tokenizer,
        forbidden_fit_source_ids={"gsm8k:gsm8k-0"},
    )
    assert {split: len(rows) for split, rows in selected.items()} == {
        "fit": 8,
        "validation": 4,
        "selection": 2,
    }
    assert all(row["source_id"] != "gsm8k:gsm8k-0" for row in selected["fit"])
    all_ids = [row["source_id"] for rows in selected.values() for row in rows]
    assert len(all_ids) == len(set(all_ids))
    assert info["used_source_count"] == 14
    validate_paper_records(selected, tokenizer=tokenizer)


def test_fit_replacements_do_not_consume_validation_selection_or_eval():
    tokenizer = PromptTokenizer(WordTokenizer())
    pools = {
        "math": [_candidate(i, family="math", dataset_name="gsm8k") for i in range(30)],
        "code": [_candidate(i, family="code", dataset_name="kodcode") for i in range(30)],
        "qa": [_candidate(i, family="qa", dataset_name="alpaca") for i in range(30)],
    }
    selected, _ = select_training_records(
        pools,
        {"fit": {"math": 10, "code": 5, "qa": 5}, "validation": {"math": 1}, "selection": {"code": 1}},
        tokenizer=tokenizer,
    )
    consumed = [row for rows in selected.values() for row in rows]
    replacements, info = select_fit_replacements(
        pools,
        selected["fit"],
        consumed + [{"source_id": "gsm8k:gsm8k-999", "prompt_hash": "eval"}],
        tokenizer=tokenizer,
    )
    assert info["target_counts"] == {"code": 1, "math": 2, "qa": 1}
    consumed_ids = {row["source_id"] for row in consumed}
    replacement_ids = {row["source_id"] for row in replacements}
    assert consumed_ids.isdisjoint(replacement_ids)
    assert {row["split"] for row in replacements} == {"fit_replacement"}


def test_projection_bootstrap_uses_fit_only_and_stratifies_by_family():
    fit = []
    for family in ("code", "math", "qa"):
        for idx in range(3):
            fit.append({"record_id": f"fit:{family}:{idx}", "family": family})
    ids = projection_bootstrap_ids(fit, count=6)
    assert ids == [
        "fit:code:0",
        "fit:math:0",
        "fit:qa:0",
        "fit:code:1",
        "fit:math:1",
        "fit:qa:1",
    ]


def test_write_manifest_set_refuses_nonidentical_overwrite(tmp_path: Path):
    tokenizer = PromptTokenizer(WordTokenizer())
    record = make_paper_record(_candidate(1), split="fit", tokenizer=tokenizer)
    write_manifest_set(tmp_path, {"fit": [record]})
    changed = dict(record)
    changed["example_id"] = "changed"
    with pytest.raises(FileExistsError):
        write_manifest_set(tmp_path, {"fit": [changed]})
    loaded = [json.loads(line) for line in (tmp_path / "fit.jsonl").read_text().splitlines()]
    assert loaded == [record]


def test_reserve_records_are_disjoint_from_consumed_and_each_other():
    tokenizer = PromptTokenizer(WordTokenizer())
    pools = {
        "math": [_candidate(i, family="math", dataset_name="gsm8k") for i in range(10)],
        "code": [
            _candidate(
                i,
                family="code",
                dataset_name="kodcode",
                prompt_text=f'def task_{i}(x):\n    """Return x."""\n',
            )
            for i in range(10)
        ],
        "qa": [_candidate(i, family="qa", dataset_name="alpaca") for i in range(10)],
    }
    for candidate in pools["code"]:
        candidate.example.metadata["canonical_solution"] = "    return x\n"
        candidate.example.metadata["test"] = "def check(candidate):\n    assert candidate(1) == 1\n"
    consumed = [
        {
            "source_id": "kodcode:kodcode-0",
            "prompt_hash": "consumed-prompt",
        }
    ]
    selected, info = select_reserve_records(
        pools,
        {
            "fit_replacement_extra": {"code": 3},
            "validation_replacement": {"math": 2, "code": 2, "qa": 2},
        },
        consumed,
        tokenizer=tokenizer,
        strict_code_syntax=True,
    )

    records = [record for rows in selected.values() for record in rows]
    assert len(records) == 9
    assert "kodcode:kodcode-0" not in {record["source_id"] for record in records}
    assert len({record["source_id"] for record in records}) == len(records)
    assert info["used_source_count"] == 9
    validate_paper_records(selected, tokenizer=tokenizer)


def test_reserve_records_reject_bad_code_syntax_and_fail_closed():
    tokenizer = PromptTokenizer(WordTokenizer())
    pools = {
        "code": [
            _candidate(
                1,
                family="code",
                dataset_name="kodcode",
                prompt_text="def broken(:\n",
            )
        ]
    }
    pools["code"][0].example.metadata["canonical_solution"] = "    return 1\n"
    pools["code"][0].example.metadata["test"] = "def check(candidate):\n    assert candidate() == 1\n"
    with pytest.raises(ValueError, match="requested 1 records but selected 0"):
        select_reserve_records(
            pools,
            {"fit_replacement_extra": {"code": 1}},
            [],
            tokenizer=tokenizer,
            strict_code_syntax=True,
        )
