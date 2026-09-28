from __future__ import annotations

from generate_tracelock_llada_traces import (
    _prompt,
    inventory_existing_traces,
    sampled_steps,
)


def test_natural_questions_uses_question_without_gold_answer() -> None:
    prompt = _prompt(
        "natural_questions",
        {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    )
    assert prompt == "Who wrote Hamlet?"
    assert "Shakespeare" not in prompt


def test_sampled_steps_is_deterministic_and_keeps_boundaries() -> None:
    first = sampled_steps(128, 0.30, 42)
    second = sampled_steps(128, 0.30, 42)
    assert first == second
    assert len(first) == round(128 * 0.30)
    assert first[0] == 0
    assert first[-1] == 127


def test_inventory_existing_traces_keeps_complete_and_rejects_partial(tmp_path) -> None:
    train = tmp_path / "train"
    validation = tmp_path / "val"
    train.mkdir()
    validation.mkdir()
    for step in range(3):
        (train / f"kodcode-00000011_step_{step:04d}.pt").touch()
    for step in range(2):
        (validation / f"kodcode-00000012_step_{step:04d}.pt").touch()

    records, audit = inventory_existing_traces(
        tmp_path, "kodcode", expected_steps=3
    )

    assert [record["sample_id"] for record in records] == ["kodcode-00000011"]
    assert records[0]["source_index"] == 11
    assert records[0]["saved_steps"] == 3
    assert audit == {
        "complete_examples": 1,
        "partial_examples": 1,
        "trace_files": 5,
    }
