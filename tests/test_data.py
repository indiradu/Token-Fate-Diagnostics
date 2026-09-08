from __future__ import annotations

import shutil

import pytest

from regret_remasking.code_eval import evaluate_humaneval_program
from regret_remasking.data import (
    build_prompt,
    extract_final_answer,
    load_examples,
    normalize_answer,
    score_exact_number,
    score_generation,
)


def test_extract_final_answer_preserves_nested_boxed_fraction():
    generation = r"Reasoning here. Therefore \boxed{\frac{13}{6}}."
    assert extract_final_answer(generation) == r"\frac{13}{6}"


def test_symbolic_grader_accepts_equivalent_numeric_forms():
    assert score_exact_number("The final answer is 42.", "42")
    assert score_exact_number("Thus, the answer is 42.", "42")
    assert score_exact_number(r"The answer is \boxed{\frac{1}{2}}.", "0.5")
    assert score_exact_number(r"The answer is \boxed{270/7}.", r"\frac{270}7\text{ degrees}")
    assert score_exact_number(r"The answer is \boxed{32348}.", r"\$32,\!348")
    assert score_exact_number(r"The answer is \boxed{\sqrt{5}}.", "sqrt(5)")
    assert score_exact_number(r"The answer is \boxed{-\infty}.", "-infinity")
    assert score_exact_number(r"The answer is \boxed{11\sqrt{2}}.", r"11\sqrt2")


def test_symbolic_grader_accepts_equivalent_expressions():
    assert score_exact_number(r"\boxed{2(x+1)}", "2*x+2")
    assert score_exact_number(r"\boxed{x=\frac{13}{6}}", r"\frac{13}{6}")


def test_symbolic_grader_handles_structured_math_answers():
    assert score_exact_number(
        r"\boxed{(-infinity,2)\cup(3,infinity)}",
        r"(-\infty, 2) \cup (3, \infty)",
    )
    assert score_exact_number(r"\boxed{\{-2,1-\sqrt{5},1+\sqrt{5}\}}", r"\{1\pm\sqrt{5},-2\}")
    assert score_exact_number(r"\boxed{(6,31,-1)}", "(6, 31, -1)")
    assert score_exact_number(r"\boxed{[-2,7]}", r"x \in [-2,7]")
    assert score_exact_number(r"\boxed{Evelyn}", r"\text{Evelyn}")
    assert score_exact_number(r"\boxed{555}", r"4210_{5}")


def test_symbolic_grader_rejects_different_answers():
    assert not score_exact_number(r"\boxed{36}", r"\frac{13}{6}")
    assert not score_exact_number(r"\boxed{5}", r"\frac{13}{6}")


def test_symbolic_grader_does_not_execute_input():
    payload = "__import__('os').system('touch /tmp/token_fate_grader_should_not_exist')"
    assert not score_exact_number(payload, "0")


def test_humaneval_loader_preserves_executable_test_metadata():
    example = load_examples(
        "humaneval",
        "test",
        limit=1,
        jsonl_path="data/humaneval.jsonl",
    )[0]
    assert example.example_id == "HumanEval/0"
    assert example.metadata["entry_point"] == "has_close_elements"
    assert "def check(candidate)" in example.metadata["test"]
    assert "Return only executable Python code" in build_prompt(example.question, example.dataset)


def test_aime_jsonl_uses_math_prompt_and_exact_answer():
    example = load_examples(
        "aime2024",
        "test",
        limit=1,
        jsonl_path="data/aime2024.jsonl",
    )[0]
    assert example.example_id.startswith("aime2024-")
    assert example.gold_answer == "204"
    assert "put the final answer in \\boxed{}" in build_prompt(example.question, example.dataset)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="HumanEval requires bubblewrap isolation")
def test_humaneval_scorer_accepts_canonical_and_rejects_broken_completion():
    example = load_examples(
        "humaneval",
        "test",
        limit=1,
        jsonl_path="data/humaneval.jsonl",
    )[0]
    assert score_generation(example.gold_answer, example)
    assert not score_generation("    return False\n", example)

    full_function_without_prompt_import = """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    return any(
        abs(left - right) < threshold
        for index, left in enumerate(numbers)
        for right in numbers[index + 1:]
    )
"""
    assert score_generation(full_function_without_prompt_import, example)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="HumanEval requires bubblewrap isolation")
def test_humaneval_sandbox_has_private_tmp(tmp_path):
    marker = "/tmp/token_fate_humaneval_isolation_marker"
    example = load_examples(
        "humaneval",
        "test",
        limit=1,
        jsonl_path="data/humaneval.jsonl",
    )[0]
    generation = (
        "    with open('" + marker + "', 'w') as handle:\n"
        "        handle.write('inside sandbox')\n"
        + example.gold_answer
    )
    result = evaluate_humaneval_program(
        generation,
        example.question,
        example.metadata["test"],
        example.metadata["entry_point"],
    )
    assert result.passed
    assert not __import__("pathlib").Path(marker).exists()


def test_code_answer_normalization_uses_code_not_last_number():
    generation = "Here is the code:\n```python\ndef answer():\n    return 42\n```"
    assert normalize_answer(generation, "humaneval") == "def answer():\n    return 42"
