from __future__ import annotations

import json
import re
import ast
import operator
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    gold_answer: str
    dataset: str
    metadata: dict[str, Any] | None = None


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text.replace("$", " "))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def normalize_answer(text: str) -> str:
    text = text.strip()
    if "####" in text:
        text = text.split("####")[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    number = extract_number(text)
    return number if number is not None else " ".join(text.lower().split())


def score_exact_number(generation: str, gold_answer: str) -> bool:
    return normalize_answer(generation) == normalize_answer(gold_answer)


def _extract_countdown_parts(question: str) -> tuple[list[int], int]:
    numbers_match = re.search(r"Numbers:\s*([0-9,\s]+)", question)
    target_match = re.search(r"Target:\s*(-?\d+)", question)
    if numbers_match is None or target_match is None:
        raise ValueError(f"could not parse countdown question: {question}")
    numbers = [int(item) for item in re.findall(r"\d+", numbers_match.group(1))]
    target = int(target_match.group(1))
    return numbers, target


def _candidate_expression(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    candidates = re.findall(r"[-+*/().\d\s]+", text)
    candidates = [item.strip() for item in candidates if any(ch.isdigit() for ch in item)]
    return candidates[-1] if candidates else text


def _eval_arithmetic_expression(expression: str) -> tuple[float, list[int]] | None:
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    used_numbers: list[int] = []

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if float(node.value).is_integer():
                used_numbers.append(int(node.value))
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            right = visit(node.right)
            if isinstance(node.op, ast.Div) and abs(right) < 1e-12:
                raise ZeroDivisionError
            return ops[type(node.op)](visit(node.left), right)
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        return visit(tree), used_numbers
    except Exception:
        return None


def score_countdown(generation: str, question: str, gold_answer: str) -> bool:
    numbers, target = _extract_countdown_parts(question)
    expression = _candidate_expression(generation)
    evaluated = _eval_arithmetic_expression(expression)
    if evaluated is not None:
        value, used_numbers = evaluated
        return abs(value - target) < 1e-6 and Counter(used_numbers) == Counter(numbers)
    return False


def score_generation(generation: str, example: Example) -> bool:
    if example.dataset.lower() == "countdown":
        return score_countdown(generation, example.question, example.gold_answer)
    if example.dataset.lower() in {"humaneval", "human_eval", "human-eval"}:
        return score_humaneval(generation, example)
    return score_exact_number(generation, example.gold_answer)


def _extract_code(generation: str) -> str:
    """Remove common Markdown wrappers without altering Python indentation."""
    fenced = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", generation, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip("\n")
    return generation.strip()


def score_humaneval(generation: str, example: Example, timeout_s: float = 3.0) -> bool:
    """Run one HumanEval task in a short-lived, resource-limited subprocess.

    HumanEval's official fields contain a function prompt and a test harness.
    The generated completion is intentionally evaluated outside this process;
    this keeps model-produced code from sharing the experiment process. This
    is a pass@1-style single-sample check, not the full official pass@k suite.
    """
    metadata = example.metadata or {}
    prompt = str(metadata.get("prompt", example.question))
    test = str(metadata.get("test", ""))
    entry_point = str(metadata.get("entry_point", ""))
    if not test or not entry_point:
        return False
    completion = _extract_code(generation)
    # The model may repeat the function header. If it emits a full definition,
    # use it directly; otherwise append the body to the dataset prompt.
    candidate = completion if re.search(r"^\s*(?:async\s+)?def\s+", completion, flags=re.MULTILINE) else prompt + completion
    harness = (
        "import contextlib\n"
        "import io\n"
        "import sys\n"
        "try:\n"
        f"    exec({candidate!r}, globals())\n"
        f"    exec({test!r}, globals())\n"
        f"    check({entry_point})\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__, file=sys.stderr)\n"
        "    raise\n"
    )
    import subprocess
    import sys
    import tempfile

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def build_prompt(question: str, dataset: str) -> str:
    dataset_key = dataset.lower()
    if dataset_key == "gsm8k":
        return (
            "Solve the math problem. Show concise reasoning and end with "
            "`#### <answer>`.\n\n"
            f"Problem: {question}"
        )
    if dataset_key == "countdown":
        return (
            "Use each provided number exactly once with +, -, *, /, and "
            "parentheses to reach the target. End with `#### <expression>`.\n\n"
            f"Problem: {question}"
        )
    if dataset_key in {"humaneval", "human_eval", "human-eval"}:
        return (
            "Complete the following Python function. Return only the completed "
            "code, with no Markdown fences or explanation.\n\n"
            f"{question}"
        )
    return (
        "Solve the problem. Show concise reasoning and put the final answer in "
        "\\boxed{}.\n\n"
        f"Problem: {question}"
    )


def _load_jsonl(path: Path, dataset: str, limit: int | None, offset: int) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < offset:
                continue
            if limit is not None and len(examples) >= limit:
                break
            row = json.loads(line)
            question = row.get("question") or row.get("problem") or row.get("prompt")
            answer = row.get("answer") or row.get("solution") or row.get("gold_answer")
            if question is None or answer is None:
                raise ValueError(f"JSONL row is missing question/answer fields: {row}")
            examples.append(
                Example(
                    example_id=str(row.get("id", f"{dataset}-{idx}")),
                    question=str(question),
                    gold_answer=str(answer),
                    dataset=dataset,
                )
            )
    return examples


def _make_countdown_example(idx: int, split: str) -> Example:
    seed_offset = {"train": 0, "validation": 100_000, "test": 200_000}.get(split, 300_000)
    rng = random.Random(seed_offset + idx)
    numbers = [rng.randint(2, 10) for _ in range(4)]
    values = [str(number) for number in numbers]
    value_numbers = numbers[:]
    while len(values) > 1:
        left_idx = rng.randrange(len(values))
        left_expr = values.pop(left_idx)
        left_value = value_numbers.pop(left_idx)
        right_idx = rng.randrange(len(values))
        right_expr = values.pop(right_idx)
        right_value = value_numbers.pop(right_idx)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            value = left_value + right_value
        elif op == "-":
            value = left_value - right_value
        else:
            value = left_value * right_value
        values.append(f"({left_expr} {op} {right_expr})")
        value_numbers.append(value)
    target = value_numbers[0]
    expression = values[0]
    question = f"Numbers: {', '.join(str(number) for number in numbers)}. Target: {target}."
    return Example(
        example_id=f"countdown-{split}-{idx}",
        question=question,
        gold_answer=expression,
        dataset="countdown",
    )


def _load_countdown(split: str, limit: int | None, offset: int) -> list[Example]:
    count = 500 if limit is None else limit
    return [_make_countdown_example(idx, split) for idx in range(offset, offset + count)]


def _load_humaneval(split: str, limit: int | None, offset: int) -> list[Example]:
    if split not in {"test", "validation"}:
        raise ValueError("HumanEval provides only a test split")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install `datasets` or pass --jsonl-path with HumanEval fields."
        ) from exc
    hf = load_dataset("openai/openai_humaneval", split="test")
    end = len(hf) if limit is None else min(len(hf), offset + limit)
    rows: list[Example] = []
    for idx in range(offset, end):
        row: dict[str, Any] = hf[idx]
        task_id = str(row.get("task_id", f"HumanEval/{idx}"))
        prompt = str(row["prompt"])
        rows.append(
            Example(
                example_id=task_id,
                question=prompt,
                gold_answer=str(row.get("canonical_solution", "")),
                dataset="humaneval",
                metadata={
                    "prompt": prompt,
                    "test": str(row["test"]),
                    "entry_point": str(row["entry_point"]),
                },
            )
        )
    return rows


def load_examples(
    dataset: str,
    split: str,
    limit: int | None,
    offset: int = 0,
    jsonl_path: str | None = None,
) -> list[Example]:
    if jsonl_path:
        return _load_jsonl(Path(jsonl_path), dataset, limit, offset)

    dataset_key = dataset.lower()
    if dataset_key == "countdown":
        return _load_countdown(split, limit, offset)
    if dataset_key in {"humaneval", "human_eval", "human-eval"} and not jsonl_path:
        return _load_humaneval(split, limit, offset)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install `datasets` or pass --jsonl-path with question/answer fields."
        ) from exc

    if dataset_key == "gsm8k":
        hf = load_dataset("gsm8k", "main", split=split)
        question_key, answer_key = "question", "answer"
    elif dataset_key in {"math500", "math-500"}:
        hf = load_dataset("HuggingFaceH4/MATH-500", split="test")
        question_key = "problem"
        answer_key = "answer"
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    rows: list[Example] = []
    end = len(hf) if limit is None else min(len(hf), offset + limit)
    for idx in range(offset, end):
        row: dict[str, Any] = hf[idx]
        rows.append(
            Example(
                example_id=str(row.get("id", f"{dataset}-{split}-{idx}")),
                question=str(row[question_key]),
                gold_answer=str(row[answer_key]),
                dataset=dataset_key,
            )
        )
    return rows
