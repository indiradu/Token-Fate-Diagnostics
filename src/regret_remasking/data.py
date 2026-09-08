from __future__ import annotations

import json
import re
import ast
import operator
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover - torch environments normally provide sympy
    sp = None


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    gold_answer: str
    dataset: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_UNSAFE_ANSWER_RE = re.compile(
    r"(?:__|\bimport\b|\beval\b|\bexec\b|\bopen\s*\(|\bsystem\s*\(|\bsubprocess\b)",
    re.IGNORECASE,
)


def extract_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text.replace("$", " "))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _last_balanced_command(text: str, commands: tuple[str, ...]) -> str | None:
    matches: list[str] = []
    for command in commands:
        start = 0
        marker = command + "{"
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            depth = 1
            cursor = idx + len(marker)
            content_start = cursor
            while cursor < len(text) and depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth == 0:
                matches.append(text[content_start : cursor - 1])
                start = cursor
            else:
                break
    return matches[-1] if matches else None


def extract_final_answer(text: str) -> str:
    """Extract a final answer without flattening symbolic LaTeX to its last number."""
    candidate = text.strip()
    if "####" in candidate:
        candidate = candidate.rsplit("####", 1)[-1].strip()
    boxed = _last_balanced_command(candidate, (r"\boxed", r"\fbox"))
    if boxed is not None:
        return boxed.strip()

    candidate = candidate.strip().strip("$").strip()
    candidate = re.sub(
        r"^(?:therefore|thus)\s*[,;:]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = candidate.rstrip(".。")

    # Gold MATH answers and concise model answers often contain symbolic LaTeX.
    # Preserve those expressions; only use the historical last-number fallback
    # for prose-like generations without an explicit answer delimiter.
    symbolic_markers = (r"\frac", r"\dfrac", r"\tfrac", r"\sqrt", r"\pi", r"\infty")
    if any(marker in candidate for marker in symbolic_markers):
        return candidate
    if "\n" not in candidate and len(candidate.split()) <= 8:
        return candidate
    number = extract_number(candidate)
    return number if number is not None else candidate


def _surface_normalize(text: str) -> str:
    text = extract_final_answer(text)
    pure_text = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]*)\}", text.strip())
    if pure_text is not None:
        text = pure_text.group(1)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\;": "",
        "−": "-",
        "∞": "infinity",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return "".join(text.lower().split()).strip("$")


def normalize_answer(text: str, dataset: str | None = None) -> str:
    if dataset and dataset.lower() in {"humaneval", "human_eval"}:
        from regret_remasking.code_eval import extract_python_code

        return "\n".join(line.rstrip() for line in extract_python_code(text).strip().splitlines())
    return _surface_normalize(text)


def _consume_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 1
    cursor = start + 1
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    return text[start + 1 : cursor - 1], cursor


def _consume_latex_argument(text: str, start: int) -> tuple[str, int] | None:
    group = _consume_group(text, start)
    if group is not None:
        return group
    if start >= len(text):
        return None
    return text[start], start + 1


def _replace_latex_fractions(text: str) -> str | None:
    while True:
        found = [(text.find(command), command) for command in (r"\dfrac", r"\tfrac", r"\frac")]
        found = [(idx, command) for idx, command in found if idx >= 0]
        if not found:
            return text
        idx, command = min(found)
        numerator = _consume_latex_argument(text, idx + len(command))
        if numerator is None:
            return None
        numerator_text, after_numerator = numerator
        denominator = _consume_latex_argument(text, after_numerator)
        if denominator is None:
            return None
        denominator_text, after_denominator = denominator
        replacement = f"(({numerator_text})/({denominator_text}))"
        text = text[:idx] + replacement + text[after_denominator:]


def _replace_latex_sqrts(text: str) -> str | None:
    while r"\sqrt" in text:
        idx = text.find(r"\sqrt")
        group = _consume_latex_argument(text, idx + len(r"\sqrt"))
        if group is None:
            return None
        radicand, after = group
        text = text[:idx] + f"sqrt({radicand})" + text[after:]
    return text


def _math_text_to_python(text: str) -> str | None:
    text = _surface_normalize(text)
    if _UNSAFE_ANSWER_RE.search(text):
        return None
    if "=" in text:
        parts = text.split("=")
        if len(parts) != 2:
            return None
        text = parts[1]
    text = re.sub(r"\\text\{[^{}]*\}", "", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\$", "").replace(r"\!", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    text = text.replace(r"\pi", "pi").replace(r"\infty", "oo")
    text = re.sub(r"\b(?:infinity|inf)\b", "oo", text)
    text = _replace_latex_fractions(text)
    if text is None:
        return None
    text = _replace_latex_sqrts(text)
    if text is None:
        return None
    text = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", text)
    text = text.replace("^", "**").replace("{", "(").replace("}", ")")
    if text.endswith(r"\%"):
        text = f"({text[:-2]})/100"
    elif text.endswith("%"):
        text = f"({text[:-1]})/100"
    text = text.replace("\\", "")

    # Insert only the common implicit products needed by MATH answer keys.
    text = text.replace("sqrt(", "SQRTFUNC(")
    text = re.sub(r"(?<=[0-9A-Za-z_)])(?=\()", "*", text)
    text = re.sub(r"(?<=[0-9)])(?=[A-Za-z])", "*", text)
    text = text.replace("SQRTFUNC*(", "sqrt(")
    return text


def _safe_sympy_parse(text: str):
    if sp is None:
        return None
    python_text = _math_text_to_python(text)
    if not python_text:
        return None
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def visit(node: ast.AST):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return sp.pi
            if node.id == "oo":
                return sp.oo
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id):
                return sp.Symbol(node.id)
            raise ValueError("unsupported symbol")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return binary_ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
                return sp.sqrt(visit(node.args[0]))
        raise ValueError(f"unsupported answer expression: {ast.dump(node)}")

    try:
        return visit(ast.parse(python_text, mode="eval"))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        if depth == 0 and text.startswith(separator, cursor):
            parts.append(text[start:cursor])
            cursor += len(separator)
            start = cursor
            continue
        cursor += 1
    parts.append(text[start:])
    return parts


def _unordered_object(items: list[Any]) -> tuple[str, tuple[Any, ...]]:
    flattened: list[Any] = []
    for item in items:
        if isinstance(item, tuple) and item and item[0] == "unordered":
            flattened.extend(item[1])
        else:
            flattened.append(item)
    return "unordered", tuple(flattened)


def _safe_math_object(text: str):
    surface = _surface_normalize(text)
    surface = surface.replace(r"\$", "").replace(r"\!", "")
    surface = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", surface)
    membership = re.search(r"\\in(?![A-Za-z])", surface)
    if membership is not None and membership.start() > 0 and membership.end() < len(surface):
        surface = surface[membership.end() :]
    matrix_start = r"\begin{pmatrix}"
    matrix_end = r"\end{pmatrix}"
    if surface.startswith(matrix_start) and surface.endswith(matrix_end):
        body = surface[len(matrix_start) : -len(matrix_end)]
        values = [_safe_math_object(item) for item in body.split(r"\\")]
        return None if any(value is None for value in values) else ("ordered", tuple(values))

    union_parts = _split_top_level(surface, r"\cup")
    if len(union_parts) > 1:
        values = [_safe_math_object(item) for item in union_parts]
        return None if any(value is None for value in values) else _unordered_object(values)

    if surface.startswith(r"\{") and surface.endswith(r"\}"):
        values = [_safe_math_object(item) for item in _split_top_level(surface[2:-2], ",")]
        return None if any(value is None for value in values) else _unordered_object(values)

    if r"\pm" in surface:
        positive = _safe_math_object(surface.replace(r"\pm", "+"))
        negative = _safe_math_object(surface.replace(r"\pm", "-"))
        if positive is None or negative is None:
            return None
        return _unordered_object([positive, negative])

    if len(surface) >= 2 and surface[0] in "([" and surface[-1] in ")]":
        parts = _split_top_level(surface[1:-1], ",")
        if len(parts) > 1:
            values = [_safe_math_object(item) for item in parts]
            if any(value is None for value in values):
                return None
            if len(values) == 2:
                return "interval_or_pair", surface[0], surface[-1], tuple(values)
            return "ordered", tuple(values)

    comma_parts = _split_top_level(surface, ",")
    if len(comma_parts) > 1:
        values = [_safe_math_object(item) for item in comma_parts]
        return None if any(value is None for value in values) else _unordered_object(values)
    base_number = re.fullmatch(r"([0-9]+)_\{?([0-9]+)\}?", surface)
    if base_number is not None and sp is not None:
        digits, base_text = base_number.groups()
        base = int(base_text)
        if base >= 2 and all(int(digit) < base for digit in digits):
            return sp.Integer(int(digits, base))
    return _safe_sympy_parse(surface)


def _math_objects_equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if sp is not None and isinstance(left, sp.Basic) and isinstance(right, sp.Basic):
        if left == right:
            return True
        try:
            return bool(sp.simplify(left - right) == 0)
        except (TypeError, ValueError):
            return False
    if not isinstance(left, tuple) or not isinstance(right, tuple) or left[0] != right[0]:
        return False
    if left[0] == "unordered":
        unmatched = list(right[1])
        for left_item in left[1]:
            match = next(
                (
                    index
                    for index, right_item in enumerate(unmatched)
                    if _math_objects_equivalent(left_item, right_item)
                ),
                None,
            )
            if match is None:
                return False
            unmatched.pop(match)
        return not unmatched
    if left[0] == "ordered":
        return len(left[1]) == len(right[1]) and all(
            _math_objects_equivalent(a, b) for a, b in zip(left[1], right[1])
        )
    if left[0] == "interval_or_pair":
        return left[1:3] == right[1:3] and all(
            _math_objects_equivalent(a, b) for a, b in zip(left[3], right[3])
        )
    return False


def answers_equivalent(candidate: str, reference: str) -> bool:
    if _UNSAFE_ANSWER_RE.search(candidate):
        return False
    candidate_surface = _surface_normalize(candidate)
    reference_surface = _surface_normalize(reference)
    if candidate_surface == reference_surface:
        return True
    return _math_objects_equivalent(
        _safe_math_object(candidate_surface),
        _safe_math_object(reference_surface),
    )


def generated_answers_equivalent(left: str, right: str, dataset: str) -> bool:
    if dataset.lower() in {"humaneval", "human_eval"}:
        return normalize_answer(left, dataset) == normalize_answer(right, dataset)
    return answers_equivalent(left, right)


def score_exact_number(generation: str, gold_answer: str) -> bool:
    return answers_equivalent(generation, gold_answer)


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


def score_generation_with_status(generation: str, example: Example) -> tuple[bool, str]:
    dataset = example.dataset.lower()
    if dataset == "countdown":
        passed = score_countdown(generation, example.question, example.gold_answer)
        return passed, "passed" if passed else "wrong_answer"
    if dataset in {"humaneval", "human_eval"}:
        from regret_remasking.code_eval import evaluate_humaneval_program

        result = evaluate_humaneval_program(
            generation,
            example.question,
            str(example.metadata["test"]),
            str(example.metadata["entry_point"]),
        )
        return result.passed, result.status
    passed = score_exact_number(generation, example.gold_answer)
    return passed, "passed" if passed else "wrong_answer"


def score_generation(generation: str, example: Example) -> bool:
    return score_generation_with_status(generation, example)[0]


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
    if dataset_key in {"humaneval", "human_eval"}:
        return (
            "Complete the Python function below. Return only executable Python code, "
            "without Markdown fences or explanation.\n\n"
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
            answer = (
                row.get("answer")
                or row.get("canonical_solution")
                or row.get("solution")
                or row.get("gold_answer")
            )
            if question is None or answer is None:
                raise ValueError(f"JSONL row is missing question/answer fields: {row}")
            examples.append(
                Example(
                    example_id=str(row.get("id") or row.get("task_id") or f"{dataset}-{idx}"),
                    question=str(question),
                    gold_answer=str(answer),
                    dataset=dataset,
                    metadata={
                        key: row[key]
                        for key in ("test", "entry_point", "canonical_solution")
                        if key in row
                    },
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

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install `datasets` or pass --jsonl-path with question/answer fields."
        ) from exc

    if dataset_key == "gsm8k":
        hf = load_dataset("openai/gsm8k", "main", split=split)
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
