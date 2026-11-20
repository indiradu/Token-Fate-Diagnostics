from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from regret_remasking.data import Example, build_prompt
from regret_remasking.locking_manifest import sha256_file, write_frozen_jsonl

SCHEMA_VERSION = 1
DEFAULT_SEED = 23
DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
KODCODE_REVISION = "45d9e3b725e585462fee2bb026e5e32a2eb0a8d7"
KODCODE_EXPECTED_SHA256 = "23f1bfadf3403848d82737d27245aedc2d7a0c37847d7811d65f69f8cb7da0c2"

TRAIN_SPLITS = (
    "fit",
    "validation",
    "selection",
    "fit_replacement",
    "fit_replacement_extra",
    "validation_replacement",
)
TRAIN_TARGETS = {
    "fit": {"math": 4000, "code": 2000, "qa": 2000},
    "validation": {"math": 200, "code": 100, "qa": 100},
    "selection": {"math": 100, "code": 100},
}
FIT_REPLACEMENT_FRACTION = 0.20
PROJECTION_BOOTSTRAP_COUNT = 64

REQUIRED_PAPER_FIELDS = (
    "record_id",
    "split",
    "family",
    "dataset_name",
    "example_id",
    "source_id",
    "source_path",
    "source_revision",
    "source_sha256",
    "prompt_text",
    "prompt_hash",
    "chat_prompt_hash",
    "question",
    "gold_answer",
    "metadata",
    "seed",
    "teacher_config",
)


@dataclass(frozen=True)
class SourceSpec:
    dataset_name: str
    family: str
    path: Path
    revision: str


@dataclass(frozen=True)
class Candidate:
    example: Example
    family: str
    dataset_name: str
    source_path: Path
    source_revision: str
    source_sha256: str
    prompt_text: str
    metadata: dict[str, Any]


class PromptTokenizer:
    def __init__(self, tokenizer: Any | None = None, *, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer(model_id)
        self.model_id = model_id

    @classmethod
    def _load_tokenizer(cls, model_id: str) -> Any | None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None
        try:
            return AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self.tokenizer is not None

    def chat_prompt(self, prompt_text: str) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return str(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass
        return f"<|user|>\n{prompt_text}\n<|assistant|>\n"

    def count(self, prompt_text: str) -> int:
        chat_prompt = self.chat_prompt(prompt_text)
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(chat_prompt, add_special_tokens=False))
        return len(_fallback_tokenize(chat_prompt))


def stable_json_hash(value: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.lower().split())


def prompt_hash(prompt_text: str) -> str:
    return stable_json_hash(normalize_prompt_text(prompt_text))


def default_teacher_config(record_key: str, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    digest = hashlib.sha256(f"{seed}:{record_key}".encode("utf-8")).digest()
    gen_length = 128 if digest[0] % 2 == 0 else 256
    divisor = (8, 16, 32)[digest[1] % 3]
    return {
        "proposal": "entropy_ranked_one_token_commitments",
        "gen_length": gen_length,
        "steps": gen_length,
        "block_length": gen_length // divisor,
        "block_divisor": divisor,
        "retained_step_fraction": 0.3,
        "seed": seed,
        "frozen": True,
    }


def make_paper_record(
    candidate: Candidate,
    *,
    split: str,
    seed: int = DEFAULT_SEED,
    tokenizer: PromptTokenizer | None = None,
) -> dict[str, Any]:
    source_id = f"{candidate.dataset_name}:{candidate.example.example_id}"
    record_id = f"{split}:{candidate.family}:{source_id}"
    tokenizer = tokenizer or PromptTokenizer()
    chat_prompt = tokenizer.chat_prompt(candidate.prompt_text)
    metadata = dict(candidate.example.metadata)
    metadata.update(candidate.metadata)
    return {
        "record_id": record_id,
        "split": split,
        "family": candidate.family,
        "dataset_name": candidate.dataset_name,
        "dataset": candidate.dataset_name,
        "example_id": candidate.example.example_id,
        "source_id": source_id,
        "source_path": str(candidate.source_path.resolve()),
        "source_revision": candidate.source_revision,
        "source_sha256": candidate.source_sha256,
        "prompt_text": candidate.prompt_text,
        "prompt_hash": prompt_hash(candidate.prompt_text),
        "chat_prompt_hash": stable_json_hash(chat_prompt),
        "question": candidate.example.question,
        "gold_answer": candidate.example.gold_answer,
        "metadata": metadata,
        "seed": seed,
        "teacher_config": default_teacher_config(record_id, seed=seed),
    }


def validate_paper_records(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tokenizer: PromptTokenizer | None = None,
    train_token_cap: int = 256,
) -> None:
    tokenizer = tokenizer or PromptTokenizer()
    owner_by_source_id: dict[str, str] = {}
    owner_by_prompt_hash: dict[str, str] = {}
    for split, records in manifests.items():
        local_record_ids: set[str] = set()
        for idx, record in enumerate(records, start=1):
            missing = [field for field in REQUIRED_PAPER_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{split}:{idx} missing required fields {missing}")
            if record["split"] != split:
                raise ValueError(f"{split}:{idx} has split {record['split']!r}")
            if not isinstance(record["metadata"], dict):
                raise ValueError(f"{split}:{idx} metadata must be an object")
            if not isinstance(record["teacher_config"], dict):
                raise ValueError(f"{split}:{idx} teacher_config must be an object")
            if record["prompt_hash"] != prompt_hash(str(record["prompt_text"])):
                raise ValueError(f"{split}:{idx} prompt_hash mismatch")
            if str(record["record_id"]) in local_record_ids:
                raise ValueError(f"{split}:{idx} duplicate record_id {record['record_id']!r}")
            local_record_ids.add(str(record["record_id"]))

            if split in TRAIN_SPLITS:
                count = tokenizer.count(str(record["prompt_text"]))
                if count > train_token_cap:
                    raise ValueError(
                        f"{split}:{idx} prompt has {count} chat tokens; cap is {train_token_cap}"
                    )
                source_id = str(record["source_id"])
                previous_source = owner_by_source_id.get(source_id)
                if previous_source is not None:
                    raise ValueError(f"{source_id!r} appears in {previous_source!r} and {split!r}")
                owner_by_source_id[source_id] = split

                phash = str(record["prompt_hash"])
                previous_prompt = owner_by_prompt_hash.get(phash)
                if previous_prompt is not None:
                    raise ValueError(f"prompt hash appears in {previous_prompt!r} and {split!r}")
                owner_by_prompt_hash[phash] = split


def write_manifest_set(
    output_dir: Path,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    written: dict[str, dict[str, Any]] = {}
    for split, records in manifests.items():
        path = output_dir / f"{split}.jsonl"
        text_hash = write_frozen_jsonl(path, records)
        written[split] = summary_entry(path, list(records), text_hash)
    return written


def summary_entry(path: Path, records: list[Mapping[str, Any]], text_hash: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": text_hash,
        "count": len(records),
        "family_counts": _count_by(records, "family"),
        "dataset_counts": _count_by(records, "dataset_name"),
        "split_counts": _count_by(records, "split"),
    }


def copy_preserved_manifest(src: Path, dst: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(src) != sha256_file(dst):
            raise FileExistsError(f"refusing to overwrite nonidentical preserved manifest: {dst}")
    else:
        shutil.copy2(src, dst)
    return sha256_file(dst)


def load_gsm8k_arrow(path: Path, *, revision: str | None = None) -> list[Candidate]:
    from datasets import Dataset

    source_sha = sha256_file(path)
    source_revision = revision or source_sha
    dataset = Dataset.from_file(str(path))
    candidates: list[Candidate] = []
    for index, row in enumerate(dataset):
        question = str(row["question"])
        answer = str(row["answer"])
        example = Example(
            example_id=f"gsm8k-train-{index}",
            question=question,
            gold_answer=answer,
            dataset="gsm8k",
            metadata={"source_index": index},
        )
        candidates.append(
            Candidate(
                example=example,
                family="math",
                dataset_name="gsm8k",
                source_path=path,
                source_revision=source_revision,
                source_sha256=source_sha,
                prompt_text=build_prompt(question, "gsm8k"),
                metadata={"answer_hash": stable_json_hash(answer)},
            )
        )
    return candidates


def load_alpaca_parquet(path: Path, *, revision: str) -> list[Candidate]:
    import pyarrow.parquet as pq

    source_sha = sha256_file(path)
    candidates: list[Candidate] = []
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches():
        rows = batch.to_pylist()
        for row in rows:
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            if not instruction or not output:
                row_index += 1
                continue
            question = _format_alpaca_question(instruction, input_text)
            example = Example(
                example_id=f"alpaca-train-{row_index}",
                question=question,
                gold_answer=output,
                dataset="alpaca",
                metadata={"source_index": row_index},
            )
            candidates.append(
                Candidate(
                    example=example,
                    family="qa",
                    dataset_name="alpaca",
                    source_path=path,
                    source_revision=revision,
                    source_sha256=source_sha,
                    prompt_text=question,
                    metadata={
                        "instruction_hash": stable_json_hash(instruction),
                        "input_hash": stable_json_hash(input_text),
                        "answer_hash": stable_json_hash(output),
                    },
                )
            )
            row_index += 1
    return candidates


def load_kodcode_jsonl(path: Path, *, revision: str = KODCODE_REVISION) -> list[Candidate]:
    source_sha = sha256_file(path)
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            solution = str(row.get("canonical_solution", "")).rstrip("\n")
            task_id = str(row.get("task_id") or f"kodcode-{row_index}")
            if not prompt or not solution:
                continue
            metadata = {
                "source_index": row_index,
                "entry_point": row.get("entry_point"),
                "test": row.get("test"),
                "canonical_solution": solution,
                "canonical_solution_hash": stable_json_hash(solution),
                "test_hash": stable_json_hash(str(row.get("test", ""))),
            }
            example = Example(
                example_id=_safe_id(task_id),
                question=prompt,
                gold_answer=solution,
                dataset="kodcode",
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
            candidates.append(
                Candidate(
                    example=example,
                    family="code",
                    dataset_name="kodcode",
                    source_path=path,
                    source_revision=revision,
                    source_sha256=source_sha,
                    prompt_text=build_prompt(prompt, "humaneval"),
                    metadata={
                        "entry_point": row.get("entry_point"),
                        "test_hash": stable_json_hash(str(row.get("test", ""))),
                        "canonical_solution_hash": stable_json_hash(solution),
                    },
                )
            )
    return candidates


def load_humaneval_reference(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_humaneval_decontaminator(records: Sequence[Mapping[str, Any]]) -> "HumanEvalDecontaminator":
    return HumanEvalDecontaminator(records)


class HumanEvalDecontaminator:
    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.prompt_norms: set[str] = set()
        self.docstring_norms: set[str] = set()
        self.solution_asts: set[str] = set()
        self.test_fingerprints: set[str] = set()
        self.prompt_ngrams: list[set[str]] = []
        for row in records:
            prompt = str(row.get("prompt", ""))
            solution = str(row.get("canonical_solution", ""))
            test = str(row.get("test", ""))
            self.prompt_norms.add(normalize_prompt_text(prompt))
            docstring = _extract_docstring(prompt)
            if docstring:
                self.docstring_norms.add(normalize_prompt_text(docstring))
            solution_ast = _canonical_ast(solution)
            if solution_ast:
                self.solution_asts.add(solution_ast)
            self.test_fingerprints.add(_test_fingerprint(test))
            self.prompt_ngrams.append(_token_ngrams(prompt, n=5))

    def reason(self, candidate: Candidate) -> str | None:
        prompt = candidate.example.question
        solution = str(candidate.example.metadata.get("canonical_solution", ""))
        test = str(candidate.example.metadata.get("test", ""))
        normalized_prompt = normalize_prompt_text(prompt)
        if normalized_prompt in self.prompt_norms:
            return "humaneval_normalized_prompt"
        docstring = _extract_docstring(prompt)
        if docstring and normalize_prompt_text(docstring) in self.docstring_norms:
            return "humaneval_normalized_docstring"
        solution_ast = _canonical_ast(solution)
        if solution_ast and solution_ast in self.solution_asts:
            return "humaneval_solution_ast"
        if _test_fingerprint(test) in self.test_fingerprints:
            return "humaneval_test_fingerprint"
        candidate_ngrams = _token_ngrams(prompt, n=5)
        if candidate_ngrams:
            for reference_ngrams in self.prompt_ngrams:
                if _jaccard(candidate_ngrams, reference_ngrams) >= 0.55:
                    return "humaneval_prompt_5gram_jaccard_ge_0.55"
        return None


def select_training_records(
    pools: Mapping[str, Sequence[Candidate]],
    targets: Mapping[str, Mapping[str, int]],
    *,
    tokenizer: PromptTokenizer,
    seed: int = DEFAULT_SEED,
    token_cap: int = 256,
    decontaminator: HumanEvalDecontaminator | None = None,
    forbidden_fit_source_ids: set[str] | None = None,
    forbidden_all_source_ids: set[str] | None = None,
    forbidden_prompt_hashes: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    forbidden_fit_source_ids = forbidden_fit_source_ids or set()
    forbidden_all_source_ids = forbidden_all_source_ids or set()
    forbidden_prompt_hashes = forbidden_prompt_hashes or set()
    used_source_ids: set[str] = set(forbidden_all_source_ids)
    used_prompt_hashes: set[str] = set(forbidden_prompt_hashes)
    filter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in targets}
    qa_code_intent_hashes: set[str] = set()

    code_pool_ngrams = [_token_ngrams(candidate.prompt_text, n=5) for candidate in pools.get("code", ())]

    for split, split_targets in targets.items():
        for family, count in split_targets.items():
            chosen = 0
            for candidate in _shuffled(pools[family], seed, split, family):
                source_id = f"{candidate.dataset_name}:{candidate.example.example_id}"
                phash = prompt_hash(candidate.prompt_text)
                reason = _candidate_filter_reason(
                    candidate,
                    split=split,
                    source_id=source_id,
                    phash=phash,
                    tokenizer=tokenizer,
                    token_cap=token_cap,
                    used_source_ids=used_source_ids,
                    used_prompt_hashes=used_prompt_hashes,
                    forbidden_fit_source_ids=forbidden_fit_source_ids,
                    decontaminator=decontaminator,
                    code_pool_ngrams=code_pool_ngrams if family == "qa" else None,
                )
                if reason is not None:
                    filter_counts[f"{split}:{family}"][reason] += 1
                    continue
                record = make_paper_record(candidate, split=split, seed=seed, tokenizer=tokenizer)
                selected[split].append(record)
                used_source_ids.add(source_id)
                used_prompt_hashes.add(phash)
                if family == "qa":
                    qa_code_intent_hashes.add(phash)
                chosen += 1
                if chosen == count:
                    break
            if chosen != count:
                raise ValueError(
                    f"{split}:{family} requested {count} records but selected {chosen}; "
                    f"filters={dict(filter_counts[f'{split}:{family}'])}"
                )

    return selected, {
        "filter_counts": {
            key: dict(counter)
            for key, counter in sorted(filter_counts.items())
            if counter
        },
        "used_source_count": len(used_source_ids) - len(forbidden_all_source_ids),
        "used_prompt_count": len(used_prompt_hashes) - len(forbidden_prompt_hashes),
        "qa_code_intent_prompt_hashes": sorted(qa_code_intent_hashes),
    }


def select_fit_replacements(
    pools: Mapping[str, Sequence[Candidate]],
    fit_records: Sequence[Mapping[str, Any]],
    consumed_records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: PromptTokenizer,
    seed: int = DEFAULT_SEED,
    token_cap: int = 256,
    decontaminator: HumanEvalDecontaminator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    consumed_source_ids = {str(record["source_id"]) for record in consumed_records}
    consumed_prompt_hashes = {
        str(record.get("prompt_hash") or record.get("content_hash"))
        for record in consumed_records
        if record.get("prompt_hash") or record.get("content_hash")
    }
    by_family = Counter(str(record["family"]) for record in fit_records)
    replacements: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    filter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for family, fit_count in sorted(by_family.items()):
        target = int(fit_count * FIT_REPLACEMENT_FRACTION)
        counts[family] = target
        chosen = 0
        for candidate in _shuffled(pools[family], seed, "fit_replacement", family):
            source_id = f"{candidate.dataset_name}:{candidate.example.example_id}"
            phash = prompt_hash(candidate.prompt_text)
            reason = _candidate_filter_reason(
                candidate,
                split="fit_replacement",
                source_id=source_id,
                phash=phash,
                tokenizer=tokenizer,
                token_cap=token_cap,
                used_source_ids=consumed_source_ids,
                used_prompt_hashes=consumed_prompt_hashes,
                forbidden_fit_source_ids=set(),
                decontaminator=decontaminator,
                code_pool_ngrams=None,
            )
            if reason is not None:
                filter_counts[family][reason] += 1
                continue
            record = make_paper_record(
                candidate,
                split="fit_replacement",
                seed=seed,
                tokenizer=tokenizer,
            )
            replacements.append(record)
            consumed_source_ids.add(source_id)
            consumed_prompt_hashes.add(phash)
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise ValueError(f"fit_replacement:{family} requested {target}, selected {chosen}")
    return replacements, {
        "target_counts": counts,
        "filter_counts": {key: dict(value) for key, value in sorted(filter_counts.items()) if value},
    }


def select_reserve_records(
    pools: Mapping[str, Sequence[Candidate]],
    targets: Mapping[str, Mapping[str, int]],
    consumed_records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: PromptTokenizer,
    seed: int = DEFAULT_SEED,
    token_cap: int = 256,
    decontaminator: HumanEvalDecontaminator | None = None,
    strict_code_syntax: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    consumed_source_ids = {str(record["source_id"]) for record in consumed_records}
    consumed_prompt_hashes = {
        str(record.get("prompt_hash") or record.get("content_hash"))
        for record in consumed_records
        if record.get("prompt_hash") or record.get("content_hash")
    }
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in targets}
    filter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    code_pool_ngrams = [_token_ngrams(candidate.prompt_text, n=5) for candidate in pools.get("code", ())]

    for split, split_targets in targets.items():
        for family, target in split_targets.items():
            chosen = 0
            for candidate in _shuffled(pools[family], seed, split, family):
                source_id = f"{candidate.dataset_name}:{candidate.example.example_id}"
                phash = prompt_hash(candidate.prompt_text)
                reason = _candidate_filter_reason(
                    candidate,
                    split=split,
                    source_id=source_id,
                    phash=phash,
                    tokenizer=tokenizer,
                    token_cap=token_cap,
                    used_source_ids=consumed_source_ids,
                    used_prompt_hashes=consumed_prompt_hashes,
                    forbidden_fit_source_ids=set(),
                    decontaminator=decontaminator,
                    code_pool_ngrams=code_pool_ngrams if family == "qa" else None,
                    strict_code_syntax=strict_code_syntax,
                )
                if reason is not None:
                    filter_counts[f"{split}:{family}"][reason] += 1
                    continue
                record = make_paper_record(candidate, split=split, seed=seed, tokenizer=tokenizer)
                selected[split].append(record)
                consumed_source_ids.add(source_id)
                consumed_prompt_hashes.add(phash)
                chosen += 1
                if chosen == target:
                    break
            if chosen != target:
                raise ValueError(
                    f"{split}:{family} requested {target} records but selected {chosen}; "
                    f"filters={dict(filter_counts[f'{split}:{family}'])}"
                )

    return selected, {
        "target_counts": {
            split: dict(split_targets) for split, split_targets in sorted(targets.items())
        },
        "filter_counts": {
            key: dict(counter)
            for key, counter in sorted(filter_counts.items())
            if counter
        },
        "used_source_count": len(consumed_source_ids)
        - len({str(record["source_id"]) for record in consumed_records}),
        "used_prompt_count": len(consumed_prompt_hashes)
        - len(
            {
                str(record.get("prompt_hash") or record.get("content_hash"))
                for record in consumed_records
                if record.get("prompt_hash") or record.get("content_hash")
            }
        ),
        "strict_code_syntax": strict_code_syntax,
    }


def projection_bootstrap_ids(fit_records: Sequence[Mapping[str, Any]], *, count: int = 64) -> list[str]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in fit_records:
        by_family[str(record["family"])].append(record)
    families = sorted(by_family)
    if not families:
        raise ValueError("fit_records is empty")
    result: list[str] = []
    index = 0
    while len(result) < count:
        family = families[index % len(families)]
        records = by_family[family]
        offset = index // len(families)
        if offset < len(records):
            result.append(str(records[offset]["record_id"]))
        index += 1
        if index > count * len(families) + sum(len(rows) for rows in by_family.values()):
            raise ValueError(f"could not select {count} bootstrap ids from fit records")
    return result


def _candidate_filter_reason(
    candidate: Candidate,
    *,
    split: str,
    source_id: str,
    phash: str,
    tokenizer: PromptTokenizer,
    token_cap: int,
    used_source_ids: set[str],
    used_prompt_hashes: set[str],
    forbidden_fit_source_ids: set[str],
    decontaminator: HumanEvalDecontaminator | None,
    code_pool_ngrams: Sequence[set[str]] | None,
    strict_code_syntax: bool = False,
) -> str | None:
    if source_id in used_source_ids:
        return "source_id_already_consumed"
    if phash in used_prompt_hashes:
        return "prompt_hash_already_consumed"
    if split == "fit" and source_id in forbidden_fit_source_ids:
        return "old_calibration_or_selection_excluded_from_fit"
    token_count = tokenizer.count(candidate.prompt_text)
    if token_count > token_cap:
        return "chat_prompt_tokens_gt_256"
    if decontaminator is not None and candidate.family == "code":
        reason = decontaminator.reason(candidate)
        if reason is not None:
            return reason
    if strict_code_syntax and candidate.family == "code":
        reason = _code_syntax_reason(candidate)
        if reason is not None:
            return reason
    if code_pool_ngrams is not None and _looks_like_code_intent(candidate.prompt_text):
        candidate_ngrams = _token_ngrams(candidate.prompt_text, n=5)
        if any(_jaccard(candidate_ngrams, ref) >= 0.35 for ref in code_pool_ngrams if ref):
            return "qa_code_intent_overlaps_code_pool"
    return None


def _format_alpaca_question(instruction: str, input_text: str) -> str:
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _fallback_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _shuffled(items: Sequence[Candidate], seed: int, *parts: str) -> list[Candidate]:
    keyed = []
    for item in items:
        key = stable_json_hash(":".join((str(seed), *parts, item.dataset_name, item.example.example_id)))
        keyed.append((key, item))
    return [item for _, item in sorted(keyed, key=lambda pair: pair[0])]


def _count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _extract_docstring(prompt: str) -> str:
    try:
        module = ast.parse(prompt)
    except SyntaxError:
        return ""
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ast.get_docstring(node, clean=True) or ""
    return ast.get_docstring(module, clean=True) or ""


def _canonical_ast(code: str) -> str:
    try:
        tree = ast.parse(_dedent_body(code))
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(node, attr, None)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _dedent_body(code: str) -> str:
    lines = code.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return code
    indent = min(len(line) - len(line.lstrip()) for line in nonempty)
    return "\n".join(line[indent:] for line in lines)


def _test_fingerprint(test: str) -> str:
    return stable_json_hash(re.sub(r"\s+", "", test.lower()))


def _token_ngrams(text: str, *, n: int) -> set[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|[-+*/%]", text.lower())
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _looks_like_code_intent(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "python",
        "function",
        "code",
        "algorithm",
        "return",
        "list of integers",
        "write a program",
        "implement",
    )
    return any(marker in lowered for marker in markers)


def _code_syntax_reason(candidate: Candidate) -> str | None:
    if not _python_syntax_ok(candidate.example.question):
        return "code_prompt_syntax_error"
    solution = str(candidate.example.metadata.get("canonical_solution", candidate.example.gold_answer))
    if not _canonical_ast(solution):
        return "code_solution_syntax_error"
    test = str(candidate.example.metadata.get("test", ""))
    if test and not _python_syntax_ok(test):
        return "code_test_syntax_error"
    return None


def _python_syntax_ok(code: str) -> bool:
    try:
        ast.parse(_dedent_body(code))
    except SyntaxError:
        return False
    return True
