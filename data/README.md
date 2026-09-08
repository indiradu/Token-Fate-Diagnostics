# Local Evaluation Data

## HumanEval

- File: `humaneval.jsonl`
- Source: `https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz`
- Records: 164
- Decompressed SHA-256:
  `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`
- License: MIT, following the upstream OpenAI HumanEval repository.

The local scorer executes generated programs only through the Bubblewrap
isolation path in `src/regret_remasking/code_eval.py`.

## AIME 2024

- File: `aime2024.jsonl`
- Source: `HuggingFaceH4/aime_2024`, revision `2fe88a2`
- Records: 30
- Converted JSONL SHA-256:
  `81bab9936923b4f858d4666f1694f329360d7d3021f0a4e7284ceea93fb4bfc9`

The JSONL is a mechanical conversion of the upstream parquet file. It keeps
the problem, answer, solution, source URL, and year fields and adds stable
`aime2024-*` ids.

## MATH-500 Level 5

- File: `math500_level5.jsonl`
- Records: 134
- Construction: the `level == 5` subset of `HuggingFaceH4/MATH-500`.

## MATH-500 Levels 1-4 Comparator

- File: `math500_levels1to4_54_seed23.jsonl`
- Records: 54, disjoint from the Level-5 population by construction.
- SHA-256:
  `c07bbddb62a831030c5a7b7df0eb060dfa818a73bf7f2b6aba8ef3615ea66230`
- Level quotas: 6 Level-1, 13 Level-2, 16 Level-3, and 19 Level-4.
- Construction: proportional stratified sampling from the 366 non-Level-5
  `HuggingFaceH4/MATH-500` rows, with per-level seeds `23 + level`, followed by
  a seed-23 shuffle.

This fixed file is used for both gen64 and gen256 so the difficulty-by-horizon
comparison does not change examples across generation lengths.
