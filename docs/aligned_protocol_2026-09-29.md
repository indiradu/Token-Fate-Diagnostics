# Aligned evaluation protocol (2026-09-29)

This is the shared setting for comparing locking methods on LLaDA. Any method, from either line of work, is comparable with the others only when it is run under this protocol. The reference campaign is `results/aligned_standard_g256_20260929/`.

## Model and hardware

| Item | Value |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct`, snapshot `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| Remote code | `modeling_llada.py` sha256 `98bac7e53fef0bb7ca01e3716c11a7f710d183e10dbb9783b88db9dbba2e3766` |
| Precision | BF16 |
| GPU | One NVIDIA RTX 5000 Ada Generation (32 GB), no other process on the device |
| Batch size | 1 |
| Seed | 23 |

## Data

124 fixed questions: 54 held-out MATH Level-5 problems and HumanEval tasks 0–69. Use the manifests unchanged.

| File | sha256 |
|---|---|
| `manifests/eval_math.jsonl` | `4fbeff1ae87b860843a8e8c70e3557c3e6024fb5151056837f09b96ce4123b37` |
| `manifests/eval_code.jsonl` | `6dbae289cbbe4049f694f08f43516c6f322e40214ad7d0d7601fedd1f9f784c7` |
| `manifests/smoke_pair.jsonl` (smoke only) | `e6f09a867963a56d0d358406779c2e4f5e24d7a0b6d80c40f60b0e22f43a1e96` |

Prompts are built from each manifest row by the repository's `build_prompt` and the LLaDA chat template.

## Baseline (B0)

This is the standard LLaDA low-confidence sampler, the same rule as Fast-dLLM's official "fixed full recompute" baseline:

- 256 generated positions and 256 steps: one token per step.
- Semi-autoregressive blocks of 32, filled left to right.
- Each step, run a full forward pass over the whole sequence with no caching. Commit the masked position in the current block whose argmax token has the highest probability. The argmax excludes the MASK token.
- Temperature 0 and no classifier-free guidance.
- Decode until no masks remain. There is no early stop at EOS, and every position must be committed.

On this panel B0 scores 43/124: 5/54 MATH and 38/70 HumanEval. The official Fast-dLLM baseline gives the same score on the same panel.

## Policies in the reference campaign

All policies share the baseline's model, data, length, seed and batch size.

| Policy id | Arm | Rule |
|---|---|---|
| `B0` | `B0` | Standard sampler above. |
| `RepKL-freeze-0.0005` | `S-sparse`, representation KL | KL(current ‖ previous) of the softmaxed final pre-vocabulary hidden state (T = 1) ≤ 0.0005. The token's all-layer state is then frozen and its query/projection/FFN/head rows are skipped. |
| `SureLock-gen-0.005` | `S-sparse`, vocabulary KL | Vocabulary-posterior KL ≤ 0.005 and uncertainty in the lowest 20% of current candidates. Generated tokens only. |
| `SureLock-prompt-0.005` | `S-sparse`, vocabulary KL | Same as above, but prompt rows are also eligible (`freeze_scope = all_unmasked`). This is the SureLock paper's Fig. 1 behavior and the collaborator's implementation choice. |
| `TraceLock-0.95` | `T-adaptive` | Trained controller (last three hidden states and two differences, 832 features, 3-layer Transformer). Commits every position in its 32-token sliding window with score > 0.95, otherwise the single highest-scoring position. It keeps its own sliding window, as the paper defines it. |
| `Confidence-0.9` | `C-block` | Fast-dLLM-style threshold decoding inside the standard blocks. Commit every masked position in the current block with confidence ≥ 0.9, otherwise the single most confident. |

Rules shared by all freeze policies:

- A token is frozen only after the model has processed it unmasked.
- EOS/EOT are never frozen.
- Frozen rows still serve their cached K/V to every active query.

Exact JSON configs are in `results/aligned_standard_g256_20260929/policies.json`.

Declared differences from the original papers:

- SureLock uses blocks of 32 rather than one block over the whole output. It uses our safe snapshot timing rather than freezing a token in the step it is revealed.
- TraceLock uses the 0.95 threshold from the paper. It does not use the released code's four-accepts-per-step cap or its EOS gate.
- Fast-dLLM computes confidence in float64 over an argmax that may include MASK. Ours uses float32 log-softmax with MASK excluded.

## Timing

- Load the model once. Run one warm-up decode per policy, then three timed repeats of every question.
- Rotate the policy order per question and repeat.
- The timer wraps the whole decode: CUDA sync, then `perf_counter`, through the final sync. It includes gates, controllers and caching. It excludes model loading, scoring and file writes.
- Per question, use the median of the three repeats.
- **Speedup** = sum of B0 medians ÷ sum of method medians − 1. Convert "time saved" g = 1 − t/t₀ with speedup = 1/(1 − g) − 1.
- Report FLOPs separately. Use the SureLock §2.4 convention (GEMMs only, 2 FLOPs per multiply-add), plus the vocabulary head and any learned controller.

## Scoring and statistics

- **MATH:** the repository scorer (`data.score_generation_with_status`: final-answer extraction plus SymPy equivalence). Not Math-Verify.
- **HumanEval:** isolated single-sample execution of the task tests (`code_eval.evaluate_humaneval_program`).
- Correctness uses repeat 1. The run is rejected if repeats differ in any output or work counter.
- Report correct counts per dataset and in total. Report the change against B0 with a 10,000-sample bootstrap over questions (never over repeats or tokens), speedup with the same bootstrap, the number of changed outputs, and right→wrong / wrong→right counts.

## What a new method must report

- Its exact config, source hash and checkpoint hashes.
- Records for all 124 × 3 runs, including `token_ids`, `nfe`, work counters and latency.
- Its numbers next to a B0 rerun in the same job on the same GPU. Never compare against B0 timings from another run.
