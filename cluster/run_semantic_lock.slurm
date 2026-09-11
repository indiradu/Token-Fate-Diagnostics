#!/usr/bin/env bash
#SBATCH -J semantic_lock
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/semantic_lock_%j.out
#SBATCH -e logs/semantic_lock_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/regret_remasking}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-2}"
OFFSET="${OFFSET:-100}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
SEMANTIC_POLICIES="${SEMANTIC_POLICIES:-confidence,margin,entropy,posterior_kl,persistence_only,token_fate_only}"
SEMANTIC_POLICY_CONFIGS="${SEMANTIC_POLICY_CONFIGS:-}"
SEMANTIC_LOCK_FRACTION="${SEMANTIC_LOCK_FRACTION:-0.06}"
SEMANTIC_MIN_CONFIDENCE="${SEMANTIC_MIN_CONFIDENCE:-0.0}"
SEMANTIC_MIN_MARGIN="${SEMANTIC_MIN_MARGIN:-0.0}"
SEMANTIC_MAX_KL="${SEMANTIC_MAX_KL:-}"
SEMANTIC_MIN_RUNLENGTH="${SEMANTIC_MIN_RUNLENGTH:-1}"
SEMANTIC_MIN_BLOCK_AGE="${SEMANTIC_MIN_BLOCK_AGE:-0}"
SEMANTIC_OPTIONAL_STEPS_PER_BLOCK="${SEMANTIC_OPTIONAL_STEPS_PER_BLOCK:--1}"
SEMANTIC_REQUIRE_POSTERIOR_HISTORY="${SEMANTIC_REQUIRE_POSTERIOR_HISTORY:-0}"
REGRET_MODEL_PATH="${REGRET_MODEL_PATH:-}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-17}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
nvidia-smi

ARGS=(
  --model-name "$MODEL_NAME"
  --dataset "$DATASET"
  --split "$SPLIT"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --output-dir "$OUTPUT_DIR"
  --steps "$STEPS"
  --gen-length "$GEN_LENGTH"
  --block-length "$BLOCK_LENGTH"
  --semantic-policies "$SEMANTIC_POLICIES"
  --representation-policies drift
  --compute-policies dense
  --semantic-lock-fraction "$SEMANTIC_LOCK_FRACTION"
  --semantic-min-confidence "$SEMANTIC_MIN_CONFIDENCE"
  --semantic-min-margin "$SEMANTIC_MIN_MARGIN"
  --semantic-min-runlength "$SEMANTIC_MIN_RUNLENGTH"
  --semantic-min-block-age "$SEMANTIC_MIN_BLOCK_AGE"
  --semantic-optional-steps-per-block "$SEMANTIC_OPTIONAL_STEPS_PER_BLOCK"
  --dtype "$DTYPE"
  --seed "$SEED"
)
if [ -n "$SEMANTIC_POLICY_CONFIGS" ]; then
  ARGS+=(--semantic-policy-configs "$SEMANTIC_POLICY_CONFIGS")
fi
if [ -n "$SEMANTIC_MAX_KL" ]; then
  ARGS+=(--semantic-max-kl "$SEMANTIC_MAX_KL")
fi
if [ "$SEMANTIC_REQUIRE_POSTERIOR_HISTORY" = "1" ]; then
  ARGS+=(--semantic-require-posterior-history)
fi
if [ -n "$REGRET_MODEL_PATH" ]; then
  ARGS+=(--regret-model-path "$REGRET_MODEL_PATH")
fi

PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$PYTHON_BIN" scripts/run_phase_lock.py "${ARGS[@]}"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$PYTHON_BIN" scripts/analyze_phase_lock.py \
  --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/paired_analysis"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$PYTHON_BIN" scripts/analyze_semantic_lock.py \
  --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/semantic_analysis"
