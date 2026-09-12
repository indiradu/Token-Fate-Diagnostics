#!/usr/bin/env bash
#SBATCH -J phase_lock
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/phase_lock_%j.out
#SBATCH -e logs/phase_lock_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/regret_remasking}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/phase_lock_audit}"
MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
EXAMPLE_INDICES="${EXAMPLE_INDICES:-}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
SEMANTIC_POLICIES="${SEMANTIC_POLICIES:-confidence,posterior,consensus}"
SEMANTIC_POLICY_CONFIGS="${SEMANTIC_POLICY_CONFIGS:-}"
REPRESENTATION_POLICIES="${REPRESENTATION_POLICIES:-drift}"
REPRESENTATION_POLICY_CONFIGS="${REPRESENTATION_POLICY_CONFIGS:-}"
COMPUTE_POLICIES="${COMPUTE_POLICIES:-row_sparse}"
REFERENCE_PLAN_SOURCE="${REFERENCE_PLAN_SOURCE:-dense_a1}"
SEMANTIC_LOCK_FRACTION="${SEMANTIC_LOCK_FRACTION:-0.06}"
REPRESENTATION_LAYER="${REPRESENTATION_LAYER:-24}"
REPRESENTATION_GATE="${REPRESENTATION_GATE:-custom}"
REPRESENTATION_THRESHOLD="${REPRESENTATION_THRESHOLD:-0.02}"
REPRESENTATION_PATIENCE="${REPRESENTATION_PATIENCE:-2}"
REPRESENTATION_MIN_AGE="${REPRESENTATION_MIN_AGE:-1}"
REPRESENTATION_LOCK_FRACTION="${REPRESENTATION_LOCK_FRACTION:-1.0}"
BETA_KL="${BETA_KL:-1.0}"
BETA_FATE="${BETA_FATE:-2.0}"
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
  --representation-policies "$REPRESENTATION_POLICIES"
  --compute-policies "$COMPUTE_POLICIES"
  --reference-plan-source "$REFERENCE_PLAN_SOURCE"
  --semantic-lock-fraction "$SEMANTIC_LOCK_FRACTION"
  --representation-layer "$REPRESENTATION_LAYER"
  --representation-gate "$REPRESENTATION_GATE"
  --representation-threshold "$REPRESENTATION_THRESHOLD"
  --representation-patience "$REPRESENTATION_PATIENCE"
  --representation-min-age "$REPRESENTATION_MIN_AGE"
  --representation-lock-fraction "$REPRESENTATION_LOCK_FRACTION"
  --beta-kl "$BETA_KL"
  --beta-fate "$BETA_FATE"
  --dtype "$DTYPE"
  --seed "$SEED"
)
if [ -n "$EXAMPLE_INDICES" ]; then
  ARGS+=(--example-indices "$EXAMPLE_INDICES")
fi
if [ -n "$SEMANTIC_POLICY_CONFIGS" ]; then
  ARGS+=(--semantic-policy-configs "$SEMANTIC_POLICY_CONFIGS")
fi
if [ -n "$REPRESENTATION_POLICY_CONFIGS" ]; then
  ARGS+=(--representation-policy-configs "$REPRESENTATION_POLICY_CONFIGS")
fi
if [ -n "$REGRET_MODEL_PATH" ]; then
  ARGS+=(--regret-model-path "$REGRET_MODEL_PATH")
fi

PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$PYTHON_BIN" scripts/run_phase_lock.py "${ARGS[@]}"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$PYTHON_BIN" scripts/analyze_phase_lock.py \
  --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/analysis"
