#!/usr/bin/env bash
#SBATCH -J phase_compute
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/phase_compute_%j.out
#SBATCH -e logs/phase_compute_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-5}"
OFFSET="${OFFSET:-0}"
EXAMPLE_INDICES="${EXAMPLE_INDICES:-}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
SEMANTIC_POLICY_CONFIGS="${SEMANTIC_POLICY_CONFIGS:?SEMANTIC_POLICY_CONFIGS is required}"
SEMANTIC_LABELS="${SEMANTIC_LABELS:-}"
REPRESENTATION_POLICY_CONFIGS="${REPRESENTATION_POLICY_CONFIGS:?REPRESENTATION_POLICY_CONFIGS is required}"
REPRESENTATION_LABEL="${REPRESENTATION_LABEL:-robust_delayed_one_update}"
TIMING_WARMUPS="${TIMING_WARMUPS:-1}"
TIMING_REPEATS="${TIMING_REPEATS:-5}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-17}"
MAX_PREEXISTING_GPU_MEMORY_MB="${MAX_PREEXISTING_GPU_MEMORY_MB:-512}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Timing on a GPU already used by an out-of-band process is not scientifically
# interpretable. Slurm exports the allocated physical GPU index on this cluster;
# fail closed if that device has substantial memory allocated before Python starts.
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES:-}"
if [[ "$ALLOCATED_GPU" =~ ^[0-9]+$ ]]; then
  PREEXISTING_GPU_MEMORY_MB="$(
    nvidia-smi --id="$ALLOCATED_GPU" --query-compute-apps=used_memory \
      --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}'
  )"
  echo "preexisting_gpu_memory_mb=$PREEXISTING_GPU_MEMORY_MB"
  if (( PREEXISTING_GPU_MEMORY_MB > MAX_PREEXISTING_GPU_MEMORY_MB )); then
    echo "Refusing contaminated timing device: GPU $ALLOCATED_GPU already uses ${PREEXISTING_GPU_MEMORY_MB} MiB" >&2
    exit 42
  fi
else
  echo "Cannot verify allocated physical GPU from CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU" >&2
  exit 43
fi

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
  --semantic-policy-configs "$SEMANTIC_POLICY_CONFIGS"
  --representation-policy-configs "$REPRESENTATION_POLICY_CONFIGS"
  --representation-label "$REPRESENTATION_LABEL"
  --timing-warmups "$TIMING_WARMUPS"
  --timing-repeats "$TIMING_REPEATS"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --dtype "$DTYPE"
  --seed "$SEED"
)
if [ -n "$EXAMPLE_INDICES" ]; then
  ARGS+=(--example-indices "$EXAMPLE_INDICES")
fi
if [ -n "$SEMANTIC_LABELS" ]; then
  ARGS+=(--semantic-labels "$SEMANTIC_LABELS")
fi

PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/benchmark_phase_lock_compute.py "${ARGS[@]}"
