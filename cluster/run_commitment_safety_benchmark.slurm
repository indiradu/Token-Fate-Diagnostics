#!/usr/bin/env bash
#SBATCH -J pl_commit_bench
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/pl_commit_bench_%j.out
#SBATCH -e logs/pl_commit_bench_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATASET="${DATASET:?DATASET is required}"
EXAMPLE_INDICES="${EXAMPLE_INDICES:?EXAMPLE_INDICES is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
PROJECTION="${PROJECTION:?PROJECTION is required}"
PREDICTOR_DIR="${PREDICTOR_DIR:?PREDICTOR_DIR is required}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/benchmark_commitment_safety_policy.py \
  --dataset "$DATASET" \
  --example-indices "$EXAMPLE_INDICES" \
  --checkpoint "$CHECKPOINT" \
  --projection "$PROJECTION" \
  --predictor-dir "$PREDICTOR_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-4}" \
  --steps "${STEPS:-128}" \
  --gen-length "${GEN_LENGTH:-128}" \
  --block-length "${BLOCK_LENGTH:-32}" \
  --methods "${METHODS:-dense:capped_confidence:confidence:trace:safety_cheap:safety_full}" \
  --maximum-extra-per-update "${MAXIMUM_EXTRA_PER_UPDATE:-2}" \
  --capped-lock-fraction "${CAPPED_LOCK_FRACTION:-0.06}" \
  --capped-confidence-threshold "${CAPPED_CONFIDENCE_THRESHOLD:-0.95}" \
  --capped-optional-steps-per-block "${CAPPED_OPTIONAL_STEPS_PER_BLOCK:-2}" \
  --timing-warmups "${TIMING_WARMUPS:-1}" \
  --timing-repeats "${TIMING_REPEATS:-2}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --seed "${SEED:-313}"
