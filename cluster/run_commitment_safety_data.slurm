#!/usr/bin/env bash
#SBATCH -J pl_commit_labels
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/pl_commit_labels_%j.out
#SBATCH -e logs/pl_commit_labels_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATASET="${DATASET:?DATASET is required}"
TRAIN_EXAMPLE_INDICES="${TRAIN_EXAMPLE_INDICES:?TRAIN_EXAMPLE_INDICES is required}"
EVAL_EXAMPLE_INDICES="${EVAL_EXAMPLE_INDICES:?EVAL_EXAMPLE_INDICES is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
PROJECTION="${PROJECTION:?PROJECTION is required}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/generate_commitment_safety_data.py \
  --dataset "$DATASET" \
  --train-example-indices "$TRAIN_EXAMPLE_INDICES" \
  --eval-example-indices "$EVAL_EXAMPLE_INDICES" \
  --checkpoint "$CHECKPOINT" \
  --projection "$PROJECTION" \
  --output-dir "$OUTPUT_DIR" \
  --candidates-per-example "${CANDIDATES_PER_EXAMPLE:-6}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE:-1}" \
  --steps "${STEPS:-128}" \
  --gen-length "${GEN_LENGTH:-128}" \
  --block-length "${BLOCK_LENGTH:-32}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --seed "${SEED:-307}"
