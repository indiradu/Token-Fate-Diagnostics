#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-indira.duisembayeva@10.127.78.113}"
REMOTE_DIR="${REMOTE_DIR:-/home/indira.duisembayeva/projects/regret_remasking}"
VENV_DIR="${VENV_DIR:-/home/indira.duisembayeva/venvs/regret_remasking}"
OUTPUT_DIR="${OUTPUT_DIR:-$REMOTE_DIR/results/phase_lock_audit}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
SKIP_SYNC="${SKIP_SYNC:-0}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=8
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
if [ "$SKIP_SYNC" != "1" ]; then
  scp "${SSH_OPTS[@]}" -r README.md pyproject.toml requirements.txt src scripts cluster tests docs \
    "$REMOTE_HOST:$REMOTE_DIR/"
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && env \
  PROJECT_DIR='$REMOTE_DIR' VENV_DIR='$VENV_DIR' OUTPUT_DIR='$OUTPUT_DIR' \
  MODEL_NAME='${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}' \
  DATASET='${DATASET:-gsm8k}' SPLIT='${SPLIT:-test}' LIMIT='${LIMIT:-10}' OFFSET='${OFFSET:-0}' \
  STEPS='${STEPS:-64}' GEN_LENGTH='${GEN_LENGTH:-64}' BLOCK_LENGTH='${BLOCK_LENGTH:-32}' \
  SEMANTIC_POLICIES='${SEMANTIC_POLICIES:-confidence,posterior,consensus}' \
  REPRESENTATION_POLICIES='${REPRESENTATION_POLICIES:-drift}' \
  COMPUTE_POLICIES='${COMPUTE_POLICIES:-row_sparse}' \
  SEMANTIC_LOCK_FRACTION='${SEMANTIC_LOCK_FRACTION:-0.06}' \
  REPRESENTATION_GATE='${REPRESENTATION_GATE:-custom}' \
  REPRESENTATION_LAYER='${REPRESENTATION_LAYER:-24}' REPRESENTATION_THRESHOLD='${REPRESENTATION_THRESHOLD:-0.02}' \
  REPRESENTATION_PATIENCE='${REPRESENTATION_PATIENCE:-2}' REPRESENTATION_MIN_AGE='${REPRESENTATION_MIN_AGE:-1}' \
  BETA_KL='${BETA_KL:-1.0}' BETA_FATE='${BETA_FATE:-2.0}' \
  REGRET_MODEL_PATH='${REGRET_MODEL_PATH:-}' DTYPE='${DTYPE:-bf16}' SEED='${SEED:-17}' \
  sbatch --parsable -p '$PARTITION' -t '$TIME_LIMIT' --export=ALL cluster/run_phase_lock.slurm"
