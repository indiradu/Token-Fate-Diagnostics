#!/usr/bin/env bash
#SBATCH -J pl_trace_p1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/pl_trace_p1_%j.out
#SBATCH -e logs/pl_trace_p1_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATASET="${DATASET:?DATASET is required}"
EXAMPLE_INDICES="${EXAMPLE_INDICES:?EXAMPLE_INDICES is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
PROJECTION="${PROJECTION:?PROJECTION is required}"
LATENCY_TABLE="${LATENCY_TABLE:?LATENCY_TABLE is required}"
POLICY="${POLICY:?POLICY is required}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
nvidia-smi
predictor_args=()
if [[ -n "${PREDICTOR_DIR:-}" ]]; then
  predictor_args=(--predictor-dir "$PREDICTOR_DIR")
fi
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/benchmark_phaselock_trace_semantic.py \
  --dataset "$DATASET" \
  --split "${SPLIT:-test}" \
  --example-indices "$EXAMPLE_INDICES" \
  --checkpoint "$CHECKPOINT" \
  --projection "$PROJECTION" \
  "${predictor_args[@]}" \
  --trace-layers "${TRACE_LAYERS:-30,31,32}" \
  --window-width "${WINDOW_WIDTH:-32}" \
  --policy "$POLICY" \
  --latency-table "$LATENCY_TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-4}" \
  --steps "${STEPS:-128}" \
  --gen-length "${GEN_LENGTH:-128}" \
  --block-length "${BLOCK_LENGTH:-32}" \
  --timing-warmups "${TIMING_WARMUPS:-1}" \
  --timing-repeats "${TIMING_REPEATS:-4}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --seed "${SEED:-251}" \
  --compute-min-locked-fraction "${COMPUTE_MIN_LOCKED_FRACTION:-0.20}" \
  --compute-min-remaining-steps "${COMPUTE_MIN_REMAINING_STEPS:-8}" \
  --compute-min-projected-saved-rows "${COMPUTE_MIN_PROJECTED_SAVED_ROWS:-1500}" \
  --compute-min-expected-gain "${COMPUTE_MIN_EXPECTED_GAIN:-0.03}"
