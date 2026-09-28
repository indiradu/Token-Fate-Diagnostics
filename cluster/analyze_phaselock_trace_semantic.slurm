#!/usr/bin/env bash
#SBATCH -J pl_trace_analyze
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH -t 00:15:00
#SBATCH -o logs/pl_trace_analyze_%j.out
#SBATCH -e logs/pl_trace_analyze_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
SELECTED_POLICY="${SELECTED_POLICY:?SELECTED_POLICY is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
EVAL_TAG="${EVAL_TAG:-20260916}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
"$VENV_DIR/bin/python" scripts/analyze_phaselock_trace_semantic.py \
  --run "results/pl_trace_p1_cropfix_eval_${SELECTED_POLICY}_gsm8k_${EVAL_TAG}" \
  --run "results/pl_trace_p1_cropfix_eval_${SELECTED_POLICY}_math500_${EVAL_TAG}" \
  --run "results/pl_trace_p1_cropfix_eval_${SELECTED_POLICY}_humaneval_${EVAL_TAG}" \
  --output-dir "$OUTPUT_DIR" \
  --bootstrap-samples 10000 \
  --seed 251
