#!/usr/bin/env bash
#SBATCH -J pl_commit_analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/pl_commit_analyze_%j.out
#SBATCH -e logs/pl_commit_analyze_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PREDICTOR_DIR="${PREDICTOR_DIR:?PREDICTOR_DIR is required}"
BENCHMARK_DIRS="${BENCHMARK_DIRS:?BENCHMARK_DIRS is required}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
arguments=()
IFS=':' read -ra roots <<< "$BENCHMARK_DIRS"
if [[ "${#roots[@]}" -lt 1 ]]; then
  echo "BENCHMARK_DIRS did not contain any benchmark directories" >&2
  exit 2
fi
for root in "${roots[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "commitment-safety benchmark directory does not exist: $root" >&2
    exit 2
  fi
  arguments+=(--benchmark-dir "$root")
done
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/analyze_commitment_safety_experiment.py \
  --predictor-dir "$PREDICTOR_DIR" \
  "${arguments[@]}" \
  --output-dir "$OUTPUT_DIR"
