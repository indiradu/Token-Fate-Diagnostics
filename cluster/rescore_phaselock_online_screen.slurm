#!/usr/bin/env bash
#SBATCH -J pl_h2_rescore
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --exclude=gpu-01,gpu-02,gpu-04
#SBATCH -o logs/pl_h2_rescore_%j.out
#SBATCH -e logs/pl_h2_rescore_%j.err
set -euo pipefail
SNAPSHOT_DIR="${SNAPSHOT_DIR:?required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?required}"
cd "$SNAPSHOT_DIR"
export PYTHONPATH="$SNAPSHOT_DIR/src:$SNAPSHOT_DIR/scripts"
export OMP_NUM_THREADS=2
"$SNAPSHOT_DIR/grader_env/bin/python" scripts/rescore_phaselock_math.py \
  --run-dirs "$OUTPUT_ROOT/dev_gsm8k" "$OUTPUT_ROOT/dev_math500" "$OUTPUT_ROOT/dev_humaneval" \
  --output-root "$OUTPUT_ROOT/symbolic_rescored"
"$SNAPSHOT_DIR/grader_env/bin/python" scripts/analyze_phaselock_online_screen.py \
  --run-dirs "$OUTPUT_ROOT/symbolic_rescored/dev_gsm8k" "$OUTPUT_ROOT/symbolic_rescored/dev_math500" "$OUTPUT_ROOT/symbolic_rescored/dev_humaneval" \
  --output-dir "$OUTPUT_ROOT/symbolic_analysis"
