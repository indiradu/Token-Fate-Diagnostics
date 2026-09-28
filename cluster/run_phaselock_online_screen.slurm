#!/usr/bin/env bash
#SBATCH -J pl_h2_screen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=gpu-01,gpu-02,gpu-04
#SBATCH -o logs/pl_h2_screen_%j.out
#SBATCH -e logs/pl_h2_screen_%j.err
set -euo pipefail
SNAPSHOT_DIR="${SNAPSHOT_DIR:?required}"
PROJECT_DIR="${PROJECT_DIR:?required}"
VENV_DIR="${VENV_DIR:?required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?required}"
cd "$SNAPSHOT_DIR"
export PYTHONPATH="$SNAPSHOT_DIR/src:$SNAPSHOT_DIR/scripts"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
nvidia-smi
"$VENV_DIR/bin/python" tests/test_phaselock_speed_gates.py
common=(--policy-config configs/phaselock_h2_controller_screen_v1.json
  --checkpoint "$PROJECT_DIR/results/tracelock_llada_8000_paper_mix_controller_20260914/best_f0_5.pt"
  --projection "$PROJECT_DIR/results/tracelock_llada_8000_paper_mix_traces_20260914/projection.npy"
  --batch-size 4 --gen-length 128 --steps 128 --block-length 32)
# All 13 arms must execute successfully before the development sweep starts.
"$VENV_DIR/bin/python" scripts/benchmark_phaselock_online_screen.py "${common[@]}" \
  --dataset gsm8k --example-indices 800:803 --timing-warmups 0 --timing-repeats 1 \
  --output-dir "$OUTPUT_ROOT/smoke_gsm8k"
for spec in gsm8k,800:807 math500,430:437 humaneval,100:107; do
  dataset="${spec%%,*}"
  indices="${spec#*,}"
  "$VENV_DIR/bin/python" scripts/benchmark_phaselock_online_screen.py "${common[@]}" \
    --dataset "$dataset" --example-indices "$indices" \
    --timing-warmups 1 --timing-repeats 3 --output-dir "$OUTPUT_ROOT/dev_$dataset"
done
"$VENV_DIR/bin/python" scripts/analyze_phaselock_online_screen.py \
  --run-dirs "$OUTPUT_ROOT/dev_gsm8k" "$OUTPUT_ROOT/dev_math500" "$OUTPUT_ROOT/dev_humaneval" \
  --output-dir "$OUTPUT_ROOT/analysis"
