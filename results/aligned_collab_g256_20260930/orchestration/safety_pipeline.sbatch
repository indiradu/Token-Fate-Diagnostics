#!/bin/bash
#SBATCH --job-name=collab-safety
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=gpu-04
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/safety_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/safety_%j.err
# Her direct-safety recipe: cluster/run_commitment_safety_data.slurm (128/128/32, 6 candidates, intervention batch 1,
# seed 307) on her slices, then cluster/train_commitment_safety.slurm (sequence_safe, max unsafe 0, min selected 8, seed 311).
# Her slices are disjoint from our 124 questions (checked: MATH-500 400-407/450-457 vs our 54; HumanEval 100+ vs 0-69).
set -euo pipefail
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} ulimit_v=$(ulimit -v) start=$(date -Is)"
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
cd "$C/source_collab"
export PYTHONPATH=src:scripts:$C/orchestration HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
declare -A TRAIN=([gsm8k]=700:707 [math500]=400:407 [humaneval]=100:107)
declare -A EVAL=([gsm8k]=820:827 [math500]=450:457 [humaneval]=120:127)
for D in gsm8k math500 humaneval; do
  echo "[job] safety data $D start=$(date -Is)"
  "$PY" "$C/orchestration/compat_run.py" scripts/generate_commitment_safety_data.py \
    --dataset "$D" --train-example-indices "${TRAIN[$D]}" --eval-example-indices "${EVAL[$D]}" \
    --checkpoint "$C/tracelock_controller/best_f0_5.pt" --projection "$C/tracelock_traces/projection.npy" \
    --output-dir "$C/safety_data/$D" --candidates-per-example 6 --intervention-batch-size 1 \
    --steps 128 --gen-length 128 --block-length 32 --dtype bf16 --device cuda --seed 307
done
echo "[job] predictor start=$(date -Is)"
"$PY" "$C/orchestration/compat_run.py" scripts/train_commitment_safety_predictor.py \
  --input-dir "$C/safety_data/gsm8k" --input-dir "$C/safety_data/math500" --input-dir "$C/safety_data/humaneval" \
  --output-dir "$C/safety_predictor" --target sequence_safe --max-observed-unsafe-rate 0 --minimum-selected 8 --seed 311
echo "[job] done=$(date -Is)"
