#!/bin/bash
#SBATCH --job-name=collab-train
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-008
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/train_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/train_%j.err
# Her TraceLock-LLaDA controller recipe (train_tracelock_llada.py defaults: 10,000 updates, batch 32, lr 1e-4,
# seed 42, best checkpoint by F0.5 at 0.95). Declared adaptation: controller max length 1024 (two panel prompts > 512 tokens).
set -euo pipefail
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} ulimit_v=$(ulimit -v) start=$(date -Is)"
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
cd "$C/source_collab"
export PYTHONPATH=src:scripts:$C/orchestration HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 COLLAB_MIN_CONTROLLER_LENGTH=1024 OMP_NUM_THREADS=8
/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python "$C/orchestration/compat_run.py" scripts/train_tracelock_llada.py \
  --trace-dir "$C/tracelock_traces" --output-dir "$C/tracelock_controller" \
  --expected-source-counts gsm8k=2000,natural_questions=2000,alpaca=2000,kodcode=2000 --seed 42
echo "[job] done=$(date -Is)"
