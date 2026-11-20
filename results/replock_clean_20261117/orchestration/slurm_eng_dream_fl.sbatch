#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, Dream engine arms. Usage: sbatch -J name -o log slurm_eng_dream.sbatch SET [SMOKE]; env FLOPS, PUB_LIMIT, ENG_ARMS
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export MODEL_TAG=dream ORIG_TAG=dream MODEL_CHECKPOINT=/l/users/rui.gao/models/Dream-v0-Instruct-7B BENCH_JSON=$R/dream/records/bench_layer_cost.json DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_ad2
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[eng-dream] $1 $2 $(date -Is)"; $PY run_eng_dream_fl.py $1 $2 || { echo "[eng-dream] FAILED"; exit 1; }
echo "[eng-dream] done $(date -Is)"
