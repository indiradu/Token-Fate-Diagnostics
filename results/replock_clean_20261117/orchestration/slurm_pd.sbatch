#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, paper_decode engine arms (ORIG, DENSE_C09, SURELOCK). Usage: sbatch -J name -o log slurm_pd.sbatch SET [SMOKE]; env MODEL_CHECKPOINT, ORIG_TAG, PUB_LIMIT
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1 MODEL_TAG=${MODEL_TAG:-original} ORIG_TAG=${ORIG_TAG:-llada8b}
export MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[pd] $1 $2 $(date -Is)"; $PY run_pd.py $1 $2 || { echo "[pd] FAILED"; exit 1; }
echo "[pd] done $(date -Is)"
