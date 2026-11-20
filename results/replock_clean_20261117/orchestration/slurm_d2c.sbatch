#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, d2Cache-framework arms. Usage: sbatch -J name -o log slurm_d2c.sbatch ARM SET [SMOKE]
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
ARM=$1; SET=$2; SM=$3
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1 MODEL_TAG=${MODEL_TAG:-original} ORIG_TAG=${ORIG_TAG:-llada8b} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}
export MASK_TOKEN_ID=126336 EOS_TOKEN_ID=126081 PAD_TOKEN_ID=126081
export PYTHONPATH=/l/users/rui.gao/envs/d2c_tf:/l/users/rui.gao/envs/d2c:/l/users/rui.gao/tools/d2Cache
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[d2c] $ARM $SET $SM $(date -Is)"; $PY run_d2c.py $ARM $SET $SM || { echo "[d2c] FAILED"; exit 1; }
echo "[d2c] done $(date -Is)"
