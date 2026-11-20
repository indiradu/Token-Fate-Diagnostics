#!/bin/bash
#SBATCH -J orig_llada8b_gsm_held
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=07:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/original/logs/slurm_O_llada8b_gsm_held_%j.log
# Stage O: original decoding (B0: one token per step, block 32) vs dense C-0.9, model llada8b, set gsm_held. Code scoring deferred.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1 MODEL_TAG=original ORIG_TAG=llada8b MODEL_CHECKPOINT=/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
echo "[O] smoke $(date -Is)"; $PY run_original.py gsm_held SMOKE || { echo "[O] smoke FAILED"; exit 1; }
echo "[O] run $(date -Is)";   $PY run_original.py gsm_held || { echo "[O] run FAILED"; exit 1; }
echo "[O] done $(date -Is)"
