#!/bin/bash
#SBATCH -J modG3
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/llada15/logs/slurm_G3_%j.log
# Stage G3: shadow-pass records (cached call vs a dense pass over the same tokens, plain S32, C-0.9) of LLaDA-1.5 on KodCode-tune (100), HumanEval (164), GSM8K-tune (100).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export FAMILY_SRC=$R/source_m1 MODEL_TAG=llada15 MODEL_CHECKPOINT=/l/users/rui.gao/models/LLaDA-1.5
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY collect_agreement.py he164 SMOKE || { echo "[G3] smoke FAILED"; exit 1; }
for ds in kod_tune he164 gsm_tune; do echo "[G3] $ds $(date -Is)"; $PY collect_agreement.py $ds || { echo "[G3] $ds FAILED"; exit 1; }; done
echo "[G3] done $(date -Is)"
