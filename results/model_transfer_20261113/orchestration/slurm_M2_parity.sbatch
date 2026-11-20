#!/bin/bash
#SBATCH -J modM2par
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=01:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/dream/logs/slurm_M2par_%j.log
# Dream-7B: adapter vs the model's own forward (if it loads), engine vs a plain dense loop (token-identical decodes), cached smoke, layer-cost cells.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export FAMILY_SRC=$R/source_ad MODEL_TAG=dream MODEL_CHECKPOINT=/l/users/rui.gao/models/Dream-v0-Instruct-7B
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY dream_parity.py || { echo "[M2par] FAILED"; exit 1; }
echo "[M2par] done $(date -Is)"
