#!/bin/bash
#SBATCH -J modM2he
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/dream/logs/slurm_M2he_%j.log
# Dream-7B stage run_rule_dream.py on: he164. Code scoring deferred.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_ad2 MODEL_TAG=dream MODEL_CHECKPOINT=/l/users/rui.gao/models/Dream-v0-Instruct-7B HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
for ds in he164; do
  echo "[M2] smoke $ds $(date -Is)"; $PY run_rule_dream.py $ds SMOKE || { echo "[M2] smoke $ds FAILED"; exit 1; }
  echo "[M2] run $ds $(date -Is)";   $PY run_rule_dream.py $ds || { echo "[M2] run $ds FAILED"; exit 1; }
done
echo "[M2] done $(date -Is)"
