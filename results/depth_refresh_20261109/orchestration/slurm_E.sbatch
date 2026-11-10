#!/bin/bash
#SBATCH -J depthE
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=05:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109/logs/slurm_E_%j.log
# Stage E: 10 arms (word lock x cache strategy) on HumanEval (scoring deferred: no bwrap on the node) then MATH 108 @512. Smoke (2 questions) first for each set.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
for set in he164 math108; do
  echo "[E] smoke $set $(date -Is)"; $PY run_depth_cells.py E_$set SMOKE || { echo "[E] smoke $set FAILED"; exit 1; }
  echo "[E] run $set $(date -Is)";   $PY run_depth_cells.py E_$set || { echo "[E] run $set FAILED"; exit 1; }
done
echo "[E] done $(date -Is)"
