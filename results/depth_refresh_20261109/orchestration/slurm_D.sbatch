#!/bin/bash
#SBATCH -J depthD
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109/logs/slurm_D_%j.log
# Stage D: MATH 108 @512 then MATH 54 @1024, one process per set, own GPU. Smoke (2 questions) first for each set.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
export DEPTH_CELLS_D='K4=dense,4;W20-dense=wdense,20;P4-l11-blank=4,11,blank;W20-blank11=wpart,20,11,blank'
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
for set in math108 math54half; do
  echo "[D] smoke $set $(date -Is)"; $PY run_depth_cells.py D_$set SMOKE || { echo "[D] smoke $set FAILED"; exit 1; }
  echo "[D] run $set $(date -Is)";   $PY run_depth_cells.py D_$set || { echo "[D] run $set FAILED"; exit 1; }
  $CPY analyze_depth_cells.py D_$set > $R/logs/analyze_D_$set.log 2>&1
done
echo "[D] done $(date -Is)"
