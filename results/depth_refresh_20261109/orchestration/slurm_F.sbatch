#!/bin/bash
#SBATCH -J depthF
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=05:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109/logs/slurm_F_%j.log
# Stage F: Elastic-Cache's evaluation protocol (HumanEval 0-69, 256 tokens), naive-diffusion family and Elastic-word-lock family, 14 arms. Scoring deferred (no bwrap on the node).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
echo "[F] smoke $(date -Is)"; $PY run_depth_cells.py F_he70 SMOKE || { echo "[F] smoke FAILED"; exit 1; }
echo "[F] run $(date -Is)";   $PY run_depth_cells.py F_he70 || { echo "[F] run FAILED"; exit 1; }
echo "[F] done $(date -Is)"
