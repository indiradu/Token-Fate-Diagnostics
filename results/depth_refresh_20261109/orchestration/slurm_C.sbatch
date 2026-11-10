#!/bin/bash
#SBATCH -J depthC
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=02:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109/logs/slurm_C_%j.log
# Stage C from scratch on a Slurm node (own GPU, no contention). Scoring deferred: no bwrap on the node; rescore afterwards with rescore_depth_records.py C_slurm.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
export DEPTH_CELLS_C='W10-dense=wdense,10;W20-dense=wdense,20;W10-blank11=wpart,10,11,blank;W20-blank11=wpart,20,11,blank;K2=dense,2;K4=dense,4'
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY run_depth_cells.py C_slurm
