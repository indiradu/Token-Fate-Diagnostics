#!/bin/bash
#SBATCH -J score_orig
#SBATCH -p ws-ia
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --propagate=NONE
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/original/logs/slurm_score_%j.log
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results
export PATH=$R/refresh_family_20261111/tools/bwrap_env/bin:$PATH LD_LIBRARY_PATH=$R/refresh_family_20261111/tools/bwrap_env/lib:$LD_LIBRARY_PATH MODEL_TAG=original FAMILY_SRC=$R/model_transfer_20261113/source_m1
cd $R/model_transfer_20261113/orchestration
hostname; which bwrap
$R/repfreeze_collect_20261006/venv_cu126/bin/python score_original.py
