#!/bin/bash
#SBATCH -J score_xfer
#SBATCH -p ws-ia
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/logs/slurm_score_xfer_%j.log
# Study 5 scoring on a CPU-only compute node (no GPU). Aborts on any sandbox failure (../ciai/score_transfer.py raises).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results
export PATH=$R/refresh_family_20261111/tools/bwrap_env/bin:$PATH LD_LIBRARY_PATH=$R/refresh_family_20261111/tools/bwrap_env/lib:$LD_LIBRARY_PATH MODEL_TAG=original FAMILY_SRC=$R/model_transfer_20261113/source_m1
cd $R/replock_clean_20261117/orchestration
hostname; which bwrap
$R/repfreeze_collect_20261006/venv_cu126/bin/python ../ciai/score_transfer.py
echo "[score] exit $? $(date -Is)"
