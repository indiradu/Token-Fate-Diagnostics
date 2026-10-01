#!/bin/bash
#SBATCH --job-name=aligned-g256
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-008
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_standard_g256_20260929/logs/aligned_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_standard_g256_20260929/logs/aligned_%j.err
# Smoke (parity + mechanism checks) -> full 124 x 6 x 3 -> finalize. Any failure stops the chain.
set -euo pipefail
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} ulimit_v=$(ulimit -v) start=$(date -Is)"
if [ "$(ulimit -v)" != "unlimited" ]; then echo "[job] address-space cap still set; aborting"; exit 3; fi
CAMPAIGN=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_standard_g256_20260929
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
cd "$CAMPAIGN"
for stage in ${STAGES:-smoke full finalize}; do
  echo "[job] stage=$stage start=$(date -Is)"
  "$PY" orchestration/run_aligned.py "$stage" 2>&1 | tee -a "logs/${stage}.log"
  echo "[job] stage=$stage done=$(date -Is)"
done
echo "[job] complete=$(date -Is)"
