#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, selective-cache engine arms (DENSE_C09, ELASTIC, R0, CTL). Usage: sbatch -J name -o log slurm_eng.sbatch SET [SMOKE]; env: MM (model key: llada8b | llada15), PUB_LIMIT, RECAL
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
MM=${MM:-llada8b}
if [ "$MM" = "llada8b" ]; then
  export MODEL_TAG=original ORIG_TAG=llada8b BENCH_JSON=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/refresh_family_20261111/records/bench_layer_cost.json
  export MODEL_CHECKPOINT=/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
else
  export MODEL_TAG=llada15 ORIG_TAG=llada15 BENCH_JSON=$R/llada15/records/bench_layer_cost.json MODEL_CHECKPOINT=/l/users/rui.gao/models/LLaDA-1.5 RECAL=1
fi
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[eng] $MM $1 $2 $(date -Is)"; $PY run_eng.py $1 $2 || { echo "[eng] FAILED"; exit 1; }
echo "[eng] done $(date -Is)"
