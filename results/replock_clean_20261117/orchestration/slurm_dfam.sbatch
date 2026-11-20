#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, Dream-architecture models (Dream, DiffuCoder). Usage: KEY=dream|diffucoder [FLOPS=1] [ENG_ARMS=a,b] sbatch -J name -o log slurm_dfam.sbatch RUNNER.py ARGS...
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
KEY=${KEY:-dream}
if [ "$KEY" = "diffucoder" ]; then export MODEL_CHECKPOINT=/l/users/rui.gao/models/DiffuCoder-7B-Instruct; else export MODEL_CHECKPOINT=/l/users/rui.gao/models/Dream-v0-Instruct-7B; fi
export MODEL_TAG=dream ORIG_TAG=$KEY BENCH_JSON=$R/dream/records/bench_layer_cost.json DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_ad2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MASK_TOKEN_ID=151666 EOS_TOKEN_ID=151643 PAD_TOKEN_ID=151643
export PYTHONPATH=/l/users/rui.gao/envs/d2c_tf:/l/users/rui.gao/envs/d2c:/l/users/rui.gao/tools/d2Cache
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[dfam] $KEY $* $(date -Is)"; $PY "$@" || { echo "[dfam] FAILED"; exit 1; }
echo "[dfam] done $(date -Is)"
