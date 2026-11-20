#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Study 5, LLaDA-MoE. Usage: [FLOPS=1] [ENG_ARMS=a,b] sbatch -J name -o log slurm_moe.sbatch RUNNER.py ARGS...   (transformers 4.57.1 private path for the model's remote code)
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export MODEL_TAG=dream ORIG_TAG=moe MODEL_CHECKPOINT=/l/users/rui.gao/models/LLaDA-MoE-7B-A1B-Instruct BENCH_JSON=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records/moe_bench_layer_cost.json
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_ad2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/l/users/rui.gao/envs/d2c_tf:/l/users/rui.gao/envs/d2c
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[moe] $* $(date -Is)"; $PY "$@" || { echo "[moe] FAILED"; exit 1; }
echo "[moe] done $(date -Is)"
