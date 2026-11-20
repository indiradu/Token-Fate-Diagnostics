#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Follow-up experiments. Usage: [KEY=llada8b|dream|moe] [N=..] [THRS=..] sbatch -J name -o log slurm_exp.sbatch SCRIPT.py ARGS...
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
KEY=${KEY:-llada8b}
export DEPTH_DEFER_SCORING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case $KEY in
  llada8b) export FAMILY_SRC=$R/source_m1 MODEL_TAG=original ORIG_TAG=llada8b BENCH_JSON=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/refresh_family_20261111/records/bench_layer_cost.json
           export MODEL_CHECKPOINT=/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 ;;
  dream)   export FAMILY_SRC=$R/source_ad2 MODEL_TAG=dream ORIG_TAG=dream MODEL_CHECKPOINT=/l/users/rui.gao/models/Dream-v0-Instruct-7B BENCH_JSON=$R/dream/records/bench_layer_cost.json ;;
  moe)     export FAMILY_SRC=$R/source_ad2 MODEL_TAG=dream ORIG_TAG=moe MODEL_CHECKPOINT=/l/users/rui.gao/models/LLaDA-MoE-7B-A1B-Instruct BENCH_JSON=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records/moe_bench_layer_cost.json PYTHONPATH=/l/users/rui.gao/envs/d2c_tf:/l/users/rui.gao/envs/d2c ;;
esac
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/orchestration || exit 1
echo "[exp] $KEY $* $(date -Is)"; $PY "$@" || { echo "[exp] FAILED"; exit 1; }
echo "[exp] done $(date -Is)"
