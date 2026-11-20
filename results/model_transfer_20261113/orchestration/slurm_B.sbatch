#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=07:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
# Stage B: one published method on one set. Usage: sbatch -J B_<method>_<set> -o <log> slurm_B.sbatch METHOD SET
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
METHOD=$1; SET=$2
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1 MODEL_TAG=original ORIG_TAG=${ORIG_TAG:-llada8b} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}
if [ "$METHOD" != "ELASTIC" ]; then export PYTHONPATH=/l/users/rui.gao/envs/tf446b HF_MODULES_CACHE=$R/original/hf_modules; mkdir -p $HF_MODULES_CACHE; fi
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
echo "[B] smoke $METHOD $SET $(date -Is)"; $PY run_published.py $METHOD $SET SMOKE || { echo "[B] smoke FAILED"; exit 1; }
echo "[B] run $METHOD $SET $(date -Is)";   $PY run_published.py $METHOD $SET || { echo "[B] run FAILED"; exit 1; }
echo "[B] done $(date -Is)"
