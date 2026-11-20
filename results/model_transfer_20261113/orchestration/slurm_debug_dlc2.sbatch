#!/bin/bash
#SBATCH -J dbg_dlc2
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/original/logs/slurm_dbg_dlc2_%j.log
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
export MODEL_TAG=original FAMILY_SRC=$R/source_m1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/l/users/rui.gao/envs/tf446b HF_MODULES_CACHE=$R/original/hf_modules
export MODEL_CHECKPOINT=/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
cd $R/orchestration
/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python debug_dlc2.py
