#!/bin/bash
#SBATCH -J modM2hf
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/dream/logs/slurm_M2hf_%j.log
# Dream-7B: adapter against the model's own forward under transformers 4.46.2 (private install on /l).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export PYTHONPATH=/l/users/rui.gao/envs/tf446b HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=$R/dream/hf_modules
mkdir -p $HF_MODULES_CACHE
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY -c "import transformers;print('transformers',transformers.__version__)"
$PY dream_hf_parity.py || { echo "[M2hf] FAILED"; exit 1; }
echo "[M2hf] done $(date -Is)"
