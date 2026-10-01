#!/bin/bash
#SBATCH --job-name=collab-traces
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-008
#SBATCH --array=0-3
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/traces_%A_%a.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/traces_%A_%a.err
# Her TraceLock-LLaDA trace recipe (generate_tracelock_llada_traces.py defaults): 128/128/32, batch 4,
# layers 30-32, projection dim 128, 30% of steps, 10% validation, prompts <= 512 tokens, seed 42,
# first 2,000 accepted prompts per source in her seeded shuffle order.
set -euo pipefail
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} ulimit_v=$(ulimit -v) start=$(date -Is)"
if [ "$(ulimit -v)" != "unlimited" ]; then echo "[job] address-space cap still set; aborting"; exit 3; fi
SOURCES=(gsm8k natural_questions alpaca kodcode)
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
cd "$C/source_collab"
export PYTHONPATH=src:scripts:$C/orchestration HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
# collab_compat only adds the two transformers-5 loader patches our own llada_trace applies.
/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python "$C/orchestration/compat_run.py" scripts/generate_tracelock_llada_traces.py \
  --source "$SOURCE" --offset 0 --limit 20000 --target-completed 2000 \
  --output-dir "$C/tracelock_traces" --seed 42
echo "[job] done=$(date -Is)"
