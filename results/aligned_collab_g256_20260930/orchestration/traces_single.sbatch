#!/bin/bash
#SBATCH --job-name=collab-trace1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=gpu-01,gpu-04
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/trace_%x_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/trace_%x_%j.err
# One source of her trace recipe (same arguments as traces.sbatch). Trace data is not timing-sensitive,
# so a shared GPU is acceptable here. RESUME=1 uses her --resume-existing path (as she did for KodCode).
set -euo pipefail
SOURCE=${SOURCE:?}; TARGET=${TARGET:-2000}; RESUME=${RESUME:-0}
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} source=$SOURCE target=$TARGET resume=$RESUME ulimit_v=$(ulimit -v) start=$(date -Is)"
if [ "$(ulimit -v)" != "unlimited" ]; then echo "[job] address-space cap still set; aborting"; exit 3; fi
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
cd "$C/source_collab"
# gpu nodes do not isolate GPUs: another user's process can hold the allocated card (212016 died that way on gpu-01).
FREE=$(/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python -c "import torch; print(torch.cuda.mem_get_info(0)[0] // 2**30)")
echo "[job] gpu_free_gib=$FREE"
if [ "$FREE" -lt 24 ]; then echo "[job] allocated GPU is occupied by another process; aborting"; nvidia-smi; exit 4; fi
export PYTHONPATH=src:scripts:$C/orchestration HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
EXTRA=(); [ "$RESUME" = 1 ] && EXTRA=(--resume-existing)
/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python "$C/orchestration/compat_run.py" scripts/generate_tracelock_llada_traces.py \
  --source "$SOURCE" --offset 0 --limit 20000 --target-completed "$TARGET" "${EXTRA[@]}" \
  --output-dir "$C/tracelock_traces" --seed 42
echo "[job] done=$(date -Is)"
