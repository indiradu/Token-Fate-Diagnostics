#!/bin/bash
#SBATCH --job-name=collab-phaseA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=gpu-04
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/phaseA_%x_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/phaseA_%x_%j.err
# One job, one GPU: smoke (our B0 + her methods on the 2 smoke cases) -> gate -> full (our B0, then her methods).
set -euo pipefail
LABEL=${LABEL:?}; METHODS=${METHODS:?}
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} label=$LABEL methods=$METHODS ulimit_v=$(ulimit -v) gpu=${CUDA_VISIBLE_DEVICES:-?} start=$(date -Is)"
if [ "$(ulimit -v)" != "unlimited" ]; then echo "[job] address-space cap still set; aborting"; exit 3; fi
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
OUT=$C/phaseA_$LABEL
mkdir -p "$OUT"
cd "$C"
collab() {  # her environment: her source + the loader shim only
  env PYTHONPATH=$C/source_collab/src:$C/source_collab/scripts:$C/orchestration HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      HF_MODULES_CACHE=$C/runtime_cache_collab TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
      "$PY" orchestration/run_collab_methods.py --methods "$METHODS" "$@"
}
echo "[job] smoke start=$(date -Is)"
"$PY" orchestration/run_our_b0.py --stage smoke --output-dir "$OUT/smoke_b0"
collab --prompts "$C/prompts_smoke.json" --output-dir "$OUT/smoke_collab" --repeats 1
"$PY" orchestration/gate.py --b0-dir "$OUT/smoke_b0" --collab-dir "$OUT/smoke_collab" --methods "$METHODS" --out "$OUT/smoke_gate.json"
echo "[job] full start=$(date -Is)"
"$PY" orchestration/run_our_b0.py --stage full --output-dir "$OUT/b0"
collab --prompts "$C/prompts_eval.json" --output-dir "$OUT/collab" --repeats 3
echo "[job] complete=$(date -Is)"
