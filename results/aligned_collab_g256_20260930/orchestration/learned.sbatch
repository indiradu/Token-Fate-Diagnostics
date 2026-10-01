#!/bin/bash
#SBATCH --job-name=collab-learned
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-008
#SBATCH --output=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/learned_%j.out
#SBATCH --error=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930/logs/learned_%j.err
# Same job, same clean workstation GPU: smoke (our B0 + her learned methods) -> gate -> full.
set -euo pipefail
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} ulimit_v=$(ulimit -v) start=$(date -Is)"
if [ "$(ulimit -v)" != "unlimited" ]; then echo "[job] address-space cap still set; aborting"; exit 3; fi
C=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_collab_g256_20260930
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
METHODS=collab-TraceLock-LLaDA,collab-open-confidence,collab-safety-full
OUT=$C/learned
mkdir -p "$OUT"
cd "$C"
learned() {
  env PYTHONPATH=$C/source_collab/src:$C/source_collab/scripts:$C/orchestration HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      HF_MODULES_CACHE=$C/runtime_cache_collab TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
      "$PY" orchestration/run_collab_learned.py --methods "$METHODS" --checkpoint "$C/tracelock_controller/best_f0_5.pt" \
      --projection "$C/tracelock_traces/projection.npy" --predictor-dir "$C/safety_predictor" "$@"
}
"$PY" orchestration/run_our_b0.py --stage smoke --output-dir "$OUT/smoke_b0"
learned --prompts "$C/prompts_smoke.json" --output-dir "$OUT/smoke_collab" --repeats 1
"$PY" orchestration/gate.py --b0-dir "$OUT/smoke_b0" --collab-dir "$OUT/smoke_collab" --methods "$METHODS" --out "$OUT/smoke_gate.json"
"$PY" orchestration/run_our_b0.py --stage full --output-dir "$OUT/b0"
learned --prompts "$C/prompts_eval.json" --output-dir "$OUT/collab" --repeats 3
echo "[job] complete=$(date -Is)"
