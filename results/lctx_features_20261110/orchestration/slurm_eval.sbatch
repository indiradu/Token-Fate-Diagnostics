#!/bin/bash
#SBATCH -J featH
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_features_20261110/logs/slurm_eval_%j.log
# Job 2 of 2 (submitted with --dependency=afterany:<featC job>). Stage H: 4 arms on MATH 108 (scored in-process) and HumanEval 164 (scoring DEFERRED: no bwrap on the node), 512 tokens, P4 cache.
# FIRST it runs check_gate.py: if the pre-registered gate did not pass (or job 1 failed, so there is no ablation result) it exits 0 WITHOUT loading the model. No human step is needed in the chain.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_features_20261110
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
hostname; echo "[job] ulimit_v=$(ulimit -v)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY check_gate.py || { echo "[H] gate did not pass (or no ablation result): nothing decoded, exiting cleanly"; exit 0; }
for set in math108 he164; do
  echo "[H] smoke $set $(date -Is)"; $PY run_feature_cells.py H_$set SMOKE || { echo "[H] smoke $set FAILED"; exit 1; }
  echo "[H] run $set $(date -Is)";   $PY run_feature_cells.py H_$set       || { echo "[H] run $set FAILED"; exit 1; }
done
echo "[H] analysis math108 (CPU) $(date -Is)"; $PY analyze_stage_h.py math108 > $R/logs/analyze_H_math108.log 2>&1 || echo "[H] analysis math108 failed (rerun by hand)"
echo "[H] done $(date -Is)  (HumanEval still needs: bwrap on PATH, rescore_records.py H_he164, analyze_stage_h.py he164, then analyze_stage_h.py both)"
