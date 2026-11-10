#!/bin/bash
#SBATCH -J refitG
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=04:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_refit_20261110/logs/slurm_eval_%j.log
# Job 2 of 2 (submitted with --dependency=afterany:<refitC job>). Stage G: 7 arms on HumanEval 164 (scoring DEFERRED: no bwrap on the node) and MATH 108 (scored in-process), 512 tokens.
# First it VERIFIES that the refit rule files exist and match their rules.json (check_rules.py); if job 1 failed this job stops here, before loading the model.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_refit_20261110
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
hostname; echo "[job] ulimit_v=$(ulimit -v)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
$PY check_rules.py || { echo "[G] rule check FAILED: the collection/fit job did not produce the rules; nothing decoded"; exit 1; }
for set in math108 he164; do
  echo "[G] smoke $set $(date -Is)"; $PY run_refit_cells.py G_$set SMOKE || { echo "[G] smoke $set FAILED"; exit 1; }
  echo "[G] run $set $(date -Is)";   $PY run_refit_cells.py G_$set       || { echo "[G] run $set FAILED"; exit 1; }
done
echo "[G] analysis math108 (CPU) $(date -Is)"; $PY analyze_stage_g.py math108 > $R/logs/analyze_G_math108.log 2>&1 || echo "[G] analysis math108 failed (rerun by hand)"
echo "[G] done $(date -Is)  (HumanEval still needs: bwrap on PATH, rescore_records.py G_he164, analyze_stage_g.py he164)"
