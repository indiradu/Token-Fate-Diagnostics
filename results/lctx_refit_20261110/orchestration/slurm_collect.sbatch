#!/bin/bash
#SBATCH -J refitC
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=02:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_refit_20261110/logs/slurm_collect_%j.log
# Job 1 of 2. One-word-per-call trace collection in the two geometries (R-S16 then R-S32c; 80 questions each, a 2-question smoke first), then THE CPU FIT ON THE NODE for both geometries
# (fit_refit.py: rule pickles + rules.json + validation/refit_*.md), then a rule check. Exit code != 0 if anything failed (job 2 then stops at its own rule check).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_refit_20261110
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
hostname; echo "[job] ulimit_v=$(ulimit -v)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
for g in S16 S32c; do
  echo "[C] smoke $g $(date -Is)"; $PY collect_traces.py $g SMOKE || { echo "[C] smoke $g FAILED"; exit 1; }
  echo "[C] run $g $(date -Is)";   $PY collect_traces.py $g       || { echo "[C] run $g FAILED"; exit 1; }
done
for g in S16 S32c; do
  echo "[C] fit $g $(date -Is)";   $PY fit_refit.py $g            || { echo "[C] fit $g FAILED"; exit 1; }
done
$PY check_rules.py || { echo "[C] rule check FAILED"; exit 1; }
echo "[C] done $(date -Is)"
