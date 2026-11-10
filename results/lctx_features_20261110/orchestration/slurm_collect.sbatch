#!/bin/bash
#SBATCH -J featC
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=03:30:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_features_20261110/logs/slurm_collect_%j.log
# Job 1 (290320) of 2. R-S32c one-word-per-call trace collection with the extra features (2-question smoke, then 80 questions; V1/V2 checks inside), then THE CPU ABLATION ON THE NODE (fit_ablation.py:
# 11 fits, 2,000-resample cluster bootstrap, permutation importance, validation/ablation_S32c.{json,md}, the rule rules/X only if the gate passes), then check_gate.py (informational here), then a DRY decode
# smoke (2 MATH questions x 4 arms with the ungated six+all rule: tests the stage-H decode path and the feature overhead; never a result). Exit != 0 only if collection or ablation failed.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/lctx_features_20261110
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
hostname; echo "[job] ulimit_v=$(ulimit -v)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
echo "[C] smoke $(date -Is)"; $PY collect_traces.py SMOKE || { echo "[C] smoke FAILED"; exit 1; }
echo "[C] run $(date -Is)";   $PY collect_traces.py       || { echo "[C] collection FAILED"; exit 1; }
echo "[C] ablation $(date -Is)"; $PY fit_ablation.py      || { echo "[C] ablation FAILED"; exit 1; }
echo "[C] gate $(date -Is)"; $PY check_gate.py; echo "[C] gate exit code $? (informational)"
echo "[C] dry decode smoke $(date -Is)"; $PY run_feature_cells.py H_math108 DRY || echo "[C] dry decode smoke FAILED (informational; stage H would hit the same problem)"
echo "[C] done $(date -Is)"
