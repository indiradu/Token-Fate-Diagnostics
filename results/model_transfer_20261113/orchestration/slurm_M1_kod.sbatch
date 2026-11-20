#!/bin/bash
#SBATCH -J modM1kod
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/llada15/logs/slurm_M1_kod_%j.log
# Stage M1: LLaDA-1.5, cost benchmark then S32/K4/P4/R0 against dense on: kod_held:C-0.9 gsm_held:C-0.9. Code scoring deferred.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1 FAMILY_SRC=$R/source_m1 MODEL_TAG=llada15 MODEL_CHECKPOINT=/l/users/rui.gao/models/LLaDA-1.5 RULE_ARMS=S32,K4,P4,R0
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
if [ ! -f $R/llada15/records/bench_layer_cost.json ]; then $PY bench_layer_cost.py || { echo "[M1] bench FAILED"; exit 1; }; fi
for pair in kod_held:C-0.9 gsm_held:C-0.9; do set=${pair%:*}; lock=${pair#*:}
  echo "[M1] smoke $set $lock $(date -Is)"; $PY run_rule.py $set $lock SMOKE || { echo "[M1] smoke FAILED"; exit 1; }
  echo "[M1] run $set $lock $(date -Is)";   $PY run_rule.py $set $lock || { echo "[M1] run FAILED"; exit 1; }
done
echo "[M1] done $(date -Is)"
