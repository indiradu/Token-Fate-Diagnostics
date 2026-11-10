#!/bin/bash
# One claim: freeze check, two smoke runs (2 questions each), Analysis 1 (noise-matched control), Analysis 2 (oracle repair), then the CPU analyses. Stops at the first failure.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/stale_cause_20261108
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
cd $R/orchestration || exit 1
test -f $R/source/SHA256.json || { echo "[chain] source/ not frozen"; exit 1; }
echo "[chain] smoke noise $(date -Is)";  $PY $R/orchestration/run_noise_control.py SMOKE  || { echo "[chain] smoke noise FAILED"; exit 1; }
echo "[chain] smoke repair $(date -Is)"; $PY $R/orchestration/run_oracle_repair.py SMOKE  || { echo "[chain] smoke repair FAILED"; exit 1; }
echo "[chain] analysis 1 $(date -Is)";   $PY $R/orchestration/run_noise_control.py        || { echo "[chain] analysis 1 FAILED"; exit 1; }
echo "[chain] analysis 2 $(date -Is)";   $PY $R/orchestration/run_oracle_repair.py        || { echo "[chain] analysis 2 FAILED"; exit 1; }
echo "[chain] CPU analyses $(date -Is)"
$CPY $R/orchestration/analyze_noise_control.py > $R/logs/analyze_noise.log 2>&1
$CPY $R/orchestration/analyze_oracle_repair.py > $R/logs/analyze_repair.log 2>&1
echo "[chain] done $(date -Is)"
