#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
cd $R/orchestration || exit 1
echo "[chain_A] smoke $(date -Is)"; $PY run_depth_cells.py A SMOKE || { echo "[chain_A] smoke FAILED"; exit 1; }
echo "[chain_A] stage A $(date -Is)"; $PY run_depth_cells.py A || { echo "[chain_A] stage A FAILED"; exit 1; }
$CPY analyze_depth_cells.py A > $R/logs/analyze_A.log 2>&1
echo "[chain_A] done $(date -Is)"
