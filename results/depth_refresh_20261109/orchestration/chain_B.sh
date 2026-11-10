#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
export DEPTH_CELLS_B='P2-l11-blank=2,11,blank;P4-l11-blank=4,11,blank;K2=dense,2;K4=dense,4'
cd $R/orchestration || exit 1
echo "[chain_B] stage B $(date -Is)"; $PY run_depth_cells.py B || { echo "[chain_B] stage B FAILED"; exit 1; }
$CPY analyze_depth_cells.py B > $R/logs/analyze_B.log 2>&1
echo "[chain_B] done $(date -Is)"
