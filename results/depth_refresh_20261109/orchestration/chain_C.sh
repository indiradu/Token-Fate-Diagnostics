#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
export DEPTH_CELLS_C='W10-dense=wdense,10;W20-dense=wdense,20;W10-blank11=wpart,10,11,blank;W20-blank11=wpart,20,11,blank;K2=dense,2;K4=dense,4'
cd $R/orchestration || exit 1
echo "[chain_C] smoke $(date -Is)"; $PY run_depth_cells.py C SMOKE || { echo "[chain_C] smoke FAILED"; exit 1; }
echo "[chain_C] stage C $(date -Is)"; $PY run_depth_cells.py C || { echo "[chain_C] stage C FAILED"; exit 1; }
$CPY analyze_depth_cells.py C > $R/logs/analyze_C.log 2>&1
echo "[chain_C] done $(date -Is)"
