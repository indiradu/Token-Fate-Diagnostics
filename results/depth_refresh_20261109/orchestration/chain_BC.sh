#!/bin/bash
# Stage B (resumes) then stage C under ONE claim (two queued claims would wait for each other). Scoring is deferred: this host has no bwrap sandbox.
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
export DEPTH_CELLS_B='P2-l11-blank=2,11,blank;P4-l11-blank=4,11,blank;K2=dense,2;K4=dense,4'
export DEPTH_CELLS_C='W10-dense=wdense,10;W20-dense=wdense,20;W10-blank11=wpart,10,11,blank;W20-blank11=wpart,20,11,blank;K2=dense,2;K4=dense,4'
cd $R/orchestration || exit 1
echo "[chain_BC] stage B $(date -Is)"; $PY run_depth_cells.py B || { echo "[chain_BC] stage B FAILED"; exit 1; }
echo "[chain_BC] smoke C $(date -Is)"; $PY run_depth_cells.py C SMOKE || { echo "[chain_BC] smoke C FAILED"; exit 1; }
echo "[chain_BC] stage C $(date -Is)"; $PY run_depth_cells.py C || { echo "[chain_BC] stage C FAILED"; exit 1; }
echo "[chain_BC] done $(date -Is)"
