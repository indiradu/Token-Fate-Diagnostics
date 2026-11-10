#!/bin/bash
# Stage C only (resumes records/stageC.jsonl after the GPU-contention abort at question 57). Scoring deferred (no bwrap on the decoding host).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
export DEPTH_DEFER_SCORING=1
export DEPTH_CELLS_C='W10-dense=wdense,10;W20-dense=wdense,20;W10-blank11=wpart,10,11,blank;W20-blank11=wpart,20,11,blank;K2=dense,2;K4=dense,4'
cd $R/orchestration || exit 1
echo "[chain_C2] stage C $(date -Is)"; $PY run_depth_cells.py C || { echo "[chain_C2] stage C FAILED"; exit 1; }
echo "[chain_C2] done $(date -Is)"
