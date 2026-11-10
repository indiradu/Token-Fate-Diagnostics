#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd $R/orchestration || exit 1
echo "[chain_bench] $(date -Is)"; $PY bench_layer_cost.py || { echo "[chain_bench] FAILED"; exit 1; }
echo "[chain_bench] done $(date -Is)"
