#!/bin/bash
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/learn2pd_control_20261108/orchestration || exit 1
echo "[chain-pilot] $(date -Is)"; $PY run_l2pd_stack.py pilot || exit 1
echo "[chain-pilot] done $(date -Is)"
