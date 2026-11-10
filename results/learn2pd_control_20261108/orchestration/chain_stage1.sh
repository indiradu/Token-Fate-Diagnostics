#!/bin/bash
# One claim. Stage 1 = HumanEval 164 (with parity + scorer guard) then MATH 108 (parity skipped: already passed in the HumanEval process). Resumable per cell record.
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/learn2pd_control_20261108/orchestration || exit 1
echo "[chain] he164 $(date -Is)"; $PY run_l2pd_stack.py he164 || { echo "[chain] he164 FAILED"; exit 1; }
echo "[chain] he164 done $(date -Is)"
echo "[chain] math108 $(date -Is)"; $PY run_l2pd_stack.py math108 NOPARITY || { echo "[chain] math108 FAILED"; exit 1; }
echo "[chain] done $(date -Is)"
