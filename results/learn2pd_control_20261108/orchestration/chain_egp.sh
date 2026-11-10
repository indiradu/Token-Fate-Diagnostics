#!/bin/bash
# One claim: EGP training traces (MATH train 56 q at gen 512, then GSM8K train 40 q at gen 256). Resumable: finished questions are skipped.
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/learn2pd_control_20261108/orchestration || exit 1
echo "[chain-egp] math $(date -Is)"; $PY collect_egp.py train_math 512 56 || exit 1
echo "[chain-egp] gsm8k $(date -Is)"; $PY collect_egp.py train_gsm8k 256 40 || exit 1
echo "[chain-egp] done $(date -Is)"
