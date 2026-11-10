#!/bin/bash
# Part A GPU queue: noise floor, then dense captures at 256 and 512 (resumable)
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/selective_cache_20261104/orchestration || exit 1
$PY partA_collect.py noise || exit 1
$PY partA_collect.py 256 || exit 1
$PY partA_collect.py 512 || exit 1
