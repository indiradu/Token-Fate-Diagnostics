#!/bin/bash
# One claim. Stage 1 (parity + HumanEval 164 + MATH 108 at 512), then the added HumanEval diagnostics (non-fatal), then stage 2 (GSM8K 200 + MATH 1024 half).
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/farcache_stack_20261105/orchestration || exit 1
echo "[chain] stage1 $(date -Is)"; $PY run_cache_stack.py 1 || { echo "[chain] stage1 FAILED"; exit 1; }
echo "[chain] stage1 done $(date -Is)"
if [ -f diag_he.py ]; then echo "[chain] diag $(date -Is)"; $PY diag_he.py run || echo "[chain] diag FAILED (non-fatal)"; echo "[chain] diag end $(date -Is)"; fi
echo "[chain] stage2 $(date -Is)"; $PY run_cache_stack.py 2 || { echo "[chain] stage2 FAILED"; exit 1; }
echo "[chain] done $(date -Is)"
