#!/bin/bash
# One GPU claim: parity gate (official model, then ours), noise floor, Part A captures at 256 and 512, then (only if the parity gate passed) the Part C smoke and run.
# All steps resumable. A failed parity step does not stop Part A; a failed gate stops before Part C.
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
cd /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/selective_cache_20261104/orchestration || exit 1
echo "[chain1] parity official $(date -Is)"; $PY parity_gate.py official; echo "[chain1] exit $?"
echo "[chain1] parity ours $(date -Is)"; $PY parity_gate.py ours; echo "[chain1] exit $?"
echo "[chain1] noise $(date -Is)"; $PY partA_collect.py noise || echo "[chain1] noise failed"
echo "[chain1] A256 $(date -Is)"; $PY partA_collect.py 256 || exit 1
echo "[chain1] A512 $(date -Is)"; $PY partA_collect.py 512 || exit 1
echo "[chain1] A done $(date -Is)"
$PY parity_ok.py || { echo "[chain1] parity gate FAILED: stopping before Part C"; exit 0; }
$PY partA_analyze.py --boot 100 > ../logs/partA_analyze.log 2>&1 || echo "[chain1] analyze failed"
$PY calibrate_theta.py > ../logs/calibrate.log 2>&1 || { echo "[chain1] calibrate failed"; exit 1; }
echo "[chain1] C smoke $(date -Is)"; $PY run_partC.py run SMOKE || { echo "[chain1] C smoke failed"; exit 1; }
echo "[chain1] C run $(date -Is)"; $PY run_partC.py run || exit 1
echo "[chain1] done $(date -Is)"
