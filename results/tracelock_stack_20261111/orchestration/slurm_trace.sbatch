#!/bin/bash
#SBATCH -J traceStack
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --exclude=ws-l2-005,ws-l2-011
#SBATCH -o /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/tracelock_stack_20261111/logs/slurm_trace_%j.log
# TraceLock (published learned word lock) under S32 / K4 / P4. Submitted with --dependency=afterany:290321 (the last queued job); one GPU job at a time. Order:
#   1 PARITY  engine dense trace lock vs paper_decode T-block on 3 real HumanEval questions (HF forward vs hand-run layers): ABORTS the whole job (exit 3) on any mismatch, before any evaluation decode
#   2 SEL     smoke (2 questions) then the matched-confidence selection on 40 disjoint questions (if it fails, records/selection.json = C-0.8 fallback and the job continues)
#   3 M512    MATH 108 @512 (smoke first; scored in-process)         4 P256  the earlier 124-case panel @256 (smoke first; HumanEval part scoring deferred)
#   5 H512    HumanEval 164 @512 (smoke first; scoring DEFERRED: rescore_trace_records.py with bwrap on PATH afterwards)
# A failure in stages 3-5 is logged and the next stage still runs (records are per question, so partial stages stay readable). Expected wall time 6-9 h (limit 12 h).
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/tracelock_stack_20261111
PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python
CPY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/.venv/bin/python
export TRACELOCK_DEFER_SCORING=1
hostname; echo "[job] ulimit_v=$(ulimit -v)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd $R/orchestration || exit 1
echo "[1] parity $(date -Is)"; $PY run_trace_cells.py PARITY; rc=$?
if [ $rc -ne 0 ]; then echo "[1] PARITY FAILED (exit $rc): aborting before any evaluation decode; see records/parity.json"; exit $rc; fi
echo "[2] selection smoke $(date -Is)"; $PY run_trace_cells.py SEL SMOKE || echo "[2] selection smoke FAILED (continuing to the real selection)"
echo "[2] selection $(date -Is)"; $PY run_trace_cells.py SEL || { echo "[2] selection FAILED: C-0.8 fallback"; $PY run_trace_cells.py SELFALLBACK; }
for st in M512 P256 H512; do
  echo "[$st] smoke $(date -Is)"; $PY run_trace_cells.py $st SMOKE || { echo "[$st] smoke FAILED: skipping this stage"; continue; }
  echo "[$st] run $(date -Is)";   $PY run_trace_cells.py $st || echo "[$st] run FAILED (exit $?): partial records kept"
done
echo "[analysis] $(date -Is)"; $CPY analyze_trace_cells.py > $R/logs/analyze_unscored.log 2>&1 || echo "[analysis] failed (expected while HumanEval is unscored)"
echo "[done] $(date -Is)"
