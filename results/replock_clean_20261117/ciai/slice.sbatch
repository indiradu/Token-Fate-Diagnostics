#!/bin/bash
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu-05,gpu-07,gpu-51,gpu-52
#SBATCH --job-name=phaselock
#SBATCH --output=/nfs-stor/rui.gao/Researches/phaselock/logs/%x-%j.log
# One short slice (30 min wall, about 25 min of work) of the GPU-transfer smoke (1 GPU is enough: LLaDA-8B bf16 at batch 1 uses about 15-18 GiB of the 40 GB). Short on purpose: backfills into free GPUs quickly, as the opichain slices do.
# Usage: sbatch [-J name] [--dependency=...] slice.sbatch STEP [STEP ...]    STEP in S7G/S7H (five arms incl. the ORIGINAL decoder, 40 prompts, GSM8K/HumanEval) | S8G/S8K/S8M (four arms, 100 prompts, GSM8K/KodCode/MATH) | S3 (sanity) | S4 (cost benchmark) | S5 (smoke decode) | S6G (comparison on GSM8K) | S6H (comparison on HumanEval); S6G/S6H resume from saved records and exit 75 when the work budget is used
# The exclude list is the one used for the opichain chain (gpu-05/07/51/52 had unusable or contended GPUs). Nothing here is submitted automatically.
set -uo pipefail
P=/nfs-stor/rui.gao/Researches/phaselock
PY=/nfs-stor/rui.gao/envs/phaselock/bin/python
export PHASELOCK_ROOT=$P MODEL_TAG=original MODEL_CHECKPOINT=/nfs-stor/rui.gao/models/LLaDA-8B-Instruct FAMILY_SRC=$P/results/model_transfer_20261113/source_m1 DEPTH_DEFER_SCORING=1
export HF_HOME=/nfs-stor/rui.gao/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/nfs-stor/rui.gao/tmp TAG=a100
export BENCH_JSON=$P/results/model_transfer_20261113/original/records/bench_layer_cost.json
START=$(date +%s)
echo "node $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} job=${SLURM_JOB_ID:-0} steps=$* $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then                      # pre-flight as in the opichain chain: the assigned GPU must be empty
  BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v vis="${CUDA_VISIBLE_DEVICES:-}" 'BEGIN{n=split(vis,a,","); for(i=1;i<=n;i++) m[a[i]]=1} (vis=="" || (($1 in m))) && $2>200 {print "gpu"$1": "$2" MiB"}')
  if [ -n "$BUSY" ]; then echo "PRE-FLIGHT FAILED on $(hostname): $BUSY"; exit 3; fi; echo "pre-flight ok"
fi
# Successor first (the opichain pattern; the QOS allows only about 4 submitted jobs per user, so one successor at a time): SUCC_DEPTH>0 means "queue the same steps again, to start when this job ends".
if [ "${SUCC_DEPTH:-0}" -gt 0 ]; then
  NEXT=$(sbatch --parsable -J pl-cmp --dependency=afterany:${SLURM_JOB_ID} --export=ALL,SUCC_DEPTH=$(( SUCC_DEPTH - 1 )) $P/ciai/slice.sbatch "$@") && echo "successor queued: $NEXT (depth left $(( SUCC_DEPTH - 1 )))" || echo "successor NOT queued (submit limit?)"
fi
cd $P/results/model_transfer_20261113/orchestration || exit 1
mkdir -p $P/results/model_transfer_20261113/original/records $P/records
for STEP in "$@"; do
  echo "=== $STEP $(date)"
  case $STEP in
    S3) $PY $P/ciai/sanity.py ;;
    S4) $PY bench_layer_cost.py ;;
    S5) $PY $P/ciai/run_transfer.py gsm_held SMOKE ;;
    S6G) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) ))
         PUB_LIMIT=40 $PY $P/ciai/run_transfer.py gsm_held ;;
    S6H) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) ))
         PUB_LIMIT=40 $PY $P/ciai/run_transfer.py he164 ;;
    S7G) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) )); TAG=a100full ARMS_LIST=ORIG,DENSE_C09,S32,R0_OWN,R0_RTX PUB_LIMIT=40 $PY $P/ciai/run_transfer.py gsm_held ;;
    S7H) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) )); TAG=a100full ARMS_LIST=ORIG,DENSE_C09,S32,R0_OWN,R0_RTX PUB_LIMIT=40 $PY $P/ciai/run_transfer.py he164 ;;
    S8G) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) )); TAG=a100n100 PUB_LIMIT=100 $PY $P/ciai/run_transfer.py gsm_held ;;
    S8K) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) )); TAG=a100n100 PUB_LIMIT=100 $PY $P/ciai/run_transfer.py kod_held ;;
    S8M) export SLICE_SECONDS=$(( 1500 - ($(date +%s) - START) )); TAG=a100n100 PUB_LIMIT=100 $PY $P/ciai/run_transfer.py math108 ;;
    *) echo "unknown step $STEP"; exit 2 ;;
  esac
  RC=$?; echo "=== $STEP exit $RC $(date)"; if [ $RC -ne 0 ]; then exit $RC; fi
done
echo "slice done $(date)"
