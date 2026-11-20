#!/bin/bash
# Chain for the GPU-transfer smoke, 30-minute slices. The QOS allows about 4 submitted jobs per user (two are the opichain jobs), so only link 1 and the first comparison slice are submitted here;
# every comparison slice queues its own successor when it starts (SUCC_DEPTH), as the opichain slices do.
#   link 1: S3+S4+S5 (sanity, this GPU's cost benchmark, 2-prompt smoke).
#   comparison slices: steps S6G then S6H (40 GSM8K prompts, then 40 HumanEval prompts), resumable; the first starts only if link 1 succeeded (afterok); up to 3 successors (afterany), each of which resumes from the
#   saved records and exits quickly when everything is done.
cd /nfs-stor/rui.gao/Researches/phaselock/ciai
J1=${1:-$(sbatch --parsable -J pl-s345 slice.sbatch S3 S4 S5)}
C1=$(SUCC_DEPTH=3 sbatch --parsable -J pl-cmp --dependency=afterok:$J1 --export=ALL,SUCC_DEPTH=3 slice.sbatch S6G S6H)
echo "submitted: s345=$J1 comparison=$C1"
