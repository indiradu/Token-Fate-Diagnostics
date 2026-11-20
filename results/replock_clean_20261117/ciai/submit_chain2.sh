#!/bin/bash
# Second ciai chain (2026-11-19): the ORIGINAL decoder as a fifth arm (40 prompts, GSM8K and HumanEval) and a 100-prompt four-arm comparison on GSM8K, KodCode and MATH. One job; each slice queues its own successor (SUCC_DEPTH).
# The cluster allows about 4 submitted jobs per user on cscc-gpu-p: if a successor is not accepted, resubmit this script (finished steps exit at once, before loading the model).
cd /nfs-stor/rui.gao/Researches/phaselock/ciai
sbatch --parsable -J pl-orig --export=ALL,SUCC_DEPTH=8 slice.sbatch S7G S7H S8G S8K S8M
