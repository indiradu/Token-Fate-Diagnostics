#!/bin/bash
# Launch (detached):  setsid nohup /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/stale_cause_20261108/orchestration/queue.sh > /dev/null 2>&1 < /dev/null &
# Waits for the GPU (see claim_gpu.sh), then runs chain.sh. Kill the waiter with:  pkill -f stale_cause_20261108/orchestration/queue.sh   (a running chain is stopped with the same pkill on chain.sh)
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/stale_cause_20261108
exec $R/orchestration/claim_gpu.sh chain -- bash $R/orchestration/chain.sh
