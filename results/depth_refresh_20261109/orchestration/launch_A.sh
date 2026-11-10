#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
POLL_S=20 setsid nohup $R/orchestration/claim_gpu.sh chainA -- bash $R/orchestration/chain_A.sh > /dev/null 2>&1 < /dev/null &
