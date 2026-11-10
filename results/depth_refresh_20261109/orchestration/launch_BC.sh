#!/bin/bash
R=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/depth_refresh_20261109
POLL_S=20 setsid nohup $R/orchestration/claim_gpu.sh chainBC -- bash $R/orchestration/chain_BC.sh > /dev/null 2>&1 < /dev/null &
