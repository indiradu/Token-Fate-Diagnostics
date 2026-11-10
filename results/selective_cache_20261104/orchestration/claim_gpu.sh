#!/bin/bash
# Usage: claim_gpu.sh LOGNAME -- COMMAND...
# Waits (poll 30 s) until nvidia-smi shows < 2 GiB used AND the other agents' GPU_IN_USE markers are absent or stale (their pid is dead) for 3 consecutive polls,
# writes our marker, runs the command, removes the marker.
ROOT=/home/rui.gao/Desktop/Token-Fate-Diagnostics
MINE=$ROOT/results/selective_cache_20261104/GPU_IN_USE
LOGS=$ROOT/results/selective_cache_20261104/logs
OTHERS="$ROOT/results/repfreeze_tests_20261029/GPU_IN_USE $ROOT/results/longlen_wordlock_20261103/GPU_IN_USE"
name=$1; shift; shift
busy() {
  for m in $OTHERS; do
    if [ -e "$m" ]; then
      pid=$(cut -d' ' -f1 "$m")
      if kill -0 "$pid" 2>/dev/null; then return 0; fi
      echo "[claim] stale marker $m (pid $pid dead) $(date -Is)" >> $LOGS/claims.log
    fi
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -ge 2000 ] && return 0
  return 1
}
ok=0
while true; do
  if busy; then ok=0; else ok=$((ok+1)); [ "$ok" -ge 3 ] && break; fi
  sleep 30
done
echo "$$ $(date -Is) selective_cache $name holds the GPU" > "$MINE"
echo "[claim] $name start $(date -Is)" >> $LOGS/claims.log
"$@" > "$LOGS/$name.log" 2>&1
code=$?
echo "[claim] $name exit $code $(date -Is)" >> $LOGS/claims.log
rm -f "$MINE"
exit $code
