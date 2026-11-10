#!/bin/bash
# Usage: claim_gpu.sh LOGNAME -- COMMAND...
# WAITS (poll every POLL_S, default 300 s) until, for 3 consecutive polls: no other agent's GPU_IN_USE marker belongs to a live pid, /tmp/dof_gpu.lock is empty/stale, nvidia-smi shows < 2 GiB used and
# no compute process, no run_l2pd*/run_cache_stack*/run_stack* python is alive, and the other agents' logs directories have been quiet for 30 min (so we do not slip into the gap between two of
# their stages). Gives up after MAXWAIT_S (default 43200 s = 12 h) with exit 3. Then writes our marker, runs COMMAND with output in logs/LOGNAME.log, and removes the marker.
ROOT=/home/rui.gao/Desktop/Token-Fate-Diagnostics
R=$ROOT/results/stale_cause_20261108
MINE=$R/GPU_IN_USE
LOGS=$R/logs
POLL_S=${POLL_S:-300}
MAXWAIT_S=${MAXWAIT_S:-43200}
OTHERS="$ROOT/results/learn2pd_control_20261108/GPU_IN_USE $ROOT/results/farcache_stack_20261105/GPU_IN_USE $ROOT/results/repfreeze_tests_20261029/GPU_IN_USE $ROOT/results/longlen_wordlock_20261103/GPU_IN_USE $ROOT/results/selective_cache_20261104/GPU_IN_USE"
QUIET="$ROOT/results/learn2pd_control_20261108/logs $ROOT/results/learn2pd_control_20261108/records $ROOT/results/farcache_stack_20261105/logs"
name=$1; shift; shift
mkdir -p $LOGS
log() { echo "[claim] $* $(date -Is)" >> $LOGS/claims.log; }
busy() {
  for m in $OTHERS; do
    if [ -e "$m" ]; then
      pid=$(cut -d' ' -f1 "$m")
      if kill -0 "$pid" 2>/dev/null; then why="marker $m live (pid $pid)"; return 0; fi
      log "stale marker $m (pid $pid dead)"
    fi
  done
  if [ -s /tmp/dof_gpu.lock ]; then
    pid=$(cut -d' ' -f1 /tmp/dof_gpu.lock)
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then why="/tmp/dof_gpu.lock live (pid $pid)"; return 0; fi
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -ge 2000 ]; then why="GPU memory used ${used} MiB"; return 0; fi
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then why="a compute process is on the GPU"; return 0; fi
  if pgrep -f "run_l2pd|run_cache_stack|run_stack|run_noise_control|run_oracle_repair" > /dev/null; then why="another run_* python is alive"; return 0; fi
  # another agent's queued claim (any other claim_gpu.sh process): yield to it, so we never start in the gap between two of their stages
  if pgrep -f "claim_gpu.sh" | grep -vx "$$" | grep -q .; then why="another claim_gpu.sh is alive (a queued or running claim of another agent)"; return 0; fi
  for d in $QUIET; do
    if [ -d "$d" ] && [ -n "$(find "$d" -maxdepth 1 -type f -mmin -30 2>/dev/null | head -1)" ]; then why="$d was written in the last 30 min"; return 0; fi
  done
  why=""
  return 1
}
ok=0; waited=0
log "$name waiting"
while true; do
  if busy; then ok=0; echo "[claim] busy: $why ($(date -Is))" > $LOGS/claim_last_poll.txt; else ok=$((ok+1)); [ "$ok" -ge 3 ] && break; fi
  sleep $POLL_S
  waited=$((waited+POLL_S))
  if [ "$waited" -ge "$MAXWAIT_S" ]; then log "$name gave up after ${waited}s (last: $why)"; exit 3; fi
done
echo "$$ $(date -Is) stale_cause $name holds the GPU" > "$MINE"
echo "$$ $(date -Is) stale_cause $name" > /tmp/dof_gpu.lock
trap 'rm -f "$MINE"; : > /tmp/dof_gpu.lock' EXIT
log "$name start"
"$@" > "$LOGS/$name.log" 2>&1
code=$?
log "$name exit $code"
exit $code
