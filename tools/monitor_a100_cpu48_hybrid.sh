#!/usr/bin/env bash
set -euo pipefail
INTERVAL="${1:-10}"
while true; do
  clear || true
  date
  echo '--- A100 ---'
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits || true
  echo '--- CPU solver processes ---'
  COUNT=$(pgrep -fc 'ecsp_cpp_solver_hybrid' || true)
  echo "active C++ workers: $COUNT"
  ps -eo pid,psr,pcpu,pmem,etime,comm,args --sort=-pcpu \
    | grep -E 'ecsp_cpp_solver_hybrid|run_nsga2_electrical_solid_loop|python' \
    | grep -v grep | head -50 || true
  echo '--- load/memory ---'
  uptime || true
  free -h || true
  sleep "$INTERVAL"
done
