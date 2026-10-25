#!/usr/bin/env bash
# M8 post-training campaign: LR tuning (3 parallel workers) -> merge -> 5 seeds (3 parallel workers) -> summary.
# Shared host: every worker runs under `nice -n 19` with AEGIS_THREADS torch threads (default 2).
# Usage: AEGIS_DEVICE=mps bash bench/m8_posttrain_all.sh [logdir]
set -euo pipefail
LOG=${1:-scratch}
mkdir -p "$LOG"
export AEGIS_THREADS=${AEGIS_THREADS:-2}
PY="nice -n 19 uv run python -m bench.m8_posttrain"
run_wave() {
  local pids=()
  for cfg in "$@"; do
    set -- $cfg
    if [ ! -f "results/m8/posttrain/tune_parts/${1}_${2}.json" ]; then
      $PY tune --method "$1" --lr "$2" > "$LOG/tune_${1}_${2}.log" 2>&1 &
      pids+=($!)
    fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
}
run_wave "sft 0.0003" "dpo 0.0003" "grpo 0.0001"
run_wave "sft 0.001" "dpo 0.001" "grpo 0.0003"
$PY tune-merge
$PY run --seeds 0 1 > "$LOG/pt_run_a.log" 2>&1 &
a=$!
$PY run --seeds 2 3 > "$LOG/pt_run_b.log" 2>&1 &
b=$!
$PY run --seeds 4 > "$LOG/pt_run_c.log" 2>&1 &
c=$!
wait $a $b $c
$PY summarize
