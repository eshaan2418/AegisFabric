#!/usr/bin/env bash
# Run every lab (privileged-container) campaign behind the paper, in order, resumably.
#   bash bench/lab_campaigns.sh [stage ...]     stages: m1 m2 m3cal ring m3 m3abl m5 (default: all)
# Each stage runs in a fresh lab container; logs go to results/logs/<stage>.log. The M3 drivers skip runs whose
# result file already exists, so re-running after an interruption resumes.
# Ablations use 3 seeds (vs 5 in the main campaign) to fit the time budget on a shared host; see DECISIONS D-012.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${LAB_IMAGE:-aegis-lab:latest}"
COMMIT="$(git rev-parse HEAD)"
DIRTY="$(test -n "$(git status --porcelain --untracked-files=no)" && echo 1 || echo 0)"
mkdir -p results/logs
LAB() {
  docker run --rm --privileged --cgroupns=host -e AEGIS_GIT_COMMIT="$COMMIT" -e AEGIS_GIT_DIRTY="$DIRTY" \
    -v "$PWD":/work -v aegis-cargo-target:/work/ckpt/target "$IMAGE" bash -c "$*"
}
stage() {
  local name="$1"; shift
  echo "=== $(date -u +%FT%TZ) start $name (commit $COMMIT dirty=$DIRTY)" | tee -a results/logs/campaigns.log
  if LAB "$@" >>"results/logs/$name.log" 2>&1; then
    echo "=== $(date -u +%FT%TZ) done $name" | tee -a results/logs/campaigns.log
  else
    echo "=== $(date -u +%FT%TZ) FAILED $name (see results/logs/$name.log)" | tee -a results/logs/campaigns.log
    return 1
  fi
}
STAGES=("${@:-m1 m2 m3cal ring m3 m3abl m5}")
for s in ${STAGES[@]}; do
  case "$s" in
    m1)    stage m1 "python -m bench.m1_baselines --seeds 5 --out results/m1" ;;
    m2)    stage m2 "python -m bench.m2_convergence --trials 10 --out results/m2" ;;
    m3cal) stage m3cal "python -m bench.m3_calibrate --out results/m3" ;;
    ring)  stage ring "python -m bench.m3_ring_micro --trials 5 --out results/m3" ;;
    m3)    stage m3 "python -m bench.m3_campaign --scenarios all --seeds 5 --out results/m3" ;;
    m3abl) stage m3abl "python -m bench.m3_campaign --ablation --scenarios steady,bw_collapse,latency_spike,site_degradation --seeds 3 --out results/m3" ;;
    m5)    stage m5 "python -m bench.m5_rotation --trials 5 --steps 60 --out results/m5" ;;
    *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
