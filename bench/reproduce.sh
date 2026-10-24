#!/usr/bin/env bash
# Reproduce AegisFabric experiments.
#   bash bench/reproduce.sh quick   (< 1 h: reduced trials/seeds, writes to results/quick/, regenerates figures)
#   bash bench/reproduce.sh full    (many hours: the full campaigns behind the paper, writes to results/)
# Requires: uv, Docker (privileged containers). Builds the lab image if missing.
set -euo pipefail
MODE="${1:-quick}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:/opt/homebrew/opt/openjdk/bin:$PATH"
IMAGE="${LAB_IMAGE:-aegis-lab:latest}"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY="$(test -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo 1 || echo 0)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || make lab-image
LAB() { docker run --rm --privileged --cgroupns=host -e AEGIS_GIT_COMMIT="$COMMIT" -e AEGIS_GIT_DIRTY="$DIRTY" \
          -v "$PWD":/work -v aegis-cargo-target:/work/ckpt/target "$IMAGE" bash -lc "$*"; }
step() { echo; echo "=== [$MODE] $* ==="; }

if [ "$MODE" = "quick" ]; then
  OUT=results/quick
  mkdir -p "$OUT"
  step "unit tests";            make test
  step "environment probe";     uv run python -m bench.probe_env --image "$IMAGE"
  step "M1 baselines (1 seed)"; LAB "python -m bench.m1_baselines --seeds 1 --nets loopback --steps 120 --target-loss 0.9 --out $OUT/m1"
  step "M2 convergence (1 trial)"; LAB "python -m bench.m2_convergence --trials 1 --out $OUT/m2"
  step "M3 calibration";        LAB "python -m bench.m3_calibrate --out $OUT/m3"
  step "M3 campaign (subset)";  LAB "python -m bench.m3_campaign --scenarios steady,bw_collapse,site_degradation --methods static,independent,joint --seeds 1 --steps 80 --out $OUT/m3"
  step "M3 analysis";           uv run python -m bench.m3_analyze --out "$OUT/m3"
  step "M5 rotation (1 trial)"; LAB "python -m bench.m5_rotation --trials 1 --steps 40 --out $OUT/m5"
  step "M9 TLA+ (small models)"; make test-formal
  [ -x bench/reproduce_extensions.sh ] && { step "extensions (quick)"; bash bench/reproduce_extensions.sh quick "$OUT"; }
  step "figures from committed results"; uv run python -m bench.make_figures
  echo; echo "quick reproduction finished: outputs in $OUT/, figures regenerated under results/*/figs/"
elif [ "$MODE" = "full" ]; then
  step "unit + lab tests";      make test test-lab
  step "M1 baselines";          LAB "python -m bench.m1_baselines --seeds 5 --out results/m1"
  step "M2 convergence";        LAB "python -m bench.m2_convergence --trials 10 --out results/m2"
  step "M3 calibration";        LAB "python -m bench.m3_calibrate --out results/m3"
  step "M3 main campaign";      LAB "python -m bench.m3_campaign --scenarios all --seeds 5 --out results/m3"
  step "M3 ablations";          LAB "python -m bench.m3_campaign --ablation --scenarios steady,bw_collapse,latency_spike,site_degradation --seeds 5 --out results/m3"
  step "M3 analysis";           uv run python -m bench.m3_analyze --out results/m3
  step "M5 rotation";           LAB "python -m bench.m5_rotation --trials 5 --steps 60 --out results/m5"
  step "M9 TLA+";               bash formal/check.sh
  [ -x bench/reproduce_extensions.sh ] && { step "extensions (full)"; bash bench/reproduce_extensions.sh full results; }
  step "figures";               uv run python -m bench.make_figures
  step "number audit";          uv run python -m bench.audit_numbers
else
  echo "usage: $0 quick|full" >&2; exit 2
fi
