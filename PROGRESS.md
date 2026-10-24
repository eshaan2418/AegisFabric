# PROGRESS

Status values: `todo / in-progress / done / blocked-with-fallback`.

| M | Title | Status | Evidence |
|---|-------|--------|----------|
| M0 | Bootstrap | done | `ENVIRONMENT.md`, `results/m0/environment.json`, `make test` + `make lint` green |
| M1 | Baseline distributed training | in-progress | code: `train/`, driver `bench/m1_baselines.py`; campaign pending |
| M2 | Network laboratory | in-progress | code: `netlab/`, lab tests `tests/test_netlab.py` (4/4 in container); `bench/m2_convergence.py` campaign pending |
| M3 | Adaptive placement & routing controller (primary) | in-progress | code: `controller/`, `bench/wan_run.py`, `bench/m3_*.py`; calibration + campaign pending |
| M4 | Crash-consistent checkpoint engine | in-progress | delegated to worktree branch (Rust `ckpt/`, envelope encryption) |
| M5 | Security layer | in-progress | CA/mTLS/rotation code in `security/`; `bench/m5_rotation.py` pending; envelope part with M4 |
| M6 | Real-time multimodal media pipeline | in-progress | delegated to worktree branch |
| M7 | Compiler/kernel extension | done | `results/m7/` (correctness, perf CPU+MPS, sm80 IR) |
| M8 | Post-training & privacy | in-progress | delegated to worktree branch |
| M9 | Formal checks (TLA+) | done | `results/m9/summary.json`, `formal/` |
| M10 | Integration, reproduction, write-up | todo | `bench/reproduce.sh`, `bench/make_figures.py`, `bench/audit_numbers.py` drafted |

## Next action
Run the lab campaigns sequentially (M1 → M2 → M3 calibrate → M3 main → M3 ablations → M5 rotation), merge the
M4/M6/M8 branches, then M10 (integration demo, paper, README, resume bullets, audit, FINAL_REVIEW).

## Campaign plan
- M1: `bench.m1_baselines --seeds 5` (DDP/FSDP2/HSDP × loopback/WAN).
- M2: `bench.m2_convergence --trials 10`.
- M3: calibrate; main = 8 scenarios × 5 methods × 5 seeds (methods interleaved per block); ablations =
  4 scenarios × (5 signal ablations + no-migration) × seeds.
- M5: `bench.m5_rotation --trials 5 --steps 60`.
