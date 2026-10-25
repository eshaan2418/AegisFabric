# PROGRESS

Status values: `todo / in-progress / done / blocked-with-fallback`.

| M | Title | Status | Evidence |
|---|-------|--------|----------|
| M0 | Bootstrap | done | `ENVIRONMENT.md`, `results/m0/environment.json`, `make test` + `make lint` green |
| M1 | Baseline distributed training | done | `results/m1/baselines.json` (3 parallelism configs × 2 networks × 5 seeds, time-to-target, per-collective share, tokens/s, MFU, RSS) |
| M2 | Network laboratory | done | `results/m2/convergence.json` (6 failure types × 10 trials); lab tests `tests/test_netlab.py`; step-time impact under static routing = M3 `static` runs |
| M3 | Adaptive placement & routing controller (primary) | in-progress | code: `controller/`, `bench/wan_run.py`, `bench/m3_*.py`; calibration + campaign pending |
| M4 | Crash-consistent checkpoint engine | done | `results/m4/` (crash campaign 0 undetected/2000, bench, dedup), `results/m4/NOTES.md` |
| M5 | Security layer | in-progress | CA/mTLS/rotation in `security/`; envelope done (`results/m5/envelope_overhead.json`); rotation campaign queued |
| M6 | Real-time multimodal media pipeline | done | `results/m6/summary.json`, `results/m6/NOTES.md` (virtual time; mixed result) |
| M7 | Compiler/kernel extension | done | `results/m7/` (correctness, perf CPU+MPS, sm80 IR) |
| M8 | Post-training & privacy | done | `results/m8/REPORT.md`, `results/m8/NOTES.md` |
| M9 | Formal checks (TLA+) | done | `results/m9/summary.json`, `formal/` |
| M10 | Integration, reproduction, write-up | todo | `bench/reproduce.sh`, `bench/make_figures.py`, `bench/audit_numbers.py` drafted |

## Next action
M1/M2/calibration/ring done (frozen at e204f85). M3 main → ablations → M5 rotation are running from frozen
worktree `../AegisFabric-campaign3` at 337cbbc (after the fixes in D-021/B-019). When done: copy
`results/{m3,m5}` back, run `bench.m3_analyze`, write the M3/M5 claims and paper sections, run the M10 demo,
then the README, resume bullets, FINAL_REVIEW, audit, `make ci` and reproduce-quick from a clean clone.

## Campaign plan
- M1: `bench.m1_baselines --seeds 5` (DDP/FSDP2/HSDP × loopback/WAN).
- M2: `bench.m2_convergence --trials 10`.
- M3: calibrate; main = 8 scenarios × 5 methods × 5 seeds (methods interleaved per block); ablations =
  4 scenarios × (5 signal ablations + no-migration) × seeds.
- M5: `bench.m5_rotation --trials 5 --steps 60`.
