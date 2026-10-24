# PROGRESS

Status values: `todo / in-progress / done / blocked-with-fallback`.

| M | Title | Status | Evidence |
|---|-------|--------|----------|
| M0 | Bootstrap | done | `ENVIRONMENT.md`, `results/m0/environment.json`, `make test` + `make lint` green |
| M1 | Baseline distributed training | todo | |
| M2 | Network laboratory | todo | |
| M3 | Adaptive placement & routing controller (primary) | todo | |
| M4 | Crash-consistent checkpoint engine | todo | |
| M5 | Security layer | todo | |
| M6 | Real-time multimodal media pipeline | todo | |
| M7 | Compiler/kernel extension | todo | |
| M8 | Post-training & privacy | todo | |
| M9 | Formal checks (TLA+) | todo | |
| M10 | Integration, reproduction, write-up | todo | |

## Next action
M1: implement `train/model.py` (tiny multimodal transformer), `train/data.py` (synthetic chart+statement
dataset), `train/trainer.py` (DDP / FSDP2 / HSDP over Gloo, telemetry hooks), and baselines on loopback.

## M0 plan (done)
- [x] Probe environment → ENVIRONMENT.md (script `bench/probe_env.py`)
- [x] Skeleton, Makefile, CI workflow, pinned deps (`uv.lock`), lab Dockerfile
- [x] Synthetic data generators (`data/synth/finance.py`): statements (+PDF), transcripts, chart videos
- [x] Accept: `make test` passes; ENVIRONMENT.md complete
