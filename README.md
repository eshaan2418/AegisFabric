# AegisFabric

A reproducible research artifact for **network-aware, secure, multimodal model training** across three
emulated sites (Toronto, Montréal, a U.S. region). The primary question:

> Does *joint* optimisation of compute placement (π), network routing (r) and checkpoint storage (s)
> reduce training time, step-time variance and failure-recovery time compared with independently tuned
> subsystems, under reproducible workload and failure conditions?

Everything runs on one machine: a CPU-only training rung inside a privileged Linux container, with real
FRRouting BGP/OSPF and kernel forwarding between network namespaces (see [Limitations](#limitations)).
Every number below is computed from a file in `results/` by `bench/claims.py`, and
`python -m bench.audit_numbers` fails if this README, the paper or the resume bullets contain a number that
no claim backs.

## Architecture

```
                     ┌──────────────────────── controller/ ────────────────────────┐
  telemetry ───────► │ Estimator (util, speed, rtt, coll, ckpt) → cost model (κ/plane)│
  (netlab, ranks)    │ Joint planner: exhaustive (π, s, r)  │  baselines: static/ECMP/ │
                     │ migration gate + hysteresis          │  independent TE / oracle │
                     └───────────┬──────────────────────────┴──────────┬───────────────┘
                                 │ route-maps via vtysh (local-pref)    │ stop / relaunch / restore
             ┌───────────────────▼──────────────────┐    ┌──────────────▼──────────────────────┐
  netlab/    │ TOR ── plane A (low-lat) ── MTL      │    │ train/  6 Gloo ranks in site netns   │
  FRR eBGP / │   \\── plane B (hi-cap) ──/  \\  /    │◄──►│ DDP / FSDP2 / HSDP, telemetry,       │
  iBGP/OSPF, │        US (both planes)              │    │ sliced async checkpoints             │
  tc/netem   │ failure injection (7 types)          │    └──────────────┬──────────────────────┘
             └──────────────────────────────────────┘                   │ routed storage prefix
                                                         ┌──────────────▼──────────────────────┐
  security/  CA (SPIFFE SVIDs), mTLS gRPC control plane, │ ckpt/  Rust engine: BLAKE3 CAS chunks, │
             rotation (two-phase CA)                     │ WAL, RocksDB index, io_uring, GC; P2:  │
                                                         │ AES-256-GCM envelope + Ed25519 manifests│
  formal/    TLA+ models of P1 (checkpoint) and P3 (rotation), checked with TLC                  │
  media/     RTP/RTCP + AV1 pipeline, ABR, backpressured inference; aiortc WebRTC ingest        │
  kernels/   Triton + Metal fused varlen prefix-LM attention                                     │
  posttrain/ privacy/   SFT / DPO / GRPO / RM; DP-SGD (Opacus) + MIA; FedAvg                     │
```

## Repository layout

| dir | what |
|---|---|
| `train/` | tiny multimodal transformer, DDP/FSDP2/HSDP trainer over Gloo, telemetry, sliced async checkpoints |
| `netlab/` | netns + FRR three-site WAN, tc shaping, BGP-policy TE, failure injection, blob store |
| `controller/` | analytical cost model, joint planner, baselines (incl. B4-style TE MILP), telemetry estimator |
| `ckpt/` | Rust checkpoint engine + Python client (`EngineBackend` is a drop-in for the blob store) |
| `security/` | test CA, SPIFFE IDs, mTLS gRPC, rotation, envelope encryption (re-exports `ckpt/crypto.py`) |
| `formal/` | TLA+ specs, model configs, TLC runner |
| `media/`, `kernels/`, `posttrain/`, `privacy/` | research extensions (M6–M8) |
| `data/synth/` | synthetic financial statements, transcripts, chart videos, QA tasks (no real PII) |
| `bench/` | experiment drivers, statistics, figures, claims registry, number audit |
| `paper/` | LaTeX paper (`make paper`) |
| `results/` | every result file (with commit, hardware, seed, config, timestamp) and `results/INDEX.md` |

## Reproducing

```
make setup             # uv sync (Python 3.12) — see DECISIONS D-004 for the package index
make lab-image         # privileged Linux lab container (FRR, tc, Rust, torch CPU)
make test              # unit tests (Python + Rust)
make ci                # lint + test + TLA+ + lab integration tests
make reproduce-quick   # < 1 h subset of every experiment into results/quick/, regenerates figures
make reproduce-full    # the full campaigns behind the paper (≈ a day on an 18-core laptop)
make demo              # end-to-end: 3-site training → failure → re-plan → encrypted restore → live WebRTC assistant
make paper             # figures + claims → paper/main.pdf
make audit             # every number in README / paper / resume bullets must be backed by results/
```

Lab campaigns run from a **frozen worktree at one commit** (`bench/lab_campaigns.sh`); each result file
records that commit. `results/INDEX.md` lists every experiment with its command, output and one-line finding.

## Results

<!-- RESULTS -->

## Limitations

- **Emulated WAN, not a real one.** veth + `tbf`/`netem` in one VM, with capacities scaled down so that a
  CPU testbed is communication-bound. Real WAN failure modes (gray links, cross traffic) are only
  approximated by scripted failures.
- **CPU, not GPU.** A tiny model on a few CPU threads per rank. Absolute times and MFU (relative to a measured
  CPU matmul peak) say nothing about GPU clusters. Claims are relative effects inside the measured regime.
- **Small search space.** 6 ranks × 3 sites allow exhaustive search; production scale would need ILP/LNS.
- **Shared host.** Other workloads ran concurrently. Methods are interleaved in seeded random order and
  load averages are recorded, but variance is inflated.
- **Prefix-level TE.** No Open vSwitch on Docker Desktop's kernel; steering is BGP local-preference per
  destination prefix and class.
- **Security ≠ privacy; FUSE/VM disk ≠ kernel FS on NVMe; virtual-time media ≠ wall-clock media; toy-scale DP
  ≠ foundation-model DP.** Each is detailed in `BLOCKERS.md`, `results/*/NOTES.md` and the paper's §Limitations.

See `FINAL_REVIEW.md` for the strongest and weakest claims, threats to validity and what a real-scale test
would need.
