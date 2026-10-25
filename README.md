# AegisFabric

**Should a training job's compute placement, network routing and checkpoint storage be planned together?**

AegisFabric is a reproducible research artifact that tests this question on an emulated three-site WAN
(Toronto, Montréal, a U.S. region). The WAN uses real FRRouting BGP/OSPF, kernel forwarding and scripted
failures. Training runs as multi-process PyTorch, the checkpoint engine is crash-tested, and the control
plane uses mTLS. Everything runs on one laptop and is rebuilt by `make`.

> Does *joint* optimisation of placement (π), routing (r) and checkpoint storage (s) reduce training time,
> step-time variance and failure-recovery time compared with independently tuned subsystems, under
> reproducible workload and failure conditions?

**Every number in this README is generated, not typed.** `bench/claims.py` computes each one from a result
file in `results/`. Each result file records its commit, hardware, seed, config and timestamp.
`make audit` fails if this README, the paper or the resume bullets contain a number that no claim backs.
Negative and null results are reported as they came out.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Results](#results)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [How results are kept honest](#how-results-are-kept-honest)
- [Limitations](#limitations)
- [Further reading](#further-reading)

## Highlights

- **A WAN lab in a laptop.** Three sites with two WAN planes are built from network namespaces, with FRR
  eBGP/iBGP/OSPF and `tc` shaping. Failures can be injected (link down, blackhole, flap, BGP reset,
  bandwidth collapse, latency spike, site degradation), and traffic engineering works through BGP
  local-preference.
- **A joint controller and its baselines.** An analytical ring all-reduce cost model is calibrated per WAN
  plane. The controller searches placement × storage × routing exhaustively and gates migrations so they
  don't thrash. It is compared against static routing, ECMP, an independently tuned B4-style min-MLU TE
  MILP, and an oracle.
- **A crash-consistent checkpoint engine in Rust.** It uses BLAKE3 content addressing, a WAL, a RocksDB
  index, io_uring and GC. Across 1000 `kill -9` trials, 500 torn writes and 500 bit flips, it had
  **0 undetected corruptions**.
- **Security that is tested by attacking it.** A SPIFFE-style CA, mTLS gRPC with TLS 1.3, two-phase CA
  rotation, and an AES-256-GCM envelope with Ed25519-signed manifests and anti-rollback. **12/12** attacks
  were rejected.
- **Formal models that found real bugs.** TLA+ models of the checkpoint protocol (P1) and certificate
  rotation (P3) caught a GC race and a clock-skew rejection in the designs as first written. Both were
  fixed in the spec and the code.
- **Research extensions.** A real-time AV1/WebRTC multimodal pipeline, Triton and Metal varlen attention
  kernels, SFT/DPO/GRPO post-training, DP-SGD with membership-inference audits, and FedAvg.

## Architecture

```
                     ┌──────────────────────── controller/ ─────────────────────────┐
  telemetry ───────► │ estimator (util, speed, rtt, coll, ckpt) → cost model (κ/plane)│
  (netlab, ranks)    │ joint planner: exhaustive (π, s, r)  │ baselines: static/ECMP/ │
                     │ migration gate + hysteresis          │ independent TE / oracle │
                     └───────────┬──────────────────────────┴──────────┬──────────────┘
                                 │ route-maps via vtysh (local-pref)    │ stop / relaunch / restore
             ┌───────────────────▼──────────────────┐    ┌──────────────▼──────────────────────┐
  netlab/    │ TOR ── plane A (low latency) ── MTL  │    │ train/  Gloo ranks in site netns     │
  FRR eBGP / │   \── plane B (high capacity) ─/     │◄──►│ DDP / FSDP2 / HSDP, telemetry,       │
  iBGP/OSPF, │         US (both planes)             │    │ sliced async checkpoints             │
  tc/netem   │ failure injection                    │    └──────────────┬──────────────────────┘
             └──────────────────────────────────────┘                   │ routed storage prefix
                                                         ┌──────────────▼──────────────────────┐
  security/  CA (SPIFFE SVIDs), mTLS gRPC control plane, │ ckpt/  Rust engine: BLAKE3 CAS, WAL, │
             two-phase CA rotation                       │ RocksDB index, io_uring, GC; P2      │
                                                         │ AES-GCM envelope, signed manifests   │
  formal/    TLA+ models of P1 (checkpoint) and P3 (rotation), checked with TLC
  media/     RTP/RTCP + AV1, ABR, backpressured inference; aiortc WebRTC ingest
  kernels/   Triton + Metal fused varlen prefix-LM attention
  posttrain/, privacy/   SFT / DPO / GRPO / RM; DP-SGD (Opacus) + MIA; FedAvg
```

The control loop works like this. Telemetry from the ranks and the network feeds an estimator. The joint
planner scores every (placement, storage site, routing) option with the calibrated cost model. When the
predicted gain beats the cost of migrating, it acts in one of two ways. It can rewrite BGP policy, which
takes effect in place. Or it can stop the job and relaunch it at new sites, restoring from the newest
*complete* encrypted checkpoint.

## Results

All numbers are mean ± 95% CI over seeds or trials. The full tables are in the paper (`make paper`) and in
`results/INDEX.md`.

### Primary question: joint vs independent control (M3)

The main campaign is still running from a frozen worktree: 8 failure scenarios × 5 methods × 5 seeds, plus
ablations. Its seed-paired, Holm-corrected comparison will go here when it finishes. The supporting
measurements are already in:

- **The cost model's ring term holds up.** The predicted slowdown from interleaving sites in the ring is
  1.92×. A microbenchmark measured 2.04× (1.34 s contiguous vs 2.74 s interleaved).
- **Calibration.** Each WAN plane gets its own efficiency factor: κ_A = 0.95 and κ_B = 1.35. A single global
  κ mispredicted which plane was faster.

### Baselines: parallelism over the WAN (M1)

| network | mode | step (s) | time to target (s) | collectives (% of step) |
|---|---|---|---|---|
| loopback | DDP | 0.84 ± 0.02 | 34 ± 1 | 4 |
| WAN | DDP | 2.04 ± 0.01 | 83 | 69 |
| WAN | FSDP2 | 4.74 ± 0.03 | 192 ± 1 | 87 |
| WAN | HSDP | 2.00 ± 0.02 | 81 ± 1 | 67 |

The WAN makes a DDP step 2.4× slower than loopback, and collectives take 69% of the step, so the testbed
is communication-bound, as the study needs. FSDP2's per-layer collectives all cross sites. HSDP (shard
within a site, replicate across sites) matches DDP.

### Network laboratory: what BGP sees and what it misses (M2)

| failure | data-plane outage | notes |
|---|---|---|
| link down | 0.13 s | carrier loss is detected locally; FIB failover in 0.14 s |
| flap | 0.14 s per event | behaves like repeated link downs |
| BGP session reset | 0.05 s | session re-established in 2.3 ± 0.5 s |
| blackhole | 2.61 ± 0.17 s | carrier stays up, so detection waits for the BGP hold timer |
| bandwidth collapse | none | throughput 189 → 19.1 Mbit/s; **BGP does not react** |
| latency spike | none | RTT 10.9 → 177 ms; **BGP does not react** |

Routing restores reachability. It is blind to *performance* failures, and those are what a controller
that knows the job can act on.

### Checkpoint engine (M4)

- **Crash consistency.** All 1000 `kill -9` trials recovered to at least the last acknowledged checkpoint.
  Of the 500 torn writes and 500 bit flips, every one either recovered cleanly or was detected:
  **0 undetected**.
- **Training stall.** The async engine stalls training less than `torch.save` or DCP does. With frozen
  layers it stalls 0.013 s, against 0.047 s for `torch.save` and 0.114 s for DCP. The cost shows up
  elsewhere: at this state size, durable latency and restore are *slower*.
- **Deduplication.** None for full fine-tuning (1.00×), 1.29× with half the layers frozen, and 2.65× for
  adapter-style updates.

### Security (M5)

- All 12 attack cases were rejected: expired or foreign certificates, wrong SPIFFE trust domain, renewing
  someone else's identity, TLS below 1.3, tampered chunks and GCM tags, tampered, replayed or rolled-back
  manifests, and wrong or foreign keys.
- **Cost of encryption and signing.** Verify-and-decrypt makes a cold restore 1.29× slower
  (1.07 → 1.38 s), and all of the extra time is in the verify phase.

### Formal verification (M9)

All 17 TLC models behaved as expected: 5 hold their invariants and 12 produce the intended
counterexamples. Two of those counterexamples were real design bugs:

- **P1, the checkpoint protocol.** GC could sweep a chunk that a commit had accepted but not yet fsync'd.
  An 11-state trace showed it.
- **P3, rotation.** Under clock skew, a CA running fast could issue a certificate that a slow verifier
  rejected (a 7-state trace). Backdating by 2S fixes it.

### Extensions (M6–M8)

| area | finding |
|---|---|
| media (M6) | Under inference overload, joint media/inference control answers in 0.92 s vs 2.93 s for independent control. That difference is **not significant after Holm** correction. Blocking inference stalls most of the session. |
| kernels (M7) | Triton and Metal varlen prefix-LM attention passed 92/92 correctness cases each. At the model's shape the Metal kernel roughly matches SDPA (0.20 vs 0.23 ms). At a larger shape it is 2.1× *slower* than SDPA. |
| post-training (M8) | At equal FLOPs, only SFT improves exact match over the warm start (0.322 → 0.543). At a larger budget, GRPO against a learned reward model reward-hacks: tool-call validity drops to 0.713. |
| privacy (M8) | With DP-SGD, utility falls to the public-model level at every ε ≤ 1024. Membership-inference AUC falls steadily with ε, from 0.74 without DP to 0.48 at ε = 1. |
| federated (M8) | FedAvg on non-IID sites reaches 29% of centralized exact match, and moves more bytes than pooling the raw data would. |

## Quick start

Requirements: macOS or Linux, Docker (privileged containers), [uv](https://docs.astral.sh/uv/) and a Rust
toolchain. No GPU is needed.

```bash
make setup             # uv sync (Python 3.12)
make lab-image         # privileged Linux lab container: FRR, tc, Rust, CPU torch
make test              # unit tests (Python + Rust)
make ci                # lint + tests + TLA+ model checking + lab integration tests
make reproduce-quick   # under an hour: a subset of every experiment → results/quick/, plus figures
make demo              # 3-site training → failure → re-plan → encrypted restore → live WebRTC assistant
make paper             # figures + claims → paper/main.pdf
make audit             # fail if any number in the docs is not backed by results/
```

`make reproduce-full` re-runs the full campaigns behind the paper, which takes about a day on a laptop.
Lab campaigns run from a **frozen worktree at one commit** (`bench/lab_campaigns.sh`), so a code edit can't
leak into a running experiment.

## Repository layout

| dir | what |
|---|---|
| `train/` | tiny multimodal transformer; DDP/FSDP2/HSDP trainer over Gloo; telemetry; sliced async checkpoints |
| `netlab/` | netns + FRR three-site WAN, `tc` shaping, BGP-policy TE, failure injection, blob store |
| `controller/` | analytical cost model, joint planner, baselines (including the B4-style TE MILP), estimator |
| `ckpt/` | Rust checkpoint engine and Python client (`EngineBackend` is a drop-in replacement for the blob store) |
| `security/` | test CA, SPIFFE IDs, mTLS gRPC, rotation, envelope encryption |
| `formal/` | TLA+ specs, model configs, TLC runner |
| `media/`, `kernels/`, `posttrain/`, `privacy/` | research extensions (M6–M8) |
| `data/synth/` | synthetic financial statements, transcripts, chart videos and QA tasks (no real PII) |
| `bench/` | experiment drivers, statistics, figures, claims registry, number audit |
| `paper/` | LaTeX paper; numbers come in as generated `\C…` macros |
| `results/` | every result file, with provenance; `results/INDEX.md` maps command → file → finding |

## How results are kept honest

- **Provenance on every file.** Each file in `results/` records the git commit, hardware, seed, config and
  timestamp.
- **Claims registry.** Each number the docs use is a named claim computed from a result file. The paper
  pulls them in as LaTeX macros, so it cannot drift from the data.
- **Statistics.** Each condition gets at least 5 seeds, reported as mean ± 95% CI. Comparisons are
  seed-paired, use Holm correction across each family, and add bootstrap CIs and Wilcoxon tests for pooled
  effects.
- **Interleaving.** Methods run in seeded random order within each block, and host load is recorded,
  because the host is shared.
- **Decision and blocker logs.** `DECISIONS.md` and `BLOCKERS.md` record every design choice and every
  problem. That includes the bugs the campaigns exposed, such as truncated checkpoints, restarts from
  step 0 and migration thrashing, and the results thrown away because of them.

## Limitations

- **Emulated WAN.** The links are veth pairs with `tbf`/`netem` in one VM, scaled down so that a CPU
  testbed is communication-bound. Real WAN failure modes (gray links, cross traffic) are only approximated.
- **CPU, not GPU.** A tiny model runs on a few CPU threads per rank. Absolute times and MFU (relative to a
  measured CPU matmul peak) say nothing about GPU clusters. The claims are relative effects inside this
  regime.
- **Small search space.** The study is small enough for exhaustive search. Production scale would need
  ILP or LNS.
- **Shared host.** Other workloads ran at the same time. Interleaving limits the bias but inflates the
  variance.
- **Prefix-level TE.** Docker Desktop's kernel has no Open vSwitch, so steering is BGP local-preference per
  destination prefix.
- **Other scope limits.** Security is not privacy. A FUSE/VM disk is not a kernel filesystem on NVMe.
  Virtual-time media is not wall-clock media. Toy-scale DP is not foundation-model DP. Each is detailed in
  `BLOCKERS.md`, `results/*/NOTES.md` and the paper's limitations section.

## Further reading

- `paper/`: the full write-up (`make paper` → `paper/main.pdf`)
- `results/INDEX.md`: every experiment, its command, output and finding
- `DECISIONS.md` and `BLOCKERS.md`: design rationale and problems hit
- `docs/protocols.md`: the checkpoint (P1/P2) and rotation (P3) protocols
- `FINAL_REVIEW.md`: strongest and weakest claims, threats to validity, and what a real-scale test would
  need (written once M3 completes)
