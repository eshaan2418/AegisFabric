# DECISIONS

Dated architecture decisions. Newest last.

## D-001 (2026-10-24) Tooling
Python 3.12 managed by `uv` (pyproject + `uv.lock`); Rust (stable, cargo) for the checkpoint engine.
One `Makefile` is the entry point for tests, lab, reproduction and the paper.

## D-002 (2026-10-24) Fallback rung: CPU-only, multi-process Gloo in network namespaces
Host is an Apple M5 Pro (18 cores, 48 GB, Metal GPU, no CUDA). MPS cannot do multi-process
collectives, and the network lab needs Linux, so training runs on CPU inside the lab container. The model is a
tiny multimodal transformer (chart-image patches + byte-level text, ~2–20 M params). Each rank runs in its own
network namespace, so every Gloo collective crosses emulated WAN links. FSDP2 (`fully_shard`) and HSDP
(2-D mesh) were both checked on Gloo/CPU (torch 2.14) before committing to this design.
**Effect on claims:** absolute step times and MFU say nothing about GPU clusters. What we claim is the
*relative* effect of joint π/r/s choices when the compute:communication ratio is in a regime where
communication matters. That ratio gets calibrated and reported.

## D-003 (2026-10-24) Network lab: custom netns lab (Mininet-equivalent) instead of containerlab
containerlab needs Docker-in-Docker and its own netns management on Docker Desktop. A single privileged
container that builds namespaces/veths with `ip` and runs one FRR instance per router namespace (FRR `-N`
pathspace) gives the same thing (real kernel forwarding, real FRR control plane, tc qdiscs on real links). It
also keeps training ranks, routers and storage servers in one filesystem, which makes orchestration simpler.
Topology is declared in Python (`netlab/topology.py`), so it is reproducible and testable.

## D-004 (2026-10-24) Package index
`files.pythonhosted.org` is unreachable from this network; packages resolve through a private mirror whose
credentials live only in the user's environment. `uv.lock` records the mirror URL (no credentials). CI and other
users can override with `UV_INDEX_URL=https://pypi.org/simple`. The lab image gets the index URL as a BuildKit
*secret*, so it never lands in image layers. `bench/secret_scan.py` runs in `make lint`.

## D-005 (2026-10-24) SDN data plane = BGP policy + kernel FIB, not Open vSwitch
The Docker Desktop LinuxKit kernel has no `openvswitch` module. Traffic engineering is therefore done in the
control plane: the controller rewrites FRR route-maps (local-preference per destination prefix, per site
gateway, per border router) through `vtysh`, and the kernel FIB forwards. Compute and storage hosts live in
separate prefixes per site, so collective flows and checkpoint flows can be steered independently. The
control plane (controller → FRR) and data plane (kernel forwarding, tc qdiscs) stay separate.

## D-006 (2026-10-24) Calibrated analytical cost model + exhaustive search (not an ILP) for the joint planner
For 6 ranks on 3 sites, the joint space is small: 10 contiguous placements × 2 cyclic orders (where
distinct), 2 storage sites, and ≤2^5 route assignments over the keys that a given (π, s) actually uses.
Exhaustive enumeration is therefore exact and runs in milliseconds, so the joint planner uses it. The
*independent TE* baseline is a B4-style min-max-utilisation MILP (PuLP + HiGHS), because that is what a
network team would run. Larger topologies would need an ILP/LNS for the joint problem too, which is noted
as future work. The cost model's Gloo ring structure (rank-order ring) was verified by microbenchmark:
interleaved placement doubles all-reduce time (`bench/collective_micro.py`).

## D-007 (2026-10-24) Experiment design for M3
- Workload: DDP (fastest mode on WAN in M1 calibration), small preset (4.9 M params), batch 16/rank, 6 ranks,
  160 steps, sliced async checkpoints every 20 steps to a storage site in {tor, mtl}. Canadian data
  residency is a hard constraint for all methods.
- Each failure scenario injects its failure twice on the tor–mtl pair: first on plane B, then on plane A.
  Every evaluated placement uses tor–mtl, so every method faces one failure on its path regardless of the
  plane it prefers.
- Background traffic: seeded UDP CBR flows under a fixed (non-controlled) routing policy.
- Methods are interleaved per (seed, scenario) in a seeded random order, because the host is shared (B-005).
- Recovery = inject → first of 3 consecutive steps ≤ 1.2× the pre-failure median. Unrecovered events are
  censored and reported separately.

## D-008 (2026-10-24) Telemetry signals and capacity knowledge
The controller reads device-reported link rate and oper-status (analogous to ifSpeed/oper-status via
gNMI), interface/flow counters (nftables per-prefix counters at gateways), 5 Hz RTT probes per WAN link,
rank-0 step/collective telemetry (online κ calibration) and checkpoint flush stats. Each can be ablated.
The oracle gets ground truth (configured link specs and the background schedule).

## D-009 (2026-10-24) Checkpoint transport for WAN experiments
Sliced async checkpoints (each rank owns ~1/n of the full state, greedily size-balanced) go over the
routed storage prefix. The transport sits behind a pluggable backend: `netlab/blobstore.py` (plain
TCP+fsync) or the M4 engine client. Migration = graceful stop with a final checkpoint, relaunch at the
new placement, restore from the old storage site.

## D-010 (2026-10-24) Control-plane security design (M5)
Python `cryptography` test CA (ECDSA P-256), SPIFFE URI SANs, grpcio mTLS. Servers use
`dynamic_ssl_server_credentials` (per-handshake reload) and clients use a rotating channel (new channel on
renewal, old one closed after a grace period). Authorization is by a SPIFFE-ID interceptor. Following the
TLC findings (M9): 2S backdating, and a two-phase CA rotation in which the new CA is distributed before it
signs. Renewal is scheduled at the midpoint of the *current cert's* validity, with an immediate renewal
on start. A functional test caught bootstrap certs expiring during worker start-up under T/2-from-start
scheduling. SPIRE was not deployed: bootstrap identities are injected by the harness as a stand-in for
attestation (B-007).

## D-011 (2026-10-24) Bug found in pre-campaign smoke test: checkpoints held ~1/6 of the state
`get_full_state` used `full_state_dict=True, cpu_offload=True`; PyTorch then returns the full state on rank 0
only and empty dicts elsewhere. Each rank sliced a different key set, so slices 1..5 were ~1 KB and restores
after migration failed (`KeyError`), or would have restored incomplete state. Fixed (`cpu_offload=False`;
tensors are on CPU already). Every slice now carries the full key-set digest, and restore rejects any
incomplete, overlapping or mismatched union. Regression tests: `tests/test_checkpointing.py`. No results had
been recorded with the buggy version (the smoke output went to `scratch/`).

## D-012 (2026-10-24) Deterministic base routing policy; campaign budget
- With no controller entries, BGP picked a plane by tie-break, and the lab measured plane B for tor→mtl,
  contradicting the cost model and D-007 ("static = plane A"). Gateways now apply a base local-preference of 150
  to plane-A routes (the operator policy "use the private wave"). Controller entries use 200, so they still
  override it. ECMP mode keeps equal preferences, which multipath needs. Any direction a plan leaves unspecified
  (e.g. reverse/ACK directions) therefore follows plane A, as the model assumes. Lab test:
  `test_default_policy_is_plane_a_everywhere`.
- Campaign budget on the shared host: M3 main = 8 scenarios × 5 methods × 5 seeds. Ablations use 3 seeds
  (4 scenarios × 6 variants) and are reported as such: they estimate each signal's contribution and are not
  the primary claim. Driver: `bench/lab_campaigns.sh`.

## D-013 (2026-10-24) M6 media: virtual-time execution, own RTP/AV1 stack, encoder re-init
- The impaired media campaign runs on a virtual-clock asyncio loop. Real codec, STT (whisper-tiny) and
  vision-stub compute is charged at its measured CPU time, and the network comes from a seeded in-process
  link model. Host load made wall-clock real-time media runs meaningless (B-005). Latencies are therefore
  model-derived. The only wall-clock media number is the unimpaired aiortc ingest leg.
- aiortc has no AV1 RTP support, so the campaign uses the repo's own RTP/RTCP + AV1-payload stack, and aiortc
  (VP8 ingest + AV1 transcode) covers the WebRTC leg.
- libsvtav1/libaom-av1 in this PyAV build ignore bitrate changes after open, so rate changes re-create the
  encoder (a keyframe), with hysteresis. Details: `results/m6/NOTES.md`.

## D-014 (2026-10-24) Per-plane calibration factor κ
In the pre-campaign smoke run, measured/predicted ring time was about 1.4 on plane A but about 0.9 on plane
B. The fluid model misses shaping and TCP effects that depend on each plane's rate-delay product. A single
global κ, calibrated online on whichever plane the job currently uses, therefore mis-predicts the other
plane, and that is exactly the comparison the joint planner makes. κ is now per plane: both are calibrated
(`bench/m3_calibrate.py` pins all routes to A, then to B, alternating trials). Online refinement updates the
κ of the bottleneck edge's plane using flush-free steps only: the trainer marks steps that overlap a
checkpoint upload (`ckpt_busy`), and those are excluded.

## D-015 (2026-10-24) M4 checkpoint engine design
Rust `aegis-ckpt`: log-structured append-only segments (`MAGIC|len|id|blake3|payload|crc32c`). A RocksDB index
`chunk_id → (segment, offset, len, epoch)` is written only after the segment fsync. The manifest WAL carries
BLAKE3 record checksums and uses group commit. GC is mark/copy/sweep with the TLC-found epoch refresh and a
sweep-time re-check (P1 as fixed in 1e7b428). I/O backends: `io_uring` (Linux; batched WRITE SQEs followed by
FDATASYNC) and a `pwrite` thread pool (portable fallback, `--io`).

## D-016 (2026-10-24) M4 chunking and client-side encryption
Fixed-size chunks per tensor, no content-defined chunking: tensor byte layouts never shift between steps, so
CDC cannot find more duplicates; `bench/m4_dedup.py` confirms chunk-size insensitivity. Chunk ids are
computed client-side (keyed BLAKE3 under P2), so only missing chunks cross the WAN and the server never sees
plaintext. `EngineBackend` adapts the engine to `SliceCheckpointer`'s put/get (one committed manifest per
blob).

## D-017 (2026-10-24) P2 implementation (`ckpt/crypto.py`, re-exported as `security/envelope.py`)
One AES-256-GCM DEK per (job, key epoch), wrapped by the KEK with AD = job‖epoch. Each chunk gets a fresh
96-bit nonce with AD = chunk_id‖epoch. The Ed25519-signed manifest binds the chunk list (a deduplicated
chunk belongs to many manifests) and carries every wrapped DEK its chunks need. Anti-rollback is a
client-held high-water mark of the manifest version.

## D-018 (2026-10-24) M8 post-training comparison rule
Methods are compared at equal analytic post-warm-start FLOPs (2N per forward token, 6N per fwd+bwd token),
all starting from the same per-seed warm start. RM training cost is reported separately. SFT/iterative
DPO/GRPO/RM are our own implementations: TRL was unavailable offline and the model is a tool-interleaved char
model. Verifiable reward = EM + 0.1·tool_ok.

## D-019 (2026-10-24) M8 DP step and MIA design
DP-SGD (Opacus; RDP accountant with PRV cross-check, Poisson sampling, C=1, δ=1e-5) is applied only to the
private supervised fine-tuning step, at example level. Membership is measured on 16-digit random canaries
inserted once. Offline LiRA shadows contain same-format reference canaries from a disjoint pool (v1 shadows
without them saturated at AUC 0.5). A clip-only (σ=0) ablation separates clipping from noise.

## D-020 (2026-10-24) M8 federated simulation
In-process FedAvg/FedProx over 3 non-IID sites. Communication is accounted as fp32 weight bytes, not
measured on a wire. Compared with centralized and local-only training at equal total local steps, plus a
rounds × local-steps sweep at fixed local work.
- *Update (2026-10-25).* The campaign calibration (`results/m3/calibration.json`, quieter host) measured
  κ_a ≈ 0.95 and κ_b ≈ 1.35, with tight trials: the planes do differ, but in the opposite direction from the
  smoke run quoted above. Those smoke values were taken under heavy host load (compute time 1.54 s vs 0.69 s
  now), which contaminated the subtraction of compute from step time. Per-plane κ stands; the smoke
  values do not.
