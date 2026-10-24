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
