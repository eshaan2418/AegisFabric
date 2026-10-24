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
