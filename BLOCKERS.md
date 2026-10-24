# BLOCKERS

| # | Date | What failed | Fallback used | Impact on claims |
|---|------|-------------|---------------|------------------|
| B-001 | 2026-10-24 | No CUDA GPU (Apple M5 Pro, Metal only); no NCCL | CPU-only rung: Gloo multi-process in network namespaces (D-002) | No GPU-cluster performance claims; MFU is relative to a measured CPU matmul peak |
| B-002 | 2026-10-24 | `openvswitch` kernel module missing in Docker Desktop VM | BGP route-map TE via FRR + kernel FIB (D-005) | Routing granularity is per destination prefix (compute vs storage prefixes per site), not per 5-tuple flow |
| B-003 | 2026-10-24 | No git remote / hosted CI runner configured (pushing is outward-facing, so not done without the user) | `.github/workflows/ci.yml` is authored; the same steps run locally as `make ci`, which gates each milestone | "CI green" means `make ci` passed locally at the recorded commit |
| B-004 | 2026-10-24 | `files.pythonhosted.org` unreachable | Private mirror via environment; public-PyPI override documented (D-004) | None |
| B-005 | 2026-10-24 | The host is shared with the user's other workloads (host load average reached 90–240 during development) | Methods interleaved in seeded random order per (seed, scenario); container load average recorded per run; ≥5 seeds | Absolute times are noisy; relative comparisons rely on interleaving. Variance is likely inflated relative to a dedicated machine |
| B-006 | 2026-10-24 | Commits are SSH-signed via 1Password and blocked while the signing prompt is pending | Work continues; commits resume once signing is approved (signing left on, since it is the user's policy) | None on results; commit granularity coarser during the outage |
| B-007 | 2026-10-24 | SPIRE not deployed (no attestation infrastructure in the lab) | Harness-issued bootstrap SVIDs (stand-in for node/workload attestation); workloads renew on start and then at half-life | Attestation is not evaluated |
