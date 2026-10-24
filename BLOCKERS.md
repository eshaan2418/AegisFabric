# BLOCKERS

| # | Date | What failed | Fallback used | Impact on claims |
|---|------|-------------|---------------|------------------|
| B-001 | 2026-10-24 | No CUDA GPU (Apple M5 Pro, Metal only); no NCCL | CPU-only rung: Gloo multi-process in network namespaces (D-002) | No GPU-cluster performance claims; MFU is relative to a measured CPU matmul peak |
| B-002 | 2026-10-24 | `openvswitch` kernel module missing in Docker Desktop VM | BGP route-map TE via FRR + kernel FIB (D-005) | Routing granularity is per destination prefix (compute vs storage prefixes per site), not per 5-tuple flow |
| B-003 | 2026-10-24 | No git remote / hosted CI runner configured (pushing is outward-facing, so not done without the user) | `.github/workflows/ci.yml` is authored; the same steps run locally as `make ci`, which gates each milestone | "CI green" means `make ci` passed locally at the recorded commit |
| B-004 | 2026-10-24 | `files.pythonhosted.org` unreachable | Private mirror via environment; public-PyPI override documented (D-004) | None |
