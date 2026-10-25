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
| B-008 | 2026-10-24 | aiortc has no AV1 RTP codec | Own RTP/AV1 stack for the impaired campaign; aiortc VP8 ingest + AV1 transcode leg | WebRTC ingest measured only unimpaired |
| B-009 | 2026-10-24 | Shared-host load breaks real-time media timing | Virtual-time execution with CPU-time-charged compute (D-013) | M6 latencies are model-derived, not wall clock |
| B-010 | 2026-10-24 | No offline vision model of useful size | Stub vision (compute proxy + deterministic chart reader); STT is real whisper-tiny | No claims about vision accuracy |
| B-011 | 2026-10-24 | SVT-AV1/libaom ignore runtime bitrate changes | Encoder re-creation with hysteresis | Extra keyframes on each rate change (both policies) |
| B-012 | 2026-10-24 | Private crate mirror quarantines `cc` 1.10.0 | `Cargo.lock` pins `cc` 1.2.x | None |
| B-013 | 2026-10-24 | `librocksdb-sys` bindgen needs libclang off the default macOS dyld path | Makefile sets `LIBCLANG_PATH`/`DYLD_FALLBACK_LIBRARY_PATH` on Darwin | None |
| B-014 | 2026-10-24 | No raw block device / NVMe; ext4/XFS are loop-mounted images on the Docker Desktop VM disk | Loopback images inside the privileged container; device bytes from `/sys/block/loopN/stat` | Absolute M4 throughput/latency include virtio + host APFS; fsync durability only as good as Docker Desktop's flush semantics; relative comparisons within one filesystem hold |
| B-015 | 2026-10-24 | M4/M8 benchmarks ran concurrently with other experiments on the shared host | `nice -n 19`, CPU caps, seeded interleaving, `small` preset for the M4 matrix | Absolute M4 timings pessimistic, CIs wide; ext4 ran before XFS while load fell, so do not compare across filesystems |
| B-016 | 2026-10-24 | No usable HF model/TRL offline, no GPU | Own 848k-param char transformer and own SFT/DPO/GRPO/RM, unit-tested | No claims about real LLM post-training |
| B-017 | 2026-10-24 | 1Password SSH signing failed for ~40 min during M4/M8 wrap-up | Work staged; committed once signing recovered | None |
