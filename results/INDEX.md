# Results index

Every experiment: command → output file → one-line finding. Numbers quoted in docs must come from these files.

| Milestone | Command | Output | Finding |
|-----------|---------|--------|---------|
| M0 | `uv run python -m bench.probe_env` | `results/m0/environment.json` | CPU-only rung; lab container has netns/netem/tbf/XFS/io_uring; no OVS kmod |
| M6 | `nice -n 19 uv run python -m bench.m6_calibrate --seeds 0 1 2` | `results/m6/calibration.json` | AV1 quality vs (bitrate, fps): 30 fps beats 15/10 fps at every rung on this content, so fps adaptation never fires |
| M6 | `nice -n 19 uv run python -m bench.m6_run --trials 5` | `results/m6/trials/*.json`, `results/m6/summary.json`, `results/m6/figs/` | Joint vs independent: lower glass-to-answer under inference overload and bandwidth drop, but not significant after Holm; neutral under loss/jitter; blocking ablation stalls most of the session |
| M6 | `nice -n 19 uv run python -m bench.m6_webrtc --trials 5` | `results/m6/webrtc_ingest.json` | aiortc VP8 ingest on localhost (wall clock, unimpaired) + AV1 transcode cost |
| M6 | (first jitter profile, with reordering) | `results/m6/trials_jitter_reorder/*.json` | Immediate NACKs under reordering cause a retransmission storm; both policies collapse (excluded from the main table, documented) |
| M4 | `python -m bench.m4_crash --kill-trials 1000 --torn-trials 500 --flip-trials 500 --seed 0` (lab) | `results/m4/crash_{kill9,torn,bitflip}.json` | 2000 fault trials, 0 undetected corruptions; kill -9 always recovers ≥ last acked checkpoint; every truncation/flip detected |
| M4 | `nice -n 19 python -m bench.m4_bench --trials 5 --ckpts 6 --steps 1 --preset small --out m4` (lab) | `results/m4/bench.json` | Async engine save stalls training less than torch.save/DCP, but durable latency and restore are slower at ~50 MB; device write amplification 0.79 with frozen layers (XFS) |
| M4 | `nice -n 19 uv run python -m bench.m4_dedup --seeds 5 --ckpts 10 --every 5 --preset base` | `results/m4/dedup.json` | Dedup across successive checkpoints: none for full fine-tuning, 1.29× half-frozen, 2.65× adapter-style |
| M5 | `nice -n 19 python -m bench.m5_envelope --trials 10 --preset base` (lab) | `results/m5/envelope_overhead.json` | P2 verify+decrypt makes a cold restore 1.29× slower, all of it in the verify phase |
| M4 | `uv run python -m bench.m4_figs` | `results/m4/figs/` | Figures/tables for the above |
| M8 | `AEGIS_DEVICE=mps bash bench/m8_posttrain_all.sh` | `results/m8/posttrain/summary.json` | At equal FLOPs only SFT improves EM over the warm start; DPO/GRPO/GRPO-RM stay at the warm-start level |
| M8 | `python -m bench.m8_posttrain run --seeds 0 1 2 3 4 --budget 1.6e13 --sub /budget1.6e13` + `summarize` | `results/m8/posttrain/budget1.6e13/summary.json` | At 4× budget SFT reaches high EM; GRPO-RM reward-hacks its learned RM |
| M8 | `python -m bench.m8_privacy shadows/shadows2/run/rescore/summarize --eps inf clip 1024 256 64 16 4 1` | `results/m8/privacy/summary.json` | Utility collapses to the public-model level at every ε ≤ 1024; MIA AUC falls monotonically with ε |
| M8 | `python -m bench.m8_federated run --seeds 0 1 2 3 4` / `comm-sweep` / `summarize` | `results/m8/federated/summary.json` | FedAvg far below centralized on non-IID sites, and moves more bytes than pooling raw data would |
| M8 | `python -m bench.m8_figs`; `python -m bench.m8_report` | `results/m8/figs/`, `results/m8/REPORT.md` | Figures and report |
| M1 | `bash bench/lab_campaigns.sh m1` (= `python -m bench.m1_baselines --seeds 5 --out results/m1`, lab, frozen worktree at e204f85) | `results/m1/baselines.json`, `results/m1/raw/*.json` | DDP/FSDP2/HSDP × loopback/WAN, 5 seeds, all reach target loss. On the WAN, collectives are ~70% of a DDP step; FSDP2 is ~2.3× slower than DDP; HSDP (shard within site) ≈ DDP. DDP is used for M3 |
| M2 | `bash bench/lab_campaigns.sh m2` (= `python -m bench.m2_convergence --trials 10 --out results/m2`, lab, frozen at e204f85) | `results/m2/convergence.json` | Link down/flap: sub-second FIB failover and outage. Blackhole: outage ≈ BGP hold time. BGP reset: short outage, multi-second re-establishment. Bandwidth collapse and latency spike are invisible to BGP |
