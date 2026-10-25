# Results index

Every experiment: command → output file → one-line finding. Numbers quoted in docs must come from these files.

| Milestone | Command | Output | Finding |
|-----------|---------|--------|---------|
| M0 | `uv run python -m bench.probe_env` | `results/m0/environment.json` | CPU-only rung; lab container has netns/netem/tbf/XFS/io_uring; no OVS kmod |
| M6 | `nice -n 19 uv run python -m bench.m6_calibrate --seeds 0 1 2` | `results/m6/calibration.json` | AV1 quality vs (bitrate, fps): 30 fps beats 15/10 fps at every rung on this content, so fps adaptation never fires |
| M6 | `nice -n 19 uv run python -m bench.m6_run --trials 5` | `results/m6/trials/*.json`, `results/m6/summary.json`, `results/m6/figs/` | Joint vs independent: lower glass-to-answer under inference overload and bandwidth drop, but not significant after Holm; neutral under loss/jitter; blocking ablation stalls most of the session |
| M6 | `nice -n 19 uv run python -m bench.m6_webrtc --trials 5` | `results/m6/webrtc_ingest.json` | aiortc VP8 ingest on localhost (wall clock, unimpaired) + AV1 transcode cost |
| M6 | (first jitter profile, with reordering) | `results/m6/trials_jitter_reorder/*.json` | Immediate NACKs under reordering cause a retransmission storm; both policies collapse (excluded from the main table, documented) |
