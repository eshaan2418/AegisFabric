# Results index

Every experiment: command → output file → one-line finding. Numbers quoted in docs must come from these files.

| Milestone | Command | Output | Finding |
|-----------|---------|--------|---------|
| M0 | `uv run python -m bench.probe_env` | `results/m0/environment.json` | CPU-only rung; lab container has netns/netem/tbf/XFS/io_uring; no OVS kmod |
