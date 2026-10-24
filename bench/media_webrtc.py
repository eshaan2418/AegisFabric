"""WebRTC (aiortc, VP8) ingest + AV1 transcode: glass-to-server latency and transcode cost.

uv run python -m bench.media_webrtc --trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import os

os.environ.setdefault("SVT_LOG", "1")

from bench.meta import write_result  # noqa: E402
from bench.stats import summarize  # noqa: E402
from media.source import build_source  # noqa: E402
from media.webrtc_ingest import run_ingest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--duration", type=float, default=10.0)
    a = ap.parse_args()
    runs = []
    for seed in range(a.trials):
        src = build_source(seed, duration_s=a.duration, use_say=False)
        r = asyncio.run(run_ingest(src.frames))
        r["seed"] = seed
        runs.append(r)
        print(r)
    keys = ["glass_to_server_ms_mean", "glass_to_server_ms_p95", "av1_transcode_ms_mean", "av1_kbps"]
    summ = {k: summarize([r[k] for r in runs if r[k] is not None]).as_dict() for k in keys}
    summ["delivery_frac"] = summarize([r["frames_received"] / r["frames_sent"] for r in runs]).as_dict()
    note = "aiortc loopback on 127.0.0.1, no impairment; dev run, machine shared"
    write_result("m6/webrtc_ingest.json", {"runs": runs, "summary": summ, "note": note}, config=vars(a))


if __name__ == "__main__":
    main()
