"""Data-plane probes: UDP echo reflector + prober (sequence numbers, 5 ms interval) for outage measurement.

    ip netns exec mtl-c3 python -m netlab.probe reflect --port 7000
    ip netns exec tor-c3 python -m netlab.probe send --dst 10.2.1.13 --port 7000 --interval 0.005 --out x.jsonl

The prober writes one JSON line per received echo: {seq, t_send, t_recv, rtt_ms}. An outage is the
largest gap between consecutive received echoes, minus the probe interval.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time

FMT = "!Qd"


def reflect(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    while True:
        data, addr = s.recvfrom(64)
        s.sendto(data, addr)


def send(dst: str, port: int, interval: float, duration: float, out: str) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.2)
    stop = time.time() + duration
    f = open(out, "w", buffering=1)

    def rx() -> None:
        while time.time() < stop + 1:
            try:
                data, _ = s.recvfrom(64)
            except TimeoutError:
                continue
            except OSError:
                continue
            seq, ts = struct.unpack(FMT, data[:16])
            now = time.time()
            f.write(json.dumps({"seq": seq, "t_send": ts, "t_recv": now, "rtt_ms": (now - ts) * 1e3}) + "\n")

    th = threading.Thread(target=rx, daemon=True)
    th.start()
    seq = 0
    nxt = time.time()
    while time.time() < stop:
        try:
            s.sendto(struct.pack(FMT, seq, time.time()), (dst, port))
        except OSError:
            pass  # no route during convergence
        seq += 1
        nxt += interval
        d = nxt - time.time()
        if d > 0:
            time.sleep(d)
    th.join()
    f.close()


def outage(records: list[dict], interval: float, after: float | None = None) -> dict:
    """Largest gap between received echoes (by send time), optionally only gaps starting after `after`."""
    recs = sorted(records, key=lambda r: r["t_send"])
    best = (0.0, None)
    for a, b in zip(recs, recs[1:], strict=False):
        if after is not None and b["t_send"] < after:
            continue
        g = b["t_send"] - a["t_send"] - interval
        if g > best[0]:
            best = (g, a["t_send"])
    return {"outage_s": max(0.0, best[0]), "gap_start": best[1], "received": len(recs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reflect", "send"])
    ap.add_argument("--dst")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--interval", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--out", default="probe.jsonl")
    a = ap.parse_args()
    if a.mode == "reflect":
        reflect(a.port)
    else:
        send(a.dst, a.port, a.interval, a.duration, a.out)


if __name__ == "__main__":
    main()
