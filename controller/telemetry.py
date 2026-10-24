"""Controller telemetry and network-state estimation.

Signals (each can be disabled for the ablation study):
  util      WAN egress interface byte counters + tc qdisc stats on every border router (SNMP/gNMI-like),
            plus per-(dst prefix) flow counters at each site gateway (NetFlow-like, nftables)
  speed     device-reported operational link rate (the tbf rate, like ifSpeed/oper-speed) and oper-status
  rtt       active UDP RTT probes across every WAN link (border ↔ border), 5 Hz
  coll      training collective/step timings reported by rank 0 (calibrates the cost model, κ)
  ckpt      checkpoint flush statistics reported by the trainer (bytes actually uploaded per save)
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from controller import model as M
from netlab.lab import nsexec, run
from netlab.topology import CLASS_OCTET, DEFAULT_WAN, SITES

ALL_SIGNALS = ("util", "speed", "rtt", "coll", "ckpt")


def install_flow_counters(lab) -> None:
    for s in SITES:
        rules = [
            "table inet aegis {",
            " chain flowcount {",
            "  type filter hook forward priority 0; policy accept;",
        ]
        for d in SITES:
            if d == s:
                continue
            for cls, o in CLASS_OCTET.items():
                rules.append(f'  ip daddr 10.{d.idx}.{o}.0/24 counter comment "{d.name}-{cls}"')
        rules += [" }", "}"]
        subprocess.run(
            nsexec(f"{s.name}-gw", "nft", "-f", "-"), input="\n".join(rules), text=True, check=True
        )


def read_flow_counters(lab, site: str) -> dict[tuple[str, str], int]:
    out = run(
        nsexec(f"{site}-gw", "nft", "-j", "list", "chain", "inet", "aegis", "flowcount"),
        capture=True,
        check=False,
    )
    res = {}
    try:
        for item in json.loads(out)["nftables"]:
            rule = item.get("rule")
            if not rule:
                continue
            comment = rule.get("comment", "")
            for e in rule.get("expr", []):
                if "counter" in e:
                    d, cls = comment.split("-")
                    res[(d, cls)] = e["counter"]["bytes"]
    except (json.JSONDecodeError, KeyError):
        pass
    return res


_TBF_RATE = re.compile(r"tbf 1: .*?rate (\d+(?:\.\d+)?)([KMG]?)bit")


def _rate_mbit(m: re.Match) -> float:
    v = float(m.group(1))
    return v * {"": 1e-6, "K": 1e-3, "M": 1.0, "G": 1e3}[m.group(2)]


class RttProbes:
    """Persistent UDP probers, one per directed WAN link, run inside the border namespaces."""

    def __init__(self, lab, out_dir: Path, interval: float = 0.2):
        self.lab, self.out_dir, self.interval = lab, out_dir, interval
        self.procs: list[subprocess.Popen] = []

    def start(self, duration: float) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        py = "python"
        for r in self.lab.topo.routers():
            if r.name.endswith("-gw"):
                continue
            self.procs.append(
                subprocess.Popen(
                    nsexec(r.name, py, "-m", "netlab.probe", "reflect", "--port", "7100"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        time.sleep(0.3)
        for lk in self.lab.topo.wan_links():
            for src, dst_ip, tag in (
                (lk.a, lk.b_ip, f"{lk.plane}_{lk.a.split('-')[0]}_{lk.b.split('-')[0]}"),
                (lk.b, lk.a_ip, f"{lk.plane}_{lk.b.split('-')[0]}_{lk.a.split('-')[0]}"),
            ):
                out = self.out_dir / f"rtt_{tag}.jsonl"
                self.procs.append(
                    subprocess.Popen(
                        nsexec(
                            src,
                            py,
                            "-m",
                            "netlab.probe",
                            "send",
                            "--dst",
                            dst_ip.split("/")[0],
                            "--port",
                            "7100",
                            "--interval",
                            str(self.interval),
                            "--duration",
                            str(duration),
                            "--out",
                            str(out),
                        ),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )

    def latest(self, plane: str, u: str, v: str, window: int = 5) -> float | None:
        """Median RTT (ms) of the last `window` echoes; None if no echo in the last 2 s (link likely down)."""
        p = self.out_dir / f"rtt_{plane}_{u}_{v}.jsonl"
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4000))
                lines = f.read().decode(errors="ignore").splitlines()[1:]
        except OSError:
            return None
        recs = []
        for line in lines[-window:]:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not recs or time.time() - recs[-1]["t_recv"] > 2.0:
            return None
        rtts = sorted(r["rtt_ms"] for r in recs)
        return rtts[len(rtts) // 2]

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()


class Estimator:
    """Maintains an estimated NetState + measured demand matrix from the enabled signals."""

    def __init__(self, lab, signals=ALL_SIGNALS, rtt: RttProbes | None = None, ewma: float = 0.5):
        self.lab = lab
        self.signals = set(signals)
        self.rtt = rtt
        self.ewma = ewma
        self.net = M.NetState.nominal()
        self.prev_tx: dict = {}
        self.prev_flow: dict = {}
        self.prev_t: float | None = None
        self.demand: dict[tuple[str, str, str], float] = {}
        self.util: dict[tuple[str, str, str], float] = {}
        self.lock = threading.Lock()
        self.history: deque = deque(maxlen=600)

    def _dev(self, plane: str, u: str, v: str) -> tuple[str, str]:
        return f"{u}-{plane}", f"w-{v}"

    def _read_router(self, ns: str) -> dict[str, dict]:
        """One namespace entry per border router: tx_bytes, carrier and tbf rate of every WAN interface."""
        script = (
            'for d in /sys/class/net/w-*; do n=$(basename $d); echo "IF $n $(cat $d/statistics/tx_bytes) '
            '$(cat $d/carrier 2>/dev/null || echo 0)"; tc qdisc show dev $n | head -1; done'
        )
        out = run(nsexec(ns, "sh", "-c", script), capture=True, check=False)
        res: dict[str, dict] = {}
        cur = None
        for line in out.splitlines():
            if line.startswith("IF "):
                _, name, tx, carrier = line.split()
                cur = res[name] = {"tx": int(tx), "up": carrier == "1"}
            elif cur is not None:
                m = _TBF_RATE.search(line)
                if m:
                    cur["rate"] = _rate_mbit(m)
        return res

    def poll(self) -> M.NetState:
        now = time.monotonic()
        dt = (now - self.prev_t) if self.prev_t else None
        net = self.net.copy()
        routers = {f"{s.name}-{p}": self._read_router(f"{s.name}-{p}") for s in SITES for p in ("a", "b")}
        for (plane, u, v), ls in net.links.items():
            ns, dev = self._dev(plane, u, v)
            info = routers.get(ns, {}).get(dev)
            if info is None:
                continue
            if "speed" in self.signals:
                if "rate" in info:
                    ls.cap_mbit = info["rate"]
                ls.up = info["up"]
            if "rtt" in self.signals and self.rtt is not None:
                r = self.rtt.latest(plane, u, v)
                if r is None:
                    ls.up = False  # no echo for 2 s: down or blackholed
                else:
                    ls.delay_ms = r / 2
                    if "speed" not in self.signals:
                        ls.up = True
            if "util" in self.signals:
                tx = info["tx"]
                if dt and (plane, u, v) in self.prev_tx:
                    self.util[(plane, u, v)] = (tx - self.prev_tx[(plane, u, v)]) * 8 / dt / 1e6
                self.prev_tx[(plane, u, v)] = tx
        if "util" in self.signals:
            for s in SITES:
                fc = read_flow_counters(self.lab, s.name)
                for (d, cls), b in fc.items():
                    key = (s.name, d, cls)
                    if dt and key in self.prev_flow:
                        rate = (b - self.prev_flow[key]) * 8 / dt / 1e6
                        old = self.demand.get(key, rate)
                        self.demand[key] = self.ewma * rate + (1 - self.ewma) * old
                    self.prev_flow[key] = b
            # background = uncontrolled ("bg" class) demand mapped onto the links its FIB path uses
            for ls in net.links.values():
                ls.bg_mbit = 0.0
            from netlab.background import BG_POLICY

            for (s, d), plane in BG_POLICY.items():
                rate = self.demand.get((s, d, "bg"), 0.0)
                for lk in M.resolve_path(net, s, d, plane):
                    net.links[lk].bg_mbit += rate
        self.prev_t = now
        with self.lock:
            self.net = net
        self.history.append({"t": time.time(), "util": dict(self.util)})
        return net

    def controlled_demand(self) -> dict[tuple[str, str, str], float]:
        return {k: v for k, v in self.demand.items() if k[2] in ("compute", "storage")}


def ground_truth(lab, bg_active: dict[tuple[str, str], float]) -> M.NetState:
    """Oracle view: configured link specs/state + the background flows currently active."""
    from netlab.background import BG_POLICY

    net = M.NetState.nominal({k: v for k, v in DEFAULT_WAN.items()})
    for lk in lab.topo.wan_links():
        u, v = lk.a.split("-")[0], lk.b.split("-")[0]
        up = (
            run(nsexec(lk.a, "cat", f"/sys/class/net/{lk.a_if}/carrier"), capture=True, check=False).strip()
            == "1"
        )
        for x, y in ((u, v), (v, u)):
            ls = net.links[(lk.plane, x, y)]
            ls.cap_mbit, ls.delay_ms = lk.spec.rate_mbit, lk.spec.delay_ms
            ls.up = up and lk.spec.loss_pct < 100
    for (s, d), rate in bg_active.items():
        for lkey in M.resolve_path(net, s, d, BG_POLICY[(s, d)]):
            net.links[lkey].bg_mbit += rate
    return net
