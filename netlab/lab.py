"""Build / control / tear down the emulated three-site WAN inside a privileged Linux container.

Every node is a network namespace. Routers run FRR (zebra+ospfd+bgpd) with per-namespace pathspace (`-N`).
WAN links get a tbf (rate, drop-tail buffer) root qdisc with a netem child (delay/jitter/loss) on both ends.
Hosts also get an out-of-band management interface (`m0`, 192.168.200.0/24) to the container's root
namespace. Control-plane RPCs use it. Data-plane traffic (collectives, checkpoints) never does.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from netlab import frr
from netlab.topology import LinkSpec, Topology, build_topology, lan_bridge

FRR_BIN = Path("/usr/lib/frr")
DAEMONS = ("zebra", "ospfd", "bgpd")
MGMT_BR = "mgmt0"
MGMT_NET = "192.168.200"


def run(cmd: list[str] | str, check: bool = True, capture: bool = False, timeout: float = 30) -> str:
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout if capture else ""


def nsexec(ns: str, *cmd: str) -> list[str]:
    return ["ip", "netns", "exec", ns, *cmd]


def ip_batch(lines: list[str], ns: str | None = None) -> None:
    if not lines:
        return
    cmd = ["ip"] + (["-n", ns] if ns else []) + ["-batch", "-"]
    r = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ip -batch failed: {r.stderr}\n" + "\n".join(lines))


def qdisc_cmds(dev: str, spec: LinkSpec, op: str = "add") -> list[str]:
    # tbf buffer ~ 1.5 x BDP-ish at 20 ms (bounded drop-tail queue); burst sized for the rate.
    rate = f"{spec.rate_mbit}mbit"
    burst = max(32_000, int(spec.rate_mbit * 1e6 / 8 / 250))  # ~4 ms of line rate
    limit = max(64_000, int(spec.rate_mbit * 1e6 / 8 * 0.020))
    netem = f"delay {spec.delay_ms}ms"
    if spec.jitter_ms:
        netem += f" {spec.jitter_ms}ms distribution normal"
    if spec.loss_pct:
        netem += f" loss {spec.loss_pct}%"
    return [
        f"tc qdisc {op} dev {dev} root handle 1: tbf rate {rate} burst {burst} limit {limit}",
        f"tc qdisc {op} dev {dev} parent 1:1 handle 10: netem {netem} limit 100000",
    ]


@dataclasses.dataclass
class Lab:
    topo: Topology
    ecmp: bool = False
    state_dir: Path = Path("/run/aegis-lab")

    # ---------------------------------------------------------------- build
    def build(self) -> None:
        self.teardown(quiet=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for n in self.topo.nodes.values():
            run(["ip", "netns", "add", n.name])
            ip_batch(["link set lo up"], n.name)
            if n.kind == "router":
                for k, v in [
                    ("net.ipv4.ip_forward", "1"),
                    ("net.ipv4.conf.all.rp_filter", "0"),
                    ("net.ipv4.conf.default.rp_filter", "0"),
                    ("net.ipv4.fib_multipath_hash_policy", "1"),
                ]:
                    run(nsexec(n.name, "sysctl", "-qw", f"{k}={v}"))
                ip_batch([f"addr add {n.loopback}/32 dev lo"], n.name)
            else:
                # WAN transfer tuning applied to every host (all methods equally): no cwnd reset after idle.
                run(nsexec(n.name, "sysctl", "-qw", "net.ipv4.tcp_slow_start_after_idle=0"))
        for gw, brs in self.topo.bridges.items():
            cmds = []
            for br, cidr in brs.items():
                cmds += [f"link add {br} type bridge", f"addr add {cidr} dev {br}", f"link set {br} up"]
            ip_batch(cmds, gw)
        root_cmds = []
        for k, lk in enumerate(self.topo.links):
            va, vb = f"v{k}a", f"v{k}b"
            root_cmds += [
                f"link add {va} type veth peer name {vb}",
                f"link set {va} netns {lk.a}",
                f"link set {vb} netns {lk.b}",
            ]
        ip_batch(root_cmds)
        for k, lk in enumerate(self.topo.links):
            va, vb = f"v{k}a", f"v{k}b"
            ip_batch(
                [f"link set {va} name {lk.a_if}", f"link set {lk.a_if} up"]
                + ([f"addr add {lk.a_ip} dev {lk.a_if}"] if lk.a_ip else []),
                lk.a,
            )
            if lk.kind == "lan":
                ip_batch(
                    [
                        f"link set {vb} name {lk.b_if}",
                        f"link set {lk.b_if} master {lan_bridge(lk.a)}",
                        f"link set {lk.b_if} up",
                    ],
                    lk.b,
                )
                gwip = self.topo.nodes[lk.a].gateway
                ip_batch([f"route add default via {gwip}"], lk.a)
            else:
                ip_batch(
                    [f"link set {vb} name {lk.b_if}", f"link set {lk.b_if} up"]
                    + ([f"addr add {lk.b_ip} dev {lk.b_if}"] if lk.b_ip else []),
                    lk.b,
                )
            if lk.kind == "wan":
                for ns, dev in ((lk.a, lk.a_if), (lk.b, lk.b_if)):
                    for c in qdisc_cmds(dev, lk.spec):
                        run(nsexec(ns, *c.split()))
        self._build_mgmt()
        (self.state_dir / "topology.json").write_text(
            json.dumps(
                {"ecmp": self.ecmp, "links": [dataclasses.asdict(lk) for lk in self.topo.links]}, default=str
            )
        )
        self.build_s = time.time() - t0

    def _build_mgmt(self) -> None:
        ip_batch(
            [
                f"link add {MGMT_BR} type bridge",
                f"addr add {MGMT_NET}.1/24 dev {MGMT_BR}",
                f"link set {MGMT_BR} up",
            ]
        )
        for j, h in enumerate(self.topo.hosts()):
            hm, rm = f"m{j}h", f"m{j}r"
            ip_batch(
                [
                    f"link add {hm} type veth peer name {rm}",
                    f"link set {hm} netns {h.name}",
                    f"link set {rm} master {MGMT_BR}",
                    f"link set {rm} up",
                ]
            )
            ip_batch(
                [f"link set {hm} name m0", f"addr add {MGMT_NET}.{10 + j}/24 dev m0", "link set m0 up"],
                h.name,
            )
            h.ifaces["m0"] = f"{MGMT_NET}.{10 + j}/24"

    def mgmt_ip(self, host: str) -> str:
        return self.topo.nodes[host].ifaces["m0"].split("/")[0]

    # ---------------------------------------------------------------- FRR
    def start_frr(self) -> None:
        for r in self.topo.routers():
            etc, rundir = Path("/etc/frr") / r.name, Path("/var/run/frr") / r.name
            etc.mkdir(parents=True, exist_ok=True)
            rundir.mkdir(parents=True, exist_ok=True)
            (etc / "frr.conf").write_text(frr.router_config(self.topo, r.name, self.ecmp))
            (etc / "vtysh.conf").write_text("service integrated-vtysh-config\n")
            (etc / "daemons").write_text("zebra=yes\nospfd=yes\nbgpd=yes\n")
            shutil.chown(etc, "frr", "frr")
            shutil.chown(rundir, "frr", "frr")
            for f in etc.iterdir():
                shutil.chown(f, "frr", "frr")
            logdir = Path("/var/log/frr")
            logdir.mkdir(exist_ok=True)
            shutil.chown(logdir, "frr", "frr")
            for d in DAEMONS:
                run(
                    nsexec(
                        r.name,
                        str(FRR_BIN / d),
                        "-d",
                        "-N",
                        r.name,
                        "-u",
                        "frr",
                        "-g",
                        "frr",
                        "-f",
                        str(etc / "frr.conf"),
                        "-i",
                        str(rundir / f"{d}.pid"),
                    )
                )
        # zebra must be up before the others parse interface config; retry boot application
        time.sleep(0.5)

    def vtysh(self, router: str, *cmds: str, check: bool = True) -> str:
        args = ["vtysh", "-N", router]
        for c in cmds:
            args += ["-c", c]
        return run(args, check=check, capture=True, timeout=30)

    def bgp_summary(self, router: str) -> dict:
        out = self.vtysh(router, "show bgp ipv4 unicast summary json", check=False)
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {}

    def established(self, router: str) -> tuple[int, int]:
        peers = self.bgp_summary(router).get("peers", {})
        est = sum(1 for p in peers.values() if p.get("state") == "Established")
        return est, len(peers)

    def wait_converged(self, timeout: float = 90.0) -> float:
        """Wait until every BGP session is Established and every gw has routes to all remote prefixes."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok = all(e == n and n > 0 for e, n in (self.established(r.name) for r in self.topo.routers()))
            if ok and all(self._gw_has_all_routes(s) for s in ("tor", "mtl", "us")):
                return time.time() - t0
            time.sleep(0.5)
        raise TimeoutError(
            "lab did not converge: "
            + json.dumps({r.name: self.established(r.name) for r in self.topo.routers()})
        )

    def _gw_has_all_routes(self, site: str) -> bool:
        out = run(nsexec(f"{site}-gw", "ip", "route", "show"), capture=True)
        need = [
            f"10.{i}.{o}.0/24"
            for i in (1, 2, 3)
            for o in (1, 2, 3)
            if f"10.{i}." not in {"tor": "10.1.", "mtl": "10.2.", "us": "10.3."}[site]
        ]
        return all(p in out for p in need)

    # ---------------------------------------------------------------- data-plane control
    def set_wan(self, plane: str, s1: str, s2: str, spec: LinkSpec) -> None:
        lk = self.topo.wan_link(plane, s1, s2)
        lk.spec = spec
        for ns, dev in ((lk.a, lk.a_if), (lk.b, lk.b_if)):
            for c in qdisc_cmds(dev, spec, op="change"):
                run(nsexec(ns, *c.split()))

    def wan_link_state(self, plane: str, s1: str, s2: str, up: bool) -> None:
        lk = self.topo.wan_link(plane, s1, s2)
        state = "up" if up else "down"
        ip_batch([f"link set {lk.a_if} {state}"], lk.a)

    def bgp_reset(self, plane: str, s1: str, s2: str) -> None:
        lk = self.topo.wan_link(plane, s1, s2)
        self.vtysh(lk.a, f"clear ip bgp {lk.b_ip.split('/')[0]}")

    def iface_counters(self, ns: str, dev: str) -> dict[str, int]:
        base = f"/sys/class/net/{dev}/statistics"
        out = run(
            nsexec(ns, "cat", f"{base}/tx_bytes", f"{base}/rx_bytes", f"{base}/tx_dropped"), capture=True
        )
        tx, rx, drop = (int(x) for x in out.split())
        return {"tx_bytes": tx, "rx_bytes": rx, "tx_dropped": drop}

    def exec_host(self, host: str, cmd: list[str], **kw) -> subprocess.Popen:
        return subprocess.Popen(nsexec(host, *cmd), **kw)

    # ---------------------------------------------------------------- teardown
    def teardown(self, quiet: bool = False) -> None:
        run("pkill -9 -f '/usr/lib/frr/' || true", check=False)
        out = run(["ip", "netns", "list"], check=False, capture=True)
        for line in out.splitlines():
            ns = line.split()[0]
            run(f"ip netns pids {ns} | xargs -r kill -9", check=False)
            run(["ip", "netns", "del", ns], check=False)
        run(["ip", "link", "del", MGMT_BR], check=False)
        # stray root-namespace veths left by an interrupted build
        run(
            "ip -o link show | awk -F': ' '{print $2}' | cut -d@ -f1 | grep -E '^(m[0-9]+[hr]|v[0-9]+[ab])$' "
            "| xargs -r -n1 ip link del",
            check=False,
        )
        for d in ("/var/run/frr",):
            if os.path.isdir(d):
                for sub in Path(d).iterdir():
                    if sub.is_dir():
                        shutil.rmtree(sub, ignore_errors=True)


def default_lab(ecmp: bool = False, compute_per_site: int = 3, wan: dict | None = None) -> Lab:
    return Lab(build_topology(compute_per_site, wan), ecmp=ecmp)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "down"])
    ap.add_argument("--ecmp", action="store_true")
    a = ap.parse_args()
    lab = default_lab(a.ecmp)
    if a.cmd == "up":
        lab.build()
        lab.start_frr()
        print(f"built in {lab.build_s:.1f}s; converged in {lab.wait_converged():.1f}s")
    else:
        lab.teardown()
