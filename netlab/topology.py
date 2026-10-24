"""Declarative three-site WAN topology (TOR, MTL, US).

Per site s (index i, AS 6500i):
  routers  s-gw (site gateway: OSPF+iBGP, originates site prefixes), s-a (plane-A border), s-b (plane-B border)
  hosts    s-c1..s-c{k} compute hosts  in 10.i.1.0/24
           s-st          storage host   in 10.i.2.0/24
           s-bg          background-traffic host in 10.i.3.0/24
Inter-site: plane A = low-latency/low-capacity full mesh between s-a routers ("private wave");
            plane B = higher-latency/higher-capacity full mesh between s-b routers ("transit").
So every site pair has ≥2 disjoint paths (direct-A, direct-B), plus 2-hop paths through the third site.
eBGP between borders, iBGP (loopbacks) + OSPF inside each site.
"""

from __future__ import annotations

import dataclasses
import itertools


@dataclasses.dataclass(frozen=True)
class Site:
    name: str
    idx: int

    @property
    def asn(self) -> int:
        return 65000 + self.idx

    def prefix(self, cls: str) -> str:
        return f"10.{self.idx}.{CLASS_OCTET[cls]}.0/24"


CLASS_OCTET = {"compute": 1, "storage": 2, "bg": 3}
SITES = [Site("tor", 1), Site("mtl", 2), Site("us", 3)]
SITE = {s.name: s for s in SITES}
PLANES = ("a", "b")


@dataclasses.dataclass
class LinkSpec:
    """One-way delay (ms), rate (Mbit/s), jitter (ms), loss (%). Applied on both directions."""

    delay_ms: float
    rate_mbit: float
    jitter_ms: float = 0.0
    loss_pct: float = 0.0


# Nominal WAN characteristics. Delays are loosely inspired by real inter-city RTTs; capacities are scaled
# down so that a CPU testbed sits in a communication-relevant regime (see DECISIONS D-002).
DEFAULT_WAN: dict[tuple[str, str, str], LinkSpec] = {
    ("a", "tor", "mtl"): LinkSpec(4, 200),
    ("a", "tor", "us"): LinkSpec(9, 200),
    ("a", "mtl", "us"): LinkSpec(11, 200),
    ("b", "tor", "mtl"): LinkSpec(10, 500),
    ("b", "tor", "us"): LinkSpec(18, 500),
    ("b", "mtl", "us"): LinkSpec(20, 500),
}


@dataclasses.dataclass
class Node:
    name: str
    kind: str  # router | host
    site: str
    loopback: str | None = None
    ifaces: dict[str, str] = dataclasses.field(default_factory=dict)  # ifname -> cidr
    gateway: str | None = None


@dataclasses.dataclass
class Link:
    a: str
    a_if: str
    b: str
    b_if: str
    a_ip: str | None
    b_ip: str | None
    kind: str  # wan | intra | lan
    plane: str | None = None
    spec: LinkSpec | None = None

    @property
    def name(self) -> str:
        return f"{self.a}:{self.a_if}--{self.b}:{self.b_if}"


@dataclasses.dataclass
class Topology:
    nodes: dict[str, Node]
    links: list[Link]
    bridges: dict[str, dict[str, str]]  # router -> {bridge_name: cidr}
    compute_per_site: int

    def wan_links(self) -> list[Link]:
        return [lk for lk in self.links if lk.kind == "wan"]

    def wan_link(self, plane: str, s1: str, s2: str) -> Link:
        s1, s2 = sorted([s1, s2], key=lambda n: SITE[n].idx)
        for lk in self.wan_links():
            if lk.plane == plane and lk.a == f"{s1}-{plane}" and lk.b == f"{s2}-{plane}":
                return lk
        raise KeyError((plane, s1, s2))

    def hosts(self, site: str | None = None, role: str | None = None) -> list[Node]:
        out = [n for n in self.nodes.values() if n.kind == "host"]
        if site:
            out = [n for n in out if n.site == site]
        if role:
            out = [n for n in out if host_role(n.name) == role]
        return sorted(out, key=lambda n: n.name)

    def host_ip(self, name: str) -> str:
        return self.nodes[name].ifaces["eth0"].split("/")[0]

    def routers(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "router"]


def host_role(name: str) -> str:
    suffix = name.split("-", 1)[1]
    if suffix.startswith("c"):
        return "compute"
    return {"st": "storage", "bg": "bg"}[suffix]


def build_topology(compute_per_site: int = 3, wan: dict | None = None) -> Topology:
    wan = dict(DEFAULT_WAN if wan is None else wan)
    nodes: dict[str, Node] = {}
    links: list[Link] = []
    bridges: dict[str, dict[str, str]] = {}
    for s in SITES:
        i = s.idx
        gw, ra, rb = f"{s.name}-gw", f"{s.name}-a", f"{s.name}-b"
        nodes[gw] = Node(gw, "router", s.name, loopback=f"10.255.{i}.1")
        nodes[ra] = Node(ra, "router", s.name, loopback=f"10.255.{i}.2")
        nodes[rb] = Node(rb, "router", s.name, loopback=f"10.255.{i}.3")
        # intra-site point-to-point links
        for k, (r, base) in enumerate([(ra, 0), (rb, 4)]):
            gw_ip, r_ip = f"10.{i}.250.{base + 1}/30", f"10.{i}.250.{base + 2}/30"
            gif, rif = f"to-{r.split('-')[1]}", "to-gw"
            nodes[gw].ifaces[gif] = gw_ip
            nodes[r].ifaces[rif] = r_ip
            links.append(Link(gw, gif, r, rif, gw_ip, r_ip, "intra"))
            del k
        # LANs: bridges inside the gateway namespace
        bridges[gw] = {}
        for cls, octet in CLASS_OCTET.items():
            br = f"br-{cls[:2]}"
            bridges[gw][br] = f"10.{i}.{octet}.1/24"
        hosts = [(f"{s.name}-c{k + 1}", "compute", 11 + k) for k in range(compute_per_site)]
        hosts += [(f"{s.name}-st", "storage", 11), (f"{s.name}-bg", "bg", 11)]
        for hname, cls, last in hosts:
            octet = CLASS_OCTET[cls]
            ip = f"10.{i}.{octet}.{last}/24"
            nodes[hname] = Node(hname, "host", s.name, ifaces={"eth0": ip}, gateway=f"10.{i}.{octet}.1")
            links.append(Link(hname, "eth0", gw, f"p-{hname.split('-', 1)[1]}", ip, None, "lan"))
    for plane in PLANES:
        for s1, s2 in itertools.combinations(SITES, 2):
            ra, rb = f"{s1.name}-{plane}", f"{s2.name}-{plane}"
            pl = 16 if plane == "a" else 17
            net = f"172.{pl}.{s1.idx}{s2.idx}"
            a_ip, b_ip = f"{net}.1/30", f"{net}.2/30"
            aif, bif = f"w-{s2.name}", f"w-{s1.name}"
            nodes[ra].ifaces[aif] = a_ip
            nodes[rb].ifaces[bif] = b_ip
            spec = wan[(plane, s1.name, s2.name)]
            links.append(Link(ra, aif, rb, bif, a_ip, b_ip, "wan", plane, dataclasses.replace(spec)))
    return Topology(nodes, links, bridges, compute_per_site)


def lan_bridge(host: str) -> str:
    return f"br-{host_role(host)[:2]}"
