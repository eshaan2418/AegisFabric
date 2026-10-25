"""FRRouting configuration generation for the three-site lab.

Route-map hooks the controller can rewrite (control plane only; forwarding stays in the kernel FIB):
  on s-gw:   FROM-A / FROM-B   inbound from the plane-A / plane-B border (local-pref per destination prefix)
  on s-a/b:  FROM-<peer>       inbound from each eBGP peer (prefer direct vs via-third-site)
Every route-map ends with a catch-all `permit 65000`. Without controller entries, the gateway prefers
plane A through a base local-preference of 150 on FROM-A (an operator's "use the private wave" policy).
This makes the default deterministic; without it, the choice would fall to BGP tie-breaks, and the lab
measured plane B winning some of those. Controller entries set local-preference 200 on the chosen plane.
In ECMP mode both planes stay at the default (100), so iBGP multipath can install both next hops.
"""

from __future__ import annotations

from netlab.topology import CLASS_OCTET, SITE, SITES, Topology

TIMERS = (1, 3)  # BGP keepalive / hold (s)


def prefix_lists() -> list[str]:
    out = []
    for s in SITES:
        for cls, octet in CLASS_OCTET.items():
            out.append(f"ip prefix-list P-{s.name}-{cls} seq 5 permit 10.{s.idx}.{octet}.0/24")
    return out


def ospf_ifaces(topo: Topology, router: str) -> list[str]:
    out = []
    for ifname, _cidr in sorted(topo.nodes[router].ifaces.items()):
        if ifname.startswith("to-"):
            out += [
                f"interface {ifname}",
                " ip ospf network point-to-point",
                " ip ospf hello-interval 1",
                " ip ospf dead-interval 3",
                "!",
            ]
    return out


def gw_config(topo: Topology, site: str, ecmp: bool = False) -> str:
    s = SITE[site]
    i = s.idx
    ka, hold = TIMERS
    lines = [
        "frr defaults traditional",
        f"hostname {site}-gw",
        "no ipv6 forwarding",
        "service integrated-vtysh-config",
        "!",
        *prefix_lists(),
        "!",
        *ospf_ifaces(topo, f"{site}-gw"),
        "router ospf",
        f" ospf router-id 10.255.{i}.1",
        " timers throttle spf 0 50 500",
        f" network 10.255.{i}.0/24 area 0",
        f" network 10.{i}.250.0/24 area 0",
        "!",
        f"router bgp {s.asn}",
        f" bgp router-id 10.255.{i}.1",
        " no bgp ebgp-requires-policy",
        " no bgp network import-check",
        f" timers bgp {ka} {hold}",
        " neighbor IBGP peer-group",
        f" neighbor IBGP remote-as {s.asn}",
        " neighbor IBGP update-source lo",
        f" neighbor IBGP timers {ka} {hold}",
        " neighbor IBGP timers connect 1",
        f" neighbor 10.255.{i}.2 peer-group IBGP",
        f" neighbor 10.255.{i}.3 peer-group IBGP",
        " address-family ipv4 unicast",
        *[f"  network 10.{i}.{o}.0/24" for o in CLASS_OCTET.values()],
        f"  neighbor 10.255.{i}.2 route-map FROM-A in",
        f"  neighbor 10.255.{i}.3 route-map FROM-B in",
        f"  maximum-paths ibgp {2 if ecmp else 1}",
        " exit-address-family",
        "!",
        "route-map FROM-A permit 65000",
        *([] if ecmp else [" set local-preference 150"]),
        "route-map FROM-B permit 65000",
        "!",
    ]
    return "\n".join(lines) + "\n"


def border_config(topo: Topology, site: str, plane: str) -> str:
    s = SITE[site]
    i = s.idx
    ka, hold = TIMERS
    me = f"{site}-{plane}"
    rid = f"10.255.{i}.{2 if plane == 'a' else 3}"
    other_local = f"10.255.{i}.{3 if plane == 'a' else 2}"
    ebgp = []  # (peer_ip, peer_asn, peer_site)
    for lk in topo.wan_links():
        if lk.a == me:
            ebgp.append((lk.b_ip.split("/")[0], SITE[lk.b.split("-")[0]].asn, lk.b.split("-")[0]))
        elif lk.b == me:
            ebgp.append((lk.a_ip.split("/")[0], SITE[lk.a.split("-")[0]].asn, lk.a.split("-")[0]))
    lines = [
        "frr defaults traditional",
        f"hostname {me}",
        "no ipv6 forwarding",
        "service integrated-vtysh-config",
        "!",
        *prefix_lists(),
        "!",
        *ospf_ifaces(topo, me),
        "router ospf",
        f" ospf router-id {rid}",
        " timers throttle spf 0 50 500",
        f" network 10.255.{i}.0/24 area 0",
        f" network 10.{i}.250.0/24 area 0",
        "!",
        f"router bgp {s.asn}",
        f" bgp router-id {rid}",
        " no bgp ebgp-requires-policy",
        f" timers bgp {ka} {hold}",
        " neighbor IBGP peer-group",
        f" neighbor IBGP remote-as {s.asn}",
        " neighbor IBGP update-source lo",
        f" neighbor IBGP timers {ka} {hold}",
        " neighbor IBGP timers connect 1",
        f" neighbor 10.255.{i}.1 peer-group IBGP",
        f" neighbor {other_local} peer-group IBGP",
    ]
    for ip, asn, _ps in ebgp:
        lines += [
            f" neighbor {ip} remote-as {asn}",
            f" neighbor {ip} timers {ka} {hold}",
            f" neighbor {ip} timers connect 1",
            f" neighbor {ip} advertisement-interval 0",
        ]
    lines += [" address-family ipv4 unicast", "  neighbor IBGP next-hop-self"]
    for ip, _asn, ps in ebgp:
        lines.append(f"  neighbor {ip} route-map FROM-{ps} in")
    lines += [" exit-address-family", "!"]
    for _ip, _asn, ps in ebgp:
        lines.append(f"route-map FROM-{ps} permit 65000")
    lines.append("!")
    return "\n".join(lines) + "\n"


def router_config(topo: Topology, router: str, ecmp: bool = False) -> str:
    site, role = router.split("-")
    if role == "gw":
        return gw_config(topo, site, ecmp)
    return border_config(topo, site, role)
