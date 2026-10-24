"""Control-plane traffic engineering: install path preferences as FRR route-map entries.

A routing decision r maps (src_site, dst_site, traffic_class) -> plane ('a' | 'b'). It is realised at the
src site gateway by raising local-preference (200) on routes for the dst site's class prefix learned from
that plane's border router. This is the only lever the controller uses. Forwarding happens in the kernel
FIB that zebra programs, so control plane and data plane stay separate. BGP handles failover on its own:
if the preferred plane's direct link is down, the border withdraws the route and the gateway falls back.
"""

from __future__ import annotations

from netlab.topology import CLASS_OCTET, SITE, SITES

Decision = dict[tuple[str, str, str], str]  # (src, dst, cls) -> plane
CONTROLLED_CLASSES = ("compute", "storage")


def seq_for(dst: str, cls: str) -> int:
    return 10 + 10 * (SITE[dst].idx * len(CLASS_OCTET) + list(CLASS_OCTET).index(cls))


def gw_commands(src: str, decisions: Decision, previous: Decision | None = None) -> list[str]:
    """vtysh config commands for gateway `src` implementing all decisions whose src is `src`."""
    cmds = ["configure terminal"]
    for (s, d, cls), plane in sorted(decisions.items()):
        if s != src:
            continue
        if previous is not None and previous.get((s, d, cls)) == plane:
            continue
        seq = seq_for(d, cls)
        other = "b" if plane == "a" else "a"
        cmds += [
            f"no route-map FROM-{other.upper()} permit {seq}",
            f"route-map FROM-{plane.upper()} permit {seq}",
            f"match ip address prefix-list P-{d}-{cls}",
            "set local-preference 200",
            "exit",
        ]
    cmds += ["end"]
    return cmds if len(cmds) > 2 else []


def clear_commands(src: str) -> list[str]:
    cmds = ["configure terminal"]
    for d in SITES:
        for cls in CLASS_OCTET:
            seq = seq_for(d.name, cls)
            cmds += [f"no route-map FROM-A permit {seq}", f"no route-map FROM-B permit {seq}"]
    return cmds + ["end"]


def apply(lab, decisions: Decision, previous: Decision | None = None) -> list[str]:
    """Push decisions to every gateway. Returns the list of gateways that changed."""
    changed = []
    for s in SITES:
        cmds = gw_commands(s.name, decisions, previous)
        if not cmds:
            continue
        lab.vtysh(f"{s.name}-gw", *cmds, check=False)
        lab.vtysh(f"{s.name}-gw", "clear ip bgp * soft in", check=False)
        changed.append(s.name)
    return changed


def reset(lab) -> None:
    for s in SITES:
        lab.vtysh(f"{s.name}-gw", *clear_commands(s.name), check=False)
        lab.vtysh(f"{s.name}-gw", "clear ip bgp * soft in", check=False)


def default_decision(plane: str = "a") -> Decision:
    return {(s.name, d.name, c): plane for s in SITES for d in SITES if s != d for c in CONTROLLED_CLASSES}


def fib_plane(lab, src: str, dst: str, cls: str) -> str | None:
    """Which plane does the kernel FIB at src-gw currently use for dst's class prefix? ('a','b','ecmp',None)."""
    from netlab.lab import nsexec, run

    pfx = f"10.{SITE[dst].idx}.{CLASS_OCTET[cls]}.0/24"
    out = run(nsexec(f"{src}-gw", "ip", "route", "show", pfx), capture=True, check=False)
    has_a, has_b = "dev to-a" in out, "dev to-b" in out
    if has_a and has_b:
        return "ecmp"
    return "a" if has_a else "b" if has_b else None
