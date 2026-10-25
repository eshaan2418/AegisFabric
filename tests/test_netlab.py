"""Lab integration tests (privileged Linux container only): `make test-lab`."""

from __future__ import annotations

import re
import time

import pytest

pytestmark = pytest.mark.netlab


@pytest.fixture(scope="module")
def lab():
    from netlab.lab import default_lab

    lb = default_lab(compute_per_site=1)
    lb.build()
    lb.start_frr()
    lb.wait_converged()
    yield lb
    lb.teardown(quiet=True)


def rtt_ms(src: str, dst_ip: str, n: int = 5) -> float:
    from netlab.lab import nsexec, run

    out = run(nsexec(src, "ping", "-c", str(n), "-i", "0.2", "-q", dst_ip), capture=True)
    return float(re.search(r"= [\d.]+/([\d.]+)/", out).group(1))


def test_converged_all_sessions(lab):
    for s in ("tor", "mtl", "us"):
        up, total = lab.established(f"{s}-gw")
        assert total > 0 and up == total


def wait_plane(lab, src, dst, cls, plane, timeout=15.0):
    from netlab import te

    deadline = time.time() + timeout
    while te.fib_plane(lab, src, dst, cls) != plane and time.time() < deadline:
        time.sleep(0.2)
    return te.fib_plane(lab, src, dst, cls)


def test_default_policy_is_plane_a_everywhere(lab):
    """No controller entries: every gateway routes every remote compute/storage prefix over plane A."""
    import itertools

    for src, dst in itertools.permutations(("tor", "mtl", "us"), 2):
        for cls in ("compute", "storage"):
            assert wait_plane(lab, src, dst, cls, "a") == "a", (src, dst, cls)


def test_cross_site_rtt_follows_plane_a(lab):
    from netlab import te
    from netlab.topology import DEFAULT_WAN

    te.apply(lab, te.default_decision("a"))
    assert wait_plane(lab, "tor", "mtl", "compute", "a") == "a"
    assert wait_plane(lab, "mtl", "tor", "compute", "a") == "a"
    rtt = rtt_ms("tor-c1", lab.topo.host_ip("mtl-c1"))
    expect = 2 * DEFAULT_WAN[("a", "tor", "mtl")].delay_ms
    assert expect * 0.8 <= rtt <= expect + 3.0, rtt


def test_te_moves_fib_to_plane_b_and_back(lab):
    from netlab import te
    from netlab.topology import DEFAULT_WAN

    base = te.default_decision("a")
    te.apply(lab, base)
    d = dict(base)
    d[("tor", "mtl", "compute")] = "b"
    d[("mtl", "tor", "compute")] = "b"
    te.apply(lab, d, base)
    assert wait_plane(lab, "tor", "mtl", "compute", "b") == "b"
    # storage class is steered independently and stays on plane A
    assert te.fib_plane(lab, "tor", "mtl", "storage") == "a"
    rtt = rtt_ms("tor-c1", lab.topo.host_ip("mtl-c1"))
    assert rtt >= 2 * DEFAULT_WAN[("b", "tor", "mtl")].delay_ms * 0.8, rtt
    te.apply(lab, base, d)
    assert wait_plane(lab, "tor", "mtl", "compute", "a") == "a"


def test_link_down_fails_over(lab):
    from netlab import te

    te.apply(lab, te.default_decision("a"))
    assert wait_plane(lab, "tor", "mtl", "compute", "a") == "a"
    lab.wan_link_state("a", "tor", "mtl", up=False)
    try:
        deadline = time.time() + 30
        while te.fib_plane(lab, "tor", "mtl", "compute") == "a" and time.time() < deadline:
            time.sleep(0.2)
        assert te.fib_plane(lab, "tor", "mtl", "compute") != "a"
        rtt_ms("tor-c1", lab.topo.host_ip("mtl-c1"), n=3)  # still reachable
    finally:
        lab.wan_link_state("a", "tor", "mtl", up=True)
        lab.wait_converged()
