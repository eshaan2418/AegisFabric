"""Analytical cost model for a synchronous data-parallel job on the three-site WAN.

Notation (per training step):
  n ranks in ring order with sites σ(0..n-1); gradient bytes S; compute time C.
  Gloo's ring all-reduce runs 2(n-1) synchronous rounds. In each round every rank sends S/n bytes to its
  ring successor, so a round costs the slowest ring edge:
      T_ring = 2(n-1) · max_e [ S/(n·bw_e) + lat_e ] · κ          (κ: online calibration factor, default 1)
  An edge between ranks in the same site is a LAN edge (bw_LAN, ~0 latency). An edge between sites uses the
  path that routing r selects for (σ(i), σ(i+1), compute). bw_e is the fair share along that path:
      share_l = max(ε, cap_l − bg_l) / (#controlled flows on l),   bw_e = min_{l ∈ path(e)} share_l.
  Checkpoint every K steps: each rank sends B_ck/n bytes to the storage site s on path r(σ(i), s, storage).
  While the flush is in flight, ring edges that share a link with a checkpoint flow get a halved share.
      D_ck  = max_i (B_ck/n) / bw_ck_i                                (flush duration)
      m     = min(K, D_ck / T_flush)                                  (steps overlapped by the flush)
      E[T]  = C + [(K − m)·T_ring + m·T_ring_flush] / K + stall_ck/K
  If the flush cannot finish within K steps, the excess is charged as a stall (the next snapshot waits).
Migrating to a different placement costs drain + restart + restore (each rank pulls B_ck from s).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import Counter

from netlab.topology import DEFAULT_WAN, SITES

SITE_NAMES = [s.name for s in SITES]
LAN_BW = 2_000e6 / 8  # bytes/s; veth+bridge inside the container, far above the WAN
EPS_BW = 1e5 / 8


def dlink(plane: str, u: str, v: str) -> tuple[str, str, str]:
    return (plane, u, v)


@dataclasses.dataclass
class LinkState:
    cap_mbit: float
    delay_ms: float
    bg_mbit: float = 0.0
    up: bool = True


@dataclasses.dataclass
class NetState:
    links: dict[tuple[str, str, str], LinkState]  # directed: (plane, u, v)

    @classmethod
    def nominal(cls, wan: dict | None = None) -> NetState:
        wan = wan or DEFAULT_WAN
        links = {}
        for (plane, a, b), spec in wan.items():
            links[(plane, a, b)] = LinkState(spec.rate_mbit, spec.delay_ms)
            links[(plane, b, a)] = LinkState(spec.rate_mbit, spec.delay_ms)
        return cls(links)

    def copy(self) -> NetState:
        return NetState({k: dataclasses.replace(v) for k, v in self.links.items()})


@dataclasses.dataclass
class JobSpec:
    n_ranks: int = 6
    grad_bytes: float = 19.7e6
    compute_s: float = 0.3
    ckpt_bytes: float = 59e6
    ckpt_every: int = 25
    ckpt_stall_s: float = 0.05
    restart_s: float = 8.0
    kappa: float = 1.0


@dataclasses.dataclass(frozen=True)
class Plan:
    placement: tuple[str, ...]  # site of each rank, in rank (= ring) order
    storage: str
    routes: tuple[tuple[tuple[str, str, str], str], ...]  # ((src, dst, cls), plane)

    @property
    def route_map(self) -> dict[tuple[str, str, str], str]:
        return dict(self.routes)


def resolve_path(net: NetState, src: str, dst: str, plane: str) -> list[tuple[str, str, str]]:
    """Links actually used. Mirrors BGP behaviour: preferred plane direct; else other plane direct; else
    two-hop through the third site on whichever plane is up."""
    if src == dst:
        return []
    other = "b" if plane == "a" else "a"
    for p in (plane, other):
        if net.links[(p, src, dst)].up:
            return [(p, src, dst)]
    third = next(s for s in SITE_NAMES if s not in (src, dst))
    for p in (plane, other):
        if net.links[(p, src, third)].up and net.links[(p, third, dst)].up:
            return [(p, src, third), (p, third, dst)]
    return []  # partitioned


@dataclasses.dataclass
class Flow:
    src: str
    dst: str
    cls: str
    nbytes: float
    path: list[tuple[str, str, str]]


def ring_flows(net: NetState, job: JobSpec, plan: Plan) -> list[Flow]:
    n = len(plan.placement)
    rm = plan.route_map
    flows = []
    for i in range(n):
        u, v = plan.placement[i], plan.placement[(i + 1) % n]
        if u == v:
            continue
        plane = rm.get((u, v, "compute"), "a")
        flows.append(Flow(u, v, "compute", job.grad_bytes / n, resolve_path(net, u, v, plane)))
    return flows


def ckpt_flows(net: NetState, job: JobSpec, plan: Plan) -> list[Flow]:
    rm = plan.route_map
    per = job.ckpt_bytes / len(plan.placement)
    out = []
    for site, k in Counter(plan.placement).items():
        if site == plan.storage:
            continue
        plane = rm.get((site, plan.storage, "storage"), "a")
        out.append(Flow(site, plan.storage, "storage", per * k, resolve_path(net, site, plan.storage, plane)))
    return out


def _shares(net: NetState, flows: list[Flow], weights: dict[int, float] | None = None) -> dict[int, float]:
    """Fair-share bandwidth (bytes/s) for each flow index, min over its path."""
    count: Counter = Counter()
    for f in flows:
        for lk in f.path:
            count[lk] += 1
    bw = {}
    for idx, f in enumerate(flows):
        if not f.path:
            bw[idx] = LAN_BW if f.src == f.dst else EPS_BW
            continue
        b = float("inf")
        for lk in f.path:
            ls = net.links[lk]
            avail = max(EPS_BW, (ls.cap_mbit - ls.bg_mbit) * 1e6 / 8)
            b = min(b, avail / count[lk])
        bw[idx] = b
    return bw


def path_latency_s(net: NetState, path: list[tuple[str, str, str]]) -> float:
    return sum(net.links[lk].delay_ms for lk in path) / 1e3


def ring_time(net: NetState, job: JobSpec, plan: Plan, extra: list[Flow] | None = None) -> float:
    rf = ring_flows(net, job, plan)
    n = len(plan.placement)
    if not rf:
        return 2 * (n - 1) * (job.grad_bytes / n / LAN_BW)
    flows = rf + (extra or [])
    bw = _shares(net, flows)
    worst = 0.0
    for idx, f in enumerate(rf):
        if not f.path:
            return float("inf")
        worst = max(worst, f.nbytes / bw[idx] + path_latency_s(net, f.path))
    return 2 * (n - 1) * worst * job.kappa


def ckpt_duration(net: NetState, job: JobSpec, plan: Plan, with_ring: bool = True) -> float:
    cf = ckpt_flows(net, job, plan)
    if not cf:
        return 0.0
    flows = cf + (ring_flows(net, job, plan) if with_ring else [])
    bw = _shares(net, flows)
    return max(
        f.nbytes / bw[i] + path_latency_s(net, f.path) if f.path else float("inf") for i, f in enumerate(cf)
    )


def expected_step(net: NetState, job: JobSpec, plan: Plan) -> dict[str, float]:
    t_ring = ring_time(net, job, plan)
    cf = ckpt_flows(net, job, plan)
    if job.ckpt_every <= 0 or not cf:
        return {"step_s": job.compute_s + t_ring, "ring_s": t_ring, "ring_flush_s": t_ring, "ckpt_s": 0.0}
    t_flush = ring_time(net, job, plan, extra=cf)
    d = ckpt_duration(net, job, plan)
    k = job.ckpt_every
    base = job.compute_s + t_ring
    during = job.compute_s + t_flush
    m = min(k, d / during) if during > 0 else 0
    overrun = max(0.0, d - k * during)  # flush slower than the checkpoint interval -> snapshot waits
    step = ((k - m) * base + m * during + job.ckpt_stall_s + overrun) / k
    return {"step_s": step, "ring_s": t_ring, "ring_flush_s": t_flush, "ckpt_s": d}


def migration_cost(net: NetState, job: JobSpec, old: Plan, new: Plan) -> float:
    """Drain + restart + every rank restoring the full state from storage (concurrent flows)."""
    if old.placement == new.placement and old.storage == new.storage:
        return 0.0
    rm = new.route_map
    flows = []
    for site, k in Counter(new.placement).items():
        if site == new.storage:
            continue
        plane = rm.get((new.storage, site, "storage"), "a")
        flows.append(
            Flow(
                new.storage, site, "storage", job.ckpt_bytes * k, resolve_path(net, new.storage, site, plane)
            )
        )
    restore = 0.0
    if flows:
        bw = _shares(net, flows)
        restore = max(
            f.nbytes / bw[i] + path_latency_s(net, f.path) if f.path else float("inf")
            for i, f in enumerate(flows)
        )
    flush = ckpt_duration(net, job, old)
    return job.restart_s + flush + restore


# ------------------------------------------------------------------ candidate enumeration
def placements(n: int = 6, max_per_site: int = 3) -> list[tuple[str, ...]]:
    """Contiguous-by-site ring placements for every feasible site-count vector and cyclic site order.

    Non-contiguous orders are dominated (more WAN edges for the same sites) and are excluded."""
    out = set()
    for counts in itertools.product(range(max_per_site + 1), repeat=len(SITE_NAMES)):
        if sum(counts) != n:
            continue
        sites = [s for s, c in zip(SITE_NAMES, counts, strict=True) if c > 0]
        for order in itertools.permutations(sites):
            if order[0] != sites[0]:  # fix rotation (ring is cyclic)
                continue
            pl = tuple(s for s in order for _ in range(counts[SITE_NAMES.index(s)]))
            out.add(pl)
    return sorted(out)


def relevant_route_keys(placement: tuple[str, ...], storage: str) -> list[tuple[str, str, str]]:
    n = len(placement)
    keys = set()
    for i in range(n):
        u, v = placement[i], placement[(i + 1) % n]
        if u != v:
            keys.add((u, v, "compute"))
    for s in set(placement):
        if s != storage:
            keys.add((s, storage, "storage"))
            keys.add((storage, s, "storage"))  # restore direction
    return sorted(keys)


def make_plan(placement, storage, route_map: dict) -> Plan:
    return Plan(tuple(placement), storage, tuple(sorted(route_map.items())))
