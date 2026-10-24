"""Planners: the joint (π, r, s) optimiser and the baselines it is compared against.

StaticPlanner       default placement/storage; routes = whatever default BGP picks (plane A); never re-plans
EcmpPlanner         same, but the lab runs iBGP multipath at the gateways (L4-hash ECMP across planes)
IndependentPlanner  placement chosen once assuming the nominal static network; storage by a fixed policy
                    (nearest allowed site to rank 0, excluding rank 0's own site); routing by a B4-style
                    traffic-engineering LP that minimises max link utilisation for the *measured* demand
                    matrix, with no knowledge of the job's structure or checkpoint schedule
JointPlanner        exhaustive search over placements × storage sites × route assignments, scored by the
                    job-aware cost model (controller/model.py). Re-places only when the predicted saving
                    over the horizon exceeds migration cost × hysteresis
OraclePlanner       JointPlanner fed the ground-truth network state (upper bound on what the model allows)
"""

from __future__ import annotations

import dataclasses
import itertools

import pulp

from controller import model as M

ALLOWED_STORAGE = ("tor", "mtl")  # data-residency policy: checkpoints of models trained on confidential
# data stay in Canada
DEFAULT_PLACEMENT = ("tor", "tor", "mtl", "mtl", "us", "us")


def default_storage(placement: tuple[str, ...]) -> str:
    """Geo-redundancy policy: store checkpoints off rank 0's site, in an allowed region."""
    for s in ALLOWED_STORAGE:
        if s != placement[0]:
            return s
    return ALLOWED_STORAGE[0]


@dataclasses.dataclass
class Decision:
    plan: M.Plan
    predicted: dict
    reason: str = ""
    migrate: bool = False


class StaticPlanner:
    name = "static"
    replans = False

    def __init__(self, job: M.JobSpec):
        self.job = job

    def initial(self, net: M.NetState) -> Decision:
        pl = DEFAULT_PLACEMENT
        st = default_storage(pl)
        plan = M.make_plan(pl, st, {})  # empty route map = default BGP (plane A)
        return Decision(plan, M.expected_step(net, self.job, plan), "default")

    def replan(self, net, current: M.Plan, demand=None) -> Decision | None:
        return None


class EcmpPlanner(StaticPlanner):
    name = "ecmp"


class IndependentPlanner(StaticPlanner):
    """Placement & storage chosen once (static assumptions); TE re-optimises routes on measured demand."""

    name = "independent"
    replans = True

    def initial(self, net: M.NetState) -> Decision:
        nominal = M.NetState.nominal()
        best = None
        for pl in M.placements(self.job.n_ranks):
            # compute-team placement: minimise nominal step time with default routing, ignoring checkpoints
            plan = M.make_plan(pl, default_storage(pl), {})
            job0 = dataclasses.replace(self.job, ckpt_every=0)
            t = M.expected_step(nominal, job0, plan)["step_s"]
            if best is None or t < best[0] - 1e-9:
                best = (t, pl)
        pl = best[1]
        plan = M.make_plan(pl, default_storage(pl), {})
        return Decision(plan, M.expected_step(net, self.job, plan), "independent-initial")

    def replan(self, net: M.NetState, current: M.Plan, demand: dict | None = None) -> Decision | None:
        if not demand:
            return None
        routes = te_min_mlu(net, demand)
        plan = M.make_plan(current.placement, current.storage, routes)
        if plan.route_map == current.route_map:
            return None
        return Decision(plan, M.expected_step(net, self.job, plan), "te-mlu")


def te_min_mlu(net: M.NetState, demand: dict[tuple[str, str, str], float]) -> dict:
    """B4-style TE: choose one plane per (src, dst, cls) demand to minimise max link utilisation.

    demand: (src, dst, cls) -> measured rate (Mbit/s). Available capacity = cap − background.
    Secondary objective: total demand-weighted latency (tie-break). Solved as a small MILP with HiGHS."""
    keys = [k for k, v in demand.items() if v > 0.01 and k[0] != k[1]]
    if not keys:
        return {}
    prob = pulp.LpProblem("te", pulp.LpMinimize)
    x = {(k, p): pulp.LpVariable(f"x_{k[0]}_{k[1]}_{k[2]}_{p}", cat="Binary") for k in keys for p in "ab"}
    z = pulp.LpVariable("mlu", lowBound=0)
    lat_terms = []
    for k in keys:
        prob += x[(k, "a")] + x[(k, "b")] == 1
    load: dict = {}
    for k in keys:
        for p in "ab":
            path = M.resolve_path(net, k[0], k[1], p)
            for lk in path:
                load.setdefault(lk, []).append(demand[k] * x[(k, p)])
            lat_terms.append(demand[k] * M.path_latency_s(net, path) * x[(k, p)])
    for lk, terms in load.items():
        ls = net.links[lk]
        avail = max(1.0, ls.cap_mbit - ls.bg_mbit)
        prob += pulp.lpSum(terms) <= z * avail
    total = sum(demand[k] for k in keys) or 1.0
    prob += z + 1e-3 * pulp.lpSum(lat_terms) / total
    prob.solve(pulp.HiGHS(msg=False, timeLimit=5))
    return {k: ("a" if (x[(k, "a")].value() or 0) > 0.5 else "b") for k in keys}


class JointPlanner:
    name = "joint"
    replans = True

    def __init__(
        self,
        job: M.JobSpec,
        horizon_s: float = 120.0,
        hysteresis: float = 1.5,
        allow_migration: bool = True,
        min_gain: float = 0.03,
    ):
        self.job = job
        self.horizon_s = horizon_s
        self.hysteresis = hysteresis
        self.allow_migration = allow_migration
        self.min_gain = min_gain

    def best_routes(self, net: M.NetState, placement, storage) -> tuple[dict, dict]:
        keys = M.relevant_route_keys(placement, storage)
        best = None
        for planes in itertools.product("ab", repeat=len(keys)):
            rm = dict(zip(keys, planes, strict=True))
            plan = M.make_plan(placement, storage, rm)
            pred = M.expected_step(net, self.job, plan)
            # tie-break: prefer plane a (fewer changes from default) via tiny penalty on 'b' choices
            score = pred["step_s"] * (1 + 1e-6 * planes.count("b"))
            if best is None or score < best[0]:
                best = (score, rm, pred)
        return best[1], best[2]

    def search(
        self, net: M.NetState, placements=None, storages=ALLOWED_STORAGE
    ) -> list[tuple[float, M.Plan, dict]]:
        out = []
        for pl in placements or M.placements(self.job.n_ranks):
            for st in storages:
                rm, pred = self.best_routes(net, pl, st)
                out.append((pred["step_s"], M.make_plan(pl, st, rm), pred))
        out.sort(key=lambda t: t[0])
        return out

    def initial(self, net: M.NetState) -> Decision:
        t, plan, pred = self.search(net)[0]
        return Decision(plan, pred, "joint-initial")

    def replan(self, net: M.NetState, current: M.Plan, demand=None) -> Decision | None:
        cur_rm, cur_pred = self.best_routes(net, current.placement, current.storage)
        stay = M.make_plan(current.placement, current.storage, cur_rm)
        cand = stay
        reason = "reroute"
        migrate = False
        if self.allow_migration:
            t_best, best, pred_best = self.search(net)[0]
            if best.placement != current.placement or best.storage != current.storage:
                gain_per_s = 1 - t_best / cur_pred["step_s"]  # fraction of wall time saved
                saved = gain_per_s * self.horizon_s
                cost = M.migration_cost(net, self.job, current, best)
                if gain_per_s > self.min_gain and saved > self.hysteresis * cost:
                    cand, cur_pred, reason, migrate = (
                        best,
                        pred_best,
                        f"migrate saved={saved:.1f}s cost={cost:.1f}s",
                        True,
                    )
        if cand == current or (not migrate and cand.route_map == current.route_map):
            return None
        # do not churn routes for negligible predicted gains
        if not migrate:
            old = M.expected_step(net, self.job, current)["step_s"]
            if old - cur_pred["step_s"] < self.min_gain * old:
                return None
        return Decision(cand, cur_pred, reason, migrate)


class OraclePlanner(JointPlanner):
    name = "oracle"
