import dataclasses

from controller import model as M
from controller import planner as P


def test_placements_contiguous_and_feasible():
    pls = M.placements(6)
    assert ("tor", "tor", "mtl", "mtl", "us", "us") in pls
    assert all(len(p) == 6 and max(p.count(s) for s in set(p)) <= 3 for p in pls)
    # every placement is contiguous by site
    for p in pls:
        changes = sum(p[i] != p[(i + 1) % 6] for i in range(6))
        assert changes == len(set(p)) or len(set(p)) == 1


def test_resolve_path_fallback_mirrors_bgp():
    net = M.NetState.nominal()
    assert M.resolve_path(net, "tor", "mtl", "a") == [("a", "tor", "mtl")]
    net.links[("a", "tor", "mtl")].up = False
    assert M.resolve_path(net, "tor", "mtl", "a") == [("b", "tor", "mtl")]
    net.links[("b", "tor", "mtl")].up = False
    assert M.resolve_path(net, "tor", "mtl", "a") == [("a", "tor", "us"), ("a", "us", "mtl")]


def test_ring_time_matches_closed_form():
    net = M.NetState.nominal()
    job = M.JobSpec(grad_bytes=16 * 2**20, ckpt_every=0)
    plan = M.make_plan(P.DEFAULT_PLACEMENT, "mtl", {})
    t = M.ring_time(net, job, plan)
    # slowest edge on plane A: mtl->us (11 ms, 200 Mbit/s)
    expect = 10 * (16 * 2**20 / 6 / (200e6 / 8) + 0.011)
    assert abs(t - expect) / expect < 1e-6


def test_joint_prefers_high_capacity_plane_and_avoids_collapsed_site():
    job = M.JobSpec(ckpt_every=0)
    net = M.NetState.nominal()
    d = P.JointPlanner(job).initial(net)
    assert set(d.plan.route_map[k] for k in d.plan.route_map if k[2] == "compute") == {"b"}
    # collapse every link touching US -> joint moves ranks out of US
    for k, ls in net.links.items():
        if "us" in k[1:]:
            ls.cap_mbit = 10
    d2 = P.JointPlanner(job).initial(net)
    assert "us" not in d2.plan.placement


def test_checkpoint_interference_increases_expected_step():
    net = M.NetState.nominal()
    j0 = M.JobSpec(ckpt_every=0)
    j1 = dataclasses.replace(j0, ckpt_every=5, ckpt_bytes=200e6)
    plan = M.make_plan(P.DEFAULT_PLACEMENT, "mtl", {})
    assert M.expected_step(net, j1, plan)["step_s"] > M.expected_step(net, j0, plan)["step_s"]


def test_te_lp_moves_demand_off_congested_plane():
    net = M.NetState.nominal()
    net.links[("b", "tor", "mtl")].bg_mbit = 480
    r = P.te_min_mlu(net, {("tor", "mtl", "compute"): 150.0, ("mtl", "us", "compute"): 150.0})
    assert r[("tor", "mtl", "compute")] == "a"  # plane B tor-mtl is nearly full of background traffic
    # mtl->us does not affect the max utilisation (0.75 on tor-mtl A), so the latency tie-break keeps A
    assert r[("mtl", "us", "compute")] == "a"


def test_joint_replan_hysteresis_blocks_tiny_gains():
    job = M.JobSpec(ckpt_every=0)
    net = M.NetState.nominal()
    jp = P.JointPlanner(job)
    d = jp.initial(net)
    assert jp.replan(net, d.plan) is None


def test_per_plane_kappa_scales_only_that_plane():
    from controller import model as M

    net = M.NetState.nominal()
    pl = ("tor", "tor", "tor", "mtl", "mtl", "mtl")
    keys = M.relevant_route_keys(pl, "tor")
    plan_a = M.make_plan(pl, "tor", {k: "a" for k in keys})
    plan_b = M.make_plan(pl, "tor", {k: "b" for k in keys})
    base = M.JobSpec(ckpt_every=0)
    ka = M.JobSpec(ckpt_every=0, kappa_plane={"a": 2.0})
    t_a, plane = M.ring_bottleneck(net, base, plan_a)
    assert plane == "a"
    assert abs(M.ring_time(net, ka, plan_a) - 2 * t_a) < 1e-9
    assert abs(M.ring_time(net, ka, plan_b) - M.ring_time(net, base, plan_b)) < 1e-9
    assert M.ring_bottleneck(net, base, plan_b)[1] == "b"
