"""One WAN training experiment: (scenario, method, seed) → results/<tag>/raw/<scenario>/<method>/seed<k>.json

Runs inside the privileged lab container:
    python -m bench.wan_run --scenario bw_collapse --method joint --seed 0

Pipeline: build the lab → storage services on tor-st/mtl-st → planner's initial plan (placement, storage,
routes) → launch the DDP job (one rank per compute-host namespace) → background traffic and failure
injection, time-aligned to the job's first step → control loop at 1 Hz (estimate → re-plan → apply routes,
or migrate) → collect step telemetry across job segments → summary metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bench.meta import resolve, write_result
from bench.scenarios import make as make_scenario
from controller import model as M
from controller import planner as PL
from controller.telemetry import ALL_SIGNALS, Estimator, RttProbes, ground_truth, install_flow_counters
from netlab import background as BG
from netlab import te
from netlab.failures import Injector
from netlab.lab import default_lab, nsexec
from train.launch import Job, Slot, read_rank_log

CKPT_PORT = 7400
MAX_CRASHES = 5  # relaunches per run before the run is abandoned (reported as timed_out + crashes)


@dataclasses.dataclass
class RunConfig:
    scenario: str
    method: str
    seed: int
    steps: int = 160
    ckpt_every: int = 20
    preset: str = "small"
    batch: int = 16
    threads: int = 2
    signals: tuple[str, ...] = ALL_SIGNALS
    tick_s: float = 1.0
    horizon_s: float = 120.0
    out: str = "results/m3"
    variant: str = ""  # label for ablations
    ckpt_backend: str = "blob"  # blob (netlab.blobstore) | engine (M4 aegis-ckpt)
    ckpt_keys: str = ""  # engine only: job key dir (P2 encryption + signed manifests); "" = plaintext
    keep_store: bool = False  # keep checkpoint stores after the run (M10 demo restores from them)

    @property
    def label(self) -> str:
        return self.method + (f"-{self.variant}" if self.variant else "")


def make_planner(cfg: RunConfig, job: M.JobSpec):
    if cfg.method == "static":
        return PL.StaticPlanner(job)
    if cfg.method == "ecmp":
        return PL.EcmpPlanner(job)
    if cfg.method == "independent":
        return PL.IndependentPlanner(job)
    if cfg.method == "joint":
        return PL.JointPlanner(job, horizon_s=cfg.horizon_s)
    if cfg.method == "joint-nomig":
        return PL.JointPlanner(job, horizon_s=cfg.horizon_s, allow_migration=False)
    if cfg.method == "oracle":
        return PL.OraclePlanner(job, horizon_s=cfg.horizon_s)
    raise ValueError(cfg.method)


class JobManager:
    """Runs the training job as a sequence of segments; a migration ends one segment and starts the next."""

    def __init__(self, lab, cfg: RunConfig, out: Path):
        self.lab, self.cfg, self.out = lab, cfg, out
        self.segment = -1
        self.job: Job | None = None
        self.plan: M.Plan | None = None
        self.ctl = out / "control.json"
        self.events: list[dict] = []
        self.port = 29800

    def slots(self, placement) -> list[Slot]:
        used: Counter = Counter()
        out = []
        for site in placement:
            used[site] += 1
            h = f"{site}-c{used[site]}"
            out.append(Slot(h, self.lab.topo.host_ip(h), "eth0"))
        return out

    def storage_addr(self, site: str) -> str:
        return f"{self.lab.topo.host_ip(f'{site}-st')}:{CKPT_PORT}"

    def launch(self, plan: M.Plan, restore_step: int = -1, restore_site: str | None = None) -> None:
        self.segment += 1
        self.plan = plan
        seg = self.out / f"seg{self.segment}"
        if self.ctl.exists():
            self.ctl.unlink()
        args = [
            "--mode",
            "ddp",
            "--preset",
            self.cfg.preset,
            "--batch",
            str(self.cfg.batch),
            "--steps",
            str(self.cfg.steps),
            "--eval-every",
            "20",
            "--threads",
            str(self.cfg.threads),
            "--seed",
            "0",
            "--control-file",
            str(self.ctl),
            "--pg-timeout",
            "300",
            "--ckpt-addr",
            self.storage_addr(plan.storage),
            "--ckpt-every",
            str(self.cfg.ckpt_every),
            "--job",
            f"{self.cfg.scenario}-{self.cfg.label}-{self.cfg.seed}",
            "--ckpt-backend",
            self.cfg.ckpt_backend,
        ]
        if self.cfg.ckpt_keys:
            args += ["--ckpt-keys", self.cfg.ckpt_keys]
        if restore_step >= 0:
            args += [
                "--restore-step",
                str(restore_step),
                "--restore-addr",
                self.storage_addr(restore_site or plan.storage),
            ]
        self.port += 1
        self.job = Job(self.slots(plan.placement), args, seg, port=self.port)
        self.job.start()
        self.events.append(
            {
                "wall": time.time(),
                "event": "launch",
                "segment": self.segment,
                "placement": plan.placement,
                "storage": plan.storage,
                "restore_step": restore_step,
            }
        )

    def newest_complete_checkpoint(self) -> tuple[int, str | None]:
        """Newest step with every slice present at one storage site (slices land via tmp+rename, so a file
        that exists is complete). Scans both sites: after a migration the newest one may be at the old site."""
        job = f"{self.cfg.scenario}-{self.cfg.label}-{self.cfg.seed}"
        n = len(self.plan.placement)
        best: tuple[int, str | None] = (-1, None)
        for site in PL.ALLOWED_STORAGE:
            names = self._stored_keys(site)
            if names is None:
                continue
            steps: Counter = Counter()
            for name in names:
                m = re.fullmatch(rf"ckpt_{re.escape(job)}_(\d+)_slice\d+of{n}", name)
                if m:
                    steps[int(m.group(1))] += 1
            for st, c in steps.items():
                if c == n and (st, site == self.plan.storage) > (best[0], best[1] == self.plan.storage):
                    best = (st, site)
        return best

    def _stored_keys(self, site: str) -> list[str] | None:
        if self.cfg.ckpt_backend == "engine":
            from ckpt.client import EngineBackend

            try:
                job = f"{self.cfg.scenario}-{self.cfg.label}-{self.cfg.seed}"
                b = EngineBackend(f"{self.lab.mgmt_ip(f'{site}-st')}:{CKPT_PORT}", job=job, timeout=30)
            except OSError:
                return None
            try:
                return sorted(b.keys())
            finally:
                b.close()
        root = self.out / f"store_{site}"
        return [f.name for f in root.iterdir()] if root.is_dir() else None

    def records(self) -> list[dict]:
        recs = []
        for k in range(self.segment + 1):
            for r in read_rank_log(self.out / f"seg{k}", 0):
                r["segment"] = k
                recs.append(r)
        return recs

    def current_step(self) -> int:
        steps = [
            r["step"] for r in read_rank_log(self.out / f"seg{self.segment}", 0) if r.get("event") == "step"
        ]
        return max(steps) if steps else -1

    def alive(self) -> bool:
        return self.job is not None and any(c is None for c in self.job.poll())

    def finished(self) -> bool:
        if self.job is None or self.alive():
            return False
        ends = [r for r in read_rank_log(self.out / f"seg{self.segment}", 0) if r.get("event") == "end"]
        return bool(ends) and ends[-1]["step"] >= self.cfg.steps

    def migrate(self, new: M.Plan, timeout: float = 240.0) -> dict:
        t0 = time.time()
        cur = self.current_step()
        stop_at = max(cur + 2, 1)
        tmp = self.ctl.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stop_at_step": stop_at, "final_ckpt": True}))
        os.replace(tmp, self.ctl)
        codes = self.job.wait(timeout)
        if any(c != 0 for c in codes):
            self.job.kill()
        ends = [r for r in read_rank_log(self.out / f"seg{self.segment}", 0) if r.get("event") == "end"]
        step = ends[-1]["step"] if ends else cur
        old_storage = self.plan.storage
        t_stop = time.time()
        self.launch(new, restore_step=step, restore_site=old_storage)
        ev = {
            "wall": t0,
            "event": "migrate",
            "from_step": step,
            "stop_s": t_stop - t0,
            "old_placement": None,
            "new_placement": new.placement,
            "new_storage": new.storage,
        }
        self.events.append(ev)
        return ev

    def kill(self) -> None:
        if self.job:
            self.job.kill()


def start_storage(lab, out: Path, backend: str = "blob") -> list:
    """One storage service per allowed site on its storage host, listening on CKPT_PORT."""
    procs: list = []
    if backend == "engine":
        from ckpt.server import ServerProcess

        for site in PL.ALLOWED_STORAGE:
            root = out / f"store_{site}"
            shutil.rmtree(root, ignore_errors=True)
            srv = ServerProcess(root, listen=f"0.0.0.0:{CKPT_PORT}", io="uring", netns=f"{site}-st")
            procs.append(srv.start())
        return procs
    for site in PL.ALLOWED_STORAGE:
        root = out / f"store_{site}"
        shutil.rmtree(root, ignore_errors=True)
        procs.append(
            subprocess.Popen(
                nsexec(
                    f"{site}-st",
                    "python",
                    "-m",
                    "netlab.blobstore",
                    "--root",
                    str(root),
                    "--port",
                    str(CKPT_PORT),
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    time.sleep(0.5)
    return procs


def bg_active(flows, t_rel: float) -> dict[tuple[str, str], float]:
    act: dict = {}
    for f in flows:
        if f.start_s <= t_rel < f.start_s + f.duration_s:
            act[(f.src, f.dst)] = act.get((f.src, f.dst), 0.0) + f.rate_mbit
    return act


def summarize_run(
    recs: list[dict], fails_log: list[dict], mgr_events: list[dict], t_job0: float, cfg: RunConfig
) -> dict:
    steps = [r for r in recs if r.get("event") == "step"]
    steps.sort(key=lambda r: r["wall"])
    starts = [r for r in recs if r.get("event") == "start"]
    first_start = min((r["wall"] for r in starts), default=t_job0)
    # wall-clock per step (includes migration downtime; first step measured from job start)
    walls = [r["wall"] for r in steps]
    st = np.array([r["step_s"] for r in steps]) if steps else np.array([])
    done = steps[-1]["wall"] if steps else float("nan")
    total = done - first_start
    eval_pts = [(r["wall"] - first_start, r["eval_loss"], r["step"]) for r in steps if "eval_loss" in r]
    # recovery per injected failure
    pre_med = (
        float(np.median(st[5:25])) if len(st) > 25 else float(np.median(st)) if len(st) else float("nan")
    )
    recov = []
    for ev in fails_log:
        if ev.get("event") != "inject":
            continue
        t_inj = ev["wall"]
        before = [r["step_s"] for r in steps if t_inj - 30 <= r["wall"] < t_inj]
        base = float(np.median(before)) if len(before) >= 3 else pre_med
        after = [r for r in steps if r["wall"] >= t_inj]
        t_rec = None
        worst = 0.0
        for i in range(len(after)):
            worst = max(worst, after[i]["step_s"])
            win = after[i : i + 3]
            if len(win) == 3 and all(w["step_s"] <= 1.2 * base for w in win):
                t_rec = max(0.0, win[0]["wall"] - win[0]["step_s"] - t_inj)
                break
        excess = sum(max(0.0, r["step_s"] - base) for r in after if r["wall"] <= t_inj + 60)
        recov.append(
            {
                "kind": ev["kind"],
                "plane": ev["plane"],
                "pair": [ev["s1"], ev["s2"]],
                "t_inject": t_inj,
                "baseline_step_s": base,
                "recovery_s": t_rec,
                "worst_step_s": worst,
                "excess_s_60": excess,
            }
        )
    migr = [e for e in mgr_events if e.get("event") == "migrate"]
    return {
        "steps_done": steps[-1]["step"] if steps else 0,
        "total_s": total,
        "step_mean_s": float(st.mean()) if len(st) else None,
        "step_p50_s": float(np.percentile(st, 50)) if len(st) else None,
        "step_p99_s": float(np.percentile(st, 99)) if len(st) else None,
        "step_std_s": float(st.std(ddof=1)) if len(st) > 1 else None,
        "step_cv": float(st.std(ddof=1) / st.mean()) if len(st) > 1 else None,
        "eval": eval_pts,
        "recovery": recov,
        "migrations": len(migr),
        "migration_events": migr,
        "step_walls": walls,
        "step_times": st.tolist(),
    }


def run(cfg: RunConfig) -> dict:
    out = resolve(cfg.out) / "raw" / cfg.scenario / cfg.label / f"seed{cfg.seed}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    load0 = os.getloadavg()
    sc = make_scenario(cfg.scenario, cfg.seed, steps=cfg.steps, ckpt_every=cfg.ckpt_every)
    lab = default_lab(ecmp=(cfg.method == "ecmp"))
    lab.build()
    lab.start_frr()
    conv = lab.wait_converged()
    install_flow_counters(lab)
    BG.apply_policy(lab)
    storage_procs = start_storage(lab, out, cfg.ckpt_backend)
    rtt = RttProbes(lab, out / "rtt")
    rtt.start(duration=sc.horizon_s + 400)
    est = Estimator(lab, signals=cfg.signals, rtt=rtt)
    cal_path = resolve(cfg.out) / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path}: run `python -m bench.m3_calibrate --out {cfg.out}` first")
    calib = json.loads(cal_path.read_text())["data"]
    job = M.JobSpec(
        n_ranks=6,
        grad_bytes=calib["grad_bytes"],
        compute_s=calib["compute_s"],
        ckpt_bytes=calib["ckpt_bytes"],
        ckpt_every=cfg.ckpt_every,
        restart_s=calib["restart_s"],
        kappa=calib["kappa"],
        kappa_plane=dict(calib.get("kappa_plane", {})),
    )
    planner = make_planner(cfg, job)
    oracle = cfg.method == "oracle"
    time.sleep(2.0)
    est.poll()
    time.sleep(1.0)
    net0 = ground_truth(lab, {}) if oracle else est.poll()
    dec = planner.initial(net0)
    plan = dec.plan
    te.apply(lab, plan.route_map)
    time.sleep(3.0)  # let route-map changes reach the FIB
    decisions = [
        {
            "wall": time.time(),
            "t_rel": None,
            "reason": dec.reason,
            "placement": plan.placement,
            "storage": plan.storage,
            "routes": {"|".join(k): v for k, v in plan.route_map.items()},
            "predicted": dec.predicted,
        }
    ]
    mgr = JobManager(lab, cfg, out)
    mgr.launch(plan)
    # align scenario clock to the first training step
    t_wait = time.time()
    while mgr.current_step() < 1 and time.time() - t_wait < 180:
        time.sleep(0.2)
    t0 = time.time()
    inj = Injector(lab, out / "failures.jsonl")
    inj.t0 = time.monotonic()
    inj.schedule(sc.failures)
    bg = BG.Background(lab, sc.bg, out / "background.jsonl")
    bg.start()
    bg.t0 = time.monotonic()
    est_hist = []
    kappa_hist = []
    deadline = t0 + 1500
    prev_routes = plan.route_map
    while time.time() < deadline:
        tick0 = time.time()
        if mgr.finished():
            break
        if not mgr.alive() and not mgr.finished():
            # crashed segment (e.g. Gloo timeout): restart from the newest complete checkpoint
            n_crash = sum(1 for e in mgr.events if e["event"] == "crash")
            last, site = mgr.newest_complete_checkpoint()
            mgr.events.append(
                {"wall": time.time(), "event": "crash", "segment": mgr.segment, "restore_step": last}
            )
            if n_crash >= MAX_CRASHES:
                break
            mgr.launch(mgr.plan, restore_step=last if last > 0 else -1, restore_site=site)
        t_rel = time.time() - t0
        if planner.replans:
            net = ground_truth(lab, bg_active(sc.bg, t_rel)) if oracle else est.poll()
            if "coll" in cfg.signals or oracle:
                # flush-free steps only: the comparison is against the flush-free ring prediction
                recs = [
                    r
                    for r in read_rank_log(out / f"seg{mgr.segment}", 0)
                    if r.get("event") == "step" and not r.get("ckpt_busy")
                ][-5:]
                if len(recs) >= 3:
                    meas = float(np.median([r["step_s"] for r in recs])) - job.compute_s
                    pred, plane = M.ring_bottleneck(
                        net, dataclasses.replace(job, kappa=1.0, kappa_plane={}), mgr.plan
                    )
                    if plane and pred > 0 and meas > 0:
                        k = min(3.0, max(0.5, meas / pred))
                        job.kappa_plane[plane] = 0.8 * job.k(plane) + 0.2 * k
                        kappa_hist.append((t_rel, plane, job.kappa_plane[plane]))
            if "ckpt" in cfg.signals or oracle:
                cks = [
                    r for r in read_rank_log(out / f"seg{mgr.segment}", 0) if r.get("event") == "ckpt_done"
                ]
                if cks:
                    job.ckpt_bytes = float(np.median([c["bytes"] for c in cks[-3:]])) * job.n_ranks
            d = planner.replan(net, mgr.plan, est.controlled_demand())
            est_hist.append(
                {"t_rel": t_rel, "links": {"|".join(k): dataclasses.asdict(v) for k, v in net.links.items()}}
            )
            if d is not None:
                rec = {
                    "wall": time.time(),
                    "t_rel": t_rel,
                    "reason": d.reason,
                    "placement": d.plan.placement,
                    "storage": d.plan.storage,
                    "routes": {"|".join(k): v for k, v in d.plan.route_map.items()},
                    "predicted": d.predicted,
                    "migrate": d.migrate,
                }
                decisions.append(rec)
                if d.migrate:
                    te.apply(lab, d.plan.route_map, prev_routes)
                    mgr.migrate(d.plan)
                else:
                    te.apply(lab, d.plan.route_map, prev_routes)
                    mgr.plan = d.plan
                prev_routes = d.plan.route_map
        dt = cfg.tick_s - (time.time() - tick0)
        if dt > 0:
            time.sleep(dt)
    timed_out = not mgr.finished()
    mgr.kill()
    bg.stop()
    rtt.stop()
    for p in storage_procs:
        p.stop() if hasattr(p, "stop") else p.kill()
    inj_log = (
        [json.loads(x) for x in (out / "failures.jsonl").read_text().splitlines()]
        if (out / "failures.jsonl").exists()
        else []
    )
    recs = mgr.records()
    summ = summarize_run(recs, inj_log, mgr.events, t0, cfg)
    summ.update(
        {
            "timed_out": timed_out,
            "n_crashes": sum(1 for e in mgr.events if e["event"] == "crash"),
            "n_migrations": sum(1 for e in mgr.events if e["event"] == "migrate"),
            "bgp_initial_convergence_s": conv,
            "n_decisions": len(decisions),
            "loadavg_start": load0,
            "loadavg_end": os.getloadavg(),
        }
    )
    payload = {
        "summary": summ,
        "decisions": decisions,
        "manager_events": mgr.events,
        "failures": inj_log,
        "kappa": kappa_hist,
        "estimates_tail": est_hist[-5:],
        "scenario": {
            "failures": [dataclasses.asdict(f) for f in sc.failures],
            "bg": [dataclasses.asdict(b) for b in sc.bg],
        },
    }
    write_result(out.with_suffix(".json"), payload, config=dataclasses.asdict(cfg), seed=cfg.seed)
    # keep raw rank-0 logs (small); drop checkpoint stores
    if not cfg.keep_store:
        for site in PL.ALLOWED_STORAGE:
            shutil.rmtree(out / f"store_{site}", ignore_errors=True)
    lab.teardown()
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="steady")
    ap.add_argument("--method", default="static")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--signals", default=",".join(ALL_SIGNALS))
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", default="results/m3")
    a = ap.parse_args()
    cfg = RunConfig(
        a.scenario,
        a.method,
        a.seed,
        steps=a.steps,
        ckpt_every=a.ckpt_every,
        signals=tuple(x for x in a.signals.split(",") if x),
        variant=a.variant,
        out=a.out,
    )
    s = run(cfg)
    print(
        json.dumps({k: v for k, v in s.items() if k not in ("step_walls", "step_times", "eval")}, default=str)
    )


if __name__ == "__main__":
    main()
