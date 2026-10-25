"""M4 benchmarks: aegis-ckpt (io_uring / pwrite) vs torch.save and torch.distributed.checkpoint, on ext4 and XFS.

Run inside the privileged lab container (needs losetup/mkfs/mount and /proc/sys/vm/drop_caches):
    nice -n 19 python -m bench.m4_bench --trials 5 --ckpts 6 --steps 1 --preset small --out m4
Writes results/m4/bench.json (summary, mean ± 95% CI over trials) and results/m4/bench_raw.json.

Both filesystems are loop-mounted image files under /var/tmp in the container, i.e. on the container's
overlay filesystem on the Docker Desktop VM's virtio disk (APFS on the host NVMe underneath), so they
share one backing device. Device-level bytes are read from /sys/block/loopN/stat (sectors written,
including journal and metadata). Workload: the real TinyVLM from train/model.py with AdamW and real
optimizer steps between checkpoints (see `Workload`). Methods run in a seeded random order per trial.

Methods: torch_save (torch.save + fsync), dcp (torch.distributed.checkpoint.save, single process,
fsync'd), aegis_uring / aegis_pwrite (P2-encrypted, async: snapshot then background flush while
training continues), aegis_uring_plain (no encryption), aegis_uring_sync (encrypted, training blocked
until the checkpoint is durable).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from bench import stats
from bench.m4_common import loadavg
from bench.meta import write_result
from ckpt.client import CheckpointClient
from ckpt.crypto import CkptCrypto
from ckpt.server import ServerProcess


# --- workload ------------------------------------------------------------------------------------
class Workload:
    """The real TinyVLM (train/model.py) + AdamW on the synthetic chart/text stream (bench/m4_common).

    Checkpoint = every tensor leaf of `train.trainer.get_full_state` (model + optimizer state, the same
    state SliceCheckpointer saves; non-tensor leaves such as param_group hyper-parameters are a few
    hundred bytes and are left out for all methods alike). Scenarios: "full" (every parameter trains,
    so AdamW touches every byte between checkpoints) and "frozen" (fine-tuning-style: embeddings and the
    lower half of the blocks frozen, see bench.m4_common.Trainer)."""

    def __init__(self, scenario: str, seed: int, preset: str = "base"):
        from bench.m4_common import Trainer

        self.tr = Trainer(preset, seed=seed, freeze="half" if scenario == "frozen" else "none")

    def step(self) -> float:
        return self.tr.one_step()

    def state(self) -> dict[str, torch.Tensor]:
        return self.tr.tensors()


def nbytes(st: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in st.values())


# --- filesystem plumbing -------------------------------------------------------------------------
def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


class LoopFS:
    def __init__(self, kind: str, size_gb: int, base: Path):
        self.kind = kind
        self.img = base / f"{kind}.img"
        self.mnt = base / f"mnt-{kind}"
        self.mnt.mkdir(parents=True, exist_ok=True)
        sh("truncate", "-s", f"{size_gb}G", str(self.img))
        self.dev = sh("losetup", "-f", "--show", str(self.img))
        if kind == "xfs":
            sh("mkfs.xfs", "-q", "-f", self.dev)
        else:
            sh("mkfs.ext4", "-q", "-F", self.dev)
        sh("mount", self.dev, str(self.mnt))
        self.name = Path(self.dev).name

    def sectors_written(self) -> int:
        return int(Path(f"/sys/block/{self.name}/stat").read_text().split()[6])

    def bytes_written(self) -> int:
        return self.sectors_written() * 512

    def close(self) -> None:
        subprocess.run(["umount", str(self.mnt)], check=False)
        subprocess.run(["losetup", "-d", self.dev], check=False)
        self.img.unlink(missing_ok=True)


def drop_caches() -> None:
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
    except OSError:
        pass


def fsync_tree(p: Path) -> None:
    for root, _, files in os.walk(p):
        for f in files:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        fd = os.open(root, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)


# --- methods -------------------------------------------------------------------------------------
def run_torch_save(fs: LoopFS, wl: Workload, n: int, steps: int) -> dict:
    d = fs.mnt / "torchsave"
    d.mkdir()
    per = []
    for i in range(n):
        for _ in range(steps):
            wl.step()
        st = wl.state()
        b0 = fs.bytes_written()
        t0 = time.perf_counter()
        p = d / f"ckpt_{i}.pt"
        with open(p, "wb") as f:
            torch.save(st, f)
            f.flush()
            os.fsync(f.fileno())
        dfd = os.open(d, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
        dt = time.perf_counter() - t0
        per.append(
            {
                "stall_s": dt,
                "durable_s": dt,
                "logical": nbytes(st),
                "stored": p.stat().st_size,
                "device": fs.bytes_written() - b0,
            }
        )
    drop_caches()
    t0 = time.perf_counter()
    torch.load(d / f"ckpt_{n - 1}.pt")
    return {"per_ckpt": per, "recovery_s": time.perf_counter() - t0}


def run_dcp(fs: LoopFS, wl: Workload, n: int, steps: int) -> dict:
    import torch.distributed.checkpoint as dcp

    d = fs.mnt / "dcp"
    d.mkdir()
    per = []
    for i in range(n):
        for _ in range(steps):
            wl.step()
        st = wl.state()
        b0 = fs.bytes_written()
        t0 = time.perf_counter()
        p = d / f"ckpt_{i}"
        dcp.save(st, checkpoint_id=str(p))
        fsync_tree(p)
        dt = time.perf_counter() - t0
        stored = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        per.append(
            {
                "stall_s": dt,
                "durable_s": dt,
                "logical": nbytes(st),
                "stored": stored,
                "device": fs.bytes_written() - b0,
            }
        )
    drop_caches()
    target = {k: torch.empty_like(v) for k, v in wl.state().items()}
    t0 = time.perf_counter()
    dcp.load(target, checkpoint_id=str(d / f"ckpt_{n - 1}"))
    return {"per_ckpt": per, "recovery_s": time.perf_counter() - t0}


def run_engine(
    fs: LoopFS, wl: Workload, n: int, steps: int, io: str, encrypt: bool, overlap_train: bool = True
) -> dict:
    root = fs.mnt / f"aegis-{io}"
    srv = ServerProcess(root, io=io, io_threads=4).start()
    crypto = CkptCrypto.generate("bench") if encrypt else None
    per = []
    base_step = float(np.median([wl.step() for _ in range(3)]))
    with CheckpointClient(srv.addr, crypto, chunk_size=1 << 20, connections=2) as c:
        for i in range(n):
            for _ in range(steps):
                wl.step()
            st = wl.state()
            b0 = fs.bytes_written()
            t0 = time.perf_counter()
            h = c.snapshot(st, step=i)
            # keep training while the flush runs in the background; measure interference
            overlap = []
            while overlap_train and not h.done.is_set():
                overlap.append(wl.step())
            s = h.wait()
            durable = time.perf_counter() - t0
            per.append(
                {
                    # sync save: training is blocked until the checkpoint is durable
                    "stall_s": s.stall_s if overlap_train else durable,
                    "snapshot_s": s.stall_s,
                    "durable_s": durable,
                    "flush_s": s.flush_s,
                    "logical": s.logical_bytes,
                    "stored": s.uploaded_bytes,
                    "device": fs.bytes_written() - b0,
                    "dedup_hits": s.dedup_hits,
                    "chunks": s.chunks,
                    "overlap_steps": len(overlap),
                    "overlap_step_s": float(np.mean(overlap)) if overlap else None,
                    "hash_s": s.hash_s,
                    "encrypt_s": s.encrypt_s,
                    "upload_s": s.upload_s,
                }
            )
        server_stats = c.stats()
        hwm, anchor = c.hwm, c.anchor
    srv.kill9()
    drop_caches()
    t0 = time.perf_counter()
    srv2 = ServerProcess(root, io=io).start()
    t_server = time.perf_counter() - t0
    with CheckpointClient(srv2.addr, crypto) as c:
        _, info = c.restore(trusted_hwm=hwm, trusted_anchor=anchor)
    rec = time.perf_counter() - t0
    srv2.stop()
    return {
        "per_ckpt": per,
        "recovery_s": rec,
        "server_recovery_s": t_server,
        "server_recovery_ms_internal": srv2.recovery.get("recovery_ms"),
        "restore_fetch_s": info.fetch_s,
        "restore_verify_s": info.verify_s,
        "baseline_step_s": base_step,
        "server_stats": server_stats,
    }


def summarize_trial(r: dict) -> dict:
    per = r["per_ckpt"]
    logical = sum(p["logical"] for p in per)
    stored = sum(p["stored"] for p in per)
    device = sum(p["device"] for p in per)
    out = {
        "stall_s": float(np.mean([p["stall_s"] for p in per[1:]])),
        "durable_s": float(np.mean([p["durable_s"] for p in per[1:]])),
        "throughput_MBps": logical / sum(p["durable_s"] for p in per) / 1e6,
        "write_amp_device": device / logical,
        "dedup_ratio": logical / stored if stored else float("nan"),
        "recovery_s": r["recovery_s"],
    }
    if "overlap_step_s" in per[-1]:
        ov = [p["overlap_step_s"] for p in per if p.get("overlap_step_s")]
        if ov:
            out["overlap_slowdown"] = float(np.mean(ov) / r["baseline_step_s"])
        out["write_amp_vs_new_bytes"] = device / stored if stored else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--ckpts", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2, help="optimizer steps between checkpoints")
    ap.add_argument("--scenarios", default="frozen,full")
    ap.add_argument("--fs", default="ext4,xfs")
    ap.add_argument(
        "--methods", default="torch_save,dcp,aegis_uring,aegis_pwrite,aegis_uring_plain,aegis_uring_sync"
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--preset", default="base", help="train/model.py preset")
    ap.add_argument("--out", default="m4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    base = Path(tempfile.mkdtemp(prefix="m4bench-", dir="/var/tmp"))
    methods = a.methods.split(",")
    trials: dict[str, list] = {}
    raw: dict = {}

    def flush_results(final: bool = False) -> None:
        results = {}
        for key, trs in trials.items():
            results[key] = {
                m: stats.summarize([tr[m] for tr in trs if tr.get(m) is not None]).as_dict()
                for m in trs[0]
                if m != "loadavg"
            }
            results[key]["state_bytes"] = raw[key][0]["per_ckpt"][0]["logical"]
        write_result(f"{a.out}/bench.json", {"summary": results, "complete": final}, vars(a), a.seed)
        if final:
            write_result(f"{a.out}/bench_raw.json", raw, vars(a), a.seed)

    try:
        for fsk in a.fs.split(","):
            for scen in a.scenarios.split(","):
                for t in range(a.trials):
                    # methods interleaved in a seeded random order per trial (shared host, B-005)
                    for meth in rng.permutation(methods):
                        key = f"{fsk}/{scen}/{meth}"
                        fs = LoopFS(fsk, 16, base)
                        try:
                            wl = Workload(scen, a.seed + t, preset=a.preset)
                            if meth == "torch_save":
                                r = run_torch_save(fs, wl, a.ckpts, a.steps)
                            elif meth == "dcp":
                                r = run_dcp(fs, wl, a.ckpts, a.steps)
                            else:
                                io = "uring" if "uring" in meth else "pwrite"
                                r = run_engine(
                                    fs,
                                    wl,
                                    a.ckpts,
                                    a.steps,
                                    io,
                                    encrypt="plain" not in meth,
                                    overlap_train="sync" not in meth,
                                )
                        finally:
                            fs.close()
                        r["loadavg"] = loadavg()
                        tr = summarize_trial(r)
                        tr["loadavg"] = r["loadavg"]
                        trials.setdefault(key, []).append(tr)
                        raw.setdefault(key, []).append(r)
                        print(
                            key,
                            t,
                            {k: round(v, 4) if isinstance(v, float) else v for k, v in tr.items()},
                            flush=True,
                        )
                    flush_results()
        flush_results(final=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
