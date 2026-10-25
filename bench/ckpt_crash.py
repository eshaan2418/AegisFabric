"""M4 crash campaign: kill -9, torn writes and bit flips against aegis-ckpt + CheckpointClient.

Run inside the privileged lab container (so `--io uring` is really io_uring):
    python -m bench.ckpt_crash --kill-trials 1000 --torn-trials 500 --flip-trials 500 --seed 0

Each outcome is one of:
  ok         : recovered latest seq satisfies durability (>= last ACKed, <= last attempted) and every retained
               checkpoint restores bit-exactly to what the client saved under that seq.
  detected   : recovery/restore raised a detected error (IntegrityError / RollbackError / missing chunk /
               corrupt WAL / server refused to start). Never silently wrong.
  undetected : restore returned data different from what was saved under that seq, or durability was
               violated without an error. This must be 0.
kill -9 kills the server process only: data the kernel already holds in the page cache survives, so this
campaign does NOT model power loss or a lying disk. The torn-write and bit-flip campaigns cover lost or
corrupted on-disk bytes.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
from pathlib import Path

import blake3
import numpy as np
import torch

from bench.meta import write_result
from ckpt.client import CheckpointClient
from ckpt.crypto import CkptCrypto
from ckpt.errors import CkptError, IntegrityError
from ckpt.server import ServerProcess

CHUNK = 64 << 10


def make_state(rng: np.random.Generator) -> dict[str, torch.Tensor]:
    def t(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return {"emb": t(256, 512), "w1": t(512, 512), "adam_m": t(512, 512), "adam_v": t(512, 512)}


def mutate(st: dict[str, torch.Tensor], rng: np.random.Generator) -> None:
    for name in rng.choice(list(st), size=int(rng.integers(1, len(st) + 1)), replace=False):
        rows = rng.integers(0, st[name].shape[0], size=int(rng.integers(1, 33)))
        st[name][torch.from_numpy(rows)] += float(rng.standard_normal())


def digest(st: dict[str, torch.Tensor]) -> str:
    h = blake3.blake3()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Workload(threading.Thread):
    """Single writer: save (and sometimes GC) until the server dies."""

    def __init__(
        self, addr: str, crypto: CkptCrypto, rng: np.random.Generator, st: dict, record: dict, hwm: int
    ):
        super().__init__(daemon=True)
        self.addr, self.crypto, self.rng, self.st, self.record = addr, crypto, rng, st, record
        self.hwm = hwm
        self.acked = hwm
        self.attempted = hwm
        self.saves = 0
        self.gcs = 0

    def run(self) -> None:
        try:
            c = CheckpointClient(self.addr, self.crypto, chunk_size=CHUNK, batch_bytes=1 << 20, timeout=30)
        except Exception:
            return
        c.hwm = self.hwm
        try:
            while True:
                mutate(self.st, self.rng)
                seq = c.hwm + 1
                self.record[seq] = digest(self.st)
                self.attempted = seq
                s = c.save(self.st, step=seq)
                assert s.seq == seq
                self.acked = seq
                self.saves += 1
                if self.rng.random() < 0.15:
                    c.gc(int(self.rng.integers(1, 4)))
                    self.gcs += 1
        except Exception:
            pass
        finally:
            try:
                c.close()
            except Exception:
                pass


def check_store(addr: str, crypto: CkptCrypto, record: dict, acked: int, attempted: int) -> tuple[str, str]:
    """Returns (outcome, detail)."""
    try:
        c = CheckpointClient(addr, crypto, chunk_size=CHUNK, timeout=30)
    except Exception as e:
        return "detected", f"connect: {e!r}"
    try:
        try:
            latest = c.latest()
        except IntegrityError as e:
            return "detected", f"latest: {e}"
        if acked > 0 and (latest is None or latest < acked):
            # durability violated; is it at least detected by the anti-rollback check?
            try:
                c.restore(trusted_hwm=acked)
            except IntegrityError as e:
                return "detected", f"rollback-detected latest={latest} acked={acked}: {type(e).__name__}"
            return "undetected", f"latest={latest} < acked={acked} without error"
        if latest is not None and latest > attempted:
            return "undetected", f"phantom seq {latest} > attempted {attempted}"
        if latest is None:
            return "ok", "empty store"
        detected = []
        for s in c.list():
            try:
                out, info = c.restore(seq=s, trusted_hwm=acked if s == latest else None)
            except IntegrityError as e:
                detected.append(f"seq{s}:{type(e).__name__}")
                continue
            if digest(out) != record.get(s):
                return "undetected", f"seq {s} content mismatch"
        if detected:
            return "detected", ";".join(detected)[:200]
        return "ok", f"latest={latest}"
    except CkptError as e:
        return "detected", f"{type(e).__name__}: {e}"[:200]
    finally:
        c.close()


def kill_campaign(trials: int, seed: int, chain_len: int = 25, max_delay: float = 0.4) -> dict:
    rng = np.random.default_rng(seed)
    outcomes = {"ok": 0, "detected": 0, "undetected": 0}
    details = []
    durability_slack = {"latest_eq_acked": 0, "latest_eq_attempted_unacked": 0}
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="m4-kill-"))
    trial = 0
    chain = 0
    recovery_ms = []
    while trial < trials:
        root = tmp / f"chain{chain}"
        io = "uring" if chain % 2 == 0 else "pwrite"
        crypto = CkptCrypto.generate(f"crash-{chain}")
        st = make_state(rng)
        record: dict[int, str] = {}
        acked = attempted = 0
        for _ in range(chain_len):
            if trial >= trials:
                break
            srv = ServerProcess(root, io=io).start()
            if srv.recovery.get("recovery_ms") is not None:
                recovery_ms.append(srv.recovery["recovery_ms"])
            w = Workload(srv.addr, crypto, rng, st, record, hwm=acked)
            w.start()
            time.sleep(float(rng.uniform(0.0, max_delay)))
            srv.kill9()
            w.join(30)
            acked, attempted = w.acked, w.attempted
            srv2 = ServerProcess(root, io=io).start()
            outcome, detail = check_store(srv2.addr, crypto, record, acked, attempted)
            try:
                latest = CheckpointClient(srv2.addr, crypto).latest()
            except Exception:
                latest = None
            srv2.stop()
            outcomes[outcome] += 1
            if latest is not None:
                if latest == acked:
                    durability_slack["latest_eq_acked"] += 1
                elif latest == attempted:
                    durability_slack["latest_eq_attempted_unacked"] += 1
                # the client's view after recovery: continue from the recovered latest
                acked = max(acked, latest)
                attempted = max(attempted, latest)
            details.append(
                {
                    "trial": trial,
                    "chain": chain,
                    "io": io,
                    "outcome": outcome,
                    "detail": detail,
                    "acked": w.acked,
                    "attempted": w.attempted,
                    "saves": w.saves,
                    "gcs": w.gcs,
                }
            )
            trial += 1
        chain += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "trials": trials,
        "outcomes": outcomes,
        "durability": durability_slack,
        "total_saves": sum(d["saves"] for d in details),
        "total_gcs": sum(d["gcs"] for d in details),
        "recovery_ms_mean": float(np.mean(recovery_ms)) if recovery_ms else None,
        "wall_s": time.time() - t_start,
        "trials_detail": details,
    }


def build_base(root: Path, crypto: CkptCrypto, rng: np.random.Generator, saves: int = 8) -> tuple[dict, int]:
    record = {}
    st = make_state(rng)
    with ServerProcess(root, io="pwrite") as srv, CheckpointClient(srv.addr, crypto, chunk_size=CHUNK) as c:
        for i in range(saves):
            mutate(st, rng)
            record[i + 1] = digest(st)
            c.save(st, step=i + 1)
            if i == saves // 2:
                c.gc(3)
    return record, saves


def candidate_files(root: Path) -> list[Path]:
    fs = [p for p in (root / "segments").iterdir()] + [root / "wal" / "manifest.wal"]
    fs += [p for p in (root / "index").iterdir() if p.is_file() and not p.name.startswith("LOG")]
    return [p for p in fs if p.stat().st_size > 0]


def fault_campaign(kind: str, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp(prefix=f"m4-{kind}-"))
    base = tmp / "base"
    crypto = CkptCrypto.generate(f"{kind}-job")
    record, acked = build_base(base, crypto, rng)
    outcomes = {"ok": 0, "detected": 0, "undetected": 0}
    by_target: dict[str, dict[str, int]] = {}
    details = []
    t0 = time.time()
    for trial in range(trials):
        root = tmp / f"t{trial}"
        shutil.copytree(base, root)
        files = candidate_files(root)
        f = files[int(rng.integers(0, len(files)))]
        size = f.stat().st_size
        target = "segment" if f.parent.name == "segments" else ("wal" if f.parent.name == "wal" else "index")
        if kind == "torn":
            cut = int(rng.integers(0, size))
            with open(f, "r+b") as fh:
                fh.truncate(cut)
            what = f"truncate {f.name} {size}->{cut}"
        else:
            nbits = int(rng.integers(1, 4))
            data = bytearray(f.read_bytes())
            offs = []
            for _ in range(nbits):
                o = int(rng.integers(0, size))
                data[o] ^= 1 << int(rng.integers(0, 8))
                offs.append(o)
            f.write_bytes(bytes(data))
            what = f"flip {nbits} bits in {f.name} at {offs}"
        try:
            srv = ServerProcess(root, io="pwrite").start(timeout=60)
        except Exception as e:
            outcome, detail = "detected", f"server refused to start: {str(e)[-150:]}"
        else:
            outcome, detail = check_store(srv.addr, crypto, record, acked, acked)
            srv.stop()
        outcomes[outcome] += 1
        by_target.setdefault(target, {"ok": 0, "detected": 0, "undetected": 0})[outcome] += 1
        details.append(
            {"trial": trial, "target": target, "fault": what, "outcome": outcome, "detail": detail}
        )
        shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "trials": trials,
        "outcomes": outcomes,
        "by_target": by_target,
        "wall_s": time.time() - t0,
        "trials_detail": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-trials", type=int, default=1000)
    ap.add_argument("--torn-trials", type=int, default=500)
    ap.add_argument("--flip-trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="m4")
    a = ap.parse_args()
    cfg = vars(a)
    if a.kill_trials:
        r = kill_campaign(a.kill_trials, a.seed)
        write_result(f"{a.out}/crash_kill9.json", r, cfg, a.seed)
        print("kill9", r["outcomes"], r["durability"], f"saves={r['total_saves']} gcs={r['total_gcs']}")
    if a.torn_trials:
        r = fault_campaign("torn", a.torn_trials, a.seed + 1)
        write_result(f"{a.out}/crash_torn.json", r, cfg, a.seed + 1)
        print("torn", r["outcomes"], r["by_target"])
    if a.flip_trials:
        r = fault_campaign("bitflip", a.flip_trials, a.seed + 2)
        write_result(f"{a.out}/crash_bitflip.json", r, cfg, a.seed + 2)
        print("bitflip", r["outcomes"], r["by_target"])


if __name__ == "__main__":
    main()
