"""M5: recovery-time (and save-time) overhead of P2 authentication + decryption vs plaintext checkpoints.

Run inside the privileged lab container (cold-cache restores need /proc/sys/vm/drop_caches):
    nice -n 19 python -m bench.m5_envelope --trials 10 --preset base
Writes results/m5/envelope_overhead.json.

One real TinyVLM checkpoint (train.trainer.get_full_state tensors, see bench.m4_common.Trainer) is saved to
two aegis-ckpt servers (io_uring), one plaintext (BLAKE3 chunk ids, integrity only) and one with the P2
envelope (keyed-BLAKE3 ids, AES-256-GCM per chunk under the epoch DEK, Ed25519-signed manifest). Each
trial restores both, in a seeded random order: cold (page cache dropped) and warm (immediately again).
Restore = manifest fetch + signature/header checks + chunk fetch + per-chunk verification
(plain: BLAKE3 recompute; P2: GCM tag + keyed-BLAKE3 recompute) + tensor reassembly.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from bench import stats
from bench.m4_bench import drop_caches
from bench.m4_common import Trainer, loadavg
from bench.meta import write_result
from ckpt.client import CheckpointClient
from ckpt.crypto import CkptCrypto
from ckpt.server import ServerProcess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--saves", type=int, default=5, help="incremental saves per mode (save-side cost)")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--io", default="uring")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--root", default="/var/tmp/aegis-m5env")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    root = Path(a.root)
    shutil.rmtree(root, ignore_errors=True)
    tr = Trainer(a.preset, seed=a.seed)
    modes = {"plain": None, "p2": CkptCrypto.generate("m5env")}
    srv = {m: ServerProcess(root / m, io=a.io).start() for m in modes}
    save = {m: [] for m in modes}
    truth = None
    try:
        for i in range(a.saves):
            tr.one_step()
            tr.one_step()
            st = {k: v.clone() for k, v in tr.tensors().items()}
            truth = st
            for m in rng.permutation(list(modes)):
                with CheckpointClient(srv[m].addr, modes[m]) as c:
                    s = c.save(st, step=i)
                save[m].append(s.as_dict())
        rows = {
            m: {"cold_s": [], "warm_s": [], "cold_verify_s": [], "cold_fetch_s": [], "warm_verify_s": []}
            for m in modes
        }
        for _t in range(a.trials):
            for m in rng.permutation(list(modes)):
                with CheckpointClient(srv[m].addr, modes[m]) as c:
                    drop_caches()
                    out, info = c.restore()
                    rows[m]["cold_s"].append(info.total_s)
                    rows[m]["cold_fetch_s"].append(info.fetch_s)
                    rows[m]["cold_verify_s"].append(info.verify_s)
                    assert all(torch.equal(out[k], truth[k]) for k in truth)
                    _, info = c.restore()
                    rows[m]["warm_s"].append(info.total_s)
                    rows[m]["warm_verify_s"].append(info.verify_s)
        state_bytes = sum(v.numel() * v.element_size() for v in truth.values())
        summ = {m: {k: stats.summarize(v).as_dict() for k, v in r.items()} for m, r in rows.items()}
        for m in modes:
            later = save[m][1:]
            summ[m]["save_flush_s"] = stats.summarize([s["flush_s"] for s in later]).as_dict()
            summ[m]["save_hash_s"] = stats.summarize([s["hash_s"] for s in later]).as_dict()
            summ[m]["save_encrypt_s"] = stats.summarize([s["encrypt_s"] for s in later]).as_dict()
            summ[m]["save_full_first_flush_s"] = save[m][0]["flush_s"]
        overhead = {
            k: {
                "p2_over_plain": summ["p2"][k]["mean"] / summ["plain"][k]["mean"],
                "abs_s": summ["p2"][k]["mean"] - summ["plain"][k]["mean"],
                "welch": stats.welch(rows["p2"][k], rows["plain"][k]),
            }
            for k in ("cold_s", "warm_s", "cold_verify_s")
        }
        write_result(
            "m5/envelope_overhead.json",
            {
                "state_bytes": state_bytes,
                "summary": summ,
                "overhead": overhead,
                "raw": rows,
                "saves": save,
                "loadavg_end": loadavg(),
                "wall_clock_end": time.time(),
            },
            vars(a),
            a.seed,
        )
        print({k: round(v["p2_over_plain"], 3) for k, v in overhead.items()})
    finally:
        for s in srv.values():
            s.stop()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
