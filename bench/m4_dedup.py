"""Dedup ratio across successive real training checkpoints, vs chunking scheme and chunk size.

    nice -n 19 uv run python -m bench.m4_dedup --seeds 5 --ckpts 10 --every 5 --preset base
Writes results/m4/dedup.json (mean ± 95% CI over seeds).

Workload: TinyVLM (train/model.py) + AdamW, checkpoint = train.trainer.get_full_state (model + optimizer),
one checkpoint every `--every` optimizer steps. Chunking schemes, all with BLAKE3 content ids (the keyed
variant used under P2 has the same equality structure, so the ratios are identical):

* tensor:  fixed-size chunks within each tensor (what CheckpointClient does)
* blob:    fixed-size chunks over the torch.save byte stream of the whole state (what EngineBackend sees
           through SliceCheckpointer)

dedup ratio = logical bytes of all checkpoints / bytes of distinct chunks. "incremental" = mean fraction
of a checkpoint's bytes that are new relative to everything stored before it (checkpoints 2..n).
No timing is measured, so this can run on the shared host.
"""

from __future__ import annotations

import argparse
import io

import blake3
import numpy as np
import torch

from bench import stats
from bench.m4_common import Trainer
from bench.meta import write_result


def tensor_chunks(tensors: dict[str, torch.Tensor], size: int):
    for k in sorted(tensors):
        t = tensors[k]
        if t.numel() == 0:
            continue
        mv = memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy())
        for o in range(0, len(mv), size):
            yield mv[o : o + size]


def blob_chunks(state: dict, size: int):
    buf = io.BytesIO()
    torch.save(state, buf)
    mv = memoryview(buf.getvalue())
    for o in range(0, len(mv), size):
        yield mv[o : o + size]


def run(scheme: str, ckpts: list, size: int) -> dict:
    seen: set[bytes] = set()
    logical = unique = 0
    incr = []
    for i, c in enumerate(ckpts):
        it = tensor_chunks(c[1], size) if scheme == "tensor" else blob_chunks(c[0], size)
        new = tot = 0
        for piece in it:
            h = blake3.blake3(piece).digest()
            tot += len(piece)
            if h not in seen:
                seen.add(h)
                new += len(piece)
        logical += tot
        unique += new
        if i > 0:
            incr.append(new / tot)
    return {"dedup_ratio": logical / unique, "incremental_new_frac": float(np.mean(incr)), "logical": logical}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ckpts", type=int, default=10)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--sizes", default="65536,262144,1048576,4194304")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sizes = [int(s) for s in a.sizes.split(",")]
    per: dict[str, list[dict]] = {}
    for scen in ("full", "frozen"):
        for seed in range(a.seeds):
            tr = Trainer(a.preset, seed=seed, freeze="half" if scen == "frozen" else "none")
            tr.one_step()  # optimizer state exists from the first checkpoint on
            ckpts = []
            for _ in range(a.ckpts):
                for _ in range(a.every):
                    tr.one_step()
                st = tr.state()
                # copy: get_full_state returns live tensors
                st = torch.utils._pytree.tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, st
                )
                tens = {k: v for k, v in _flat_tensors(st).items()}
                ckpts.append((st, tens))
            for scheme in ("tensor", "blob"):
                for size in sizes:
                    r = run(scheme, ckpts, size)
                    per.setdefault(f"{scen}/{scheme}/{size}", []).append(r)
                    print(scen, seed, scheme, size, {k: round(v, 4) for k, v in r.items()}, flush=True)
    summary = {
        k: {
            "dedup_ratio": stats.summarize([r["dedup_ratio"] for r in v]).as_dict(),
            "incremental_new_frac": stats.summarize([r["incremental_new_frac"] for r in v]).as_dict(),
            "state_bytes": v[0]["logical"] / a.ckpts,
        }
        for k, v in per.items()
    }
    write_result("m4/dedup.json", {"summary": summary, "per_seed": per}, vars(a), 0)


def _flat_tensors(st: dict) -> dict[str, torch.Tensor]:
    from train.checkpointing import flatten

    return {k: v for k, v in flatten(st).items() if isinstance(v, torch.Tensor)}


if __name__ == "__main__":
    main()
