import threading

import torch

from netlab.blobstore import Client, Handler, Server
from train.checkpointing import SliceCheckpointer, flatten, unflatten
from train.model import PRESETS, TinyVLM


def test_flatten_roundtrip():
    d = {
        "model": {"a.w": torch.ones(2)},
        "opt": {
            "state": {"a.w": {"step": torch.tensor(3.0)}},
            "param_groups": [{"lr": 0.1, "params": ["a.w"], "betas": (0.9, 0.99)}],
        },
    }
    r = unflatten(flatten(d))
    assert torch.equal(r["model"]["a.w"], d["model"]["a.w"])
    assert r["opt"]["param_groups"][0]["params"] == ["a.w"]
    assert r["opt"]["param_groups"][0]["betas"] == (0.9, 0.99)


def test_slices_roundtrip_through_blobstore(tmp_path):
    srv = Server(("127.0.0.1", 0), Handler)
    srv.root = tmp_path
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    addr = f"127.0.0.1:{srv.server_address[1]}"
    m = TinyVLM(PRESETS["tiny"])
    opt = torch.optim.AdamW(m.parameters())
    m.loss(torch.rand(2, 32, 32), torch.randint(1, 255, (2, 128))).backward()
    opt.step()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    msd, osd = get_state_dict(m, opt, options=StateDictOptions(full_state_dict=True))
    full = {"model": msd, "opt": osd}
    world = 3
    for r in range(world):
        SliceCheckpointer(Client(addr), r, world).save(7, full, blocking=True)
    got = SliceCheckpointer(Client(addr), 0, world).restore(7)
    for k, v in msd.items():
        assert torch.equal(got["model"][k], v)
    k0 = next(iter(osd["state"]))
    assert torch.equal(got["opt"]["state"][k0]["exp_avg"], osd["state"][k0]["exp_avg"])
    srv.shutdown()


def _store(tmp_path):
    srv = Server(("127.0.0.1", 0), Handler)
    srv.root = tmp_path
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


def test_restore_rejects_slices_cut_from_different_states(tmp_path):
    """Regression: ranks slicing different key sets (e.g. rank 0 full state, others empty) must not
    restore silently or crash obscurely; the completeness check names the problem."""
    import pytest

    srv, addr = _store(tmp_path)
    m = TinyVLM(PRESETS["tiny"])
    full = {"model": m.state_dict(), "opt": {"param_groups": [{"lr": 0.1, "params": list(range(12))}]}}
    partial = {"opt": {"param_groups": [{"lr": 0.1, "params": list(range(12))}]}}
    world = 3
    SliceCheckpointer(Client(addr), 0, world).save(5, full, blocking=True)
    for r in (1, 2):
        SliceCheckpointer(Client(addr), r, world).save(5, partial, blocking=True)
    with pytest.raises(ValueError):
        SliceCheckpointer(Client(addr), 0, world).restore(5)
    srv.shutdown()


def _rank_main(rank, world, port, q):
    import torch.distributed as dist

    from train.trainer import get_full_state

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.manual_seed(0)
    m = TinyVLM(PRESETS["tiny"])
    ddp = torch.nn.parallel.DistributedDataParallel(m)
    opt = torch.optim.AdamW(ddp.parameters())
    m.loss(torch.rand(2, 32, 32), torch.randint(1, 255, (2, 128))).backward()
    opt.step()
    st = get_full_state(ddp, opt)
    q.put((rank, len(flatten(st))))
    dist.destroy_process_group()


def test_full_state_is_full_on_every_rank():
    """Regression for cpu_offload=True + full_state_dict=True returning {} on non-zero ranks."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_rank_main, args=(r, 2, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
    sizes = dict(q.get(timeout=5) for _ in ps)
    assert sizes[0] == sizes[1] > 10, sizes
