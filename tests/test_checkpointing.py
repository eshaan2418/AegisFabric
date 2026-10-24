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
