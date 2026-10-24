"""Distributed trainer (DDP / FSDP2 / HSDP over Gloo) with telemetry.

Launched one process per rank, with torchrun-compatible env (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT),
usually by `train/launch.py`, which can put each rank in its own network namespace.

Parallelism modes:
  ddp   data parallel, gradient all-reduce (bucketed, ring in Gloo)
  fsdp  FSDP2 fully_shard over all ranks (all-gather params, reduce-scatter grads)
  hsdp  FSDP2 on a 2-D mesh (replicate across sites, shard within a site), via --sites
  local single process, no communication (compute-only reference)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from train.data import ShardedStream, eval_batch
from train.model import PRESETS, TinyVLM, count_params, flops_per_token
from train.telemetry import CollectiveTimer, TelemetrySink, measure_peak_matmul_flops


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ddp", "fsdp", "hsdp", "local"], default="ddp")
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--batch", type=int, default=8, help="per-rank micro-batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-time", type=float, default=0, help="stop after this many seconds (0 = off)")
    ap.add_argument("--target-loss", type=float, default=0, help="stop when eval loss <= target (0 = off)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sites", type=int, default=1, help="HSDP replicate-dim size (ranks contiguous by site)")
    ap.add_argument("--out", type=Path, default=Path("scratch/train"))
    ap.add_argument("--telemetry-udp", default=None)
    ap.add_argument("--control-file", type=Path, default=None, help="JSON {stop_at_step:N}")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None, help="full-state file written by --save-final")
    ap.add_argument("--save-final", type=Path, default=None)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--ckpt-addr", default=None, help="aegis-ckpt server host:port (M4 engine)")
    ap.add_argument("--pg-timeout", type=float, default=180)
    ap.add_argument("--peak-flops", type=float, default=0, help="per-rank peak FLOP/s (0 = measure)")
    return ap.parse_args(argv)


def build(args, rank: int, world: int):
    torch.manual_seed(args.seed)
    cfg = dataclasses.replace(PRESETS[args.preset])
    model = TinyVLM(cfg)
    dp_rank, dp_world = rank, world
    if args.mode == "ddp" and world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, bucket_cap_mb=25)
    elif args.mode in ("fsdp", "hsdp") and world > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        if args.mode == "fsdp":
            mesh = init_device_mesh("cpu", (world,))
        else:
            assert world % args.sites == 0, "world must be divisible by --sites"
            mesh = init_device_mesh("cpu", (args.sites, world // args.sites), mesh_dim_names=("rep", "shard"))
        for blk in model.blocks:
            fully_shard(blk, mesh=mesh)
        fully_shard(model, mesh=mesh)
    return cfg, model, dp_rank, dp_world


def core(model):
    return model.module if hasattr(model, "module") else model


def lm_loss(model, imgs, toks):
    """Next-byte loss computed through the *wrapper's* forward (arms DDP reducer / FSDP gathers)."""
    logits = model(imgs, toks[:, :-1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1), ignore_index=0
    )


def get_full_state(model, opt) -> dict:
    """Full (unsharded) model+optimizer state on every rank, CPU tensors. Placement-independent."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    msd, osd = get_state_dict(model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {"model": msd, "opt": osd}


def set_full_state(model, opt, state: dict) -> None:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

    set_state_dict(
        model,
        opt,
        model_state_dict=state["model"],
        optim_state_dict=state["opt"],
        options=StateDictOptions(full_state_dict=True),
    )


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def evaluate(model, args, cfg, rank, world) -> float:
    imgs, toks = eval_batch(args.data_seed, args.eval_n, cfg.text_len, cfg.img_size)
    per = args.eval_n // max(world, 1)
    sl = slice(rank * per, (rank + 1) * per)
    model.eval()
    with torch.no_grad():
        loss = lm_loss(model, imgs[sl], toks[sl])
    model.train()
    t = torch.tensor([loss.item()])
    if world > 1:
        dist.all_reduce(t)
        t /= world
    return float(t.item())


def read_control(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def main(argv=None) -> int:
    args = parse(argv)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    peak = args.peak_flops or measure_peak_matmul_flops(args.threads)

    timer = CollectiveTimer()
    timer.install()
    t_init0 = time.perf_counter()
    if world > 1 and args.mode != "local":
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=dt.timedelta(seconds=args.pg_timeout)
        )
    cfg, model, dp_rank, dp_world = build(args, rank, world if args.mode != "local" else 1)
    if args.mode == "ddp" and world > 1:
        model.register_comm_hook(None, timer.ddp_hook())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step0 = args.start_step
    if args.resume and args.resume.exists():
        st = torch.load(args.resume, weights_only=False)
        set_full_state(model, opt, st)
        step0 = st["step"]
    stream = ShardedStream(
        args.data_seed, dp_rank, dp_world, args.batch, cfg.text_len, cfg.img_size, start_step=step0
    )
    n_params = count_params(TinyVLM(cfg))
    n_ne = count_params(TinyVLM(cfg), non_embedding=True)
    fpt = flops_per_token(cfg, n_ne)
    tok_per_step = args.batch * cfg.seq_len * (world if args.mode != "local" else 1)
    sink = TelemetrySink(args.out / f"rank{rank}.jsonl", args.telemetry_udp, rank)
    sink.emit(
        {
            "event": "start",
            "mode": args.mode,
            "preset": args.preset,
            "params": n_params,
            "params_nonembed": n_ne,
            "flops_per_token": fpt,
            "peak_flops_per_rank": peak,
            "world": world,
            "init_s": time.perf_counter() - t_init0,
            "start_step": step0,
            "pid": os.getpid(),
        }
    )
    timer.drain()
    t_train0 = time.perf_counter()
    reached = None
    last_eval = float("nan")
    step = step0
    stop_reason = "steps"
    while step < args.steps:
        ctl = read_control(args.control_file)
        if "stop_at_step" in ctl and step >= int(ctl["stop_at_step"]):
            stop_reason = "control"
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        t0 = time.perf_counter()
        imgs, toks = stream.next()
        loss = lm_loss(model, imgs, toks)
        t_fwd = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t_bwd = time.perf_counter()
        opt.step()
        t1 = time.perf_counter()
        step += 1
        coll = timer.drain()
        dt_step = t1 - t0
        rec = {
            "event": "step",
            "step": step,
            "loss": float(loss.item()),
            "step_s": dt_step,
            "fwd_s": t_fwd - t0,
            "bwd_s": t_bwd - t_fwd,
            "opt_s": t1 - t_bwd,
            "elapsed_s": t1 - t_train0,
            "tok_s": tok_per_step / dt_step,
            "mfu": fpt * tok_per_step / dt_step / (peak * (world if args.mode != "local" else 1)),
            "coll": coll,
        }
        if args.eval_every and step % args.eval_every == 0:
            last_eval = evaluate(model, args, cfg, rank, world if args.mode != "local" else 1)
            rec["eval_loss"] = last_eval
            if args.target_loss and reached is None and last_eval <= args.target_loss:
                reached = time.perf_counter() - t_train0
                rec["reached_target_s"] = reached
        sink.emit(rec)
        if args.target_loss and reached is not None:
            stop_reason = "target"
            break
        if args.max_time and time.perf_counter() - t_train0 > args.max_time:
            stop_reason = "max_time"
            break
    if args.save_final:
        st = get_full_state(model, opt)
        if rank == 0:
            args.save_final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**st, "step": step}, args.save_final)
    sink.emit(
        {
            "event": "end",
            "step": step,
            "train_s": time.perf_counter() - t_train0,
            "eval_loss": last_eval,
            "reached_target_s": reached,
            "stop_reason": stop_reason,
        }
    )
    sink.close()
    if world > 1 and args.mode != "local":
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
