"""M10 end-to-end demo: train across 3 sites → failure → controller re-plans → encrypted checkpoint/restore
→ serve a live WebRTC assistant session with the restored model.

Part 1, inside the lab container (`make demo` runs both parts):
    python -m bench.m10_demo lab --out results/m10
  A 6-rank DDP job runs on the emulated WAN under the joint controller in the `site_degradation` scenario.
  Every WAN link of one site collapses for the rest of the run; routing alone cannot fix that, so the
  controller has to re-place the job. Checkpoints go to the M4 engine (io_uring) with P2 encryption and
  signed manifests, using one job key directory for all segments. After the run, the newest complete
  checkpoint is restored from the engine store with full verification. The trained weights are written to
  scratch/m10/model.pt, a local hand-off file, not a result.
Part 2, on the host (aiortc is installed there):
    python -m bench.m10_demo assistant --out results/m10
  A live aiortc (ICE/DTLS-SRTP, VP8) session streams synthetic chart video. A frame selector forwards
  2 frames/s to an inference worker. When the worker is busy, the frame is dropped rather than queued
  (backpressure). The worker greedily decodes the restored model's statement text for the frame.
  Recorded: glass-to-answer (capture → answer), frames delivered, selected, dropped, answered.
  The model is a 4.9M-parameter toy trained for a few minutes: the demo shows the systems path works,
  not answer quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time

import numpy as np

from bench.meta import REPO_ROOT, resolve, write_result

HANDOFF = REPO_ROOT / "scratch" / "m10" / "model.pt"


def lab_part(a) -> None:
    import torch

    from bench.wan_run import CKPT_PORT, RunConfig, run  # noqa: F401
    from ckpt.client import EngineBackend
    from ckpt.crypto import init_keydir, load_keydir
    from ckpt.server import ServerProcess
    from train.checkpointing import SliceCheckpointer

    out = resolve(a.out)
    keys = REPO_ROOT / "scratch" / "m10" / "keys"
    cfg = RunConfig(
        "site_degradation",
        "joint",
        a.seed,
        steps=a.steps,
        out=str(out / "demo"),
        ckpt_backend="engine",
        ckpt_keys=str(keys),
        keep_store=True,
    )
    job = f"{cfg.scenario}-{cfg.label}-{cfg.seed}"
    init_keydir(keys, job)
    cal = out / "demo" / "calibration.json"
    if not cal.exists():
        src = resolve("results/m3/calibration.json")
        cal.parent.mkdir(parents=True, exist_ok=True)
        cal.write_text(src.read_text())  # the demo reuses the M3 calibration (same workload)
    summ = run(cfg)
    raw = out / "demo" / "raw" / cfg.scenario / cfg.label / f"seed{cfg.seed}"
    run_data = json.loads(raw.with_suffix(".json").read_text())["data"]
    # newest complete checkpoint across the two (residency-allowed) storage sites
    best = None
    for site in ("tor", "mtl"):
        root = raw / f"store_{site}"
        if not root.is_dir():
            continue
        srv = ServerProcess(root, listen="127.0.0.1:0", io="uring").start()
        b = EngineBackend(srv.addr, job=job, crypto=load_keydir(keys, job))
        steps: dict[int, int] = {}
        for k in b.keys():
            if k.endswith("of6"):
                st = int(k.split("_")[-2])
                steps[st] = steps.get(st, 0) + 1
        b.close()
        srv.stop()
        for st, c in steps.items():
            if c == 6 and (best is None or st > best[0]):
                best = (st, site, root)
    if best is None:
        raise SystemExit("no complete checkpoint found")
    step, site, root = best
    srv = ServerProcess(root, listen="127.0.0.1:0", io="uring").start()
    t0 = time.perf_counter()
    b = EngineBackend(srv.addr, job=job, crypto=load_keydir(keys, job))
    state = SliceCheckpointer(b, 0, 6, job=job).restore(step)
    restore_s = time.perf_counter() - t0
    b.close()
    srv.stop()
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state["model"], "step": step, "preset": cfg.preset}, HANDOFF)
    timeline = sorted(
        [
            {"t": e["wall"], "what": e["event"], **{k: v for k, v in e.items() if k not in ("wall",)}}
            for e in run_data["manager_events"]
        ]
        + [
            {"t": f["wall"], "what": f["event"] + ":" + f["kind"], "plane": f["plane"]}
            for f in run_data["failures"]
        ]
        + [
            {
                "t": d["wall"],
                "what": "decision",
                "reason": d["reason"],
                "placement": d["placement"],
                "storage": d["storage"],
            }
            for d in run_data["decisions"]
        ],
        key=lambda x: x["t"],
    )
    write_result(
        out / "demo_lab.json",
        {
            "summary": summ,
            "restored": {"step": step, "site": site, "verified_restore_s": restore_s, "encrypted": True},
            "timeline": timeline,
        },
        config=vars(a),
        seed=a.seed,
    )
    print(
        json.dumps(
            {
                "migrations": summ["n_migrations"],
                "crashes": summ["n_crashes"],
                "restored_step": step,
                "site": site,
                "restore_s": round(restore_s, 2),
            }
        )
    )


def assistant_part(a) -> None:
    import torch

    from bench.stats import summarize
    from media.source import build_source
    from media.webrtc_ingest import run_ingest
    from train.model import PRESETS, TinyVLM

    torch.set_num_threads(2)
    ck = torch.load(HANDOFF, weights_only=False)
    model = TinyVLM(PRESETS[ck["preset"]])
    model.load_state_dict(ck["model"])
    model.eval()
    c = model.cfg
    src = build_source(a.seed, duration_s=a.duration, use_say=False)
    lock = threading.Lock()
    busy = threading.Event()
    log = {"selected": 0, "dropped_busy": 0, "answers": []}
    last_sel = [-1e9]

    def infer(idx: int, rgb: np.ndarray, t_cap: float) -> None:
        g = rgb.mean(axis=2) / 255.0
        f = g.shape[0] // c.img_size
        img = torch.tensor(
            g[: c.img_size * f, : c.img_size * f].reshape(c.img_size, f, c.img_size, f).mean(axis=(1, 3)),
            dtype=torch.float32,
        )[None]
        toks = torch.tensor([[2]])
        with torch.no_grad():
            for _ in range(a.answer_len):
                nxt = model(img, toks)[:, -1].argmax(-1, keepdim=True)
                toks = torch.cat([toks, nxt], 1)
        text = bytes(int(t) for t in toks[0, 1:] if 0 < int(t) < 128).decode("ascii", "replace")
        with lock:
            log["answers"].append({"frame": idx, "glass_to_answer_s": time.monotonic() - t_cap, "text": text})
        busy.clear()

    def on_frame(idx: int, rgb: np.ndarray, t_cap: float) -> None:
        now = time.monotonic()
        if now - last_sel[0] < 1.0 / a.select_fps:
            return
        last_sel[0] = now
        log["selected"] += 1
        if busy.is_set():
            log["dropped_busy"] += 1  # backpressure: drop, never stall the media path
            return
        busy.set()
        threading.Thread(target=infer, args=(idx, rgb, t_cap), daemon=True).start()

    r = asyncio.run(run_ingest(src.frames, on_frame=on_frame))
    time.sleep(2.0)  # let a last in-flight answer finish
    g2a = [x["glass_to_answer_s"] for x in log["answers"]]
    data = {
        "ingest": r,
        "restored_from_step": ck["step"],
        "frames_selected": log["selected"],
        "frames_dropped_backpressure": log["dropped_busy"],
        "answers": log["answers"],
        "glass_to_answer_s": summarize(g2a).as_dict() if g2a else None,
        "note": "aiortc loopback on 127.0.0.1, wall clock, shared host; toy model (answer quality not claimed)",
    }
    write_result(resolve(a.out) / "demo_assistant.json", data, config=vars(a), seed=a.seed)
    print(json.dumps({k: v for k, v in data.items() if k not in ("answers", "ingest")}, default=str)[:800])
    if log["answers"]:
        print("sample answer:", repr(log["answers"][0]["text"][:80]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["lab", "assistant"])
    ap.add_argument("--out", default="results/m10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--select-fps", type=float, default=2.0)
    ap.add_argument("--answer-len", type=int, default=48)
    a = ap.parse_args()
    (lab_part if a.part == "lab" else assistant_part)(a)


if __name__ == "__main__":
    main()
