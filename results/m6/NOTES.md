# M6 notes: live financial-assistant media path (AV1 RTP, joint vs independent adaptation)

All numbers below come from `results/m6/summary.json` (aggregated from `results/m6/trials/*.json`),
`results/m6/webrtc_ingest.json` and `results/m6/calibration.json`. Per-figure tables are in
`results/m6/figs/TABLES.md`. Format: mean ± 95% CI half-width over n = 5 seeds.

## What was built

* `media/rtp.py`, `media/av1rtp.py`: RTP/RTCP (SR, RR, REMB, PLI, NACK) and the AV1 RTP payload format
  (aggregation header, OBU fragmentation). There is a receiver jitter buffer with a playout deadline, and A/V
  sync via RTCP SR NTP↔RTP mapping. Median |A/V skew| is about 2.5 ms in every non-blocking condition.
* `media/codec.py`: AV1 through SVT-AV1 (PyAV, CBR low-delay, 1 thread in the campaign) and dav1d decode.
  Opus carries the audio.
* `media/adapt.py`: GCC-style loss-based controller (RR fraction lost), receiver delay-gradient estimator
  that sends REMB, and a bitrate ladder. **Independent** policy: ABR only, fixed 30 fps, fixed 10 fps frame
  sampling, FIFO inference queue. **Joint** policy: ABR rung, encoder fps from the offline
  Q(bitrate, fps)→VMAF table, frame-selection rate capped at 0.8× the measured inference service rate,
  and a latest-wins vision slot in which STT/answer jobs jump the queue.
* `media/inference.py`: a single shared "accelerator" that never blocks the media path. Its jobs are
  STT (**whisper-tiny**, a real offline model run through `transformers`) and a **vision stub**: a
  random-weight CNN used as a compute proxy, a deterministic chart reader, and a service time floor of
  50 ms, or 150 ms in the overload condition. Speech is synthesised with macOS `say` from the synthetic
  transcripts (`media/source.py`; all 55 trials record `speech: say`, `stt: whisper-tiny`). Video is a
  256×256 30 fps chart screen-share built from `data/synth/finance.py`.
* `media/webrtc_ingest.py`: a separate aiortc leg (real ICE/DTLS-SRTP on localhost). It negotiates VP8,
  and the server transcodes to AV1.
* `media/netem.py`: seeded in-process link model with delay, jitter, Gilbert-Elliott loss, rate limit,
  drop-tail and an optional order-preserving jitter.

## Execution model (read this before quoting any latency)

**The campaign runs in virtual time** (`media/vloop.py`, `Runtime(mode="virtual")`). Every component is an
asyncio task on one virtual-clock loop. The codec, whisper and vision calls really execute,
single-threaded. Each call's measured CPU time (`time.process_time`, floored at the configured vision
service time) is charged as its latency in virtual time. Network delay, jitter and serialisation come
from the emulated link model, not from sockets.

What this means for the claims:
* Glass-to-answer, glass-to-inference, stall and playout numbers are **model-derived latencies**: real
  compute cost plus emulated network delay. They are not wall-clock measurements of a deployed system.
  They leave out OS scheduling, GIL contention, socket and kernel-stack latency, and any parallelism
  between the codec, STT and vision. Everything is serialised as if one core ran it all, but compute
  jobs overlap with network time.
* Virtual time makes results independent of host load. That was the reason for it: the host is shared,
  and its 1-min load average across the campaign ran from 50 to 296 on 18 cores (mean 131;
  `summary.json: loadavg_1m`). CPU time still varies somewhat with cache and frequency contention.
* The only wall-clock measurement is the aiortc ingest leg, `results/m6/webrtc_ingest.json`. It uses
  localhost with no impairment.
* Emulated network ≠ real network. The link is an in-process model (`media/netem.py`), not kernel
  netem/tc. It has no cross traffic and one bottleneck. RTCP feedback uses a separate fixed-delay
  reverse link.

Host-sharing rules followed: everything ran at `nice -n 19`, encoder/STT/vision ran with 1 thread in the
campaign, and the aiortc AV1 transcode was capped at 2 threads. The calibration in `calibration.json` was run
earlier by the prior session with 4 encoder threads; its config records `threads: 4`.

## Conditions

Profiles (`media/netem.py`):
* `clean`: 30 ms delay, 3 ms jitter, 1.5 Mbps. Control.
* `lossy`: 40 ms delay, 15 ms jitter, bursty GE loss of about 3%.
* `bwdrop`: 1.2 Mbps, falling to 250 kbps at 8 s and back at 16 s, with 0.5% loss.
* `jitter`: 50 ms delay, normal jitter sd 40 ms, order preserved, 1% loss, 1.5 Mbps.
* `overload`: clean link, vision service time 150 ms instead of 50 ms.

Policies: independent and joint, plus `blocking` (an ablation in which the media path waits on the
inference queue). Each trial is a 24 s session with 3 spoken client questions; seeds 0–4. Total: 55
trials (11 conditions × 5 seeds).

## Results (joint vs independent)

Glass-to-answer means the capture time of the end of the question to the time the answer is emitted
(virtual time).

| profile | g2a indep (s) | g2a joint (s) | VMAF indep | VMAF joint | stall indep (s/24 s) | stall joint |
|---|---|---|---|---|---|---|
| clean | 1.05 ± 0.12 | 1.12 ± 0.05 | 76.0 ± 0.4 | 75.9 ± 0.8 | 0 | 0 |
| lossy | 1.05 ± 0.11 | 1.03 ± 0.12 | 66.0 ± 12 | 67.8 ± 14 | 2.2 ± 3.8 | 1.6 ± 4.5 |
| bwdrop | 1.96 ± 0.47 | 1.33 ± 0.12 | 51.5 ± 2.9 | 55.4 ± 6.7 | 6.2 ± 0.8 | 4.9 ± 2.5 |
| jitter | 0.79 ± 0.06 | 0.83 ± 0.07 | 71.4 ± 1.0 | 71.1 ± 1.4 | 0.007 | 0.007 |
| overload | 2.93 ± 1.3 | 0.92 ± 0.40 | 76.0 ± 0.6 | 74.1 ± 5.6 | 0 | 0.63 ± 1.7 |

Statistics: Welch t, Mann-Whitney U and a paired t over seeds; the same seed gives the same content and
the same link RNG stream. All are in `summary.json: tests / pooled_tests / ablation_tests`.

* **bwdrop**: glass-to-answer is lower with joint (1.33 vs 1.96 s, Welch p = 0.019, MWU p = 0.016,
  paired p = 0.019, Hedges g = −2.0). This **does not survive Holm** across the 36 per-profile tests
  (p_holm = 0.51). VMAF (+3.9), stall time (−1.2 s) and network frame loss (0.22 vs 0.28) point the
  same way, but none of them is significant (p ≥ 0.16).
* **overload** (inference-bound): glass-to-answer 0.92 vs 2.93 s (Welch p = 0.011, MWU p = 0.008,
  p_holm = 0.33). Frames dropped by backpressure: 4.2 vs 51 (p_holm = 0.030). Glass-to-inference p95:
  0.35 vs 2.99 s. Joint pays 2 VMAF points and 0.6 s of stalls, neither significant. Caveat: the
  independent baseline samples at a fixed 10 fps while vision serves at most about 6.7 fps. Its FIFO
  therefore overflows and STT waits behind vision jobs. The gain comes mostly from service-rate-aware
  sampling plus STT priority, and the size of the gap depends on that choice of fixed rate.
* **clean, lossy, jitter**: no significant difference in glass-to-answer, VMAF or stalls. In clean and
  jitter, joint is slightly *slower* to answer (+0.07 s and +0.04 s, n.s.).
* **Pooled over the three impairment profiles** (lossy, bwdrop, jitter; 15 pairs): glass-to-answer 1.07
  vs 1.26 s (paired p = 0.063, Holm 0.37, n.s.). VMAF 64.8 vs 63.0 (n.s.). Stall time 2.2 vs 2.8 s
  (n.s.). Glass-to-inference p95 0.36 vs 0.65 s (paired p = 0.007, Holm 0.050). Joint consistently drops
  more frames to backpressure (7.1 vs 0.7, Holm p < 0.001). This is by design: latest-wins replaces a
  stale pending frame instead of queueing it.
* **The joint encoder knob was inert.** The calibration table rates 30 fps above 15 or 10 fps at every
  rung (e.g. 400 kbps: 96.5 @30 vs 85.0 @15 vs 90.1 @10), so joint picked 30 fps in all 25 joint
  trials (`mean_fps_decision = 30`). Its bitrate decisions use the same ABR as independent. Any
  joint-vs-independent difference therefore comes from frame selection and inference scheduling, not
  from encoder adaptation.
* **Backpressure ablation** (`overload-blocking`): when the media path blocks on the inference queue, the
  stream collapses. Stall time is 21.3 of 24 s vs 0.63 s for joint, rendered VMAF is 5.0 vs 74.1,
  answer correctness is 0 (stale vision), and A/V skew grows to about 8 s (Welch p < 1e-4). Dropping or
  deferring frames instead of blocking is what keeps media alive.
* Answer correctness (intent recognised from whisper text plus a fresh-enough chart read within
  tolerance) is 0.53–0.73 everywhere, with no significant differences. Whisper-tiny mis-transcriptions
  (e.g. "a count's in 23,186" for an account id) dominate.

Overall, this is a **mixed result.** Joint adaptation helps when the inference worker is the
bottleneck, which is a large effect, and probably helps under a bandwidth collapse, which is significant
uncorrected but not after Holm. It is neutral under random loss and jitter. With this content and
calibration, encoder-fps adaptation contributes nothing.

### Pathology found: NACK storm under reordering jitter
The first `jitter` profile used netem-style independent per-packet jitter (sd 40 ms), which reorders
packets. These trials are kept in `results/m6/trials_jitter_reorder/` and are excluded from the summary.
Both policies collapsed: rendered VMAF 18.7–47.3, stall time 7.9–17.1 s of 24 s, about 2.7–3.6 k
retransmissions and 1.5–2.3 k drop-tail drops per trial, and about 0.9 Mbps sent on a 1.5 Mbps link.
The receiver NACKs a gap within 10 ms with no reordering tolerance (`media/session.py:_nack_loop`).
Reordered packets therefore trigger retransmissions, which fill the bottleneck queue and cause real loss.
The reported `jitter` profile preserves order (`LinkProfile.reorder=False`). Proper fix (not done): a
reorder-tolerant NACK delay, as in libwebrtc.

### WebRTC ingest leg (wall clock, aiortc, localhost, no impairment)
`results/m6/webrtc_ingest.json`, 5 runs of 10 s each:
* glass-to-server latency is 36.3 ± 0.6 ms mean and 39.7 ± 2.1 ms p95;
* 99.6% of frames delivered;
* the AV1 re-encode takes 2.5 ± 0.4 ms/frame at 422 kbps (2 threads).

aiortc negotiates VP8 (it has no AV1 RTP support), so AV1 appears there only as a transcode. The
impaired campaign does **not** run through aiortc; it uses the repo's own RTP stack (above). The first
attempt delivered 0 frames because aiortc's default public STUN server led to a dead candidate pair on
this host. The fix is host candidates only (`RTCConfiguration(iceServers=[])`).

## Limitations / scoped down
* The campaign uses the in-repo RTP/RTCP/AV1-payload stack in virtual time, not aiortc. aiortc is only
  the unimpaired ingest leg.
* The network is emulated (in-process model). No kernel netem, no real WAN.
* The latencies are virtual-time model outputs (see above), not deployed wall-clock latencies.
* The vision model is a stub (compute proxy plus a deterministic chart reader); only the STT is a real
  model.
* VMAF uses `vmaf_v0.6.1` on 256×256 synthetic screen content, so only relative comparisons are
  meaningful.
* n = 5 seeds per condition, so power is low. After Holm, only the backpressure-drop and inference-latency
  effects are significant.
* The independent baseline's fixed 10 fps sampling exceeds vision capacity in `overload` (see above).
* Provenance:
  * 37 trials were produced at commit `968685d` with `dirty=true`. The only uncommitted change was a
    docstring in `media/inference.py`, committed later in `f084e0c`.
  * 2 trials were produced at `f084e0c` while uncommitted edits touched only the aggregation code.
  * The remaining trials are clean, at `860fd02` (6) and `24f4bb1` (10, the jitter reruns).
  * `calibration.json` records commit `f17548b` (`dirty=false`) because the media code was still untracked
    when it ran. It was produced by the code committed in `8aae4fb`.
* The prior session's driver names `bench/media_*.py` were renamed to `bench/m6_*.py`. Trial
  `command` fields still say `media_run.py`.

## Proposed DECISIONS entries
* **D-0xx M6 virtual-time execution.** The media campaign runs on a virtual-clock asyncio loop. Real
  codec/STT/vision compute is charged at its measured CPU time, and the network comes from a seeded
  in-process link model. Reason: host load (load average 50–296 on 18 cores) made wall-clock real-time
  media runs meaningless. Effect: latencies are model-derived, and a wall-clock number exists only for
  the unimpaired aiortc ingest leg.
* **D-0xx M6 transport.** We use our own RTP/RTCP and AV1-payload stack for the impaired campaign. aiortc
  (VP8 ingest plus AV1 transcode) covers the WebRTC leg, because aiortc has no AV1 RTP support and
  cannot run under the virtual clock.
* **D-0xx M6 encoder re-init.** libsvtav1 (and libaom-av1) in this PyAV/FFmpeg build ignore bit-rate
  changes after open (measured). Rate changes therefore re-create the encoder, which costs a keyframe,
  with hysteresis.

## Proposed BLOCKERS rows
| # | Date | What failed | Fallback used | Impact on claims |
|---|---|---|---|---|
| B-0xx | 2026-10-24 | aiortc has no AV1 RTP codec | Own RTP/AV1 stack for the campaign; aiortc VP8 ingest + AV1 transcode leg | WebRTC ingest is measured only unimpaired |
| B-0xx | 2026-10-24 | Shared host load (load average 50–296) breaks real-time media timing | Virtual-time execution with CPU-time-charged compute | Latencies are model-derived, not wall clock |
| B-0xx | 2026-10-24 | No offline vision model of a useful size | Stub vision (compute proxy + deterministic chart reader); STT is real whisper-tiny | No claims about vision-model accuracy |
| B-0xx | 2026-10-24 | SVT-AV1/libaom ignore runtime bitrate changes | Encoder re-creation with hysteresis | Extra keyframes on every rate change (both policies) |

## Proposed INDEX rows
| Milestone | Command | Output | Finding |
|---|---|---|---|
| M6 | `nice -n 19 uv run python -m bench.m6_calibrate --seeds 0 1 2` | `results/m6/calibration.json` | AV1 Q(bitrate, fps): 30 fps beats 15/10 fps at every rung on this content |
| M6 | `nice -n 19 uv run python -m bench.m6_run --trials 5` | `results/m6/trials/*.json`, `results/m6/summary.json`, `results/m6/figs/m6_metrics.png` | Joint vs independent: g2a 0.92 vs 2.93 s under inference overload and 1.33 vs 1.96 s under bandwidth drop (neither survives Holm); neutral under loss and jitter; blocking ablation stalls 21 of 24 s |
| M6 | `nice -n 19 uv run python -m bench.m6_webrtc --trials 5` | `results/m6/webrtc_ingest.json` | aiortc VP8 ingest on localhost: about 36 ms glass-to-server; AV1 transcode about 2.5 ms/frame |
| M6 | (first jitter profile, reordering) | `results/m6/trials_jitter_reorder/*.json` | Immediate NACKs under reordering cause a retransmission storm; both policies collapse |
