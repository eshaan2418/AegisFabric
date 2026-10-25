# M4 (checkpoint engine) + M5 checkpoint envelope — notes

All timings in this directory were measured on a **shared host under concurrent load** (another agent's
timing-sensitive network campaign ran at the same time; host load average 50–60 on 18 cores during the
runs; the container load average, 6–19, is recorded per trial in `bench_raw.json`). Benchmarks ran at
`nice -n 19` in a container capped at 4 CPUs with ≤4 torch/io threads. Absolute numbers are therefore
pessimistic and noisy. Methods are interleaved in a seeded random order within each (fs, scenario, trial).
The two filesystems were **not** interleaved: ext4 ran first, at higher load. Compare methods within a
filesystem, not ext4 against XFS.

## Headline results (every number below is from the named file; ± = 95% CI half-width)

**Crash consistency** (`bench/m4_crash.py`, rerun at commit a0e6ce4 in the lab container):

| campaign | trials | ok | detected | **undetected** | file |
|---|---|---|---|---|---|
| kill -9 of the server at a random time during save/GC (io_uring and pwrite alternate per chain) | 1000 | 1000 | 0 | **0** | `crash_kill9.json` |
| torn write: truncate a random store file (segment / WAL / index) at a random offset | 500 | 325 | 175 | **0** | `crash_torn.json` |
| bit flip: 1–3 random bit flips in a random store file | 500 | 327 | 173 | **0** | `crash_bitflip.json` |

The kill -9 trials performed 4601 saves and 641 GCs. After recovery, the latest checkpoint equalled the
last acked one in 956 trials, and was a durable-but-unacked newer one in 37. In the rest, no save had
completed. Every truncation or flip that hit the WAL or a segment was detected (corrupt-WAL refusal,
missing chunk, checksum or GCM mismatch). Damage to the RocksDB index was always repaired, either by the
cross-check or by a rebuild from segments, or hit dead bytes. The earlier session's run (commit d8b74ee)
gave identical counts.

**Benchmark matrix** (`bench/m4_bench.py` → `bench.json`; TinyVLM `small` + AdamW, 6 checkpoints, 1 step
apart, 5 trials; state 59.1 MB in "full", 39.2 MB in "frozen"; table for every metric in `figs/TABLES.md`):

| metric (XFS, full) | torch.save | DCP | aegis io_uring async (P2) | aegis pwrite async (P2) | aegis io_uring sync (P2) |
|---|---|---|---|---|---|
| training stall / ckpt (s) | 0.18 ± 0.19 | 0.35 ± 0.29 | 0.079 ± 0.073 | 0.057 ± 0.058 | 0.49 ± 0.44 |
| durable latency / ckpt (s) | 0.18 ± 0.19 | 0.35 ± 0.29 | 1.06 ± 0.78 | 0.82 ± 0.55 | 0.49 ± 0.44 |
| device bytes / logical bytes | 1.00 | 1.01 | 1.00 | 1.00 | 1.00 |
| recovery to tensors (s) | 0.12 ± 0.17 | 0.60 ± 0.59 | 0.69 ± 0.69 | 0.47 ± 0.32 | 0.81 ± 1.1 |

| metric (XFS, frozen) | torch.save | DCP | aegis io_uring async (P2) | aegis pwrite async (P2) | aegis io_uring sync (P2) |
|---|---|---|---|---|---|
| training stall / ckpt (s) | 0.047 ± 0.011 | 0.11 ± 0.03 | 0.014 ± 0.015 | 0.012 ± 0.012 | 0.18 ± 0.03 |
| device bytes / logical bytes | 1.00 | 1.01 | 0.79 | 0.79 | 0.79 |
| logical / uploaded bytes (6 ckpts) | 1.00 | 0.99 | 1.27 | 1.27 | 1.27 |

What these show:
- **Async save cuts the training stall** to the snapshot copy. With XFS and frozen layers it is 0.012–0.014 s
  for the engine against 0.047 s for torch.save and 0.11 s for DCP. On ext4 the CIs of the engine and
  torch.save overlap (0.06–0.13 vs 0.16–0.18 s). A **sync** engine save stalls 5–13× longer than the async one
  (0.18–0.55 s).
- **The engine is not faster to durability** at this state size. Its durable latency (0.3–1.7 s) is
  worse than torch.save (0.05–0.18 s), because the Python client's hashing and encryption plus the TCP
  hop cost more than a ~50 MB sequential write. During the background flush, training steps run
  1.1–2.0× slower (`overlap_slowdown`), because the flush competes for the same 4 CPUs.
- **Write amplification** at the device is 1.0 for full training (no dedup is possible) and 0.79 with
  frozen layers on XFS. ext4 shows 0.94–1.22 for the engine against 1.05 for torch.save (journal and
  metadata on many small appends).
- **io_uring vs pwrite**: no difference distinguishable at this size and load (overlapping CIs everywhere).
- **Recovery**: a cold torch.load of one file (0.04–0.12 s) beats the engine (0.3–1.1 s). The engine
  number is end to end: restarting the server process after kill -9, the WAL scan and index cross-check
  (internal 23–546 ms, in `bench_raw.json`), and a fully verified restore.

**Dedup across successive training checkpoints** (`bench/m4_dedup.py` → `dedup.json`; TinyVLM `base`,
10 checkpoints 5 AdamW steps apart, 5 seeds; ratios were identical across seeds, so CI ≈ 0):

| scenario | trainable | dedup ratio (per-tensor chunks, any size 64 KiB–4 MiB) | new bytes per incremental ckpt |
|---|---|---|---|
| full fine-tune | all | 1.00 | 100% |
| frozen embeddings + lower half of blocks | ~upper half | 1.29 | 75% |
| adapter (last block + norm + head) | last block | 2.65 | 31% |

Chunk size does not matter for per-tensor chunking. Chunking the torch.save byte stream (what
`EngineBackend` sees) matches it up to 256 KiB and degrades slightly at 1–4 MiB (1.28 / 1.24 for
"frozen"). This supports fixed-size chunking without CDC.

**P2 envelope overhead** (`bench/m5_envelope.py` → `results/m5/envelope_overhead.json`; 174 MB `base`
checkpoint, io_uring, 10 interleaved trials):

| | plaintext (BLAKE3 ids) | P2 (AES-256-GCM + keyed BLAKE3 + Ed25519) | ratio | Welch p |
|---|---|---|---|---|
| cold restore (s) | 1.07 ± 0.22 | 1.38 ± 0.32 | 1.29× | 0.096 |
| warm restore (s) | 1.02 ± 0.19 | 1.25 ± 0.32 | 1.22× | 0.18 |
| verify phase, cold (s) | 0.48 ± 0.09 | 0.85 ± 0.27 | 1.76× | 0.014 |

The P2 restore costs about +0.3 s per 174 MB, all of it in the verify/decrypt phase. That difference is
significant on its own (p = 0.014) but not on total restore time at n = 10 under this load.
Save-side encryption cost 0.23 ± 0.10 s per incremental save.

## What is where

| Piece | File |
|---|---|
| Rust engine (segments, RocksDB index, manifest WAL, group commit, recovery, GC, io_uring / pwrite) | `ckpt/src/*.rs` |
| Python client: snapshot → background flush, verified restore | `ckpt/client.py` (`CheckpointClient`) |
| Drop-in blob backend for `SliceCheckpointer` (`put(key, bytes)`, `get(key)`) | `ckpt/client.py` (`EngineBackend`) |
| P2 envelope (AES-256-GCM DEKs wrapped by a KEK, keyed-BLAKE3 ids, Ed25519 manifests) | `ckpt/crypto.py`, re-exported as `security/envelope.py` |
| Server launcher | `ckpt/server.py` |
| Tests | `ckpt/src/store.rs` (`cargo test`), `tests/test_ckpt_client.py`, `tests/test_ckpt_backend.py` |
| Drivers | `bench/m4_bench.py`, `bench/m4_dedup.py`, `bench/m4_crash.py`, `bench/m5_envelope.py`, `bench/m4_figs.py` |

Provenance of the engine: it was written in an earlier session on branch
`worktree-agent-a7e1cec597105cb2a` (commits 83b66cf, d8b74ee, 81c0f60) and merged here. A second,
independently written engine from this session (same protocol, plus a fsync "durability log" for
power-loss-accurate torn-write injection and a `bug-no-epoch-refresh` mutation switch) is kept on branch
`m4-alt-engine-a21f` for reference only. It is not used by anything here.

## How to enable the engine in the WAN runs (not done here: `bench/wan_run.py` is owned by another agent)

1. At each storage site, start `ckpt/target/release/aegis-ckpt serve --root <dir> --listen 0.0.0.0:<port> --io uring`
   in the storage namespace instead of `python -m netlab.blobstore` (or use `ckpt.server.ServerProcess(root,
   listen=..., io="uring", netns="<site>-st")`).
2. Pass `--ckpt-backend engine` to `train.trainer`. It already constructs
   `ckpt.client.EngineBackend(addr, job=args.job, rank=rank)`, and `SliceCheckpointer` uses it unchanged.
3. Optional: `EngineBackend(..., crypto=CkptCrypto(...))` for P2 encryption/signing. The trainer does not
   pass key material yet; that needs the KEK/signing key from the security service.
Several ranks may share one storage server: commits race on the server's manifest sequence and the
client re-derives seq/prev and re-signs on conflict (`CheckpointClient._commit`, tested with 3
concurrent ranks in `tests/test_ckpt_backend.py`).

## Proposed DECISIONS entries

- **D-0xx Checkpoint engine design.** Rust `aegis-ckpt`: log-structured append-only segments
  (`MAGIC|len|id|blake3|payload|crc32c`), RocksDB (`rocksdb` crate 0.25, built from source; no fallback
  needed) index `chunk_id → (segment, offset, len, epoch)` inserted only after the segment fsync, manifest
  WAL with BLAKE3 record checksums, group commit, mark/copy/sweep GC with the TLC-found epoch refresh
  and sweep-time re-check (P1 as fixed in 1e7b428). I/O backends: `io_uring` (Linux; segment writes as WRITE
  SQEs batched into one submission, followed by an io_uring FDATASYNC) and `pwrite` thread pool (portable fallback, selectable with `--io`; the macOS host always uses it).
- **D-0xx Chunking: fixed-size, per tensor, no CDC.** `CheckpointClient` chunks each tensor's bytes
  separately at a fixed size (1 MiB in the benchmarks; 256 KiB for `EngineBackend` blobs). Tensor byte
  layouts never shift between training steps (no insertions), so content-defined chunking cannot find
  more duplicates than aligned fixed chunks. It would only add hashing cost and variable chunk sizes.
  `bench/m4_dedup.py` measures the chunk-size sensitivity (see Results).
- **D-0xx Client-side chunking and encryption.** Chunk ids are computed client-side (keyed BLAKE3 under
  P2), so only chunks the server lacks cross the WAN (HAS before PUT). The server never sees plaintext
  or plaintext hashes.
- **D-0xx Blob adapter for SliceCheckpointer.** `EngineBackend` maps `put(key, bytes)` to one committed
  manifest whose single shard is the blob (`extra.key = key`). `get(key)` restores the newest manifest
  carrying the key, after full verification. This keeps `SliceCheckpointer` and the trainer unchanged.
- **D-0xx P2 implementation lives in `ckpt/crypto.py`** (data path), re-exported as `security/envelope.py`.
  Following the normative P2 (not a per-chunk key), there is one AES-256-GCM DEK per (job, key epoch),
  wrapped by the KEK with AD = job‖epoch. Each chunk uses a fresh 96-bit nonce and AD = chunk_id‖epoch.
  The manifest binds the chunk list through its Ed25519 signature (a deduplicated chunk belongs to many
  manifests, so the chunk AD cannot name a single one), and it carries every wrapped DEK its chunks need,
  so old epochs stay readable after `rotate_dek`.

## Proposed BLOCKERS rows

| # | Date | What failed | Fallback used | Impact on claims |
|---|------|-------------|---------------|------------------|
| B-0xx | 2026-10-24 | Private crate mirror quarantines `cc` 1.10.0 (HTTP 403) | `Cargo.lock` pins `cc` 1.2.x | None |
| B-0xx | 2026-10-24 | `librocksdb-sys` bindgen needs libclang, which is off the default dyld path on macOS | Makefile sets `LIBCLANG_PATH`/`DYLD_FALLBACK_LIBRARY_PATH` for cargo on Darwin | None |
| B-0xx | 2026-10-24 | No raw block device; ext4/XFS are loop-mounted image files on the Docker Desktop VM's virtual disk (APFS-backed) | Loopback images inside the privileged container; device bytes from `/sys/block/loopN/stat` | Absolute throughput/latency include the VM's virtio and host-APFS layers. fsync durability is only as good as Docker Desktop's flush semantics. Relative comparisons hold |
| B-0xx | 2026-10-24 | Host shared with a concurrent timing-sensitive campaign (load avg 50–60) | `nice -n 19`, `--cpus 4`, seeded interleaving, `small` preset instead of `base` for the benchmark matrix | Absolute numbers pessimistic, CIs wide |

## Proposed results/INDEX.md rows

| Command | Output | Finding |
|---|---|---|
| `python -m bench.m4_crash --kill-trials 1000 --torn-trials 500 --flip-trials 500 --seed 0` (lab) | `results/m4/crash_{kill9,torn,bitflip}.json` | 2000 fault trials, 0 undetected corruptions. kill -9 always recovers ≥ last acked checkpoint; every WAL/segment truncation or flip is detected |
| `nice -n 19 python -m bench.m4_bench --trials 5 --ckpts 6 --steps 1 --preset small --out m4` (lab) | `results/m4/bench.json`, `bench_raw.json` | Async engine save stalls training 0.012–0.13 s/ckpt vs 0.05–0.18 s torch.save and 0.11–0.65 s DCP. Durable latency and recovery are slower than torch.save at ~50 MB. Device write amp 0.79 with frozen layers on XFS |
| `nice -n 19 uv run python -m bench.m4_dedup --seeds 5 --ckpts 10 --every 5 --preset base` | `results/m4/dedup.json` | Dedup across successive checkpoints: 1.00 (full fine-tune), 1.29 (half frozen), 2.65 (adapter). Insensitive to chunk size for per-tensor chunking |
| `nice -n 19 python -m bench.m5_envelope --trials 10 --preset base` (lab) | `results/m5/envelope_overhead.json` | P2 auth+decrypt makes a 174 MB restore 1.29× slower (1.07 → 1.38 s cold), all of it in verify (1.76×) |
| `uv run python -m bench.m4_figs` | `results/m4/figs/*.png`, `figs/TABLES.md` | Figures and tables for all of the above |

## Honest limitations

- **kill -9 is not power loss.** The kill campaign kills only the server process, so the page cache
  survives. Lost or garbled bytes are covered separately by the torn-write campaign, which truncates a
  random file of a *quiescent* store at a random offset (this includes fsync'd data, hence the many
  "detected" outcomes), and by the bit-flip campaign. No campaign models a disk that lies about fsync.
- **Torn writes are injected after the fact** (truncate a random file at a random point), not at the
  exact moment of an in-flight write. The alternative engine on `m4-alt-engine-a21` has a fsync
  durability log meant for in-flight torn-tail injection. It was not used for the reported numbers.
- **The "full" scenario cannot dedup.** AdamW updates every parameter and both moments each step, so
  successive full-fine-tune checkpoints share almost no chunks. Incrementality only appears for frozen
  parameters (the "frozen" scenario: embeddings plus the lower half of the blocks).
- **Small model, CPU only.** The benchmark matrix uses the `small` TinyVLM preset (state 39–59 MB),
  because `base` took ~7 min per run at host load ≈ 60 (that first attempt was stopped and discarded).
  Per-checkpoint fsync, process and Python overheads weigh more at this size than they would for GB-scale
  states, so the durable-latency comparison with torch.save says little about large models. The dedup and
  envelope runs use `base` (state 174 MB).
- **ext4 vs XFS is confounded with load drift.** The two filesystems ran one after the other while host
  load fell (container load 13–19 during ext4, 6–15 during XFS). Methods were interleaved only within a
  filesystem.
- **The CIs are wide** (n = 5 under shared load). Several engine-vs-baseline differences listed above are
  within noise, and the notes say so where they are.
- An incomplete, earlier `ckpt_bench.json` from the previous session's worktree (synthetic MLP workload,
  ext4/frozen only) was **not** copied: it was superseded by `bench.json`, which uses the real model, both
  filesystems and both scenarios.
- **Write amplification is device-level** (loop device sectors written, including fs journal/metadata).
  It is below 1 for the engine because dedup removes repeated bytes before they are written.
- The FUSE adapter was skipped (optional).
- `EngineBackend.get(key)` finds a key by scanning manifest headers (cached). That is O(#retained
  manifests) on first use, which is fine with GC retention but not with unbounded histories.
- Anti-rollback state (`trusted` hwm per key, or `trusted_hwm`/`trusted_anchor` in `restore`) lives in
  the client process. P2 places it at the controller, and that wiring is not done.
