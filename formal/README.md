# formal/ — TLA+ models of P1 (checkpoint store) and P3 (cert rotation)

```
bash formal/get_tools.sh          # pinned tla2tools.jar v1.8.0, SHA-256 in formal/tla2tools.sha256
bash formal/check.sh              # all models -> results/m9/<model>.out + results/m9/summary.json
uv run pytest tests/test_formal.py  # small models only; skipped if java/jar missing
```

`formal/models.py` is the registry: each model has a spec, constants, invariants, and an
**expected outcome**. Fixed protocols must pass. Buggy variants and sanity (non-vacuity) checks must
produce a counterexample. `check.sh` exits non-zero if any outcome differs from what was expected.
Traces are kept in the `.out` files. The `.cfg` files are generated from `models.py`.

## P1 — `ckpt/CkptStore.tla`

Modelled: chunk PUT with dedup, append to the active segment, group-commit fsync, index insert
only after fsync, ack, client resend on a missing ack, COMMIT with the all-chunks-indexed check and
reject→re-upload, WAL append and fsync, commit ack after the WAL is durable, and crash at any point.
A crash keeps an arbitrary subset of un-fsync'd chunks and an arbitrary prefix of un-fsync'd WAL
records, and may leave one torn WAL record. Also modelled: recovery (truncating the torn tail,
dropping index entries without data, reporting the latest manifest), and mark/sweep GC with epochs,
retaining K manifests, concurrent with uploads and commits.

Invariants: `I1` (retained committed manifests reference only durable, indexed chunks), `I2`
(recovery returns the latest committed manifest and never needs the error path), `I3` (GC never
removes chunks of a retained committed manifest or of an accepted, not-yet-fsync'd commit),
`AckedDurable` (no acknowledged commit is lost), and liveness `AllDone` (under weak fairness every
checkpoint is eventually committed and acked).

Bounds: 3 chunks, 2 manifests sharing one chunk, K = 1. The small model has 1 crash and 1 GC
cycle; `p1_fixed_2crash_2gc` has 2 crashes and 2 GC cycles (safety only).

Abstractions (not checked): one logical copy per chunk id, so segment compaction is atomic removal of
the dead copy. RocksDB writes are synchronous and atomic. BLAKE3/CRC detection of torn or corrupt
records is assumed perfect, so `I2`'s "never partial" holds by construction, and what TLC checks is
that the error branch is never reached. Bit-flips in durable data, multiple clients per manifest,
manifest signing (P2) and network partitions are not modelled. Multi-chunk records are modelled as
single atomic chunk writes.

## P3 — `rotation/CertRotation.tla`

Modelled: discrete real time `now`, per-node clock offsets in `[-S, S]` relative to real time, chosen
nondeterministically at start (CA, workload, 2 verifiers). Also: renewal when workload-local time
≥ `notAfter − T/RenewDiv`, issuance latency in `1..Lmax`, up to `MaxFail` failed issuances with
retry back-off `R`, and CA rotation (publish ca2, propagate to each verifier within P = 2 ticks,
switch issuer, remove ca1 once CA-local time ≥ max notAfter(ca1) + S, propagate the removal). Urgent
actions must fire before time advances. The workload presents its newest cert.

Invariant `R1`: every verifier accepts the presented cert at every state. `R1Weak` only requires
that *some* held cert is accepted.

Bounds: T = 10 (T = 16 for the 2-failure model), S = 1, Lmax = 1, R = 1, MaxFail = 1 (or 2), and a
horizon of 26–36 ticks, which covers ≥ 2 renewals and a full CA rotation (non-vacuity is checked by
`p3_sanity_reachable`).

Abstractions: clock drift is not modelled (offsets are fixed per run). There is one workload, and
connection-level behaviour (TLS session reuse) is out of scope. Revocation is not modelled.

## Notes on numbers

`distinct_states` for passing models is deterministic. For violating models, TLC stops at the first
counterexample, so the counts depend on worker scheduling.
