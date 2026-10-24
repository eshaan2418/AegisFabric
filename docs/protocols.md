# Protocols (normative; implemented in `ckpt/`, `security/`, model-checked in `formal/`)

## P1. Checkpoint commit / recovery (ckpt engine, server side)

On-disk layout under a store root:

- `segments/seg-NNNNNNNN.log`: append-only chunk records
  `MAGIC(4) | len(u32) | chunk_id(32) | blake3(payload)(32) | payload(len) | crc32c(header)(4)`.
- `index/`: RocksDB, `chunk_id → (segment, offset, len, epoch)`. An entry is inserted **only after** the
  segment bytes it points to are fsync'd.
- `wal/manifest.wal`: append-only records `MAGIC | seq(u64) | len(u32) | manifest | blake3(record)(32)`.

A **manifest** is an opaque byte string to the server (client signs/encrypts it, see P2), with a cleartext header
listing the `chunk_id`s it references and a monotonically increasing `seq`.

Commit, from the server's point of view:
1. `PUT_CHUNK(id, bytes)`: if `id` is already indexed, dedup hit → ack. Otherwise append to the active
   segment. Group-commit: fsync the segment, then insert index entries, then ack. Never ack before fsync.
2. `COMMIT(manifest)`: reject unless every referenced `chunk_id` is indexed (durable). Append the WAL
   record and fsync, **then** ack. A checkpoint is *committed* iff its WAL record is durable and its
   checksum is intact.
3. Recovery after a crash: scan the WAL; accept the longest prefix of records with valid checksums; truncate
   a torn tail. The index is cross-checked against the segments, and entries pointing past a segment's valid
   end are dropped. The *latest committed checkpoint* is the last accepted WAL record. `RESTORE` re-verifies
   blake3 of every chunk; any mismatch or missing chunk returns a **detected error** (never silently wrong data).
4. GC: retain the last K committed manifests. Mark their chunks live. Copy live chunks out of victim
   segments into a new segment, fsync, update the index, and only then delete victim segments. Chunks
   indexed in epochs ≥ the GC start epoch are always retained, because they may belong to an in-flight,
   uncommitted checkpoint.

Invariants (checked by tests and TLA+):
- **I1** No committed manifest references a chunk that is not durably stored.
- **I2** Recovery returns either the last committed manifest with all chunks intact, or a detected error.
  It never returns a partially written checkpoint.
- **I3** GC never deletes a chunk referenced by a retained committed manifest, or by an in-flight commit.

## P2. Checkpoint confidentiality and authenticity (client side)

- Per-job **DEK** (AES-256-GCM key), wrapped by a **KEK** (AES-256 key-wrap via AES-GCM with the KEK). In the
  lab the KEK lives in the security service, standing in for a KMS/HSM. DEKs rotate per *key epoch*, and the
  manifest records the wrapped DEK and its epoch.
- `chunk_id = BLAKE3-keyed(job_mac_key, plaintext)`, so dedup works across checkpoints without exposing
  plaintext hashes to the server. Ciphertext = `nonce(12) || AES-GCM(DEK, nonce, plaintext, aad=chunk_id)`.
- Manifest = `{job, seq, prev_manifest_hash, step, created_at, key_epoch, wrapped_dek, shards:[{name, chunks:[id..], len}]}`,
  signed with **Ed25519** (job signing key; public key pinned in the trust config).
- Restore verifies: signature, `seq` ≥ trusted high-water mark (anti-rollback/replay; the high-water mark is
  stored by the controller), `prev_manifest_hash` chain, AES-GCM tag of every chunk, `chunk_id` recomputed
  from plaintext. Any failure means reject.

## P3. Workload certificate rotation (security service)

- A test CA (Python `cryptography`) issues X.509 certs with a SPIFFE URI SAN `spiffe://aegis.lab/<site>/<role>/<id>`
  and lifetime `T`. `notBefore = now − S`, where `S` is the max modeled clock skew.
- Each workload renews at `T/2 ± jitter`, retrying on failure. Old and new certs are both valid during the overlap.
- Peers verify the chain to the trust bundle, the validity window with skew tolerance `S`, and that the SPIFFE ID is
  authorized for the RPC.
- CA rotation: the new CA joins the bundle before it signs anything. The old CA is removed only after every cert it
  issued has expired (+ `S`).
- gRPC servers reload credentials per handshake. Established connections are not torn down by rotation.

Invariant **R1**: under clock skew ≤ `S` and issuance latency + retries < `T/2 − S`, every live workload holds,
at every instant, at least one cert that all peers accept.
