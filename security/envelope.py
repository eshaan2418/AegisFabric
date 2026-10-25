"""Checkpoint envelope encryption / authenticity (protocol P2) — security-side entry point.

The implementation lives with the checkpoint client in `ckpt/crypto.py` (it is used on the data path
by `ckpt.client.CheckpointClient`); this module re-exports it so security code and docs can reference
one place. Summary of what it does (see docs/protocols.md P2):

* `KeyService`: holds the KEK (lab stand-in for a KMS/HSM); wraps per-epoch DEKs with AES-256-GCM,
  associated data = job + key epoch.
* `CkptCrypto`: chunk_id = BLAKE3-keyed(job MAC key, plaintext); chunk ciphertext =
  epoch || nonce || AES-256-GCM(DEK_epoch, plaintext, aad = chunk_id || epoch); Ed25519 manifest
  signatures with a pinned verify key; DEK rotation (`rotate_dek`) keeps older epochs readable.
* Restore-side checks (in `CheckpointClient.restore`): signature, server header vs signed manifest,
  seq vs trusted high-water mark, prev-manifest-hash chain, GCM tag and recomputed chunk id per chunk.

Negative tests: tests/test_ckpt_client.py (tampered chunk, tampered manifest, replayed manifest,
rollback, wrong keys, DEK rotation). Overhead: bench/m5_envelope.py -> results/m5/envelope_overhead.json.
Only `cryptography` (AES-GCM, Ed25519) and `blake3` (keyed mode) primitives are used.
"""

from ckpt.crypto import CkptCrypto, KeyService
from ckpt.errors import IntegrityError, RollbackError, SignatureError

__all__ = ["CkptCrypto", "IntegrityError", "KeyService", "RollbackError", "SignatureError"]
