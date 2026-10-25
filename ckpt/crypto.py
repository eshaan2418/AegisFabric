"""Protocol P2 (docs/protocols.md): envelope encryption, keyed chunk ids, signed manifests.

Only established primitives from `cryptography` (AES-256-GCM, Ed25519) and `blake3` (keyed mode) are
used; this module composes them, it implements no primitive.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import blake3
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ckpt.errors import IntegrityError, SignatureError

NONCE = 12


class KeyService:
    """Stand-in for a KMS/HSM: holds the KEK and wraps/unwraps data keys. The KEK never leaves it."""

    def __init__(self, kek: bytes | None = None):
        self._kek = AESGCM(kek or AESGCM.generate_key(bit_length=256))

    def wrap(self, key: bytes, context: bytes) -> bytes:
        nonce = os.urandom(NONCE)
        return nonce + self._kek.encrypt(nonce, key, context)

    def unwrap(self, wrapped: bytes, context: bytes) -> bytes:
        try:
            return self._kek.decrypt(wrapped[:NONCE], wrapped[NONCE:], context)
        except InvalidTag as e:
            raise IntegrityError("DEK unwrap failed (wrong KEK or tampered wrapped key)") from e


@dataclass
class CkptCrypto:
    job: str
    kms: KeyService
    verify_key: Ed25519PublicKey  # pinned in trust config
    mac_key: bytes  # 32 bytes, keyed-BLAKE3 chunk ids
    signing_key: Ed25519PrivateKey | None = None  # None for restore-only parties
    key_epoch: int = 1
    _deks: dict[int, bytes] = field(default_factory=dict, repr=False)
    _wrapped: dict[int, bytes] = field(default_factory=dict, repr=False)

    @classmethod
    def generate(cls, job: str, kms: KeyService | None = None) -> CkptCrypto:
        sk = Ed25519PrivateKey.generate()
        c = cls(
            job=job,
            kms=kms or KeyService(),
            verify_key=sk.public_key(),
            mac_key=os.urandom(32),
            signing_key=sk,
        )
        c._new_dek(1)
        return c

    def _ctx(self, epoch: int) -> bytes:
        return b"aegis-dek|" + self.job.encode() + b"|" + struct.pack("<I", epoch)

    def _new_dek(self, epoch: int) -> None:
        dek = AESGCM.generate_key(bit_length=256)
        self._deks[epoch] = dek
        self._wrapped[epoch] = self.kms.wrap(dek, self._ctx(epoch))
        self.key_epoch = epoch

    def rotate_dek(self) -> int:
        self._new_dek(self.key_epoch + 1)
        return self.key_epoch

    def wrapped_deks(self, epochs: set[int] | None = None) -> dict[int, bytes]:
        return {e: w for e, w in self._wrapped.items() if epochs is None or e in epochs}

    def dek(self, epoch: int, wrapped: bytes | None = None) -> AESGCM:
        if epoch not in self._deks:
            if wrapped is None:
                raise IntegrityError(f"no DEK for key epoch {epoch}")
            self._deks[epoch] = self.kms.unwrap(wrapped, self._ctx(epoch))
            self._wrapped[epoch] = wrapped
        return AESGCM(self._deks[epoch])

    # --- chunks --------------------------------------------------------------------------------
    def chunk_id(self, plaintext: bytes | memoryview) -> bytes:
        return blake3.blake3(plaintext, key=self.mac_key).digest()

    def encrypt_chunk(self, cid: bytes, plaintext: bytes | memoryview) -> bytes:
        ep = struct.pack("<I", self.key_epoch)
        nonce = os.urandom(NONCE)
        return ep + nonce + self.dek(self.key_epoch).encrypt(nonce, bytes(plaintext), cid + ep)

    def decrypt_chunk(self, cid: bytes, blob: bytes, wrapped: dict[int, bytes]) -> bytes:
        if len(blob) < 4 + NONCE + 16:
            raise IntegrityError("chunk ciphertext too short")
        (epoch,) = struct.unpack("<I", blob[:4])
        aes = self.dek(epoch, wrapped.get(epoch))
        try:
            pt = aes.decrypt(blob[4 : 4 + NONCE], blob[4 + NONCE :], cid + blob[:4])
        except InvalidTag as e:
            raise IntegrityError("chunk AES-GCM tag mismatch") from e
        if self.chunk_id(pt) != cid:
            raise IntegrityError("chunk id does not match plaintext")
        return pt

    # --- manifests -----------------------------------------------------------------------------
    def sign(self, body: bytes) -> bytes:
        if self.signing_key is None:
            raise SignatureError("no signing key")
        return self.signing_key.sign(body)

    def verify(self, sig: bytes, body: bytes) -> None:
        try:
            self.verify_key.verify(sig, body)
        except InvalidSignature as e:
            raise SignatureError("manifest signature invalid") from e


# --- job key material on disk (lab stand-in for a KMS + secret distribution) ---------------------
def init_keydir(path, job: str) -> None:
    """Create the job's key material once: KEK (stands in for the KMS), keyed-hash key, Ed25519 manifest
    signing key, and the epoch-1 DEK *wrapped* by the KEK. Every rank of every segment of the job loads the
    same material, so a relaunched job (migration, crash restart) can decrypt and verify earlier checkpoints."""
    from pathlib import Path

    from cryptography.hazmat.primitives import serialization

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    c = CkptCrypto.generate(job, KeyService(kek := AESGCM.generate_key(bit_length=256)))
    files = {
        "kek.bin": kek,
        "mac.bin": c.mac_key,
        "sign.pem": c.signing_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
        "dek1.wrapped": c.wrapped_deks()[1],
        "job.txt": job.encode(),
    }
    for name, data in files.items():
        fd = os.open(p / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)


def load_keydir(path, job: str) -> CkptCrypto:
    from pathlib import Path

    from cryptography.hazmat.primitives import serialization

    p = Path(path)
    if (p / "job.txt").read_bytes().decode() != job:
        raise IntegrityError(f"key directory {p} belongs to another job")
    sk = serialization.load_pem_private_key((p / "sign.pem").read_bytes(), password=None)
    c = CkptCrypto(
        job=job,
        kms=KeyService((p / "kek.bin").read_bytes()),
        verify_key=sk.public_key(),
        mac_key=(p / "mac.bin").read_bytes(),
        signing_key=sk,
    )
    c.dek(1, (p / "dek1.wrapped").read_bytes())  # unwrap through the KMS stand-in
    c.key_epoch = 1
    return c
