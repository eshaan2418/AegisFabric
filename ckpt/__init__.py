"""Python side of the AegisFabric checkpoint engine.

- `ckpt.wire`    : binary TCP protocol client for the Rust `aegis-ckpt` server
- `ckpt.crypto`  : P2 envelope encryption, keyed chunk ids, Ed25519 manifests (docs/protocols.md)
- `ckpt.client`  : `CheckpointClient` (snapshot → background flush, verified restore)
- `ckpt.server`  : helpers to launch / kill the server process
"""

from ckpt.errors import (
    CkptError,
    IntegrityError,
    NotFoundError,
    RollbackError,
    ServerError,
    SignatureError,
)

__all__ = ["CkptError", "IntegrityError", "NotFoundError", "RollbackError", "ServerError", "SignatureError"]
