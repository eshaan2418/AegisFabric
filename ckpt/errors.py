class CkptError(Exception):
    """Base class for checkpoint errors."""


class ServerError(CkptError):
    """Server-side I/O or protocol error."""


class InvalidRequest(CkptError):
    """Server rejected a malformed or out-of-order request (e.g. non-monotonic seq)."""


class NotFoundError(CkptError):
    """Requested manifest/chunk does not exist (or referenced chunks are not durable)."""


class IntegrityError(CkptError):
    """A *detected* integrity violation: checksum/tag mismatch, missing chunk, corrupt WAL, ..."""


class SignatureError(IntegrityError):
    """Manifest signature or server/manifest header mismatch."""


class RollbackError(IntegrityError):
    """Manifest older than the trusted high-water mark, or broken hash chain (rollback/replay)."""
