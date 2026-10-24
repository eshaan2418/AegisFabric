"""mTLS gRPC plumbing with live certificate rotation (no restarts, no dropped RPCs).

Server: `grpc.dynamic_ssl_server_credentials` with a fetcher, so every *new* handshake picks up the
current server cert and trust bundle. Established HTTP/2 connections are unaffected.
Client: `RotatingChannel` builds a new channel whenever the workload's identity changes, swaps it in
atomically, and closes the old one after a grace period, so in-flight RPCs finish on the old connection.
Authorization: a server interceptor maps the peer's SPIFFE ID (URI SAN of the verified client cert) to
the set of methods it may call.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable

import grpc

from security.ca import Issued, TrustBundle, cert_spiffe


class Identity:
    """Holds the current workload cert; renews at T/2 ± jitter via `issue_fn` (retrying on failure)."""

    def __init__(
        self, issue_fn: Callable[[], Issued], ttl_s: float, jitter: float = 0.1, retry_s: float = 0.5
    ):
        self.issue_fn = issue_fn
        self.ttl_s = ttl_s
        self.jitter = jitter
        self.retry_s = retry_s
        self.current = issue_fn()
        self.version = 0
        self.lock = threading.Lock()
        self.renewals: list[dict] = []
        self.failures = 0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.listeners: list[Callable[[Issued], None]] = []

    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait = self.ttl_s / 2 * (1 + random.uniform(-self.jitter, self.jitter))
            if self._stop.wait(wait):
                return
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    new = self.issue_fn()
                except Exception:
                    self.failures += 1
                    self._stop.wait(self.retry_s)
                    continue
                with self.lock:
                    self.current = new
                    self.version += 1
                self.renewals.append(
                    {
                        "wall": time.time(),
                        "issue_s": time.perf_counter() - t0,
                        "not_after": new.not_after.timestamp(),
                    }
                )
                for cb in self.listeners:
                    cb(new)
                break

    def get(self) -> tuple[int, Issued]:
        with self.lock:
            return self.version, self.current


def server_credentials(identity: Identity, bundle: TrustBundle) -> grpc.ServerCredentials:
    def config():
        _, cur = identity.get()
        return grpc.ssl_server_certificate_configuration(
            [(cur.key_pem, cur.cert_pem)], root_certificates=bundle.pem()
        )

    return grpc.dynamic_ssl_server_credentials(config(), config, require_client_authentication=True)


class SpiffeAuthz(grpc.ServerInterceptor):
    """policy: method-suffix -> list of allowed SPIFFE-ID prefixes."""

    def __init__(self, policy: dict[str, list[str]]):
        self.policy = policy
        self.denied = 0

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        method = handler_call_details.method.rsplit("/", 1)[-1]
        allowed = self.policy.get(method, [])
        if handler is None or handler.unary_unary is None:
            return handler
        inner = handler.unary_unary

        def wrapper(request, context):
            pem = context.auth_context().get("x509_pem_cert", [b""])[0]
            sid = cert_spiffe(pem) if pem else None
            if sid is None or not any(sid.startswith(p) for p in allowed):
                self.denied += 1
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, f"spiffe id {sid!r} not allowed for {method}"
                )
            context.spiffe_id = sid  # type: ignore[attr-defined]
            return inner(request, context)

        return grpc.unary_unary_rpc_method_handler(
            wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def peer_spiffe(context) -> str | None:
    pem = context.auth_context().get("x509_pem_cert", [b""])[0]
    return cert_spiffe(pem) if pem else None


class RotatingChannel:
    def __init__(
        self, target: str, identity: Identity, bundle: TrustBundle, server_name: str, grace_s: float = 5.0
    ):
        self.target, self.identity, self.bundle, self.server_name = target, identity, bundle, server_name
        self.grace_s = grace_s
        self.lock = threading.Lock()
        self.version = -1
        self.channel: grpc.Channel | None = None
        self.handshakes: list[float] = []
        self._ensure()

    def _build(self, cur: Issued) -> grpc.Channel:
        creds = grpc.ssl_channel_credentials(
            root_certificates=self.bundle.pem(), private_key=cur.key_pem, certificate_chain=cur.cert_pem
        )
        return grpc.secure_channel(
            self.target,
            creds,
            options=[
                ("grpc.ssl_target_name_override", self.server_name),
                ("grpc.use_local_subchannel_pool", 1),
            ],
        )

    def _ensure(self) -> grpc.Channel:
        v, cur = self.identity.get()
        with self.lock:
            if v != self.version or self.channel is None:
                old = self.channel
                ch = self._build(cur)
                t0 = time.perf_counter()
                try:
                    grpc.channel_ready_future(ch).result(timeout=10)
                    self.handshakes.append(time.perf_counter() - t0)
                except grpc.FutureTimeoutError:
                    pass  # surface as RPC error on use
                self.channel, self.version = ch, v
                if old is not None:
                    threading.Timer(self.grace_s, old.close).start()
            return self.channel

    def get(self) -> grpc.Channel:
        return self._ensure()


def insecure_channel(target: str) -> grpc.Channel:
    return grpc.insecure_channel(target)
