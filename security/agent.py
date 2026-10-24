"""Workload-side control agent: periodic Report() over mTLS with self-renewing identity (via Renew())."""

from __future__ import annotations

import datetime as dt
import threading
import time

import grpc

from security import control_pb2 as pb
from security import control_pb2_grpc as pbg
from security.ca import Issued, TrustBundle
from security.mtls import Identity, RotatingChannel


class Reporter:
    def __init__(
        self,
        target: str,
        bootstrap: Issued,
        bundle_pem: bytes,
        ttl_s: float,
        job: str,
        rank: int,
        server_name: str = "controller.aegis.lab",
        interval_s: float = 1.0,
        secure: bool = True,
    ):
        self.job, self.rank, self.interval_s = job, rank, interval_s
        self.bundle = TrustBundle([bundle_pem])
        self.secure = secure
        self.latest: dict = {}
        self.directive = pb.Directive(stop_at_step=-1)
        self.errors: list[str] = []
        self.rpc_lat: list[tuple[float, float]] = []  # (wall, seconds)
        self._stop = threading.Event()
        if secure:
            self._boot = bootstrap
            self.identity = Identity(self._renew, ttl_s)
            self.chan = RotatingChannel(target, self.identity, self.bundle, server_name)
        else:
            self.chan = None
            self.plain = grpc.insecure_channel(target)

    def _renew(self) -> Issued:
        if not hasattr(self, "identity"):
            return self._boot  # first identity: bootstrap credential (stand-in for SPIRE attestation)
        stub = pbg.ControlStub(self.chan.get())
        rep = stub.Renew(pb.RenewRequest(spiffe_id=self._boot.spiffe), timeout=10)
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        cas = x509.load_pem_x509_certificates(rep.bundle_pem)
        with self.bundle.lock:
            self.bundle.cas = [c.public_bytes(serialization.Encoding.PEM) for c in cas]

        return Issued(
            rep.cert_pem,
            rep.key_pem,
            b"",
            dt.datetime.now(dt.UTC),
            dt.datetime.fromtimestamp(rep.not_after, dt.UTC),
            self._boot.spiffe,
        )

    def start(self) -> None:
        if self.secure:
            self.identity.start()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self.secure:
            self.identity.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            lat = self.latest
            req = pb.StepReport(
                job=self.job,
                rank=self.rank,
                step=int(lat.get("step", 0)),
                step_s=float(lat.get("step_s", 0.0)),
                coll_s=float(lat.get("coll_s", 0.0)),
                wall=time.time(),
            )
            t0 = time.perf_counter()
            try:
                ch = self.chan.get() if self.secure else self.plain
                self.directive = pbg.ControlStub(ch).Report(req, timeout=5)
                self.rpc_lat.append((time.time(), time.perf_counter() - t0))
            except grpc.RpcError as e:
                self.errors.append(f"{time.time():.3f} {e.code()} {e.details()}")
