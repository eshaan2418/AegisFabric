"""Controller-side gRPC service: telemetry reports, directives, and workload cert renewal (test CA)."""

from __future__ import annotations

import threading
import time
from concurrent import futures

import grpc

from security import control_pb2 as pb
from security import control_pb2_grpc as pbg
from security.ca import CA, TrustBundle, spiffe_id
from security.mtls import Identity, SpiffeAuthz, peer_spiffe, server_credentials

WORKLOAD_PREFIX = "spiffe://aegis.lab/"
POLICY = {
    "Report": [WORKLOAD_PREFIX],  # any workload in the trust domain may report
    "Renew": [WORKLOAD_PREFIX],  # ...and renew *its own* identity (checked in the handler)
}


class ControlServicer(pbg.ControlServicer):
    def __init__(self, ca_ref: dict, bundle: TrustBundle, ttl_s: float):
        self.ca_ref = ca_ref  # {"ca": CA}, swappable for CA rotation
        self.bundle = bundle
        self.ttl_s = ttl_s
        self.lock = threading.Lock()
        self.latest: dict[tuple[str, int], pb.StepReport] = {}
        self.directives: dict[str, pb.Directive] = {}
        self.reports = 0
        self.renews = 0

    def Report(self, request, context):
        with self.lock:
            self.latest[(request.job, request.rank)] = request
            self.reports += 1
            return self.directives.get(request.job, pb.Directive(stop_at_step=-1))

    def Renew(self, request, context):
        caller = peer_spiffe(context)
        if request.spiffe_id and request.spiffe_id != caller:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "may only renew own identity")
        iss = self.ca_ref["ca"].issue(caller, self.ttl_s)
        with self.lock:
            self.renews += 1
        return pb.RenewReply(
            cert_pem=iss.cert_pem,
            key_pem=iss.key_pem,
            bundle_pem=self.bundle.pem(),
            not_after=iss.not_after.timestamp(),
        )

    def set_directive(self, job: str, stop_at_step: int, final_ckpt: bool = True) -> None:
        with self.lock:
            self.directives[job] = pb.Directive(
                stop_at_step=stop_at_step, final_ckpt=final_ckpt, epoch=int(time.time())
            )


class ControlPlane:
    """CA + controller gRPC server (mTLS, SPIFFE authz)."""

    def __init__(
        self,
        bind: str = "0.0.0.0:7500",
        ttl_s: float = 60.0,
        skew_s: float = 1.0,
        server_name: str = "controller.aegis.lab",
        server_ips: list[str] | None = None,
        secure: bool = True,
    ):
        self.ca = CA(skew_s=skew_s)
        self.ca_ref = {"ca": self.ca}
        self.bundle = TrustBundle([self.ca.cert_pem])
        self.ttl_s = ttl_s
        self.server_name = server_name
        self.server_id = spiffe_id("ctl", "controller", "0")
        self.server_identity = Identity(
            lambda: self.ca_ref["ca"].issue(self.server_id, ttl_s, dns=[server_name], ips=server_ips or []),
            ttl_s,
        )
        self.servicer = ControlServicer(self.ca_ref, self.bundle, ttl_s)
        self.authz = SpiffeAuthz(POLICY)
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=16), interceptors=[self.authz] if secure else []
        )
        pbg.add_ControlServicer_to_server(self.servicer, self.server)
        if secure:
            self.port = self.server.add_secure_port(
                bind, server_credentials(self.server_identity, self.bundle)
            )
        else:
            self.port = self.server.add_insecure_port(bind)
        self.secure = secure

    def start(self) -> None:
        self.server.start()
        if self.secure:
            self.server_identity.start()

    def stop(self) -> None:
        self.server_identity.stop()
        self.server.stop(grace=1)

    def begin_ca_rotation(self) -> CA:
        """Phase 1 (P3): the new CA joins the trust bundle. Workloads pick up the bundle on their next
        renewal, so after one lifetime T every verifier has it. The new CA does NOT sign anything yet."""
        self.next_ca = CA(name="aegis-root-2", skew_s=self.ca.skew_s)
        self.bundle.add(self.next_ca.cert_pem)
        return self.next_ca

    def commit_ca_rotation(self) -> CA:
        """Phase 2: after ≥ T + 2S since phase 1, switch signing to the new CA. Returns the old CA. Remove it
        from the bundle only after its last issued cert expired (+2S); see `retire_ca`."""
        old = self.ca_ref["ca"]
        self.ca_ref["ca"] = self.next_ca
        return old

    def retire_ca(self, old: CA) -> None:
        self.bundle.remove(old.cert_pem)
