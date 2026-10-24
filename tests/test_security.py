import datetime as dt
import socket
import ssl
import time

import grpc
import pytest

from security import control_pb2 as pb
from security import control_pb2_grpc as pbg
from security.agent import Reporter
from security.ca import CA, TrustBundle, cert_spiffe, spiffe_id
from security.mtls import Identity, RotatingChannel
from security.service import ControlPlane


@pytest.fixture()
def cp():
    c = ControlPlane(bind="127.0.0.1:0", ttl_s=3.0, skew_s=0.2, server_ips=["127.0.0.1"])
    c.start()
    yield c
    c.stop()


def _stub(cp, issued, bundle_pem=None):
    ident = Identity(lambda: issued, ttl_s=3600)
    ch = RotatingChannel(
        f"127.0.0.1:{cp.port}", ident, TrustBundle([bundle_pem or cp.ca.cert_pem]), cp.server_name
    )
    return pbg.ControlStub(ch.get())


def test_spiffe_san_and_backdating():
    ca = CA(skew_s=1.5)
    iss = ca.issue(spiffe_id("tor", "trainer", "0"), ttl_s=60)
    assert cert_spiffe(iss.cert_pem) == "spiffe://aegis.lab/tor/trainer/0"
    assert (dt.datetime.now(dt.UTC) - iss.not_before).total_seconds() >= 3.0 - 0.5  # 2S backdating


def test_valid_client_reports_and_gets_directive(cp):
    stub = _stub(cp, cp.ca.issue(spiffe_id("tor", "trainer", "0"), 60))
    cp.servicer.set_directive("j", 42)
    d = stub.Report(pb.StepReport(job="j", rank=0, step=1), timeout=5)
    assert d.stop_at_step == 42


def test_expired_cert_rejected(cp):
    past = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    stub = _stub(cp, cp.ca.issue(spiffe_id("tor", "trainer", "0"), 60, now=past))
    with pytest.raises(grpc.RpcError) as e:
        stub.Report(pb.StepReport(job="j"), timeout=3)
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


def test_unknown_ca_rejected(cp):
    rogue = CA(name="rogue")
    stub = _stub(cp, rogue.issue(spiffe_id("tor", "trainer", "0"), 60))
    with pytest.raises(grpc.RpcError) as e:
        stub.Report(pb.StepReport(job="j"), timeout=3)
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


def test_wrong_trust_domain_san_denied(cp):
    stub = _stub(cp, cp.ca.issue("spiffe://evil.example/tor/trainer/0", 60))
    with pytest.raises(grpc.RpcError) as e:
        stub.Report(pb.StepReport(job="j"), timeout=3)
    assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_cannot_renew_someone_elses_identity(cp):
    stub = _stub(cp, cp.ca.issue(spiffe_id("tor", "trainer", "0"), 60))
    with pytest.raises(grpc.RpcError) as e:
        stub.Renew(pb.RenewRequest(spiffe_id=spiffe_id("us", "trainer", "5")), timeout=3)
    assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_server_negotiates_tls13(cp):
    iss = cp.ca.issue(spiffe_id("tor", "probe", "0"), 60)
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    (d / "c.pem").write_bytes(iss.cert_pem)
    (d / "k.pem").write_bytes(iss.key_pem)
    (d / "ca.pem").write_bytes(cp.ca.cert_pem)
    ctx = ssl.create_default_context(cafile=str(d / "ca.pem"))
    ctx.load_cert_chain(str(d / "c.pem"), str(d / "k.pem"))
    ctx.set_alpn_protocols(["h2"])
    with socket.create_connection(("127.0.0.1", cp.port)) as s:
        with ctx.wrap_socket(s, server_hostname=cp.server_name) as t:
            assert t.version() == "TLSv1.3"


def test_live_rotation_no_failed_rpcs(cp):
    boot = cp.ca.issue(spiffe_id("tor", "trainer", "0"), 3.0)
    r = Reporter(f"127.0.0.1:{cp.port}", boot, cp.ca.cert_pem, ttl_s=3.0, job="j", rank=0, interval_s=0.05)
    r.start()
    time.sleep(0.5)
    cp.begin_ca_rotation()  # phase 1: new CA distributed to every verifier via renewals
    time.sleep(3.0 + 0.4)  # >= T + 2S
    old = cp.commit_ca_rotation()  # phase 2: new CA signs
    time.sleep(3.0 + 0.4)  # all certs from the old CA expired (+2S)
    cp.retire_ca(old)
    time.sleep(2.0)
    r.stop()
    assert old.cert_pem not in cp.bundle.cas
    assert r.errors == []
    assert len(r.identity.renewals) >= 2
    assert len(r.rpc_lat) > 50
