"""Test certificate authority and workload identities (protocol P3 in docs/protocols.md).

Uses only the `cryptography` library (no custom crypto). Keys are ECDSA P-256. Workload certs carry a
SPIFFE ID as a URI SAN (spiffe://aegis.lab/<site>/<role>/<id>) and have short lifetimes. notBefore is
backdated by 2·S (S = max modeled clock skew), per the fix TLC found in M9.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import threading
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TRUST_DOMAIN = "aegis.lab"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def spiffe_id(site: str, role: str, ident: str) -> str:
    return f"spiffe://{TRUST_DOMAIN}/{site}/{role}/{ident}"


@dataclasses.dataclass
class Issued:
    cert_pem: bytes
    key_pem: bytes
    chain_pem: bytes  # issuing CA cert
    not_before: dt.datetime
    not_after: dt.datetime
    spiffe: str


class CA:
    def __init__(self, name: str = "aegis-root", lifetime_days: int = 30, skew_s: float = 1.0):
        self.key = ec.generate_private_key(ec.SECP256R1())
        now = _now()
        subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        self.cert = (
            x509.CertificateBuilder()
            .subject_name(subj)
            .issuer_name(subj)
            .public_key(self.key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=lifetime_days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(self.key, hashes.SHA256())
        )
        self.skew_s = skew_s
        self.lock = threading.Lock()
        self.issued = 0

    @property
    def cert_pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    def issue(
        self,
        spiffe: str,
        ttl_s: float,
        dns: list[str] | None = None,
        ips: list[str] | None = None,
        now: dt.datetime | None = None,
    ) -> Issued:
        import ipaddress

        now = now or _now()
        key = ec.generate_private_key(ec.SECP256R1())
        nb = now - dt.timedelta(seconds=2 * self.skew_s)
        na = now + dt.timedelta(seconds=ttl_s)
        sans: list[x509.GeneralName] = [x509.UniformResourceIdentifier(spiffe)]
        sans += [x509.DNSName(d) for d in (dns or [])]
        sans += [x509.IPAddress(ipaddress.ip_address(i)) for i in (ips or [])]
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, spiffe.rsplit("/", 1)[-1])]))
            .issuer_name(self.cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(nb)
            .not_valid_after(na)
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .sign(self.key, hashes.SHA256())
        )
        with self.lock:
            self.issued += 1
        return Issued(
            cert.public_bytes(serialization.Encoding.PEM),
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
            self.cert_pem,
            nb,
            na,
            spiffe,
        )


class TrustBundle:
    """Set of accepted CA certs. CA rotation: add the new CA everywhere *before* it signs anything; remove
    the old one only after all its certs expired (+ skew)."""

    def __init__(self, cas: list[bytes] | None = None):
        self.cas: list[bytes] = list(cas or [])
        self.lock = threading.Lock()

    def add(self, pem: bytes) -> None:
        with self.lock:
            if pem not in self.cas:
                self.cas.append(pem)

    def remove(self, pem: bytes) -> None:
        with self.lock:
            self.cas = [c for c in self.cas if c != pem]

    def pem(self) -> bytes:
        with self.lock:
            return b"".join(self.cas)


def cert_spiffe(cert_pem: bytes) -> str | None:
    cert = x509.load_pem_x509_certificate(cert_pem)
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return None
    uris = san.get_values_for_type(x509.UniformResourceIdentifier)
    return next((u for u in uris if u.startswith("spiffe://")), None)


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.chmod(0o600)
    tmp.replace(path)


def load_issued(cert_pem: bytes, key_pem: bytes) -> Issued:
    cert = x509.load_pem_x509_certificate(cert_pem)
    return Issued(
        cert_pem,
        key_pem,
        b"",
        cert.not_valid_before_utc,
        cert.not_valid_after_utc,
        cert_spiffe(cert_pem) or "",
    )
