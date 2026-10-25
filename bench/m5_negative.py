"""M5 negative tests: every attack the security layer must reject, run and recorded as a result file.

    uv run python -m bench.m5_negative            # → results/m5/negative.json

Each case is a pytest test that performs the attack against the real component (mTLS gRPC control plane,
or the checkpoint engine + P2 client) and asserts that it is rejected with the expected error. `rejected` is
true iff that test passed. Cases whose test was skipped (e.g. the engine binary is not built) are recorded as
skipped, not as rejected.
"""

from __future__ import annotations

import pytest

from bench.meta import REPO_ROOT, write_result

CASES = {
    "expired_cert": "tests/test_security.py::test_expired_cert_rejected",
    "unknown_ca": "tests/test_security.py::test_unknown_ca_rejected",
    "wrong_san_trust_domain": "tests/test_security.py::test_wrong_trust_domain_san_denied",
    "renew_other_identity": "tests/test_security.py::test_cannot_renew_someone_elses_identity",
    "tls_below_1_3": "tests/test_security.py::test_server_negotiates_tls13",
    "tampered_chunk": "tests/test_ckpt_client.py::test_tampered_chunk_rejected",
    "gcm_tag_tamper": "tests/test_ckpt_client.py::test_gcm_tamper_detected_client_side",
    "tampered_manifest": "tests/test_ckpt_client.py::test_tampered_manifest_rejected",
    "replayed_manifest": "tests/test_ckpt_client.py::test_replayed_manifest_rejected",
    "rolled_back_manifest": "tests/test_ckpt_client.py::test_rollback_rejected",
    "wrong_keys": "tests/test_ckpt_client.py::test_wrong_keys_rejected",
    "foreign_job_keys": "tests/test_ckpt_backend.py::test_keydir_lets_a_relaunched_job_restore",
}


class _Collect:
    def __init__(self):
        self.outcomes: dict[str, str] = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.outcomes[report.nodeid] = report.outcome


def main() -> int:
    col = _Collect()
    code = pytest.main(["-q", "-p", "no:cacheprovider", "-o", "addopts=", *CASES.values()], plugins=[col])
    rows = {}
    for case, node in CASES.items():
        out = col.outcomes.get(node, "not-run")
        rows[case] = {"test": node, "outcome": out, "rejected": out == "passed"}
    n_rej = sum(r["rejected"] for r in rows.values())
    n_skip = sum(r["outcome"] == "skipped" for r in rows.values())
    write_result(
        "m5/negative.json",
        {
            "cases": rows,
            "n_cases": len(rows),
            "n_rejected": n_rej,
            "n_skipped": n_skip,
            "pytest_exit": int(code),
        },
        config={"cases": CASES, "cwd": str(REPO_ROOT)},
    )
    print(f"{n_rej}/{len(rows)} attacks rejected ({n_skip} skipped)")
    return 0 if n_rej + n_skip == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
