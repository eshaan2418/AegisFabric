"""Fail if tracked/staged files contain credentials (URL-embedded passwords, private keys, tokens).

Test PKI material is generated at runtime into temp dirs and never committed."""

from __future__ import annotations

import re
import subprocess
import sys

from bench.meta import REPO_ROOT

PATTERNS = [
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),  # user:pass@host
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:password|passwd|secret|token)\s*[=:]\s*['\"][^'\"]{12,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
SKIP_SUFFIX = (".png", ".pdf", ".npy", ".lock", ".jpg", ".ivf", ".y4m", ".mp4")


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.split()
    bad = []
    for f in files:
        if f.endswith(SKIP_SUFFIX) or f == "bench/secret_scan.py":
            continue
        try:
            text = (REPO_ROOT / f).read_text(errors="ignore")
        except (FileNotFoundError, IsADirectoryError):
            continue
        for pat in PATTERNS:
            for m in pat.finditer(text):
                bad.append(f"{f}: {m.group(0)[:40]}...")
    if bad:
        print("secret_scan: possible credentials found:\n  " + "\n  ".join(bad))
        return 1
    print(f"secret_scan: {len(files)} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
