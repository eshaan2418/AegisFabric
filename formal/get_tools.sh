#!/usr/bin/env bash
# Download the pinned TLA+ tools jar into formal/tools/ (gitignored) and verify its SHA-256.
set -euo pipefail
VERSION="v1.8.0"
URL="https://github.com/tlaplus/tlaplus/releases/download/${VERSION}/tla2tools.jar"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/tools"
JAR="$HERE/tools/tla2tools.jar"
EXPECTED="$(cat "$HERE/tla2tools.sha256" 2>/dev/null || true)"
[ -f "$JAR" ] || curl -sSfL -o "$JAR" "$URL"
ACTUAL=$(shasum -a 256 "$JAR" | awk '{print $1}')
if [ -n "$EXPECTED" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "tla2tools.jar SHA-256 mismatch: got $ACTUAL expected $EXPECTED" >&2
  exit 1
fi
echo "tla2tools.jar $VERSION sha256=$ACTUAL"
