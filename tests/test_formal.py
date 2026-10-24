"""Model-check the small TLA+ configs (skipped when java or formal/tools/tla2tools.jar is missing)."""

import sys
from pathlib import Path

import pytest

# formal/ is not an installed package (see pyproject packages list); import it from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formal import run_tlc  # noqa: E402
from formal.models import MODELS  # noqa: E402

JAVA = run_tlc.find_java()
pytestmark = [
    pytest.mark.slow,  # ~85 s; run with `make test-formal`
    pytest.mark.skipif(
        not JAVA or not run_tlc.JAR.exists(), reason="java or tla2tools.jar missing (run formal/get_tools.sh)"
    ),
]


@pytest.mark.parametrize("model", [m for m in MODELS if m.small], ids=lambda m: m.name)
def test_model_outcome(model, tmp_path):
    model.cfg.write_text(model.render())
    r = run_tlc.run(model, JAVA, workers=2, out_dir=tmp_path)
    assert r["outcome"] == model.expect, (r, (tmp_path / f"{model.name}.out").read_text()[-3000:])
    if model.expect == "violation":
        assert r["trace_len"] > 0
