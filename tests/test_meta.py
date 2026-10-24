import json

from bench.meta import write_result


def test_write_result_has_provenance(tmp_path):
    p = write_result(tmp_path / "x.json", {"v": 1}, config={"a": 1}, seed=5)
    d = json.loads(p.read_text())
    assert d["data"] == {"v": 1}
    m = d["meta"]
    for k in ("git", "hardware", "seed", "config", "timestamp", "command"):
        assert k in m
    assert m["seed"] == 5
