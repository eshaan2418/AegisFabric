import json

import numpy as np

from data.synth import finance as F


def test_statement_deterministic_and_balanced():
    a = F.make_statement(np.random.default_rng(7))
    b = F.make_statement(np.random.default_rng(7))
    assert a == b
    assert a.closing_cents == a.opening_cents + sum(t.amount_cents for t in a.transactions)
    assert a.account_id.startswith("SYN-")
    assert a.currency == F.CURRENCY[a.site]


def test_transcript_consistent_with_statement():
    rng = np.random.default_rng(1)
    st = F.make_statement(rng)
    tr = F.make_transcript(rng, st)
    assert any(F.fmt_money(st.closing_cents) in t["text"] for t in tr)
    assert all(t["end_s"] > t["start_s"] for t in tr)


def test_fmt_money():
    assert F.fmt_money(-12345) == "-123.45"
    assert F.fmt_money(5) == "0.05"


def test_video_and_chart_shapes():
    rng = np.random.default_rng(2)
    st = F.make_statement(rng)
    assert F.render_chart(st, 32).shape == (32, 32)
    v = F.make_video_frames(rng, st, n_frames=8, size=32)
    assert v.shape == (8, 32, 32, 3) and v.dtype == np.uint8


def test_generate_corpus(tmp_path):
    info = F.generate_corpus(tmp_path, n=10, seed=3, pdfs=2, videos=1)
    assert info["n_statements"] == 10
    lines = (tmp_path / "statements.jsonl").read_text().splitlines()
    assert len(lines) == 10 and "text" in json.loads(lines[0])
    assert (tmp_path / "statement_0.pdf").read_bytes()[:4] == b"%PDF"
    assert np.load(tmp_path / "video_0.npy").ndim == 4
