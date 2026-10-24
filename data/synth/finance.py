"""Synthetic financial documents: account statements, call transcripts and chart frames.

Everything is generated from a seeded RNG out of fixed fictional vocabularies. No real
person, institution or account is referenced; names are drawn from a small list of
invented given/family names and account numbers are random digits prefixed with "SYN-".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np

GIVEN = [
    "Avery",
    "Jordan",
    "Riley",
    "Quinn",
    "Morgan",
    "Casey",
    "Rowan",
    "Emerson",
    "Sasha",
    "Devon",
    "Parker",
    "Hayden",
    "Kai",
    "Noor",
    "Imani",
    "Tenzin",
    "Ayla",
    "Ezra",
    "Mika",
    "Laken",
]
FAMILY = [
    "Ardent",
    "Bellweather",
    "Cobalt",
    "Driftwood",
    "Elmsworth",
    "Fairlow",
    "Grantham",
    "Holloway",
    "Ivers",
    "Juniper",
    "Kestrel",
    "Larkspur",
    "Marchetti",
    "Northcote",
    "Oakhurst",
    "Pemberly",
]
MERCHANTS = [
    "Maple Grocers",
    "Laurentian Transit",
    "Harbourview Hydro",
    "Northern Telecom Co",
    "Cedar Pharmacy",
    "Lakeshore Fuel",
    "Bluefin Cafe",
    "Summit Insurance",
    "Prairie Books",
    "Granite Gym",
    "Aurora Streaming",
    "Birchwood Rent",
]
CATEGORIES = {
    "Maple Grocers": "groceries",
    "Laurentian Transit": "transport",
    "Harbourview Hydro": "utilities",
    "Northern Telecom Co": "utilities",
    "Cedar Pharmacy": "health",
    "Lakeshore Fuel": "transport",
    "Bluefin Cafe": "dining",
    "Summit Insurance": "insurance",
    "Prairie Books": "shopping",
    "Granite Gym": "health",
    "Aurora Streaming": "entertainment",
    "Birchwood Rent": "housing",
}
SITES = ["TOR", "MTL", "US"]
CURRENCY = {"TOR": "CAD", "MTL": "CAD", "US": "USD"}


@dataclasses.dataclass
class Transaction:
    date: str
    merchant: str
    category: str
    amount_cents: int  # negative = debit


@dataclasses.dataclass
class Statement:
    account_id: str
    holder: str
    site: str
    currency: str
    period_start: str
    period_end: str
    opening_cents: int
    transactions: list[Transaction]

    @property
    def closing_cents(self) -> int:
        return self.opening_cents + sum(t.amount_cents for t in self.transactions)

    def to_text(self) -> str:
        lines = [
            f"STATEMENT {self.account_id} holder={self.holder} site={self.site} ccy={self.currency}",
            f"period {self.period_start}..{self.period_end} opening={fmt_money(self.opening_cents)}",
        ]
        for t in self.transactions:
            lines.append(f"{t.date} {t.merchant} [{t.category}] {fmt_money(t.amount_cents)}")
        lines.append(f"closing={fmt_money(self.closing_cents)}")
        return "\n".join(lines)


def fmt_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}{c // 100}.{c % 100:02d}"


def make_statement(rng: np.random.Generator, site: str | None = None, n_tx: int | None = None) -> Statement:
    site = site or str(rng.choice(SITES))
    holder = f"{rng.choice(GIVEN)} {rng.choice(FAMILY)}"
    acct = "SYN-" + "".join(str(d) for d in rng.integers(0, 10, size=8))
    start = dt.date(2025, 1, 1) + dt.timedelta(days=int(rng.integers(0, 330)))
    end = start + dt.timedelta(days=29)
    n_tx = n_tx if n_tx is not None else int(rng.integers(4, 12))
    txs = []
    for _ in range(n_tx):
        m = str(rng.choice(MERCHANTS))
        day = start + dt.timedelta(days=int(rng.integers(0, 30)))
        amt = -int(rng.integers(300, 250_000))
        txs.append(Transaction(day.isoformat(), m, CATEGORIES[m], amt))
    if rng.random() < 0.8:  # payroll deposit
        txs.append(
            Transaction(
                (start + dt.timedelta(days=14)).isoformat(),
                "Payroll Deposit",
                "income",
                int(rng.integers(150_000, 900_000)),
            )
        )
    txs.sort(key=lambda t: t.date)
    return Statement(
        acct,
        holder,
        site,
        CURRENCY[site],
        start.isoformat(),
        end.isoformat(),
        int(rng.integers(0, 2_000_000)),
        txs,
    )


TRANSCRIPT_TEMPLATES = [
    ("client", "Hi, I'm calling about account {acct}. Can you tell me how much I spent on {cat} this month?"),
    ("advisor", "Of course. I can see {n} {cat} transactions totalling {total} {ccy}."),
    ("client", "And what's my closing balance?"),
    ("advisor", "Your closing balance for the period ending {end} is {closing} {ccy}."),
    ("client", "Thanks. Could you flag the largest charge for review?"),
    ("advisor", "The largest charge was {big_amt} at {big_m} on {big_d}. I've flagged it."),
]


def make_transcript(rng: np.random.Generator, st: Statement) -> list[dict]:
    debit_cats = sorted({t.category for t in st.transactions if t.amount_cents < 0})
    cat = str(rng.choice(debit_cats)) if debit_cats else "groceries"
    sel = [t for t in st.transactions if t.category == cat]
    big = min(st.transactions, key=lambda t: t.amount_cents)
    ctx = dict(
        acct=st.account_id,
        cat=cat,
        n=len(sel),
        total=fmt_money(-sum(t.amount_cents for t in sel)),
        ccy=st.currency,
        end=st.period_end,
        closing=fmt_money(st.closing_cents),
        big_amt=fmt_money(-big.amount_cents),
        big_m=big.merchant,
        big_d=big.date,
    )
    t0 = 0.0
    out = []
    for spk, tmpl in TRANSCRIPT_TEMPLATES:
        text = tmpl.format(**ctx)
        dur = 0.35 * len(text.split()) + float(rng.uniform(0.2, 0.8))
        out.append({"speaker": spk, "start_s": round(t0, 2), "end_s": round(t0 + dur, 2), "text": text})
        t0 += dur + float(rng.uniform(0.1, 0.6))
    return out


def render_chart(st: Statement, size: int = 32) -> np.ndarray:
    """Render a running-balance chart as a (size, size) uint8 grayscale image."""
    bal = np.cumsum([st.opening_cents] + [t.amount_cents for t in st.transactions]).astype(float)
    img = np.zeros((size, size), dtype=np.uint8)
    lo, hi = bal.min(), bal.max()
    span = hi - lo if hi > lo else 1.0
    xs = np.linspace(0, size - 1, len(bal)).astype(int)
    ys = (size - 1 - (bal - lo) / span * (size - 1)).astype(int)
    for i in range(len(bal) - 1):
        n = max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i]), 1)
        for k in range(n + 1):
            x = xs[i] + (xs[i + 1] - xs[i]) * k // n
            y = ys[i] + (ys[i + 1] - ys[i]) * k // n
            img[y, x] = 255
    img[-1, :] = 96  # x-axis
    return img


def make_video_frames(
    rng: np.random.Generator, st: Statement, n_frames: int = 48, size: int = 64
) -> np.ndarray:
    """Synthetic 'advisor screen-share' video: chart revealed left-to-right plus a moving placeholder head.

    Returns uint8 array (T, H, W, 3)."""
    chart = render_chart(st, size)
    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    cx0, cy0 = size * 0.8, size * 0.25
    for f in range(n_frames):
        fr = np.full((size, size, 3), 24, dtype=np.uint8)
        reveal = int((f + 1) / n_frames * size)
        fr[:, :reveal, 1] = np.maximum(fr[:, :reveal, 1], chart[:, :reveal])
        # talking-head placeholder: disc whose radius oscillates ("speaking")
        r = size * 0.1 * (1 + 0.15 * np.sin(f / 2.0))
        cy = cy0 + 1.5 * np.sin(f / 5.0)
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (yy - cy) ** 2 + (xx - cx0) ** 2 <= r**2
        fr[mask] = (200, 160, 120)
        fr += rng.integers(0, 6, size=fr.shape, dtype=np.uint8)  # sensor noise
        frames[f] = fr
    return frames


def write_pdf(st: Statement, path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(f"Synthetic statement {st.account_id}")
    y = 750
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "SYNTHETIC DATA - NOT A REAL STATEMENT")
    c.setFont("Helvetica", 10)
    for line in st.to_text().splitlines():
        y -= 16
        c.drawString(50, y, line)
    c.save()


def generate_corpus(out: Path, n: int, seed: int, pdfs: int = 5, videos: int = 3) -> dict:
    """Write statements.jsonl, transcripts.jsonl, a few PDFs and video .npy files to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stmts = [make_statement(rng) for _ in range(n)]
    with open(out / "statements.jsonl", "w") as f:
        for s in stmts:
            f.write(json.dumps(dataclasses.asdict(s) | {"text": s.to_text()}) + "\n")
    with open(out / "transcripts.jsonl", "w") as f:
        for s in stmts:
            f.write(json.dumps({"account_id": s.account_id, "turns": make_transcript(rng, s)}) + "\n")
    for i, s in enumerate(stmts[:pdfs]):
        write_pdf(s, out / f"statement_{i}.pdf")
    for i, s in enumerate(stmts[:videos]):
        np.save(out / f"video_{i}.npy", make_video_frames(rng, s))
    return {"n_statements": n, "pdfs": min(pdfs, n), "videos": min(videos, n), "seed": seed}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/corpus")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate_corpus(Path(a.out), a.n, a.seed))
