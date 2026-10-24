"""Synthetic financial-document reasoning + tool-use task built on `data.synth.finance` statements.

An episode is a question about one (hidden) statement. The model never sees the statement itself; it has to
call tools that do:

    C: total(<category>)     -> sum of debits in category (positive amount)
    C: count(<category>)     -> number of debit transactions in category
    C: closing()             -> closing balance
    C: largest()             -> "<merchant> <amount>" of the largest debit
    C: calc(<a>-<b>) / calc(<a>+<b>)  -> calculator on two decimal amounts

The environment appends `R: <result>` after each call line. The model ends with `A: <answer>`.
Some questions use synonyms ("eating out" -> dining) or need two lookups plus a calculation, so mapping
language to tool arguments and copying numbers is non-trivial for a small model.
"""

from __future__ import annotations

import dataclasses
import re
from decimal import Decimal

import numpy as np

from data.synth import finance as F

DEBIT_CATEGORIES = sorted(set(F.CATEGORIES.values()))

SYNONYMS = {
    "groceries": ["groceries", "food shopping"],
    "transport": ["transport", "getting around"],
    "utilities": ["utilities", "bills like power and phone"],
    "health": ["health", "wellness"],
    "dining": ["dining", "eating out"],
    "insurance": ["insurance", "coverage premiums"],
    "shopping": ["shopping", "retail purchases"],
    "entertainment": ["entertainment", "streaming and fun"],
    "housing": ["housing", "rent"],
}

TEMPLATES = {
    "total": ["How much was spent on {c}?", "Total {c} spending?", "What did I spend on {c} this period?"],
    "count": ["How many {c} transactions were there?", "Number of {c} charges?"],
    "closing": ["What is the closing balance?", "Closing balance please.", "How much is left at period end?"],
    "largest_m": ["Which merchant had the largest charge?", "Where was the biggest charge?"],
    "largest_a": ["What was the largest charge amount?", "How big was the biggest charge?"],
    "diff": ["How much more on {c} than {d}?", "Difference between {c} and {d} spending?"],
}
QTYPES = list(TEMPLATES)

CALL_RE = re.compile(r"^C: (total|count|closing|largest|calc)\(([^()]*)\)$")
NUM_RE = re.compile(r"^-?\d+\.\d{2}$")


@dataclasses.dataclass
class Episode:
    qtype: str
    prompt: str  # "Q: ...\n"
    gold_completion: str  # reference trajectory incl. R: lines
    answer: str
    statement: F.Statement


def _money(cents: int) -> str:
    return F.fmt_money(cents)


class Ledger:
    """Tool environment bound to one statement."""

    def __init__(self, st: F.Statement):
        self.st = st

    def debits(self, cat: str) -> list[F.Transaction]:
        return [t for t in self.st.transactions if t.amount_cents < 0 and t.category == cat]

    def run(self, line: str) -> str | None:
        """Execute a call line. Returns result string, or None if the call is invalid."""
        m = CALL_RE.match(line.strip())
        if not m:
            return None
        name, arg = m.group(1), m.group(2).strip()
        if name in ("total", "count"):
            if arg not in DEBIT_CATEGORIES:
                return None
            ds = self.debits(arg)
            return _money(-sum(t.amount_cents for t in ds)) if name == "total" else str(len(ds))
        if name == "closing":
            return _money(self.st.closing_cents) if arg == "" else None
        if name == "largest":
            if arg != "":
                return None
            big = min(self.st.transactions, key=lambda t: t.amount_cents)
            return f"{big.merchant} {_money(-big.amount_cents)}"
        if name == "calc":
            mm = re.match(r"^(-?\d+\.\d{2})([+-])(-?\d+\.\d{2})$", arg)
            if not mm:
                return None
            a, op, b = Decimal(mm.group(1)), mm.group(2), Decimal(mm.group(3))
            r = a - b if op == "-" else a + b
            return f"{r:.2f}"
        return None


def make_episode(
    rng: np.random.Generator,
    st: F.Statement | None = None,
    qtype: str | None = None,
    qtype_probs: dict[str, float] | None = None,
) -> Episode:
    st = st or F.make_statement(rng)
    led = Ledger(st)
    if qtype is None:
        if qtype_probs:
            ks = list(qtype_probs)
            p = np.array([qtype_probs[k] for k in ks], float)
            qtype = str(rng.choice(ks, p=p / p.sum()))
        else:
            qtype = str(rng.choice(QTYPES))
    present = sorted({t.category for t in st.transactions if t.amount_cents < 0})
    c = str(rng.choice(present)) if present else "groceries"
    others = [x for x in DEBIT_CATEGORIES if x != c]
    d = str(rng.choice(others))
    tmpl = str(rng.choice(TEMPLATES[qtype]))
    q = tmpl.format(c=str(rng.choice(SYNONYMS[c])), d=str(rng.choice(SYNONYMS[d])))
    lines: list[str] = []

    def call(s: str) -> str:
        r = led.run(s)
        assert r is not None, s
        lines.append(s)
        lines.append("R: " + r)
        return r

    if qtype == "total":
        ans = call(f"C: total({c})")
    elif qtype == "count":
        ans = call(f"C: count({c})")
    elif qtype == "closing":
        ans = call("C: closing()")
    elif qtype in ("largest_m", "largest_a"):
        r = call("C: largest()")
        merchant, amt = r.rsplit(" ", 1)
        ans = merchant if qtype == "largest_m" else amt
    else:  # diff
        a = call(f"C: total({c})")
        b = call(f"C: total({d})")
        ans = call(f"C: calc({a}-{b})")
    lines.append("A: " + ans)
    return Episode(qtype, f"Q: {q}\n", "\n".join(lines) + "\n", ans, st)


def make_split(
    n: int, seed: int, qtype_probs: dict[str, float] | None = None, site: str | None = None
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [make_episode(rng, F.make_statement(rng, site=site), qtype_probs=qtype_probs) for _ in range(n)]


def score(ep: Episode, completion: str) -> dict:
    """Grade a model trajectory (with environment-inserted R: lines).

    exact: final answer equals gold. tool_ok: >=1 call emitted and every call line was valid/executable."""
    calls = [ln for ln in completion.split("\n") if ln.startswith("C: ")]
    led = Ledger(ep.statement)
    tool_ok = len(calls) > 0 and all(led.run(c) is not None for c in calls)
    ans = None
    for ln in completion.split("\n"):
        if ln.startswith("A: "):
            ans = ln[3:].strip()
            break
    return {"exact": ans == ep.answer, "tool_ok": tool_ok, "n_calls": len(calls)}


# ---- non-IID site distributions for the federated simulation ---------------------------------------------
SITE_MERCHANT_WEIGHT = {
    "TOR": {"Maple Grocers": 4, "Laurentian Transit": 3, "Bluefin Cafe": 3, "Lakeshore Fuel": 2},
    "MTL": {"Birchwood Rent": 4, "Harbourview Hydro": 3, "Cedar Pharmacy": 3, "Granite Gym": 2},
    "US": {"Aurora Streaming": 4, "Prairie Books": 3, "Summit Insurance": 3, "Northern Telecom Co": 2},
}
SITE_QTYPE_PROBS = {
    "TOR": {
        "total": 0.45,
        "count": 0.35,
        "closing": 0.05,
        "largest_m": 0.05,
        "largest_a": 0.05,
        "diff": 0.05,
    },
    "MTL": {
        "total": 0.05,
        "count": 0.05,
        "closing": 0.40,
        "largest_m": 0.05,
        "largest_a": 0.05,
        "diff": 0.40,
    },
    "US": {"total": 0.05, "count": 0.05, "closing": 0.05, "largest_m": 0.40, "largest_a": 0.40, "diff": 0.05},
}


def make_site_statement(rng: np.random.Generator, site: str) -> F.Statement:
    """Statement whose debit merchants follow a site-specific skewed mix (weight 0.25 for unlisted merchants)."""
    st = F.make_statement(rng, site=site)
    w = np.array([SITE_MERCHANT_WEIGHT[site].get(m, 0.25) for m in F.MERCHANTS], float)
    txs = []
    for t in st.transactions:
        if t.amount_cents < 0:
            m = str(rng.choice(F.MERCHANTS, p=w / w.sum()))
            t = F.Transaction(t.date, m, F.CATEGORIES[m], t.amount_cents)
        txs.append(t)
    return dataclasses.replace(st, transactions=txs)


def make_site_split(n: int, seed: int, site: str) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [
        make_episode(rng, make_site_statement(rng, site), qtype_probs=SITE_QTYPE_PROBS[site])
        for _ in range(n)
    ]
