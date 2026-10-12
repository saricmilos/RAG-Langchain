"""Build the evaluation set: numeric, text and unanswerable questions with gold evidence.

Numeric answers come from the XBRL facts tagged in each filing, so nobody has to
look numbers up by hand. Text questions are written by an LLM from a sampled
passage, with a verbatim evidence quote that is checked against the passage.
Unanswerable questions are written by hand: companies, years or forecasts the
filings do not cover, where the right answer is to say so.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from .chunking import Chunk
from .config import COMPANIES, JUDGE_MODEL, WORK
from .llm import chat_json

# key -> (XBRL concepts, period kind, forms, question templates)
NUMERIC_SPECS = {
    "revenue": (["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                 "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax"], "duration", ("10-K", "10-Q"),
                ["What were {short}'s total revenues for {period}?",
                 "How much revenue did {short} report for {period}?"]),
    "net_income": (["us-gaap:NetIncomeLoss"], "duration", ("10-K", "10-Q"),
                   ["What was {short}'s net income for {period}?",
                    "How much net income did {short} earn in {period}?"]),
    "operating_income": (["us-gaap:OperatingIncomeLoss"], "duration", ("10-K", "10-Q"),
                         ["What was {short}'s operating income in {period}?"]),
    "rnd": (["us-gaap:ResearchAndDevelopmentExpense",
             "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"], "duration", ("10-K", "10-Q"),
            ["How much did {short} spend on research and development in {period}?"]),
    "eps_diluted": (["us-gaap:EarningsPerShareDiluted"], "duration", ("10-K", "10-Q"),
                    ["What were {short}'s diluted earnings per share for {period}?"]),
    "cash": (["us-gaap:CashAndCashEquivalentsAtCarryingValue"], "instant", ("10-K", "10-Q"),
             ["How much cash and cash equivalents did {short} hold as of {date}?"]),
    "total_assets": (["us-gaap:Assets"], "instant", ("10-K", "10-Q"),
                     ["What were {short}'s total assets as of {date}?"]),
    "capex": (["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
               "us-gaap:PaymentsToAcquireProductiveAssets"], "duration", ("10-K",),
              ["How much did {short} spend on purchases of property and equipment in {period}?"]),
    "buybacks": (["us-gaap:PaymentsForRepurchaseOfCommonStock"], "duration", ("10-K",),
                 ["How much cash did {short} spend repurchasing its common stock in {period}?"]),
}
ANNUAL_ORDER = ["revenue", "net_income", "rnd", "operating_income", "eps_diluted", "cash", "capex", "buybacks",
                "total_assets"]
QUARTER_ROTATION = ["revenue", "net_income", "operating_income", "eps_diluted", "cash"]

UNANSWERABLE = [
    ("What was Netflix's total revenue for fiscal year 2024?", None, "company not in the corpus"),
    ("How much did Intel spend on research and development in 2024?", None, "company not in the corpus"),
    ("What was Walmart's net income for fiscal 2025?", None, "company not in the corpus"),
    ("What were Oracle's cloud services revenues in fiscal 2025?", None, "company not in the corpus"),
    ("What was JPMorgan Chase's net interest income in 2024?", None, "company not in the corpus"),
    ("What were Apple's total net sales in fiscal year 2015?", "AAPL", "year not covered by the filings"),
    ("How much revenue did Microsoft report for fiscal year 2012?", "MSFT", "year not covered by the filings"),
    ("What was Tesla's net income in 2016?", "TSLA", "year not covered by the filings"),
    ("What was NVIDIA's revenue for fiscal year 2019?", "NVDA", "year not covered by the filings"),
    ("What was Alphabet's total revenue in 2010?", "GOOGL", "year not covered by the filings"),
    ("How many employees did NVIDIA have in 2015?", "NVDA", "year not covered by the filings"),
    ("What was Amazon's net sales in the first quarter of 2026?", "AMZN", "period after the latest filing"),
    ("What was Microsoft's net income for the quarter ended March 31, 2026?", "MSFT",
     "period after the latest filing"),
    ("What was Berkshire Hathaway's Class A share price on March 2, 2026?", "BRK-B",
     "market data not in the filings"),
    ("How many vehicles will Tesla deliver in 2030?", "TSLA", "forecast not disclosed"),
    ("What will Alphabet's revenue be in 2028?", "GOOGL", "forecast not disclosed"),
    ("What is Goldman Sachs's price target for Amazon stock?", "AMZN", "analyst data not in the filings"),
    ("What was Eli Lilly's revenue from Mounjaro in 2019?", "LLY", "product not yet launched in that year"),
    ("How much did Broadcom pay to acquire Intel?", "AVGO", "event never happened"),
    ("What is Mark Zuckerberg's personal net worth?", "META", "not in the filings"),
]

TEXT_ITEMS_10K = {"1": 1, "1A": 2, "7": 2, "7A": 1, "1C": 1, "3": 1}   # sampling weights
TEXT_ITEMS_10Q = {"2", "1A"}


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def _days(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 0
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _current_facts(facts: list[dict], filing: dict, kind: str) -> list[dict]:
    """Consolidated (no segment) facts for the filing's own period."""
    end = filing["period_end"]
    lo, hi = (350, 380) if filing["form"] == "10-K" else (80, 100)
    out = []
    for f in facts:
        if f["segment"] or f["end"] != end:
            continue
        if kind == "instant" and f["instant"]:
            out.append(f)
        elif kind == "duration" and not f["instant"] and lo <= _days(f["start"], f["end"]) <= hi:
            out.append(f)
    return out


def _numeric_question(spec_key: str, filing: dict, facts: list[dict], rng: random.Random) -> dict | None:
    concepts, kind, forms, templates = NUMERIC_SPECS[spec_key]
    if filing["form"] not in forms:
        return None
    current = [f for f in _current_facts(facts, filing, kind) if f["concept"] in concepts]
    money = [f for f in current if ("share" in f["unit"]) == (spec_key == "eps_diluted")]
    if not money:
        return None
    best = max(money, key=lambda f: f["value"]) if spec_key == "revenue" else money[0]
    # Every place the filing shows this number (table cell, rounded mention in the text).
    shown = sorted({f["shown"] for f in facts
                    if f["concept"] == best["concept"] and f["start"] == best["start"] and f["end"] == best["end"]
                    and not f["segment"] and abs(f["value"] - best["value"]) <= 0.001 * abs(best["value"]) + 1e-9})
    end = date.fromisoformat(filing["period_end"])
    end_label = f"{end:%B} {end.day}, {end.year}"
    period = f"fiscal year {filing['fiscal_year']}" if filing["form"] == "10-K" else f"the quarter ended {end_label}"
    short = COMPANIES[filing["ticker"]]["short"]
    question = rng.choice(templates).format(short=short, period=period, date=end_label)
    return {
        "type": "numeric", "ticker": filing["ticker"], "doc_id": filing["doc_id"], "form": filing["form"],
        "question": question,
        "gold": {"metric": spec_key, "concept": best["concept"], "value": best["value"],
                 "unit": "usd_per_share" if spec_key == "eps_diluted" else "usd", "shown": shown,
                 "period_end": filing["period_end"]},
    }


def numeric_questions(filings: list[dict], facts_by_doc: dict, rng: random.Random) -> list[dict]:
    out = []
    for filing in filings:
        facts = facts_by_doc[filing["doc_id"]]
        amazon = filing["ticker"] == "AMZN"
        if filing["form"] == "10-K":
            keys, limit = ANNUAL_ORDER, (9 if amazon else 6)
        else:
            start = sum(1 for f in filings if f["form"] == "10-Q" and f["doc_id"] < filing["doc_id"])
            keys = QUARTER_ROTATION[start % len(QUARTER_ROTATION):] + QUARTER_ROTATION[:start % len(QUARTER_ROTATION)]
            limit = 4 if amazon else 1
        made = 0
        for key in keys:
            if made >= limit:
                break
            q = _numeric_question(key, filing, facts, rng)
            if q:
                out.append(q)
                made += 1
    return out


TEXT_PROMPT = """You write evaluation questions for a search system over SEC filings.

Below is a passage from {company}'s {form} for {period}, from {where}.

Write ONE question an investor or analyst might type, which this passage answers specifically.
- Name the company and the year or period in the question.
- Paraphrase the way a real user would ask; do not copy distinctive phrases from the passage.
- It must be answerable from this passage alone, and not be a yes/no question.
- If the passage is boilerplate, a cross-reference, or has no specific fact, return {{"skip": true}}.

Return JSON: {{"question": "...", "answer": "1-2 sentence reference answer",
"evidence": "an exact quote of 12 to 30 consecutive words copied verbatim from the passage that supports the answer"}}

Passage:
\"\"\"{text}\"\"\""""


def _text_question(chunk: Chunk) -> dict | None:
    where = f"Part {chunk.part}, Item {chunk.item}. {chunk.item_title}" if chunk.part else f"Item {chunk.item}"
    data = chat_json(JUDGE_MODEL, TEXT_PROMPT.format(
        company=chunk.company, form=chunk.form, period=chunk.period_label, where=where, text=chunk.text))
    if not data or data.get("skip") or not data.get("question"):
        return None
    evidence = data.get("evidence", "")
    if len(evidence.split()) < 10 or _norm(evidence) not in _norm(chunk.text):
        return None
    return {
        "type": "text", "ticker": chunk.ticker, "doc_id": chunk.doc_id, "form": chunk.form,
        "question": data["question"].strip(),
        "gold": {"answer": data["answer"].strip(), "evidence": evidence.strip(), "chunk_id": chunk.id,
                 "item": chunk.item, "item_title": chunk.item_title},
    }


def _sample_passages(chunks: list[Chunk], filings: list[dict], rng: random.Random) -> list[Chunk]:
    by_doc = defaultdict(list)
    for c in chunks:
        if c.kind == "text" and len(c.text) >= 600:
            by_doc[c.doc_id].append(c)
    picked = []
    for ticker in COMPANIES:
        tenks = [f for f in filings if f["ticker"] == ticker and f["form"] == "10-K"]
        tenqs = [f for f in filings if f["ticker"] == ticker and f["form"] == "10-Q"]
        amazon = ticker == "AMZN"
        per_10k = 16 if amazon else (7 if len(tenks) == 1 else 4)
        for f in tenks:
            pool = [c for c in by_doc[f["doc_id"]] if c.item in TEXT_ITEMS_10K]
            weights = [TEXT_ITEMS_10K[c.item] for c in pool]
            seen = set()
            while len(seen) < min(per_10k, len(pool)):
                seen.add(rng.choices(range(len(pool)), weights=weights)[0])
            picked += [pool[i] for i in sorted(seen)]
        for f in (tenqs if amazon else rng.sample(tenqs, 1)):
            pool = [c for c in by_doc[f["doc_id"]] if c.item in TEXT_ITEMS_10Q]
            picked += rng.sample(pool, min(3 if amazon else 2, len(pool)))
    return picked


def text_questions(chunks: list[Chunk], filings: list[dict], rng: random.Random) -> list[dict]:
    passages = _sample_passages(chunks, filings, rng)
    with ThreadPoolExecutor(8) as pool:
        made = list(pool.map(_text_question, passages))
    return [q for q in made if q]


def unanswerable_questions() -> list[dict]:
    return [{"type": "unanswerable", "ticker": t, "doc_id": None, "form": None, "question": q,
             "gold": {"reason": why}} for q, t, why in UNANSWERABLE]


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    filings = json.loads((WORK / "filings.json").read_text(encoding="utf-8"))
    facts_by_doc = defaultdict(list)
    with (WORK / "facts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            fact = json.loads(line)
            facts_by_doc[fact["doc_id"]].append(fact)
    with (WORK / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]

    questions = numeric_questions(filings, facts_by_doc, rng)
    questions += text_questions(chunks, filings, rng)
    questions += unanswerable_questions()
    for i, q in enumerate(questions):
        q["id"] = f"q{i + 1:03d}"
    return questions
