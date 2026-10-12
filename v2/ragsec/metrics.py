"""Scoring: did retrieval find the evidence, and is the answer right?

The same rules score v1 and v2. v1's chunks are raw SGML, so their tags are
stripped before matching; that only helps v1.
"""
from __future__ import annotations

import html
import re

from .config import JUDGE_MODEL
from .llm import chat_json

TAG_RE = re.compile(r"<[^>]*>")
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mm": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "trillion": 1e12, "t": 1e12}
NUM_RE = re.compile(
    r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\)?\s*(thousand|million|billion|trillion|bn|mm|[kmbt])?\b",
    re.I)


def normalize(text: str) -> str:
    text = html.unescape(TAG_RE.sub(" ", text))
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("—", "-").replace("–", "-").replace("\xa0", " "))
    return re.sub(r"\s+", " ", text).strip().lower()


def _shingles(text: str, n: int = 6) -> set[str]:
    words = re.findall(r"[a-z0-9$%']+", text)
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def is_hit(question: dict, chunk_text: str, chunk_ticker: str | None) -> bool:
    """Whether a retrieved chunk holds the question's gold evidence."""
    if question["type"] == "numeric":
        if chunk_ticker != question["ticker"]:
            return False
        text = normalize(chunk_text)
        return any(re.search(rf"(?<![\d,.]){re.escape(s.lower())}(?![\d]|,\d|\.\d)", text)
                   for s in question["gold"]["shown"])
    if question["type"] == "text":
        gold = _shingles(normalize(question["gold"]["evidence"]))
        text = normalize(chunk_text)
        found = sum(1 for s in gold if s in text)
        return found / len(gold) >= 0.5
    return False


def retrieval_scores(hits: list[bool], ks=(1, 3, 5, 10)) -> dict:
    first = next((i for i, h in enumerate(hits) if h), None)
    out = {f"hit@{k}": float(first is not None and first < k) for k in ks}
    out["mrr@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
    return out


def numbers_in(text: str) -> list[tuple[float, float, bool]]:
    """(value, half a unit of its last digit, has an explicit scale word) for every number in the text."""
    out = []
    for m in NUM_RE.finditer(text):
        whole, frac, scale = m.group(1), m.group(2), (m.group(3) or "").lower()
        value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        mult = SCALE.get(scale, 1.0)
        out.append((value * mult, 0.5 * 10 ** (-len(frac or "")) * mult, bool(scale)))
    return out


def numeric_correct(answer: str, gold: dict) -> bool:
    """True if the answer states the gold value, to the precision it gives (and within 5%)."""
    target = abs(gold["value"])
    for value, half_unit, scaled in numbers_in(answer):
        candidates = [(value, half_unit)] if scaled else [(value * s, half_unit * s) for s in (1, 1e3, 1e6, 1e9)]
        for v, tol in candidates:
            err = abs(v - target)
            if err <= max(0.005 * target, tol) and err <= 0.05 * target:
                return True
    return False


JUDGE_PROMPT = """You grade answers from a question-answering system over SEC filings.

Question: {question}
Reference answer: {reference}
Supporting passage from the filing: "{evidence}"

System answer:
\"\"\"{answer}\"\"\"

Grade the system answer against the reference and passage:
- "correct": states the substance of the reference answer (extra correct detail is fine).
- "partial": gets part of it right but misses or garbles a key element.
- "incorrect": wrong, unrelated, or contradicts the reference.
- "declined": says it cannot find or answer it.
Return JSON: {{"grade": "correct|partial|incorrect|declined", "why": "one sentence"}}"""

DECLINE_PROMPT = """A question-answering system over SEC filings was asked a question its documents cannot answer
({reason}). The right behaviour is to say the filings do not contain the answer, rather than give a figure or claim.

Question: {question}
System answer:
\"\"\"{answer}\"\"\"

Did the system decline (say it could not find it, or that the filings do not cover it) without stating a specific
answer as fact? Return JSON: {{"declined": true|false, "why": "one sentence"}}"""

REFUSAL_RE = re.compile(
    r"(couldn'?t|could not|can'?t|cannot|unable to|not able to|don'?t|do not|didn'?t|did not) (find|locate|answer|"
    r"provide|determine|have)|no (information|data|mention|record)|not (available|included|provided|contain|found|"
    r"mentioned|disclosed)|isn'?t (available|included|in)|does not (contain|include|provide|mention)", re.I)


def judge_text(question: dict, answer: str) -> dict:
    g = question["gold"]
    data = chat_json(JUDGE_MODEL, JUDGE_PROMPT.format(
        question=question["question"], reference=g["answer"], evidence=g["evidence"], answer=answer)) or {}
    grade = data.get("grade", "incorrect")
    return {"grade": grade, "score": {"correct": 1.0, "partial": 0.5}.get(grade, 0.0), "why": data.get("why", "")}


def judge_unanswerable(question: dict, answer: str) -> dict:
    data = chat_json(JUDGE_MODEL, DECLINE_PROMPT.format(
        reason=question["gold"]["reason"], question=question["question"], answer=answer)) or {}
    declined = bool(data.get("declined"))
    return {"grade": "declined" if declined else "answered", "score": float(declined), "why": data.get("why", "")}


def looks_declined(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))
