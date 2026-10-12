"""v2 answers: retrieve, then ask the model to answer only from numbered excerpts and cite them.

Same model as v1 (gpt-4o-mini). The differences are what it is shown (clean excerpts
labelled with company, filing, Item and page) and what it is told (cite every claim,
say so when the excerpts do not answer).
"""
from __future__ import annotations

import re
import time

from .chunking import Chunk
from .config import ANSWER_MODEL
from .llm import chat
from .retrieve import RetrievalConfig, Retriever

SYSTEM = """You answer questions about public companies using only excerpts from their SEC filings (10-K and 10-Q).

Rules:
- Use only the numbered excerpts below. Never use outside knowledge, even if you know the answer.
- After each claim, cite the excerpt(s) it comes from, like [2] or [1][3].
- Give figures exactly as the filing states them, with units and the period (for example "$416,161 million in
  fiscal 2025"). Tables are in millions unless the excerpt says otherwise.
- If the excerpts do not contain the answer (wrong company, wrong period, or not covered), reply with
  "The filings I have don't answer this." plus one sentence on what they do cover, and cite nothing.
- Be concise: two to five sentences, or a short list."""

CITE_RE = re.compile(r"\[(\d{1,2})\]")


def format_excerpt(n: int, c: Chunk) -> str:
    where = f"Item {c.item}. {c.item_title}" if c.item else c.item_title
    page = f", p. {c.page}" if c.page else ""
    return f"[{n}] {c.company} {c.form}, {c.period_label}, {where}{page}\n{c.text}"


def answer(retriever: Retriever, question: str, cfg: RetrievalConfig, k_context: int = 8) -> dict:
    t = time.time()
    ranked = retriever.search(question, cfg)
    retrieval_seconds = time.time() - t
    context = [retriever.chunks[i] for i, _ in ranked[:k_context]]
    excerpts = "\n\n".join(format_excerpt(n, c) for n, c in enumerate(context, 1))
    text, meta = chat(ANSWER_MODEL, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ], max_tokens=500)
    cited_numbers = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(context)})
    return {
        "answer": text.strip(),
        "context_ids": [c.id for c in context],
        "retrieved_ids": [retriever.chunks[i].id for i, _ in ranked],
        "cited": [{"n": n, "chunk_id": context[n - 1].id} for n in cited_numbers],
        "tokens_in": meta["tokens_in"],
        "tokens_out": meta["tokens_out"],
        "cost": meta["cost"],
        "seconds": round(retrieval_seconds + meta["seconds"], 2),
    }
