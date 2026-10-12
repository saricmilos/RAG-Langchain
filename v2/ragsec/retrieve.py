"""v2 retrieval: dense, BM25, hybrid fusion, filters parsed from the question, reranking.

Each step is a switch in RetrievalConfig so the ablation can turn them on one at a time.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import requests

from .chunking import Chunk
from .config import COMPANIES
from .embed import chunk_matrix, query_matrix

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_RE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),\s*(\d{{4}})", re.I)
FY_RE = re.compile(r"\b(?:fiscal(?: year)?|fy)\s*(\d{4})\b|\b(\d{4})\s+(?:annual report|10-k|fiscal year)\b", re.I)
STOP = set("the a an of in for and to on by what was were is are did does how much many which with its at as from "
           "that this be or it their they has have had during".split())


@dataclass(frozen=True)
class RetrievalConfig:
    embedder: str = "bge-small"
    dense: bool = True
    bm25: bool = False
    filters: bool = False
    rerank: str | None = None    # None, "minilm-ce", "cohere"
    pool: int = 40               # candidates passed from fusion to the reranker
    k: int = 10


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 over a sparse term matrix, vectorised with numpy."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            counts: dict[int, int] = {}
            for tok in doc:
                j = self.vocab.setdefault(tok, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            rows += [i] * len(counts)
            cols += list(counts)
            vals += list(counts.values())
        from scipy.sparse import csc_matrix
        tf = csc_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        lengths = np.asarray([len(d) for d in docs], dtype=np.float32)
        df = np.diff(tf.indptr).astype(np.float32)
        self.idf = np.log((len(docs) - df + 0.5) / (df + 0.5) + 1.0)
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        weighted = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = csc_matrix((weighted, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: list[str]) -> np.ndarray:
        ids = [self.vocab[t] for t in set(query) if t in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, ids] @ self.idf[ids]).ravel()


def parse_filters(question: str, chunks_meta: dict) -> dict:
    """Company and period named in the question, kept only if the corpus has them."""
    q = question.lower()
    tickers = [t for t, info in COMPANIES.items()
               if any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", q) for a in info["aliases"])]
    out: dict = {"tickers": tickers}
    if not tickers:
        return out
    docs = [d for d in chunks_meta["docs"] if d["ticker"] in tickers]
    m = DATE_RE.search(question)
    if m:
        when = date(int(m.group(3)), list(MONTHS.split("|")).index(m.group(1).lower()) + 1, int(m.group(2)))
        match = [d["doc_id"] for d in docs if d["period_end"] == when.isoformat()]
        if match:
            out["doc_ids"] = match
            return out
    m = FY_RE.search(question) or re.search(r"\b(20\d{2})\b", question)
    if m:
        year = int(next(g for g in m.groups() if g))
        match = [d["doc_id"] for d in docs if d["form"] == "10-K" and d["fiscal_year"] == year]
        if match:
            out["doc_ids"] = match
    return out


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.ticker = np.asarray([c.ticker for c in chunks])
        self.doc_id = np.asarray([c.doc_id for c in chunks])
        seen, docs = set(), []
        for c in chunks:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                docs.append({"doc_id": c.doc_id, "ticker": c.ticker, "form": c.form,
                             "fiscal_year": c.fiscal_year, "period_end": c.period_end})
        self.meta = {"docs": docs}
        self._dense: dict[str, np.ndarray] = {}
        self._bm25: BM25 | None = None
        self._ce = None

    def dense_matrix(self, key: str) -> np.ndarray:
        if key not in self._dense:
            self._dense[key] = chunk_matrix(key, self.chunks)
        return self._dense[key]

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(c.indexed_text) for c in self.chunks])
        return self._bm25

    def mask(self, question: str) -> tuple[np.ndarray | None, dict]:
        f = parse_filters(question, self.meta)
        if "doc_ids" in f:
            return np.isin(self.doc_id, f["doc_ids"]), f
        if f["tickers"]:
            return np.isin(self.ticker, f["tickers"]), f
        return None, f

    def search(self, question: str, cfg: RetrievalConfig, qvec: np.ndarray | None = None) -> list[tuple[int, float]]:
        mask, _ = self.mask(question) if cfg.filters else (None, {})
        rankings = []
        if cfg.dense:
            if qvec is None:
                qvec = query_matrix(cfg.embedder, [question])[0]
            s = self.dense_matrix(cfg.embedder) @ qvec
            rankings.append(self._top(s, mask, max(cfg.pool, cfg.k)))
        if cfg.bm25:
            s = self.bm25.scores(tokenize(question))
            rankings.append(self._top(s, mask, max(cfg.pool, cfg.k)))
        fused = rankings[0] if len(rankings) == 1 else _rrf(rankings)
        if cfg.rerank:
            fused = self._rerank(question, [i for i, _ in fused[: cfg.pool]], cfg.rerank)
        return fused[: cfg.k]

    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray | None, n: int) -> list[tuple[int, float]]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        idx = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def _rerank(self, question: str, ids: list[int], how: str) -> list[tuple[int, float]]:
        docs = [self.chunks[i].indexed_text for i in ids]
        if how == "minilm-ce":
            if self._ce is None:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
            scores = self._ce.predict([(question, d) for d in docs], batch_size=16, show_progress_bar=False)
        elif how == "cohere":
            scores = cohere_rerank(question, docs)
        else:
            raise ValueError(how)
        order = np.argsort(-np.asarray(scores))
        return [(ids[i], float(scores[i])) for i in order]


def _rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


def cohere_rerank(question: str, docs: list[str], model: str = "rerank-v3.5") -> list[float]:
    for attempt in range(5):
        r = requests.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"},
            json={"model": model, "query": question, "documents": docs, "top_n": len(docs)},
            timeout=60,
        )
        if r.status_code == 429:   # trial keys allow 10 calls a minute
            time.sleep(6 * (attempt + 1))
            continue
        r.raise_for_status()
        scores = [0.0] * len(docs)
        for res in r.json()["results"]:
            scores[res["index"]] = res["relevance_score"]
        return scores
    raise RuntimeError("Cohere rate limit")
