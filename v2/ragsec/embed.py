"""Chunk and query embeddings for every embedder in the ablation, cached in work/emb."""
from __future__ import annotations

import json
import os
import threading

import numpy as np
from dotenv import load_dotenv

from .chunking import Chunk
from .config import REPO, WORK

load_dotenv(REPO / ".env")

EMB_DIR = WORK / "emb"
EMB_DIR.mkdir(exist_ok=True)

BGE_QUERY = "Represent this sentence for searching relevant passages: "

# key -> (backend, model, embed the header line too?, query prefix)
EMBEDDERS = {
    "minilm-nohdr": ("st", "sentence-transformers/all-MiniLM-L6-v2", False, ""),
    "minilm": ("st", "sentence-transformers/all-MiniLM-L6-v2", True, ""),
    "bge-small": ("st", "BAAI/bge-small-en-v1.5", True, BGE_QUERY),
    "oai-small": ("openai", "text-embedding-3-small", True, ""),
    "oai-large": ("openai", "text-embedding-3-large", True, ""),
}

_st_models: dict = {}
_query_lock = threading.Lock()


def load_chunks() -> list[Chunk]:
    with (WORK / "chunks.jsonl").open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def _st(model: str):
    if model not in _st_models:
        from sentence_transformers import SentenceTransformer
        _st_models[model] = SentenceTransformer(model, device="cpu")
    return _st_models[model]


def _openai_embed(model: str, texts: list[str], batch: int = 200) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120)
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
        print(f"  {model}: {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)


def embed_texts(key: str, texts: list[str], is_query: bool) -> np.ndarray:
    backend, model, _, prefix = EMBEDDERS[key]
    if is_query and prefix:
        texts = [prefix + t for t in texts]
    if backend == "st":
        m = _st(model).encode(texts, batch_size=32, show_progress_bar=not is_query, convert_to_numpy=True)
    else:
        m = _openai_embed(model, texts)
    return _normalize(m.astype(np.float32))


def chunk_matrix(key: str, chunks: list[Chunk] | None = None) -> np.ndarray:
    """The (n_chunks, dim) matrix for an embedder, computed once and cached."""
    path = EMB_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    chunks = chunks or load_chunks()
    with_header = EMBEDDERS[key][2]
    texts = [c.indexed_text if with_header else c.text for c in chunks]
    m = embed_texts(key, texts, is_query=False)
    np.save(path, m)
    return m


def query_matrix(key: str, questions: list[str]) -> np.ndarray:
    """Query embeddings, cached per embedder and question text."""
    path = EMB_DIR / f"{key}.queries.json"
    with _query_lock:   # answer threads share this file
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        missing = [q for q in dict.fromkeys(questions) if q not in cache]
        if missing:
            for q, v in zip(missing, embed_texts(key, missing, is_query=True)):
                cache[q] = v.tolist()
            path.write_text(json.dumps(cache), encoding="utf-8")
        return np.asarray([cache[q] for q in questions], dtype=np.float32)
