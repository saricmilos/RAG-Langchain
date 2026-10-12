"""Step 4: the retrieval ablation. Every system answers the same questions; we score its top 10.

    python scripts/04_retrieval.py            # all systems whose embeddings exist
Writes eval/retrieval.json (per question ranks) and eval/retrieval_summary.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec.config import EVAL, REPO, WORK
from ragsec.embed import EMB_DIR, embed_texts, load_chunks, query_matrix
from ragsec.metrics import is_hit, retrieval_scores
from ragsec.retrieve import RetrievalConfig, Retriever

# The ablation, in order: each row changes one thing from the row before it.
SYSTEMS = [
    ("v2 chunks, v1's embedder (MiniLM)", RetrievalConfig(embedder="minilm-nohdr")),
    ("+ company/period header in each chunk", RetrievalConfig(embedder="minilm")),
    ("bge-small embedder", RetrievalConfig(embedder="bge-small")),
    ("OpenAI text-embedding-3-small", RetrievalConfig(embedder="oai-small")),
    ("OpenAI text-embedding-3-large", RetrievalConfig(embedder="oai-large")),
    ("+ BM25 keyword search (hybrid)", RetrievalConfig(embedder="oai-large", bm25=True)),
    ("+ company/period filters from the question", RetrievalConfig(embedder="oai-large", bm25=True, filters=True)),
    ("+ MiniLM cross-encoder reranker", RetrievalConfig(embedder="oai-large", bm25=True, filters=True, rerank="minilm-ce")),
    ("+ Cohere rerank-v3.5 instead", RetrievalConfig(embedder="oai-large", bm25=True, filters=True, rerank="cohere")),
]
V1_NAME = "v1 as shipped (Amazon-only index, raw SGML chunks)"

# What v2 answers with. Cohere's reranker found the evidence slightly more often in the top
# 5-10 (+2-3 points, a handful of questions) but less often first, and would tie v2 to a
# rate-limited trial key, so the setup before it wins.
FINAL = "+ company/period filters from the question"


def final_system() -> tuple[str, RetrievalConfig]:
    return FINAL, dict(SYSTEMS)[FINAL]


def load_questions() -> list[dict]:
    with (EVAL / "questions.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def v1_ranks(questions: list[dict]) -> dict:
    """v1's own index and embedder: MiniLM question vectors against its FAISS file, top 10."""
    import faiss
    index = faiss.read_index(str(REPO / "data" / "vector_stores" / "faiss_index" / "index.faiss"))
    with (WORK / "v1_chunks.jsonl").open(encoding="utf-8") as f:
        v1 = [json.loads(line) for line in f]
    qv = embed_texts("minilm", [q["question"] for q in questions], is_query=True)
    _, ids = index.search(qv, 10)
    out = {}
    for q, row in zip(questions, ids):
        out[q["id"]] = {"ids": [int(i) for i in row],
                        "hits": [is_hit(q, v1[i]["text"], v1[i]["ticker"]) for i in row]}
    return out


def v2_ranks(retriever: Retriever, questions: list[dict], cfg: RetrievalConfig) -> dict:
    qv = query_matrix(cfg.embedder, [q["question"] for q in questions])
    out = {}
    for q, vec in zip(questions, qv):
        ranked = retriever.search(q["question"], cfg, qvec=vec)
        chunks = [retriever.chunks[i] for i, _ in ranked]
        out[q["id"]] = {"ids": [c.id for c in chunks], "hits": [is_hit(q, c.indexed_text, c.ticker) for c in chunks]}
    return out


def summarize(ranks: dict, questions: list[dict]) -> dict:
    def avg(qs):
        rows = [retrieval_scores(ranks[q["id"]]["hits"]) for q in qs]
        return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]} | {"n": len(qs)}
    groups = {
        "all": questions,
        "numeric": [q for q in questions if q["type"] == "numeric"],
        "text": [q for q in questions if q["type"] == "text"],
        "amazon": [q for q in questions if q["ticker"] == "AMZN"],
    }
    return {name: avg(qs) for name, qs in groups.items() if qs}


def main() -> None:
    questions = [q for q in load_questions() if q["type"] != "unanswerable"]
    cache_path = EVAL / "retrieval.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    retriever = Retriever(load_chunks())

    if V1_NAME not in cache.get("ranks", {}):
        cache.setdefault("ranks", {})[V1_NAME] = v1_ranks(questions)
    for name, cfg in SYSTEMS:
        if name in cache["ranks"] and "--force" not in sys.argv:
            continue
        if not (EMB_DIR / f"{cfg.embedder}.npy").exists():
            print(f"skip (no embeddings yet): {name}")
            continue
        print(f"running: {name}", flush=True)
        cache["ranks"][name] = v2_ranks(retriever, questions, cfg)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    order = [n for n in [V1_NAME] + [n for n, _ in SYSTEMS] if n in cache["ranks"]]
    cache["summary"] = {name: summarize(cache["ranks"][name], questions) for name in order}
    cache["configs"] = {name: cfg.__dict__ for name, cfg in SYSTEMS}
    cache["final"] = FINAL
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    lines = ["| System | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Numeric hit@10 | Text hit@10 | Amazon hit@10 |",
             "|---|---|---|---|---|---|---|---|"]
    for name in order:
        s = cache["summary"].get(name)
        if not s:
            continue
        a = s["all"]
        lines.append(f"| {name} | {a['hit@1']:.0%} | {a['hit@5']:.0%} | {a['hit@10']:.0%} | {a['mrr@10']:.2f} | "
                     f"{s['numeric']['hit@10']:.0%} | {s['text']['hit@10']:.0%} | {s['amazon']['hit@10']:.0%} |")
    n = cache["summary"][V1_NAME]["all"]["n"]
    na = cache["summary"][V1_NAME]["amazon"]["n"]
    table = "\n".join(lines)
    (EVAL / "retrieval_summary.md").write_text(
        f"Retrieval on {n} answerable questions ({na} about Amazon). A hit is a top-10 chunk holding the gold "
        f"evidence: the XBRL-tagged number for numeric questions, the quoted passage for text questions.\n\n"
        f"{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
