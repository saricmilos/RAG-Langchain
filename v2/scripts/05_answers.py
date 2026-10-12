"""Step 5: v2 answers every question (and the showcase set) with the chosen retrieval setup.

    python scripts/05_answers.py
Writes work/v2_answers.jsonl, one row per question, resumable.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec import llm
from ragsec.answer import answer
from ragsec.config import EVAL, WORK
from ragsec.embed import load_chunks, query_matrix
from ragsec.retrieve import Retriever

sys.path.insert(0, str(Path(__file__).resolve().parent))
FINAL_NAME, FINAL_CFG = import_module("04_retrieval").final_system()

OUT = WORK / "v2_answers.jsonl"


def main() -> None:
    questions = [json.loads(line) for name in ("questions.jsonl", "showcase.jsonl")
                 for line in (EVAL / name).open(encoding="utf-8")]
    done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    retriever = Retriever(load_chunks())
    retriever.dense_matrix(FINAL_CFG.embedder)
    retriever.bm25  # build once before the threads start
    query_matrix(FINAL_CFG.embedder, [q["question"] for q in questions])   # one batch, cached

    def run(q: dict) -> dict:
        return {"id": q["id"], "system": FINAL_NAME, **answer(retriever, q["question"], FINAL_CFG)}

    with OUT.open("a", encoding="utf-8") as out, ThreadPoolExecutor(6) as pool:
        for row in pool.map(run, todo):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    print(f"{len(todo)} answered with '{FINAL_NAME}', LLM cost ${llm.total_cost():.3f}")


if __name__ == "__main__":
    main()
