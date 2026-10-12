"""Run v1's own engine (v1/app/engine_deploy.py, untouched) on every question.

Run with v1's environment:
    ..\\rag_venv\\Scripts\\python.exe scripts\\v1_run_agent.py [--limit N]

Three things differ from production, all forced by what still exists:
- The Pinecone index v1 served from has been deleted, so the vector store is v1's
  local FAISS index, built by the same pipeline from the same Amazon filings.
- Embeddings run locally with the same model (all-MiniLM-L6-v2) instead of the
  Hugging Face endpoint.
- The Alpha Vantage news tool answers "rate limit reached", so the evaluation does
  not depend on a live news feed (the free tier allows 25 calls a day anyway).
Each question gets a fresh session. Writes work/v1_answers.jsonl.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"       # keep evaluation runs out of the LangSmith account
os.environ["LANGCHAIN_TRACING_V2"] = "false"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v1"))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

import app.engine_deploy as v1  # noqa: E402

OUT = REPO / "v2" / "work" / "v1_answers.jsonl"


@tool
def get_financial_news(ticker: str) -> str:
    """
    Fetch real-time financial news and sentiment from Alpha Vantage.
    Input MUST be stock ticker symbol only (example: AAPL, NVDA)
    """
    return "Alpha Vantage rate limit reached. Try again later."


class Recorder:
    """Wraps the agent so the shipped ask_with_sources_async runs unchanged and we keep its messages."""

    def __init__(self, inner):
        self.inner = inner
        self.responses: dict[str, dict] = {}

    async def ainvoke(self, inputs, config=None, **kwargs):
        resp = await self.inner.ainvoke(inputs, config=config, **kwargs)
        self.responses[config["configurable"]["session_id"]] = resp
        return resp


def build_engine():
    v1.get_financial_news = get_financial_news
    engine = v1.ChatEngine()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": "cpu"})
    engine.vector_store = FAISS.load_local(str(REPO / "data" / "vector_stores" / "faiss_index"), embeddings,
                                           allow_dangerous_deserialization=True)
    engine.create_conversational_chain()
    engine.agent_executor = Recorder(engine.agent_executor)
    return engine


async def run_one(engine, q: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t = time.time()
        row = {"id": q["id"]}
        try:
            result = await engine.ask_with_sources_async(question=q["question"], session_id=q["id"])
            row.update(answer=result["answer"], sources=result["sources"])
        except Exception as e:  # recorded, not hidden: an error is v1's answer to that question
            row.update(answer="", sources=[], error=f"{type(e).__name__}: {e}")
        row["seconds"] = round(time.time() - t, 2)
        messages = engine.agent_executor.responses.get(q["id"], {}).get("messages", [])
        row["tool_calls"] = [{"name": c["name"], "args": c["args"]}
                             for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
        row["retrieved"] = [m.content for m in messages
                            if isinstance(m, ToolMessage) and m.name == "search_knowledge_base"]
        usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
        row["tokens_in"] = sum(u["input_tokens"] for u in usage)
        row["tokens_out"] = sum(u["output_tokens"] for u in usage)
        return row


async def main() -> None:
    questions = [json.loads(line) for line in (REPO / "v2" / "eval" / "questions.jsonl").open(encoding="utf-8")]
    extra = REPO / "v2" / "eval" / "showcase.jsonl"
    if extra.exists():
        questions += [json.loads(line) for line in extra.open(encoding="utf-8")]
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")}
    todo = [q for q in questions if q["id"] not in done]
    if "--limit" in sys.argv:
        todo = todo[: int(sys.argv[sys.argv.index("--limit") + 1])]
    engine = build_engine()
    sem = asyncio.Semaphore(6)
    with OUT.open("a", encoding="utf-8") as out:
        for coro in asyncio.as_completed([run_one(engine, q, sem) for q in todo]):
            row = await coro
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{row['id']} {row['seconds']}s {'ERROR ' + row['error'][:80] if 'error' in row else ''}",
                  flush=True)


if __name__ == "__main__":
    asyncio.run(main())
