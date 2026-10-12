"""Step 6: score v1's and v2's answers with the same rules.

Numeric answers are checked against the XBRL value in code; text answers and refusals
are graded by gpt-4.1 against the reference answer and evidence. Writes
eval/answers_scored.jsonl (each answer with its grade), eval/answer_summary.md and
eval/summary.json.
"""
import hashlib
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec import llm
from ragsec.config import EVAL, WORK
from ragsec.embed import load_chunks
from ragsec.metrics import is_hit, judge_text, judge_unanswerable, looks_declined, numeric_correct

COHERE_PER_SEARCH = 0.002   # rerank-v3.5: $2 per 1,000 searches
JUDGE_CACHE = WORK / "judge_cache.json"   # an answer is graded once, so re-scoring is stable and free
_cache_lock = threading.Lock()


def load(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def grade(q: dict, answer: str, error: bool, cache: dict) -> dict:
    if error or not answer.strip():
        return {"grade": "error", "score": 0.0, "why": "no answer"}
    if q["type"] == "numeric":
        ok = numeric_correct(answer, q["gold"])
        declined = not ok and looks_declined(answer)
        return {"grade": "correct" if ok else ("declined" if declined else "incorrect"), "score": float(ok), "why": ""}
    key = hashlib.sha1(f"{q['id']}\n{answer}".encode("utf-8")).hexdigest()
    if key not in cache:
        result = judge_text(q, answer) if q["type"] == "text" else judge_unanswerable(q, answer)
        with _cache_lock:
            cache[key] = result
    return cache[key]


def main() -> None:
    questions = [json.loads(line) for line in (EVAL / "questions.jsonl").open(encoding="utf-8")]
    v1 = load(WORK / "v1_answers.jsonl")
    v2 = load(WORK / "v2_answers.jsonl")
    chunks = {c.id: c for c in load_chunks()}
    reranked_with_cohere = "Cohere" in next(iter(v2.values()))["system"]
    cache = json.loads(JUDGE_CACHE.read_text(encoding="utf-8")) if JUDGE_CACHE.exists() else {}

    def score(q: dict) -> dict:
        a1, a2 = v1[q["id"]], v2[q["id"]]
        row = {"id": q["id"], "type": q["type"], "ticker": q["ticker"], "question": q["question"]}
        # v1: its sources never name a filing, so its citations cannot point at the evidence.
        v1_found = q["type"] != "unanswerable" and any(is_hit(q, t, "AMZN") for t in a1["retrieved"])
        row["v1"] = {"answer": a1["answer"], "sources": a1["sources"],
                     **grade(q, a1["answer"], "error" in a1, cache), "evidence_retrieved": v1_found,
                     "cites_evidence": False, "seconds": a1["seconds"],
                     "cost": llm.cost_of("gpt-4o-mini", a1["tokens_in"], a1["tokens_out"]),
                     "error": a1.get("error")}
        context_hit = q["type"] != "unanswerable" and any(
            is_hit(q, chunks[i].indexed_text, chunks[i].ticker) for i in a2["context_ids"])
        cited_hit = q["type"] != "unanswerable" and any(
            is_hit(q, chunks[c["chunk_id"]].indexed_text, chunks[c["chunk_id"]].ticker) for c in a2["cited"])
        row["v2"] = {"answer": a2["answer"], "cited": [c["chunk_id"] for c in a2["cited"]],
                     **grade(q, a2["answer"], False, cache), "evidence_retrieved": context_hit, "cites_evidence": cited_hit,
                     "seconds": a2["seconds"],
                     "cost": a2["cost"] + (COHERE_PER_SEARCH if reranked_with_cohere else 0.0)}
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(score, questions))
    JUDGE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    with (EVAL / "answers_scored.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    summary = {}
    for system in ("v1", "v2"):
        s = {}
        for name, sel in [("all", rows), ("amazon", [r for r in rows if r["ticker"] == "AMZN"])]:
            num = [r[system]["score"] for r in sel if r["type"] == "numeric"]
            txt = [r[system]["score"] for r in sel if r["type"] == "text"]
            una = [r[system]["score"] for r in sel if r["type"] == "unanswerable"]
            answerable = [r for r in sel if r["type"] != "unanswerable"]
            s[name] = {
                "numeric_correct": pct(num), "text_score": pct(txt), "unanswerable_declined": pct(una),
                "answerable_score": pct([r[system]["score"] for r in answerable]),
                "evidence_retrieved": pct([r[system]["evidence_retrieved"] for r in answerable]),
                "cites_evidence": pct([r[system]["cites_evidence"] for r in answerable]),
                "false_declines": pct([r[system]["grade"] == "declined" for r in answerable]),
                "errors": sum(r[system]["grade"] == "error" for r in sel),
                "n": len(sel), "n_numeric": len(num), "n_text": len(txt), "n_unanswerable": len(una),
            }
        s["median_seconds"] = statistics.median(r[system]["seconds"] for r in rows)
        s["cost_per_100"] = round(100 * statistics.mean(r[system]["cost"] for r in rows), 3)
        summary[system] = s
    summary["judge_cost"] = round(llm.total_cost(), 3)
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    a, b = summary["v1"], summary["v2"]
    lines = [
        "| | v1 as shipped | v2 |", "|---|---|---|",
        f"| Numeric answers correct ({a['all']['n_numeric']}) | {a['all']['numeric_correct']}% | {b['all']['numeric_correct']}% |",
        f"| Text answers, graded score ({a['all']['n_text']}) | {a['all']['text_score']}% | {b['all']['text_score']}% |",
        f"| Unanswerable questions declined ({a['all']['n_unanswerable']}) | {a['all']['unanswerable_declined']}% | {b['all']['unanswerable_declined']}% |",
        f"| Evidence in what the model was shown | {a['all']['evidence_retrieved']}% | {b['all']['evidence_retrieved']}% |",
        f"| Answer cites the passage holding the evidence | {a['all']['cites_evidence']}% | {b['all']['cites_evidence']}% |",
        f"| Crashed (no answer) | {a['all']['errors']} | {b['all']['errors']} |",
        f"| Amazon only: answerable score ({a['amazon']['n'] - a['amazon']['n_unanswerable']}) | {a['amazon']['answerable_score']}% | {b['amazon']['answerable_score']}% |",
        f"| Median seconds per answer | {a['median_seconds']:.1f} | {b['median_seconds']:.1f} |",
        f"| Cost per 100 questions | ${a['cost_per_100']:.2f} | ${b['cost_per_100']:.2f} |",
    ]
    (EVAL / "answer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"judge cost ${summary['judge_cost']}")


if __name__ == "__main__":
    main()
