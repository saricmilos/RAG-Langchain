"""Step 7: write the demo's data into the website (saricmilos.com, src/content/sec-filings-rag).

    python scripts/07_export_site.py [path to the site repo]
Everything the page shows is precomputed here: no model runs when someone opens it.
"""
import json
import random
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec.config import COMPANIES, EVAL, WORK
from ragsec.embed import load_chunks
from ragsec.metrics import is_hit, looks_declined

SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\dev\sites\saricmilos.com")
OUT = SITE / "src" / "content" / "sec-filings-rag"

GROUPS = {
    "showcase": "v1's own test questions",
    "numeric": "Numbers",
    "text": "Risks, strategy, business",
    "unanswerable": "Not in the filings",
}
PICK = {"numeric": 16, "text": 14, "unanswerable": 6}
FEATURED = "q078"
UNANSWERABLE_PICK = ["Netflix", "fiscal year 2015", "first quarter of 2026", "deliver in 2030", "Goldman",
                     "acquire Intel"]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def short_period(c) -> str:
    if c.form == "10-K":
        return f"FY{c.fiscal_year}"
    end = date.fromisoformat(c.period_end)
    return f"{c.fiscal_period} FY{c.fiscal_year}, quarter ended {end:%b} {end.day}, {end.year}"


def chunk_label(c) -> str:
    parts = [f"{COMPANIES[c.ticker]['short']} {c.form} {short_period(c)}"]
    if c.item:
        parts.append(f"Item {c.item}{(' ' + c.item_title) if len(c.item_title) < 40 else ''}")
    if c.page:
        parts.append(f"p. {c.page}")
    return " · ".join(parts)


def fragment_url(url: str, text: str) -> str:
    """Link to the filing on sec.gov, scrolled to and highlighting the passage where browsers support it."""
    words = re.findall(r"[A-Za-z0-9$%][\w$%.,'’-]*", text)[:7]
    if len(words) < 4 or "|" in text.split("\n")[0]:
        return url
    return f"{url}#:~:text={quote(' '.join(words))}"


def money(value: float, unit: str) -> str:
    if unit == "usd_per_share":
        return f"${value:,.2f} per share"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e6:,.0f} million"


def pick_questions(questions: list[dict], scored: dict) -> list[dict]:
    rng = random.Random(11)
    chosen = []
    for kind in ("numeric", "text"):
        # A zero shown as a dash (Amazon bought back no stock in 2024) is a flaw in the answer key, not a question
        pool = [q for q in questions if q["type"] == kind and (kind != "numeric" or q["gold"]["value"] != 0)]
        rng.shuffle(pool)
        wrong = [q for q in pool if scored[q["id"]]["v2"]["score"] < 1][:2]   # show where v2 still fails
        picked = list(wrong)
        for ticker in list(COMPANIES) + ["AMZN", "AMZN"]:
            for q in pool:
                if len(picked) >= PICK[kind]:
                    break
                if q["ticker"] == ticker and q not in picked:
                    picked.append(q)
                    break
        chosen += picked
    for needle in UNANSWERABLE_PICK:
        chosen += [q for q in questions if q["type"] == "unanswerable" and needle in q["question"]][:1]
    return chosen


def main() -> None:
    chunks = {c.id: c for c in load_chunks()}
    filings = {f["doc_id"]: f for f in json.loads((WORK / "filings.json").read_text(encoding="utf-8"))}
    questions = jsonl(EVAL / "questions.jsonl")
    showcase = jsonl(EVAL / "showcase.jsonl")
    scored = {r["id"]: r for r in jsonl(EVAL / "answers_scored.jsonl")}
    v1 = {r["id"]: r for r in jsonl(WORK / "v1_answers.jsonl")}
    v2 = {r["id"]: r for r in jsonl(WORK / "v2_answers.jsonl")}

    items = []
    for q in showcase + pick_questions(questions, scored):
        a1, a2, s = v1[q["id"]], v2[q["id"]], scored.get(q["id"])
        gold = None
        if q["type"] == "numeric":
            f = filings[q["doc_id"]]
            gold = {"text": money(q["gold"]["value"], q["gold"]["unit"]),
                    "source": f"{COMPANIES[q['ticker']]['short']} {f['form']}, {f['period_label']}", "url": f["url"]}
        elif q["type"] == "text":
            c = chunks[q["gold"]["chunk_id"]]
            gold = {"text": q["gold"]["answer"], "evidence": q["gold"]["evidence"], "source": chunk_label(c),
                    "url": fragment_url(c.url, q["gold"]["evidence"])}
        elif q["type"] == "unanswerable":
            gold = {"text": f"Nothing to find: {q['gold']['reason']}."}

        shown_v1 = a1["retrieved"][0] if a1["retrieved"] else ""
        cited = {c["chunk_id"] for c in a2["cited"]}
        items.append({
            "id": q["id"],
            "group": GROUPS[q["type"]],
            "type": q["type"],
            "ticker": q["ticker"],
            "q": q["question"],
            "gold": gold,
            "v1": {
                "answer": a1["answer"] or "",
                "error": (a1.get("error") or "").split(":")[0] or None,
                "grade": s["v1"]["grade"] if s else None,
                "sources": a1["sources"],
                "searched": [c["args"].get("query") or c["args"].get("ticker") for c in a1["tool_calls"]],
                "shown": shown_v1[:900],
                "shownChars": len(shown_v1),
                "found": s["v1"]["evidence_retrieved"] if s else None,
            },
            "v2": {
                "answer": a2["answer"],
                "grade": s["v2"]["grade"] if s else None,
                "found": s["v2"]["evidence_retrieved"] if s else None,
                "context": [
                    {
                        "n": n,
                        "label": chunk_label(chunks[cid]),
                        "url": fragment_url(chunks[cid].url, chunks[cid].text),
                        "text": chunks[cid].text if cid in cited else None,
                        "evidence": bool(q["type"] in ("numeric", "text")
                                         and is_hit(q, chunks[cid].indexed_text, chunks[cid].ticker)),
                    }
                    for n, cid in enumerate(a2["context_ids"], 1)
                ],
            },
        })

    # The page opens on a sharp contrast, chosen by hand: v1 says NVIDIA's fiscal 2025 net income was
    # "approximately $0.7 billion" (it was $72,880 million), v2 gets it right from the 10-K.
    ids = {i["id"] for i in items}
    featured = FEATURED if FEATURED in ids else next(
        (i["id"] for i in items if i["type"] == "numeric" and i["v1"]["grade"] == "incorrect"
         and i["v2"]["grade"] == "correct" and not looks_declined(i["v1"]["answer"])), None)

    retrieval = json.loads((EVAL / "retrieval.json").read_text(encoding="utf-8"))
    summary = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))
    v1_index = json.loads((EVAL / "v1_index_stats.json").read_text(encoding="utf-8"))
    filings_list = list(filings.values())
    order = list(retrieval["summary"])
    results = {
        "corpus": {
            "companies": len(COMPANIES),
            "filings": len(filings_list),
            "tenK": sum(f["form"] == "10-K" for f in filings_list),
            "tenQ": sum(f["form"] == "10-Q" for f in filings_list),
            "chunks": len(chunks),
        },
        "questions": {k: sum(q["type"] == k for q in questions) for k in ("numeric", "text", "unanswerable")},
        "v1Index": v1_index,
        "answers": summary,
        "ablation": [
            {"label": name, **{k: retrieval["summary"][name]["all"][k] for k in ("hit@1", "hit@5", "hit@10", "mrr@10")},
             "numeric": retrieval["summary"][name]["numeric"]["hit@10"],
             "text": retrieval["summary"][name]["text"]["hit@10"],
             "amazon": retrieval["summary"][name]["amazon"]["hit@10"]}
            for name in order
        ],
        "final": next(iter(v2.values()))["system"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "demo.json").write_text(json.dumps({"featured": featured, "items": items}, ensure_ascii=False,
                                              separators=(",", ":")),
                                   encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} demo questions, {(OUT / 'demo.json').stat().st_size // 1024} KB -> {OUT}")


if __name__ == "__main__":
    main()
