"""Step 2: build eval/questions.jsonl (numeric, text and unanswerable questions)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec import llm
from ragsec.config import EVAL
from ragsec.evalset import build


def main() -> None:
    out = EVAL / "questions.jsonl"
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} exists; pass --force to rebuild it (the results depend on it)")
    questions = build()
    with out.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(Counter(q["type"] for q in questions))
    print(Counter(q["ticker"] for q in questions))
    print(f"LLM cost ${llm.total_cost():.2f}")


if __name__ == "__main__":
    main()
