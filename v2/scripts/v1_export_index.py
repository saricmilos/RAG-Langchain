"""Copy the text of v1's local FAISS index out of its LangChain pickle, read-only.

Run with v1's environment (rag_venv), which has the LangChain classes the pickle needs:
    ..\\rag_venv\\Scripts\\python.exe scripts\\v1_export_index.py

Writes work/v1_chunks.jsonl: position in the FAISS index, company, raw text.
"""
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "data" / "vector_stores" / "faiss_index"
OUT = REPO / "v2" / "work" / "v1_chunks.jsonl"


def main() -> None:
    with (INDEX / "index.pkl").open("rb") as f:
        docstore, index_to_id = pickle.load(f)
    with OUT.open("w", encoding="utf-8") as out:
        for pos in range(len(index_to_id)):
            doc = docstore._dict[index_to_id[pos]]
            out.write(json.dumps({"pos": pos, "ticker": doc.metadata.get("company_name"),
                                  "source": doc.metadata.get("source"), "text": doc.page_content}) + "\n")
    print(f"{len(index_to_id)} chunks -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
