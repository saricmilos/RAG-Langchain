"""What v1's index actually holds: companies, markup, and how much of each chunk its embedder read.

Needs work/v1_chunks.jsonl (scripts/v1_export_index.py). Writes eval/v1_index_stats.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec.config import EVAL, WORK

TAG_RE = re.compile(r"<[^>]*>")
UUENCODE_RE = re.compile(r"^M[!-`]{60}$", re.M)   # lines of uuencoded images and PDFs


def main() -> None:
    from transformers import AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()

    rows = [json.loads(line) for line in (WORK / "v1_chunks.jsonl").open(encoding="utf-8")]
    texts = [r["text"] for r in rows]
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    lengths = np.asarray([len(ids) for ids in tok(texts)["input_ids"]])
    window = 256   # all-MiniLM-L6-v2 max_seq_length: the rest of a chunk is cut before embedding
    visible = sum(len(re.sub(r"\s+", " ", TAG_RE.sub(" ", t)).strip()) for t in texts)
    stats = {
        "chunks": len(rows),
        "companies": dict(Counter(r["ticker"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "chunk_chars_median": int(np.median([len(t) for t in texts])),
        "tokens_median": int(np.median(lengths)),
        "pct_chunks_over_window": round(100 * float((lengths > window).mean()), 1),
        "pct_tokens_never_embedded": round(100 * float(np.clip(lengths - window, 0, None).sum() / lengths.sum()), 1),
        "pct_chars_markup": round(100 * (1 - visible / sum(len(t) for t in texts)), 1),
        "pct_chunks_with_uuencoded_lines": round(100 * sum(bool(UUENCODE_RE.search(t)) for t in texts) / len(texts), 1),
    }
    (EVAL / "v1_index_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
