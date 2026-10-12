"""Step 1: parse the 45 filings into chunks and XBRL facts.

Writes work/chunks.jsonl, work/facts.jsonl and work/filings.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec.chunking import chunk_filing
from ragsec.config import WORK
from ragsec.filings import find_submissions, load_filing


def main() -> None:
    paths = find_submissions()
    chunks_out = (WORK / "chunks.jsonl").open("w", encoding="utf-8")
    facts_out = (WORK / "facts.jsonl").open("w", encoding="utf-8")
    filings_meta = []
    for path in paths:
        filing = load_filing(path)
        chunks = chunk_filing(filing)
        for c in chunks:
            chunks_out.write(json.dumps(c.to_dict()) + "\n")
        for fact in filing.facts:
            facts_out.write(json.dumps({"doc_id": filing.doc_id, **fact}) + "\n")
        no_page = sum(c.page is None for c in chunks)
        items = Counter(c.item for c in chunks)
        filings_meta.append({
            "doc_id": filing.doc_id, "ticker": filing.ticker, "company": filing.company,
            "form": filing.form, "accession": filing.accession, "cik": filing.cik,
            "period_end": filing.period_end.isoformat(), "fiscal_year": filing.fiscal_year,
            "fiscal_period": filing.fiscal_period, "period_label": filing.period_label,
            "url": filing.url, "submission": str(path), "chunks": len(chunks),
            "text_chars": sum(len(c.text) for c in chunks), "facts": len(filing.facts),
        })
        print(f"{filing.doc_id:24} {filing.fiscal_period:>2} FY{filing.fiscal_year} "
              f"chunks {len(chunks):4}  no page {no_page:4}  items {len(items):2}  facts {len(filing.facts)}")
    chunks_out.close()
    facts_out.close()
    (WORK / "filings.json").write_text(json.dumps(filings_meta, indent=1), encoding="utf-8")
    print("filings", len(filings_meta), "chunks", sum(f["chunks"] for f in filings_meta))


if __name__ == "__main__":
    main()
