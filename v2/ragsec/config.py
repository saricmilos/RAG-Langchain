"""Paths and the company list shared by every v2 step."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "v2"

# The filings v1 downloaded with sec-edgar-downloader, read-only. Amazon's sit in
# data/completed (v1 had uploaded them to Pinecone), the other nine in data/not_completed.
RAW_DIRS = [REPO / "data" / "not_completed", REPO / "data" / "completed"]

WORK = V2 / "work"          # parsed filings, chunks, indexes: rebuilt by the scripts, not in git
EVAL = V2 / "eval"          # the question set and results: in git
WORK.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

# Names a question may use for each company, used for the company filter.
COMPANIES = {
    "AAPL": {"name": "Apple Inc.", "short": "Apple", "aliases": ["apple", "aapl"]},
    "AMZN": {"name": "Amazon.com, Inc.", "short": "Amazon", "aliases": ["amazon", "amzn", "aws"]},
    "AVGO": {"name": "Broadcom Inc.", "short": "Broadcom", "aliases": ["broadcom", "avgo"]},
    "BRK-B": {"name": "Berkshire Hathaway Inc.", "short": "Berkshire Hathaway",
              "aliases": ["berkshire", "brk", "brk-b", "brk.b"]},
    "GOOGL": {"name": "Alphabet Inc.", "short": "Alphabet", "aliases": ["alphabet", "google", "googl", "goog"]},
    "LLY": {"name": "Eli Lilly and Company", "short": "Eli Lilly", "aliases": ["eli lilly", "lilly", "lly"]},
    "META": {"name": "Meta Platforms, Inc.", "short": "Meta", "aliases": ["meta", "facebook"]},
    "MSFT": {"name": "Microsoft Corporation", "short": "Microsoft", "aliases": ["microsoft", "msft"]},
    "NVDA": {"name": "NVIDIA Corporation", "short": "NVIDIA", "aliases": ["nvidia", "nvda"]},
    "TSLA": {"name": "Tesla, Inc.", "short": "Tesla", "aliases": ["tesla", "tsla"]},
}

ANSWER_MODEL = "gpt-4o-mini"   # v1's model, kept for v2 so the gains come from retrieval and grounding
JUDGE_MODEL = "gpt-4.1"        # writes the text questions and grades text answers
