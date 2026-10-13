# v2: the SEC filings assistant, rebuilt and measured

v1 (in [`../v1`](../v1), February 2026) was a LangChain agent that answered questions about
public companies from their SEC filings. It had never been tested against known answers. v2
started from that test: 223 questions with known answers, which v1's own code answered first,
then every change to v2 was measured against the same set.

Built in October 2026. The results are on
[saricmilos.com/projects/sec-filings-rag](https://www.saricmilos.com/projects/sec-filings-rag).

## Results

Same 223 questions, same answering model (gpt-4o-mini):

| | v1 as shipped | v2 |
|---|---|---|
| Numeric answers correct (120) | 0.8% | 94.2% |
| Text answers, graded score (83) | 41.6% | 84.3% |
| Unanswerable questions declined (20) | 60.0% | 100.0% |
| Evidence in what the model was shown | 2.0% | 75.9% |
| Answer cites the passage holding the evidence | 0.0% | 71.9% |
| Crashed (no answer) | 5 | 0 |
| Amazon only: answerable score (39) | 33.3% | 91.0% |
| Median seconds per answer | 3.6 | 4.3 |
| Cost per 100 questions | $0.12 | $0.03 |

v1's 42% on text questions is not grounded: it had the evidence in front of it about 2% of the
time, and the grader credits the generic risk-factor answers gpt-4o-mini writes from its own
knowledge. Its numbers show the difference: 1 right out of 120.

Finding the evidence, one change at a time (each row changes one thing from the row above; the
first row is v1's own index, which holds only Amazon):

| System | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Numeric hit@10 | Text hit@10 | Amazon hit@10 |
|---|---|---|---|---|---|---|---|
| v1 as shipped (Amazon-only index, raw SGML chunks) | 0% | 0% | 2% | 0.01 | 0% | 5% | 10% |
| v2 chunks, v1's embedder (MiniLM) | 8% | 15% | 18% | 0.11 | 17% | 20% | 20% |
| + company/period header in each chunk | 17% | 36% | 47% | 0.26 | 56% | 34% | 41% |
| bge-small embedder | 33% | 60% | 69% | 0.44 | 88% | 42% | 59% |
| OpenAI text-embedding-3-small | 38% | 63% | 71% | 0.49 | 90% | 43% | 64% |
| OpenAI text-embedding-3-large | 43% | 66% | 72% | 0.53 | 93% | 42% | 64% |
| + BM25 keyword search (hybrid) | 45% | 67% | 73% | 0.54 | 94% | 43% | 67% |
| + company/period filters from the question | 46% | 73% | 76% | 0.57 | 98% | 45% | 67% |
| + MiniLM cross-encoder reranker | 39% | 68% | 74% | 0.51 | 94% | 45% | 64% |
| + Cohere rerank-v3.5 instead | 41% | 76% | 78% | 0.56 | 99% | 47% | 69% |

The header naming company, period and Item on every chunk is the largest single gain. The free
bge-small embedder, run locally, comes within 2 points of OpenAI's small model. v2 answers with
"+ company/period filters". The ms-marco cross-encoder made things worse, and
Cohere's reranker found the evidence slightly more often in the top 5 to 10 but less often
first; neither earned its place.

What is still wrong:

- **Line items with near twins.** Asked for net income, v2 sometimes gives the figure before
  noncontrolling interests (Berkshire, Tesla), with the right line in front of it. Asked for
  Amazon's purchases of property and equipment, it gave the figure net of proceeds; the gross
  line was not among its excerpts.
- **Text questions with many good passages.** Hit@10 on text is 45%: a question about "risks from
  IT failures" has several fair answers in a 10-K, and only the passage it was written from counts
  as a hit. The graded answers (84%) say more about these.
- **The set is also the development set.** The choices made on it were few and coarse (which
  embedder, whether to filter or rerank), but there is no held-out split.
- One numeric answer key is a zero printed as a dash (Amazon bought back no stock in 2024); it
  counts as a miss for both versions.

## The corpus

The 10-K and 10-Q filings v1 downloaded from EDGAR for ten companies (Apple, Amazon, Broadcom,
Berkshire Hathaway, Alphabet, Eli Lilly, Meta, Microsoft, NVIDIA, Tesla): 13 annual and 32
quarterly reports, filed from early 2025 to early 2026. They sit in `../data` (not in git) and are
only read.

## The evaluation set (`eval/questions.jsonl`)

| Kind | How the answer key was made | Count |
|---|---|---|
| Numeric | From the XBRL facts tagged in the filing itself (revenue, net income, R&D, operating income, diluted EPS, cash, capex, buybacks, total assets) for the filing's own period, consolidated only. Nobody typed a number. | 120 |
| Text | gpt-4.1 wrote one question per randomly drawn passage (risk factors, MD&A, business, market risk, cybersecurity, legal), with a reference answer and a verbatim evidence quote. A question was kept only if the quote appears in the passage. | 83 |
| Unanswerable | Written by hand: companies not in the corpus, years before it, quarters after it, forecasts, market data. The right answer is to say the filings don't cover it. | 20 |

Amazon has more questions (41) than the others because it is the only company v1 had indexed, so
it is where the two versions' methods can be compared fairly.

`eval/showcase.jsonl` holds the twelve Apple questions v1's README listed and never answered.
They have no answer key and are only shown on the site.

## How it is scored

- **Retrieval:** a retrieved chunk is a hit if it holds the gold evidence: the tagged number as the
  filing prints it (for the right company), or at least half of the evidence quote's six-word
  shingles. v1's chunks are raw SGML, so their tags are stripped before matching, which only helps
  v1. Hit@k and MRR@10 over the 203 answerable questions.
- **Numeric answers:** checked in code. Right if a number in the answer matches the tagged value to
  the precision the answer gives (and within 5%), with its scale word ("billion") applied.
- **Text answers:** graded by gpt-4.1 against the reference answer and evidence: correct (1),
  partial (0.5), incorrect or declined (0).
- **Unanswerable:** right only if the answer declines rather than states something (gpt-4.1 judges).
- **Citations:** whether a passage the answer cites holds the gold evidence. v1's sources are one
  fixed string, so it scores 0 by construction.

Both versions answer with gpt-4o-mini, so the differences come from what the model is shown and
told, not from a stronger model.

## How v1 was run

`scripts/v1_run_agent.py` imports v1's `app/engine_deploy.py` unchanged and calls its own
`ask_with_sources_async`, one fresh session per question. Three things had to differ from
production: the Pinecone index it served from has been deleted, so it searches v1's local FAISS
index (built by the same pipeline from the same Amazon filings); the embeddings run locally with
the same model; and the Alpha Vantage news tool answers "rate limit reached", so no answer depends
on a live feed. LangSmith tracing is off for the run.

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_parse.py` | Takes each submission's primary document, turns its HTML into paragraphs and whole tables, tags each block with its Part, Item and printed page, and reads its XBRL facts (`ragsec/filings.py`). Cuts chunks that never cross an Item, keep tables whole (or split by rows with the header repeated) and fit the embedder's 512 tokens (`ragsec/chunking.py`). 12,995 chunks. |
| 2 | `scripts/02_build_evalset.py` | Builds the evaluation set (`ragsec/evalset.py`). |
| 3 | `scripts/03_embed.py` | Embeds every chunk with each embedder in the ablation (`ragsec/embed.py`). |
| 4 | `scripts/04_retrieval.py` | The retrieval ablation (`ragsec/retrieve.py`, `ragsec/metrics.py`). |
| 5 | `scripts/05_answers.py` | v2 answers every question with the chosen setup (`ragsec/answer.py`). |
| 6 | `scripts/06_score.py` | Scores v1's and v2's answers with the same rules. |
| 7 | `scripts/07_export_site.py` | Writes the website's demo data. |

v1 side: `scripts/v1_export_index.py` copies the text out of v1's FAISS pickle (run with v1's
`rag_venv`), `scripts/v1_index_stats.py` measures it, `scripts/v1_run_agent.py` runs v1's engine.

## Running it

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/01_parse.py
```

Steps 2 to 6 need `OPENAI_API_KEY` (and `COHERE_API_KEY` for the Cohere reranker) in `../.env`.
The parsed filings, chunks and embeddings go to `work/` (not in git); the evaluation set and every
result go to `eval/` (in git).
