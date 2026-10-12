# SEC filings assistant

Questions about public companies, answered from their SEC filings, with the passage each answer
came from. Built twice:

- **[v1/](v1/)**: the first version, built by hand in February 2026. A LangChain agent on
  gpt-4o-mini over Pinecone, served by FastAPI on Fly.io. Kept exactly as it was (tag `v1.0`).
- **[v2/](v2/)**: the rebuild, made with Claude Code in October 2026. It started from an answer key
  of 223 questions that both versions answer, then changed one thing at a time.

| Same 223 questions, same model | v1 | v2 |
|---|---|---|
| Numbers answered right | 1% | 94% |
| Text answers, graded | 42% | 84% |
| Says so when the filings can't answer | 60% | 100% |
| Cites the passage holding the evidence | 0% | 72% |

How it was measured, and what is still wrong: [v2/README.md](v2/README.md). Try both versions
side by side at [saricmilos.com/projects/sec-filings-rag](https://www.saricmilos.com/projects/sec-filings-rag).

The filings themselves (`data/`, about 1 GB from SEC EDGAR) are not in git.
