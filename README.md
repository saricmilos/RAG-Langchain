# SEC filings assistant

Questions about public companies' SEC filings, answered from the filings themselves, built twice.

- **[v1/](v1/)**: the first version, built by hand in February 2026: a LangChain agent on gpt-4o-mini
  over Pinecone, served by FastAPI on Fly.io. Kept exactly as it was (tag `v1.0`).
- **[v2/](v2/)**: the rebuild, made with Claude Code in October 2026, starting from an evaluation
  set that both versions answer.

The filings themselves (`data/`, about 1 GB from SEC EDGAR) are not in git.
