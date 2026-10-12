Retrieval on 203 answerable questions (39 about Amazon). A hit is a top-10 chunk holding the gold evidence: the XBRL-tagged number for numeric questions, the quoted passage for text questions.

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
