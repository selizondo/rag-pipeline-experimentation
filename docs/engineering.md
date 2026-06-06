# Design and Tradeoffs

---

## Local SentenceTransformers over OpenAI Embeddings

This project uses `all-MiniLM-L6-v2` (384-dim) and `all-mpnet-base-v2` (768-dim) from SentenceTransformers rather than OpenAI's embedding API. Multi-paper ingestion with 12 grid configs across ~5,000 chunks per paper would incur meaningful API cost at OpenAI rates. Local embeddings also enable offline experimentation and direct model-to-model comparison without API versioning variability.

Tradeoff: SentenceTransformers models top out below OpenAI text-embedding-3-large at MTEB benchmarks. `rag-pipeline-systematic-evals` explores the OpenAI embedding side; this project explores the local side.

---

## YAML-Driven Grid over Code-Only Config

Experiment grids are defined in `config/experiments/baseline.yaml` and loaded via `build_grid_from_yaml()`. A researcher can add a new chunking strategy or alpha value without touching `experiment.py`. Pydantic model validation at load time surfaces YAML errors before any embedding runs.

Tradeoff: YAML parsing errors fail at runtime, not at import time. Config validation is the guard.

---

## Index Sharing Across Retrieval Methods

FAISS indices are keyed by `{chunk_config}__{embed_config}`, not by the full experiment ID. Dense, BM25, and hybrid cells for the same chunk+embed config share one index instead of rebuilding it three times. This cuts ingest time by 3x per chunk+embed pair.

Tradeoff: index directories must not be shared across runs with different embedding models. A `{chunk}__minilm` index and a `{chunk}__mpnet` index cannot coexist in the same path. The label scheme enforces this at naming time.

---

## Judge Model is Optional

Answer generation and judge scoring are gated behind `judge_model: str | None = None`. When unset, `generation_metrics` is empty and the result is retrieval-only. This is the common case: compare 12 retrieval configs without paying per-call judge costs for each. When set, the first `judge_n` queries are scored on 4 dimensions (relevance, accuracy, completeness, citation quality).

---

## Real qrels over Synthetic QA

This project uses hand-authored `qrels.json` with real queries across multiple arXiv papers rather than LLM-generated QA pairs. Queries were written by humans reading the papers; relevant chunks were identified by content, not by which chunk was used to generate a question. This is a stronger ground truth.

Tradeoff: the qrels set is small (~50 queries before filtering) because hand-authoring is expensive. At 50 queries, MRR variance is high; a 1-rank difference in one query moves MRR by 0.02.

---

## Subprocess Isolation for Embedding and FAISS

`sentence-transformers` and `faiss-cpu` crash when both are loaded in the same process on Intel Mac (two OpenMP implementations conflict). `ingest_grid.py` spawns one subprocess per (chunker, embedder) pair. Each subprocess exits after writing its index, fully releasing model weights. `evaluate.py` pre-computes all query embeddings in isolated subprocesses before any FAISS index loads.

This is not a workaround applied after the crash was discovered. It is the architecture: embedding runs are always isolated because the conflict is deterministic on this platform.

---

## Document-Level IR Scoring

`qrels.json` is ground truth at document level, not chunk level. Retrieved chunks are deduplicated to unique document IDs before computing MRR/Recall. A retriever that returns 5 chunks from 2 documents produces 2 doc IDs for scoring: the correct measure for a document search system.

---

## Precision@5 Target is Mathematically Impossible with These qrels

The original spec states a Precision@5 target of >0.60. With qrels where each query has exactly one relevant chunk, the maximum achievable Precision@5 is 1/5 = 0.20. The target applies to a recall-precision regime where multiple chunks are relevant per query. Actual Precision@5 results (0.12-0.20) are correct. MRR and Recall@K are the meaningful metrics.
