# Design Decisions and Tradeoffs

## Local SentenceTransformers embeddings over OpenAI API

This project uses `all-MiniLM-L6-v2` (384-dim) and `all-mpnet-base-v2` (768-dim) from SentenceTransformers rather than OpenAI's embedding API. Reasons: (1) multi-paper ingestion with 12 grid configs × ~5,000 chunks per paper would incur significant API cost; (2) local embeddings enable offline experimentation; (3) comparing two SentenceTransformers models across retrieval strategies is the experimental variable, not embedding provider. The tradeoff: SentenceTransformers models top out below OpenAI large at MTEB benchmarks. rag-pipeline-systematic-evals explores the OpenAI embedding side; this project explores local.

## YAML-driven grid over code-only config

Experiment grids are defined in `config/experiments/baseline.yaml` and loaded via `build_grid_from_yaml()`. This separates experiment definitions from the orchestration code — a researcher can add a new chunking strategy or alpha value without touching `experiment.py`. The tradeoff: YAML parsing errors fail at runtime, not import time. Config validation uses Pydantic models at load time to surface errors early.

## Index sharing across retrieval methods (same chunk+embed config)

FAISS indices are keyed by `{chunk_config}__{embed_config}`, not by the full experiment ID. Dense, BM25, and hybrid cells for the same chunk+embed config share one index instead of rebuilding it three times. This cuts ingest time by 3× for each chunk+embed pair. The tradeoff: the index directory must not be shared across runs with different embedding models — a `{chunk}__minilm` index and a `{chunk}__mpnet` index cannot coexist in the same path. The label scheme enforces this.

## judge_model is optional (not always run)

Answer generation and judge scoring are gated behind `judge_model: str | None = None`. When unset, `generation_metrics` is empty `{}` and the result is retrieval-only. This matches the common case where you want to compare 12 retrieval configs quickly without paying per-call judge costs for each. When set, the first `judge_n` queries are scored on 4 dimensions (relevance, accuracy, completeness, citation_quality). The tradeoff: generation metrics are not available for the full grid unless explicitly triggered.

## Multi-paper qrels over synthetic QA

This project uses a hand-authored `qrels.json` with real queries across multiple arXiv papers rather than LLM-generated QA pairs. This is a stronger ground truth — queries were written by humans who read the papers, and relevant chunks were identified by content, not by which chunk was used to generate a question. The tradeoff: the qrels set is small (~50 queries) because hand-authoring is expensive. At 50 queries, MRR variance is high; a 1-rank difference in one query moves MRR by 0.02.

## Precision@5 target (>0.60) is mathematically impossible with 1-to-1 qrels

The spec states a Precision@5 target of >0.60. With qrels where each query has exactly one relevant chunk, the maximum achievable Precision@5 is 1/5 = 0.20. The spec target applies to a recall-precision regime where multiple chunks are relevant per query (e.g., paragraph-level retrieval with 3-5 relevant chunks). This is a spec authoring error — the target was carried over from a different data regime. Actual Precision@5 results are 0.12–0.20, which is correct.

## Incomplete grid (6 of 12 baseline configs)

The baseline grid defines 12 configurations: 4 chunk strategies × 3 retrieval methods × partial embed models. As of this writing, 6 results exist (fixed and recursive × minilm × dense and hybrid). The 6 missing configs (recursive × mpnet, sliding_window × both models) require downloading arXiv PDFs and embedding ~25,000 chunks locally — approximately 2 hours of CPU time. They are planned but not yet run.
