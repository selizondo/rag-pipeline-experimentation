# RAG Pipeline Experimentation

![Tests](https://github.com/selizondo/rag-pipeline-experimentation/actions/workflows/ci.yml/badge.svg)

Teams pick RAG configuration from blog posts and intuition. This project puts that intuition to a test: 18 configurations across 100 arXiv ML papers with real ground-truth labels. Hybrid retrieval at alpha=0.6 achieves MRR 0.960. BM25 alone is the floor at every chunk size.

**Stack:** Python · FAISS · SentenceTransformers · rank-bm25 · OpenAI · rag-common

## Results

18-cell grid (3 chunkers x 2 embedders x 3 retrievers), 100 arXiv papers, 281 queries from Open RAG Benchmark:

Best configuration: `fixed_512_ol64__minilm-l6__hybrid_a0.6`

| Metric | Value |
|--------|-------|
| MRR | 0.960 |
| NDCG@5 | 0.960 |
| Recall@5 | 0.960 |
| LLM judge score | 4.53 / 5.0 |

Hybrid retrieval consistently outperforms dense-only on this corpus. ML papers contain exact technical terminology: model names, dataset names, metric abbreviations. BM25 anchors results to those exact terms; dense retrieval handles semantic paraphrase around them. BM25 alone is the floor: keyword search without embeddings produces the weakest results across all chunking strategies.

## How It Works

### Real ground truth, not synthetic

`qrels.json` uses BEIR/TREC format from the Open RAG Benchmark (vectara/open_ragbench): 1,000 arXiv papers, 3,045 queries authored by humans reading actual papers, 400 genuinely relevant papers and 600 hard negatives. Hard negatives are topically adjacent papers that do not answer the question, making this more realistic than benchmarks where non-relevant documents are randomly sampled.

Retrieval metrics (MRR, Recall@K, NDCG@K) are computed against these labels with no LLM involved.

### Subprocess isolation fixes the Intel Mac embedding conflict

`sentence-transformers` and `faiss-cpu` crash when both are loaded in the same process on Intel Mac. `ingest_grid.py` spawns one subprocess per (chunker, embedder) pair: each subprocess exits after writing its index, fully releasing model weights before the next model loads. `evaluate.py` pre-computes all query embeddings in isolated subprocesses before any FAISS index loads. 61 tests pass without triggering these paths.

### Judge scores reflect corpus coverage, not model quality

Judge scores with 20 papers: 2.52/5. With 100 papers: 4.53/5. The improvement is not a model change: with 20 papers, many queries reference papers outside the index, and the LLM produces plausible-sounding but unsupported answers. The judge correctly penalizes those. With 100 papers, most queries have the answer available in context. This is the measurement that shows corpus coverage is the bottleneck before retrieval strategy.

**Companion post:** [Building a RAG System That Can Actually Experiment](docs/blog_post.md)
**Related projects:** [rag-pipeline-systematic-evals](https://github.com/selizondo/rag-pipeline-systematic-evals) (single-PDF grid search with synthetic QA; different evaluation methodology) · [rag-common](https://github.com/selizondo/rag-common) (shared chunkers, retrievers, metrics used by this pipeline)

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| Evaluation methodology | [Methodology](docs/methodology.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
