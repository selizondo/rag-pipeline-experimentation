# RAG Pipeline — Experimentation

Modular Retrieval-Augmented Generation system for research papers. Swap chunking strategy, embedding model, and retrieval method via YAML config and measure what actually works best on your corpus.

---

## The Problem

Researchers and students deal with hundreds of academic papers across conferences and journals. Finding specific results, methods, or conclusions buried inside dense PDFs is painfully slow. A simple keyword search misses semantic meaning — "attention optimization techniques" doesn't match "transformer efficiency improvements" without understanding that the concepts are related. Reading entire papers end-to-end is impractical when surveying a field.

The deeper problem is that RAG systems have many moving parts — chunking strategy, embedding model, retrieval method, hybrid fusion weight — and there is no one-size-fits-all configuration. A chunking strategy that works well for government budget documents performs poorly on ML paper abstracts. An embedding model that excels on dense technical prose may be overkill for short FAQ-style text. The only way to know what works for a specific corpus is to measure it.

This project builds an intelligent QA system that retrieves relevant passages and generates cited answers, then wraps that system in an experiment grid so you can measure every configuration dimension against real ground-truth labels — and stop guessing.

---

## Dataset: Open RAG Benchmark

**Source:** [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) on HuggingFace — License: CC-BY-NC-4.0

The Open RAG Benchmark is a real-world evaluation dataset built from 1,000 arXiv research papers across machine learning, NLP, and computer science. Unlike synthetic benchmarks, the questions were authored by humans reading actual papers, and the relevance labels are document-level ground truth derived from those papers' content.

| Component | Count |
|---|---|
| PDF documents | 1,000 (400 genuinely relevant + 600 hard negatives) |
| QA pairs | 3,045 (1,793 abstractive + 1,252 extractive) |
| Query types | 1,914 text-only, 763 text+image, 148 text+table, 220 mixed |

**Hard negatives** are documents that are topically adjacent but do not answer the question. Their presence makes this benchmark more realistic than datasets where non-relevant documents are randomly sampled — a retriever that finds vaguely related papers won't score well here.

**Ground truth format:** `qrels.json` uses the BEIR standard format — the same format used by major IR benchmarks like TREC and MS MARCO. Each query maps to one or more paper IDs with relevance scores. This is what enables computing Precision, Recall, MRR, and NDCG without manual labeling.

### Downloading papers

The dataset does not bundle PDFs. `pdf_urls.json` contains arXiv URLs for each paper. The download script handles rate limiting (1s delay between requests as required by arXiv):

```bash
python scripts/download_dataset.py --limit 100   # recommended for full evaluation
python scripts/download_dataset.py --limit 20    # fast iteration during development
python scripts/download_dataset.py --no-pdfs     # metadata only
```

This project ships with 100 PDFs already downloaded and a filtered `qrels_filtered.json` covering the 281 queries whose relevant papers are in the 100-paper subset.

---

## What the Experiment Grid Measures

RAG quality depends on three independent decisions. This project tests each systematically:

### Chunking strategy — how documents are split

Before a document can be retrieved, it must be split into smaller passages. The chunking strategy determines where those boundaries fall:

| Strategy | How it works | When it wins |
|---|---|---|
| **Fixed-size** (512 tok, 64 overlap) | Slide a character window, snap to word boundaries | Consistent structure, predictable index size |
| **Recursive** (512 tok, 100 overlap) | Split on paragraphs → sentences → words in order | Documents with explicit section structure |
| **Sliding window** (w=10, step=5) | Overlapping sentence windows | High recall needed — answers straddling boundaries |

**Why chunking matters:** Chunks too small lose context — the answer is there but incomplete. Chunks too large dilute signal — the relevant sentence is buried among irrelevant content. Splitting mid-sentence produces degraded embeddings because the embedded text is semantically incomplete. The right strategy depends on document structure, not intuition.

### Embedding model — how text is converted to vectors

An embedding model converts text into a dense vector (list of numbers) where similar meanings produce numerically close vectors. The model was trained on billions of examples to place semantically related text nearby in vector space, regardless of surface word overlap.

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast (~5× faster) | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better on domain-specific text |

**Why multiple models matter:** Higher dimensions allow finer-grained distinctions between similar concepts — `all-mpnet-base-v2` can separate "attention mechanism for transformers" from "attention mechanism for CNNs" more reliably. Whether that precision is worth the 2× storage and 5× inference cost depends on your corpus. This project measures the trade-off on arXiv ML papers specifically.

### Retrieval method — how queries are matched to chunks

| Method | Mechanism | Strength |
|---|---|---|
| **Dense** | Cosine similarity between query and chunk embeddings via FAISS | Captures semantic paraphrase |
| **BM25** | TF-IDF-style keyword scoring (rank-bm25) | Exact technical term matching |
| **Hybrid** (α=0.6) | `0.6 × dense_norm + 0.4 × bm25_norm` | Both at once |

**Why hybrid retrieval:** Dense retrieval captures semantic meaning but can blur technical distinctions — model names, metric names, and dataset names that appear verbatim in only one paper. BM25 anchors results to those exact terms. Combining them at α=0.6 (60% dense, 40% BM25) captures both semantic and lexical relevance.

**Score fusion detail:** BM25 scores are unbounded positive floats; cosine similarities are bounded in [−1, 1]. Before combining, both are min-max normalised to [0, 1] so one retriever doesn't dominate. See [docs/tradeoffs.md](docs/tradeoffs.md) for the score fusion analysis.

---

## Evaluation Metrics

All metrics are computed using `qrels_filtered.json` as ground truth — no LLM involved, no synthetic labels, no cheating.

**MRR (Mean Reciprocal Rank):** For each query, find the rank of the first relevant document. Score = 1/rank (rank 1 → 1.0, rank 3 → 0.33). Average over all queries. Primary metric when users care about getting at least one correct result quickly.

**Recall@K:** What fraction of relevant documents appeared in the top-K results? At K=5, did the system surface the answer in its top 5? Important when the downstream LLM generator needs the answer to be somewhere in the retrieved context.

**NDCG@K (Normalised Discounted Cumulative Gain):** Rewards relevant results at higher ranks more than lower ranks. Logarithmic discount — rank 1 counts for much more than rank 5. Best single metric when both relevance and rank position matter.

**Precision@K:** Of the K results returned, what fraction were relevant? Note: with one relevant document per query, the maximum achievable Precision@5 is 1/5 = 0.20 regardless of retriever quality. Included for completeness; MRR and Recall@K are the meaningful primary metrics for this dataset structure.

**LLM-as-Judge (generation quality):** After retrieval, the top-K chunks are passed to `gpt-4o-mini` to generate a cited answer. A separate judge call scores the answer on four dimensions: Relevance, Accuracy, Completeness, and Citation Quality (1–5 scale each). This is the only step that uses an LLM API.

---

## Results (18-cell grid, 100 papers, 281 queries)

```
3 chunking × 2 embedding × 3 retrieval = 18 experiments
```

**Best configuration:** `fixed_512_ol64__minilm-l6__hybrid_a0.6`

| Metric | Value |
|---|---|
| MRR | **0.960** |
| NDCG@5 | 0.960 |
| Recall@5 | 0.960 |
| Avg judge score | 4.53 / 5.0 |

**Key findings:**

- **Hybrid retrieval consistently outperforms dense-only** on this corpus. ML paper abstracts use precise technical terminology (model names, metric names, dataset names) that BM25 anchors reliably, while dense retrieval handles semantic paraphrase. The combination wins.
- **BM25-only is the floor** — keyword search without embeddings produces the weakest results across all chunking strategies, confirming that semantic understanding is necessary for this query type.
- **Judge scores (4.53/5) vs 20-paper subset (2.52/5):** The dramatic improvement reflects corpus coverage. With 20 papers, many queries referenced papers outside the index — the LLM produced plausible-sounding but unsupported answers, which the judge correctly penalised. With 100 papers, most queries have the answer available in the retrieved context.

Full leaderboard and per-query details in `experiments/results/`.

---

## Quick start

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- OpenAI-compatible API key (only required for answer generation and judging — all retrieval evaluation runs without it)

### 2. Install

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/newline"
uv sync
```

Or with pip:
```bash
pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_API_KEY to your OpenAI key
```

`.env` variables (`llm_utils` naming convention):
```dotenv
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_JUDGE_MODEL=gpt-4o-mini
```

### 4. Verify setup

```bash
python -m pytest tests/ -q
# Expected: 61 passed — all offline, no API calls
```

---

## Running the pipeline

### Step 1: Download papers

```bash
python scripts/download_dataset.py --limit 100
```

Output: `data/papers/*.pdf` + `data/qrels_filtered.json`

### Step 2: Build FAISS indices

Pre-embeds all documents. Runs one subprocess per (chunker, embedder) pair so each embedding model is fully isolated — prevents memory conflicts on CPU-only machines.

```bash
python scripts/ingest_grid.py data/papers/ \
    --config config/experiments/baseline.yaml \
    --index-dir data/indices
```

This takes 20–120 minutes on CPU (most time is embedding 100 papers × 2 models).

### Step 3: Run evaluation grid

Evaluates all 18 configs against 281 queries. Pre-computes query embeddings in isolated subprocesses before FAISS search to avoid library conflicts on Intel Mac.

```bash
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
    --config config/experiments/baseline.yaml \
    -o experiments/results/ --force
```

Results land in `experiments/results/{experiment_id}.json`. Interrupted runs resume automatically (skip already-written results unless `--force`).

### Step 4: Generate answers, judge, and visualize

Runs 10 queries through the best config's index, generates cited answers, judges them, saves 6 charts, and appends to the iteration log.

```bash
python scripts/run_full_pipeline.py --n-queries 10
```

Requires `LLM_API_KEY` in `.env`. Cost: ~$0.10–0.20 for 10 queries.

---

## Experiment configs

| Config | Cells | Description |
|---|---|---|
| `config/experiments/smoke.yaml` | 1 | 1 chunker × 1 embedder × 1 retriever, 3 queries — wiring check |
| `config/experiments/baseline.yaml` | 18 | 3 chunkers × 2 embedders × 3 retrievers, 50 queries per cell |

---

## Iteration log

Every experiment run appends to `experiments/iteration_log.jsonl`. Each entry records the config snapshot, before/after metrics, and the reason for the change. This makes every configuration decision traceable to a specific experiment result — you can explain why config X outperformed config Y based on data, not intuition.

---

## Interactive interfaces

### Streamlit web UI

```bash
streamlit run app.py
```

Sidebar selects index directory, embedding model, retrieval method, top-K, and LLM model. Displays retrieved chunks, citations, and per-query timing.

### CLI REPL

```bash
python scripts/serve.py -i data/indices/fixed_512_ol64__minilm-l6__hybrid_a0.6
```

Type questions at the prompt. `quit` exits. `clear` resets the screen.

---

## Project layout

```
rag-pipeline-experimentation/
├── app.py                          # Streamlit web UI (PaperSearch)
├── config/experiments/             # YAML experiment grid definitions
│   ├── smoke.yaml                  # 1-cell wiring check
│   └── baseline.yaml               # 18-cell baseline grid
├── scripts/
│   ├── download_dataset.py         # fetch arXiv PDFs + build qrels_filtered.json
│   ├── ingest_grid.py              # pre-build FAISS indices (subprocess isolated)
│   ├── precompute_queries.py       # embed all queries into cache (no FAISS)
│   ├── evaluate.py                 # run experiment grid, save results
│   ├── ingest.py                   # ingest a single PDF directory into a named index
│   ├── serve.py                    # interactive CLI REPL
│   └── run_full_pipeline.py        # load best result → generate → judge → visualize
├── src/
│   ├── config.py                   # YAML loading, ChunkConfig, EmbedConfig, ExperimentConfig
│   ├── pipeline.py                 # RAGPipeline: ingest() + query() + load()
│   ├── embedders.py                # SentenceTransformersEmbedder (lazy-load + disk cache)
│   ├── evaluator.py                # qrels I/O, per-query scoring, ExperimentResult
│   ├── experiment.py               # build_chunker/embedder/pipeline, run_grid()
│   ├── generator.py                # LLM answer generation with [N] citation extraction
│   ├── judge.py                    # LLM-as-Judge (4 dimensions, 1–5 scale)
│   ├── models.py                   # Document, Citation, QAResponse, JudgeScore
│   ├── iteration_log.py            # append-only experiment history
│   ├── visualizer.py               # Matplotlib/Seaborn charts
│   ├── base.py                     # BaseChunker, BaseEmbedder, BaseReranker
│   └── chunkers_ext.py             # RecursiveChunker, SlidingWindowChunker
├── tests/                          # 61 tests (all offline, no API key required)
├── data/
│   ├── papers/                     # downloaded arXiv PDFs
│   ├── qrels_filtered.json         # 281 queries → 100 papers
│   ├── query_cache/                # pre-computed query embeddings (per model)
│   ├── embed_cache/                # chunk embeddings cached per (model, chunk_label)
│   └── indices/                    # FAISS indices per (chunk_config × embed_model)
├── experiments/
│   ├── results/                    # one JSON per completed experiment cell
│   └── iteration_log.jsonl         # append-only run history
├── visualizations/                 # PNG charts (regenerated by run_full_pipeline.py)
└── docs/
    ├── tradeoffs.md                # design decisions and rationale
    └── failures.md                 # known limitations and failure modes
```

---

## Key design decisions

**Document-level IR evaluation** — `qrels.json` is ground truth at document level, not chunk level. Retrieved chunks are de-duplicated to unique document IDs before computing MRR/Recall. A retriever that returns 5 chunks from 2 documents produces only 2 doc IDs for scoring — which is the right thing to measure for a document search system.

**Pre-computed query embeddings** — `evaluate.py` and `run_full_pipeline.py` embed all queries in an isolated subprocess before loading any FAISS index. This avoids a known crash on Intel Mac where `sentence-transformers` and `faiss-cpu` conflict when both are loaded in the same process. Cached embeddings in `data/query_cache/` are reused across runs.

**Subprocess-isolated index builds** — `ingest_grid.py` spawns one subprocess per (chunker, embedder) pair. Each subprocess exits after writing its index, fully releasing model weights and embeddings to the OS before the next model loads. Prevents OOM on memory-constrained machines.

**Per-experiment FAISS index** — each (chunk_config × embed_model) pair gets its own index directory. Different chunker + embedder combinations produce incompatible FAISS indices; separate directories prevent clobber. Retrieval method (dense/BM25/hybrid) is applied at query time against the same index.

**Lazy model loading** — `SentenceTransformersEmbedder._load()` is only called on first `embed()` call. Importing `embedders.py` in tests or evaluation scripts does not pay the ~1s model-load cost.

---

## Common issues

**`LLM_API_KEY is not set`**
`.env` must use `LLM_API_KEY` (not `OPENAI_API_KEY`) — `llm_utils` uses provider-agnostic naming. Copy `.env.example` and fill in your key.

**`ModuleNotFoundError: no module named 'rag_common'`**
Run `uv sync` from this directory with `UV_PROJECT_ENVIRONMENT` set. The shared library installs as an editable package from the git dependency.

**`No indices loaded` in Streamlit**
Run `scripts/ingest_grid.py` to build at least one index before launching the UI.

**Slow first run**
First run downloads SentenceTransformers model weights (~90MB for MiniLM, ~420MB for MPNet). Both are cached after first run — subsequent runs are much faster.

**Charts not generating on headless servers**
Set `MPLBACKEND=Agg` in your environment.

---

## Related projects

| Repo | Relationship |
|---|---|
| [rag-common](../rag-common) | Shared chunkers, retrievers, vector store, and metrics used by this pipeline |
| [rag-pipeline-systematic-evals](../rag-pipeline-systematic-evals) | Grid search over a single PDF with synthetic QA — different evaluation methodology |
