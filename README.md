# RAG Pipeline — Experimentation (P4)

Modular RAG system for research papers with a configurable experiment grid: swap chunking strategy, embedding model, and retrieval method via YAML config and measure what actually works.

**Dataset:** [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) — 1,000 arXiv PDFs, 3,045 human-authored QA pairs with ground-truth relevance labels.

**Key result:** Recursive chunking + `all-mpnet-base-v2` + hybrid retrieval (α=0.6) produces the best MRR on 100-paper runs.

---

## What this project does

| Phase | Steps |
|---|---|
| **Ingest** | PDF → parse (PyMuPDF) → chunk → embed (SentenceTransformers) → FAISS index → save |
| **Query** | question → retrieve (dense / BM25 / hybrid) → LLM answer with [N] citations |
| **Evaluate** | qrels queries → IR metrics (MRR, MAP, Recall@K, Precision@K, NDCG@K) + optional LLM-as-Judge |

**Grid search:** any combination of chunking strategy, embedding model, and retrieval method, defined in a YAML config. Resume-aware: completed cells are skipped.

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- OpenAI API key (for LLM answer generation and judging)
- A Groq or local LLM API key for `llm_utils` (optional — only needed for answer generation)

### 2. Clone and install

```bash
# From the repo root
cd rag_pipeline_experimentation

# Install shared packages (local editable)
uv pip install -e ../rag_common/
uv pip install -e ../llm_utils/

# Install project + dev deps
uv pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini         # used for answer generation
OPENAI_JUDGE_MODEL=gpt-4o-mini   # used for LLM-as-Judge scoring
```

### 4. Verify setup

```bash
python -m pytest tests/ -q
# Expected: 61 passed, 3 warnings
```

Tests are fully offline — no OpenAI or PDF calls. All mocked.

---

## Dataset

```bash
# Download 50 PDFs (default)
python scripts/download_dataset.py

# Download 100 PDFs (recommended for full evaluation)
python scripts/download_dataset.py --limit 100

# Metadata only, skip PDF download
python scripts/download_dataset.py --no-pdfs

# Re-download already-existing files
python scripts/download_dataset.py --limit 100 --force
```

Output:
- `data/papers/*.pdf` — downloaded arXiv PDFs
- `data/qrels_filtered.json` — queries filtered to the downloaded papers only

---

## Running experiments

```bash
# Smoke test — 1 cell, 2 PDFs, 3 queries (~16s)
python scripts/evaluate.py data/smoke_papers/ data/qrels_smoke.json \
  --config config/experiments/smoke.yaml

# 12-cell baseline grid — 20 PDFs (fast iteration / debugging)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --limit 20

# 12-cell baseline grid — full 100 PDFs, 281 queries (~30 min)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml

# Re-run all cells (ignore cached results)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --force

# Show top-10 configs in summary table (default: top-3)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --top-k 10

# Write results to a custom directory
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --out-dir experiments/results_v2
```

Results land in `experiments/results/{experiment_id}.json`. Interrupted runs resume automatically.

### Experiment configs

| Config | Cells | Description |
|---|---|---|
| `config/experiments/smoke.yaml` | 1 | 1 chunker × 1 embedder × 1 retriever, 3 queries — wiring check |
| `config/experiments/baseline.yaml` | 12 | 3 chunkers × 2 embedders × 2 retrievers, 50 queries |

---

## Running tests

```bash
# All tests (no API key required)
python -m pytest tests/ -q

# Specific test file
python -m pytest tests/test_pipeline.py -v

# Single test
python -m pytest tests/test_evaluator.py::TestEvaluate::test_perfect_retrieval_mrr_one -v
```

---

## Interactive interfaces

### Streamlit web UI

```bash
# Build an index first (or use an existing one from a grid run)
streamlit run app.py
```

Sidebar lets you select the index directory, embedding model, retrieval method, top-K, and LLM model. Displays retrieved chunks, citations, and per-query timing.

### CLI REPL

```bash
# Interactive QA against a saved index
python scripts/serve.py -i data/indices/recursive_512_ol100__minilm-l6__dense

# With hybrid retrieval
python scripts/serve.py -i data/indices/recursive_512_ol100__mpnet-base__hybrid \
  --retrieval hybrid --alpha 0.6 --top-k 10

# With a specific LLM
python scripts/serve.py -i data/indices/my_index --model gpt-4o-mini
```

Type `quit` or `exit` to stop. `clear` resets the screen.

### Build a standalone index

```bash
# Ingest a directory of PDFs into a named index
python scripts/ingest.py data/papers/ -o data/indices/my_index \
  --chunk-strategy recursive --chunk-size 512 --overlap 100 \
  --embed-model all-MiniLM-L6-v2
```

---

## Inspecting results

```bash
# Pretty-print one result
python3 -c "
import json
with open('experiments/results/fixed_512_ol64__minilm-l6__dense.json') as f:
    d = json.load(f)
print('metrics:', json.dumps(d['metrics'], indent=2))
print('n_queries:', d['n_queries'])
"

# List all results sorted by MRR
python3 -c "
import json, glob
results = []
for f in glob.glob('experiments/results/*.json'):
    d = json.load(open(f))
    results.append((d['metrics']['mrr'], f))
for mrr, name in sorted(results, reverse=True):
    print(f'{mrr:.4f}  {name}')
"
```

---

## Project layout

```
rag_pipeline_experimentation/
├── app.py                       # Streamlit web UI (PaperSearch)
├── config/experiments/          # YAML experiment grid definitions
│   ├── smoke.yaml               # 1-cell wiring check
│   └── baseline.yaml            # 12-cell baseline grid
├── scripts/
│   ├── download_dataset.py      # fetch arXiv PDFs + build qrels_filtered.json
│   ├── evaluate.py              # run experiment grid, save results
│   ├── ingest.py                # ingest a PDF directory into a named FAISS index
│   ├── serve.py                 # interactive CLI REPL (Rich UI)
│   └── run_full_pipeline.py     # end-to-end: download → ingest → evaluate
├── src/
│   ├── config.py                # YAML loading, ChunkConfig, EmbedConfig, RetrievalConfig,
│   │                            # ExperimentConfig, build_grid_from_yaml()
│   ├── pipeline.py              # RAGPipeline: ingest() + query() + load() orchestration
│   ├── embedders.py             # SentenceTransformersEmbedder (lazy-load + disk cache)
│   ├── evaluator.py             # qrels I/O, per-query scoring, ExperimentResult
│   ├── experiment.py            # build_chunker/embedder/pipeline, run_experiment(), run_grid()
│   ├── generator.py             # LLM answer generation with [N] citation extraction
│   ├── judge.py                 # LLM-as-Judge (4 dimensions, 1–5 scale)
│   ├── models.py                # Document, Citation, QAResponse, JudgeScore, ExperimentResult
│   ├── iteration_log.py         # append-only experiment history log
│   ├── visualizer.py            # Matplotlib/Seaborn charts from results
│   ├── base.py                  # re-exports: BaseChunker, BaseEmbedder, BaseReranker, BaseLLM
│   └── chunkers_ext.py          # re-exports: RecursiveChunker, SlidingWindowChunker
├── tests/                       # 61 tests (all mocked; no API key required)
│   ├── test_chunkers_ext.py
│   ├── test_embedders.py
│   ├── test_evaluator.py
│   ├── test_experiment.py
│   └── test_pipeline.py
├── data/
│   ├── papers/                  # downloaded arXiv PDFs
│   ├── smoke_papers/            # 2 PDFs for smoke test
│   ├── qrels_filtered.json      # 281 queries → 100 papers
│   ├── qrels_smoke.json         # 3 queries → 2 papers
│   ├── embed_cache/             # pickled embeddings keyed by (model, chunk_label)
│   └── indices/                 # FAISS indices per experiment cell
├── experiments/results/         # one JSON per completed experiment cell
├── visualizations/              # PNG charts
├── pyproject.toml
└── .env.example
```

---

## Key design decisions

**Document-level IR evaluation** — P4 uses pre-authored qrels (ground-truth at document level, not chunk level). Retrieved chunks are de-duplicated to unique document IDs before computing MRR/Recall. This means even a perfect retriever that returns 5 chunks from 2 documents produces only 2 doc IDs for scoring.

**Per-experiment FAISS index** — each grid cell (`chunk_config × embed_model × retrieval_method`) has its own index directory. Different chunker + embedder combinations produce incompatible FAISS indices; separate directories prevent clobber.

**SentenceTransformers embedding cache** — keyed by `(model_name, chunk_label)`. Cache hit is validated by checking that stored chunk IDs are a subset of the current chunk set. Stale caches (chunk config changed) are detected by UUID mismatch.

**Lazy model loading** — the SentenceTransformers model is loaded on first `embed()` call. Importing `embedders.py` in tests doesn't pay the ~1s model-load cost.

**Chunker is irrelevant after index load** — `pipeline.load()` restores chunks from the saved FAISS store. The chunker passed to `RAGPipeline()` is only used during `ingest()`. The Streamlit app and CLI REPL pass a dummy `FixedSizeChunker` when loading pre-built indices.

---

## Common issues

**`ModuleNotFoundError: No module named 'rag_common'`**
Install the shared library: `uv pip install -e ../rag_common/`

**`ModuleNotFoundError: No module named 'llm_utils'`**
Install the shared LLM utilities: `uv pip install -e ../llm_utils/`

**`OPENAI_API_KEY is not set`**
Make sure `.env` exists in the `rag_pipeline_experimentation/` directory and you're running from that directory.

**`No indices loaded` in Streamlit**
Run `scripts/evaluate.py` (or `scripts/ingest.py`) to build at least one index before launching the UI.

**Slow first run**
First run downloads SentenceTransformers model weights (~90 MB for MiniLM, ~420 MB for MPNet) and generates embeddings. Both are cached after first run — subsequent runs are much faster.

**Charts not generating**
Matplotlib requires a display. On headless servers, set `MPLBACKEND=Agg` in your environment.
