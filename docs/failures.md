# Failure Scenarios

Documented failure modes encountered during development.

---

## Failure 1: Index Clobber Across Retrieval Methods (Fixed)

### What broke
In the initial implementation, each of the 12 grid configs wrote its FAISS index to `data/indices/{experiment_id}/`. Dense, BM25, and hybrid cells for the same chunk+embed config each had a different `experiment_id` and therefore rebuilt the index three times, tripling ingest time.

Worse, if the directory naming scheme changed, an old index from a different config could be loaded for a new config, returning embeddings from a mismatched embedding model with no error: just silently wrong retrieval results.

### Detection mechanism
Not automatically detected in the initial version. The bug was identified by observing that 12-config runs took 3× longer than expected and that result directories were inconsistently structured.

### Fix applied
Index path is now keyed by `{chunk_config}__{embed_config}` only, not by the full experiment_id. Dense, BM25, and hybrid cells for the same chunk+embed config share one index. The fix is in `run_experiment()` in `src/experiment.py`:

```python
index_dir = index_base_dir / f"{config.chunk.label()}__{config.embed.label()}"
```

### Verification
Running all 3 retrieval methods for the same `fixed_512_ol64__minilm` config produces identical ingestion logs for the first cell and "index already exists, loading" for the second and third.

---

## Failure 2: generation_metrics Always Empty {} (Fixed)

### What broke
`ExperimentResult` has a `generation_metrics` field that was designed to hold judge scores (relevance, accuracy, completeness, citation_quality). In the original `evaluate()` function, this field was never populated: the judge was never called. Every result file had `"generation_metrics": {}`.

### Why it matters
Retrieval metrics (MRR, Recall@K) measure whether the right chunks were found. They don't measure whether the generated answer was correct or faithful to the retrieved context. `generation_metrics` is the only signal for answer quality: without it, this project can't claim end-to-end RAG evaluation.

### Detection mechanism
Identified during staff review: `ExperimentResult.generation_metrics` field existed in the model but no code path populated it. grep confirmed `judge_answer` was never called from `evaluate()`.

### Fix applied
`evaluate()` now accepts `judge_model: str | None = None` and `judge_n: int = 5`. When `judge_model` is set, the first `judge_n` queries are re-run through `generate_answer()` → `judge_answer()`, and the 4 judge dimensions are aggregated into `generation_metrics`. The fix is in `src/evaluator.py`.

---

## Failure 3: Precision@5 Spec Target is Mathematically Unreachable

### What breaks
The project spec states Precision@5 target >0.60. With qrels where each query has exactly one relevant chunk, the maximum achievable Precision@5 is 1/5 = 0.20. No implementation can meet the 0.60 target with this ground truth structure.

### Why it matters
If treated as a real target, the pipeline would appear to fail a spec requirement regardless of retrieval quality. Engineers reading the results would incorrectly diagnose a retrieval problem.

### Detection mechanism
Caught during metrics analysis: actual Precision@5 results are 0.12–0.20 (correct for 1-relevant-per-query qrels). The spec target of 0.60 would require 3 relevant chunks per query on average.

### Resolution
Documented as a spec authoring error: the Precision@5 target was written for a multi-relevant-chunk regime (e.g., paragraph-level retrieval where a question may be answered by 3-5 chunks). MRR and Recall@5 are the primary targets for this dataset structure. The spec target is noted in this file but not used to evaluate results.

---

## Failure 4: Incomplete Experiment Grid (Resolved)

### What broke
The baseline grid defines 18 configurations (3 chunkers × 2 embedding models × 3 retrieval methods). Initially only 6 result files existed: recursive × mpnet and all sliding_window configs were missing due to compute time (~45 min per config on CPU).

### Resolution
All 18 configs are now complete. Result files in `experiments/results/`:
- `fixed_512_ol64` × minilm-l6 × dense/bm25/hybrid
- `fixed_512_ol64` × mpnet-base × dense/bm25/hybrid
- `recursive_512_ol100` × minilm-l6 × dense/bm25/hybrid
- `recursive_512_ol100` × mpnet-base × dense/bm25/hybrid
- `sliding_w10_s5` × minilm-l6 × dense/bm25/hybrid
- `sliding_w10_s5` × mpnet-base × dense/bm25/hybrid

### Status
Resolved: all 18 of 18 configs run and committed.

---

## Failure 5: Missing FAISS Index at Query Time

### What breaks
`FAISSVectorStore.load()` calls `faiss.read_index()` on a path that does not exist. The original code propagated a cryptic `RuntimeError` from the FAISS C++ layer: no path, no actionable message.

This surfaces in two cases:
1. **Streamlit UI or `serve.py`** points at an index directory from a different label scheme (e.g., after a chunker config change that altered the label format).
2. **`run_grid()` resumes** a partial run but a cell's index directory was deleted manually.

### Detection mechanism
Surfaced during staff review by tracing the `pipeline.load()` → `FAISSVectorStore.load()` → `faiss.read_index()` call chain. There was no existence check before the FAISS call.

### Fix applied
`pipeline.load()` now checks for the `faiss_index/` subdirectory before calling `FAISSVectorStore.load()` and raises a clear `FileNotFoundError` with the full path and a recovery instruction:

```python
faiss_path = index_dir / "faiss_index"
if not faiss_path.exists():
    raise FileNotFoundError(
        f"No FAISS index found at {faiss_path}. "
        "Run scripts/evaluate.py or scripts/ingest.py to build the index first."
    )
```

### Why not a graceful fallback
There is no sensible default index to fall back to: the caller explicitly specified the index path. Raising `FileNotFoundError` is correct; the error is at the call site, not a transient runtime failure. The fix improves debuggability without changing semantics.
