"""
Grid-search orchestrator for P4.

Execution order minimises redundant work:
    for each chunk_config:
        for each embed_config:
            build & ingest pipeline (FAISS index cached per experiment_id)
            for each retrieval_config:
                run evaluation, save result

Resume logic: if experiments/results/{experiment_id}.json already exists the
cell is skipped unless force=True.
"""

from __future__ import annotations

from pathlib import Path

from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker

from src.base import BaseChunker
from src.chunkers_ext import RecursiveChunker, SlidingWindowChunker
from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, ExperimentConfig,
    RetrievalMethod,
)
from src.embedders import SentenceTransformersEmbedder
from src.evaluator import evaluate, load_result, save_result
from src.models import ExperimentResult
from src.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def build_chunker(config: ChunkConfig) -> BaseChunker:
    if config.strategy == ChunkStrategy.FIXED:
        return FixedSizeChunker(config.chunk_size, config.overlap)
    if config.strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(config.chunk_size, config.overlap)
    if config.strategy == ChunkStrategy.SLIDING_WINDOW:
        return SlidingWindowChunker(config.window_size, config.step)
    if config.strategy == ChunkStrategy.SENTENCE:
        return SentenceBasedChunker(config.sentences_per_chunk, config.overlap_sentences)
    raise ValueError(f"Unsupported chunk strategy: {config.strategy}")


def build_embedder(config: EmbedConfig) -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder(
        model_name=config.model.value,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
    )


def patch_embed_fn(pipeline: RAGPipeline, query_cache: dict) -> None:
    """
    Replace the retriever's live embed_fn with a cache lookup.

    Prevents sentence-transformers and faiss-cpu from being active in the same
    process simultaneously, which causes a segfault on Intel Mac.
    """
    import numpy as np

    def cached_embed(texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in query_cache]
        if missing:
            raise KeyError(f"Query not in cache: {missing[0]!r}")
        return np.array([query_cache[t] for t in texts], dtype=np.float32)

    retriever = pipeline._retriever
    if hasattr(retriever, "_embed_fn"):
        retriever._embed_fn = cached_embed
    if hasattr(retriever, "dense") and hasattr(retriever.dense, "_embed_fn"):
        retriever.dense._embed_fn = cached_embed


def build_pipeline(config: ExperimentConfig) -> RAGPipeline:
    return RAGPipeline(
        chunker=build_chunker(config.chunk),
        embedder=build_embedder(config.embed),
        retrieval_method=config.retrieval.method.value,
        alpha=config.retrieval.alpha,
    )


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    pdf_paths: list[Path],
    qrels: dict[str, dict],
    result_dir: Path,
    index_base_dir: Path = Path("data/indices"),
    force: bool = False,
    judge_model: str | None = None,
    judge_n: int = 5,
    query_cache: dict | None = None,
) -> ExperimentResult:
    """
    Run one experiment cell: ingest PDFs, evaluate against qrels, save result.

    The FAISS index is written to `index_base_dir/{experiment_id}/` so that
    different configs do not clobber each other.

    Args:
        config:        ExperimentConfig for this cell.
        pdf_paths:     List of PDF paths to ingest.
        qrels:         Loaded qrels dict (query_id → entry).
        result_dir:    Directory for result JSON files.
        index_base_dir: Root for per-experiment FAISS indices.
        force:         Re-run even if result already exists.
        judge_model:   Optional LLM model for answer generation + judge scoring.
                       When set, generation_metrics are populated in the result.
        judge_n:       Number of queries to score with the judge.

    Returns:
        ExperimentResult (loaded from disk if skipped, freshly computed otherwise).
    """
    result_path = result_dir / f"{config.experiment_id}.json"

    if not force and result_path.exists():
        return load_result(result_path)

    # Key by chunk+embed only — retrieval method doesn't affect embeddings,
    # so dense/hybrid/bm25 cells for the same chunk+embed share one index.
    index_dir = index_base_dir / f"{config.chunk.label()}__{config.embed.label()}"
    pipeline = build_pipeline(config)
    if (index_dir / "faiss_index" / "index.faiss").exists():
        pipeline.load(index_dir)
    else:
        pipeline.ingest(
            pdf_paths=pdf_paths,
            index_dir=index_dir,
            chunk_label=config.chunk.label(),
        )

    # Patch embed_fn with pre-computed cache to avoid sentence-transformers + FAISS conflict.
    # BM25 retrieval ignores embed_fn so patching is safe for all retrieval methods.
    if query_cache is not None:
        patch_embed_fn(pipeline, query_cache)

    result = evaluate(qrels, pipeline, config, judge_model=judge_model, judge_n=judge_n)
    save_result(result, result_path)
    return result


# ---------------------------------------------------------------------------
# Full grid runner
# ---------------------------------------------------------------------------

def run_grid(
    configs: list[ExperimentConfig],
    pdf_paths: list[Path],
    qrels: dict[str, dict],
    result_dir: Path,
    index_base_dir: Path = Path("data/indices"),
    force: bool = False,
    progress_cb=None,
    judge_model: str | None = None,
    judge_n: int = 5,
    query_caches: dict[str, dict] | None = None,
) -> list[ExperimentResult]:
    """
    Run all experiment cells, returning results in grid order.

    Args:
        configs:      List of ExperimentConfig (typically from build_grid_from_yaml).
        pdf_paths:    PDF files to ingest.
        qrels:        Loaded qrels dict.
        result_dir:   Output directory for result JSONs.
        index_base_dir: Root for FAISS indices.
        force:        Re-run completed cells.
        progress_cb:  Optional callable(i, total, experiment_id) for progress reporting.
        judge_model:  Optional LLM model for generation + judge scoring across all cells.
        judge_n:      Number of queries to score per cell with the judge.

    Returns:
        List of ExperimentResult in the same order as `configs`.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    for i, config in enumerate(configs):
        if progress_cb:
            progress_cb(i, len(configs), config.experiment_id)
        embed_label = config.embed.label()
        query_cache = query_caches.get(embed_label) if query_caches else None
        result = run_experiment(
            config=config,
            pdf_paths=pdf_paths,
            qrels=qrels,
            result_dir=result_dir,
            index_base_dir=index_base_dir,
            force=force,
            judge_model=judge_model,
            judge_n=judge_n,
            query_cache=query_cache,
        )
        results.append(result)

    return results
