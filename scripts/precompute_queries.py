"""
Pre-compute query embeddings for all queries in a qrels file.

Runs sentence-transformers only — FAISS is never imported. This avoids the
library conflict where sentence-transformers and faiss-cpu crash when loaded
in the same process on Intel Mac.

Usage:
    python scripts/precompute_queries.py data/qrels_filtered.json \\
        --model all-MiniLM-L6-v2 \\
        --out-dir data/query_cache

Output:
    data/query_cache/{model_label}.pkl  — dict[query_text → np.ndarray(D,)]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EmbedModelName
from src.evaluator import load_qrels


def _model_label(model_name: str) -> str:
    """Convert model name to filesystem-safe label (mirrors EmbedConfig.label())."""
    return model_name.lower().replace("all-", "").split("-v")[0]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="precompute_queries",
        description="Pre-embed all qrels queries for one embedding model.",
    )
    p.add_argument("qrels", type=Path, help="Path to qrels JSON file")
    p.add_argument("--model", type=str, default=EmbedModelName.MINILM.value,
                   help=f"Embedding model name (default: {EmbedModelName.MINILM.value})")
    p.add_argument("--out-dir", type=Path, default=Path("data/query_cache"),
                   help="Output directory for cached embeddings (default: data/query_cache)")
    p.add_argument("--cache-dir", type=Path, default=Path("data/embed_cache"),
                   help="Sentence-transformers model cache dir (default: data/embed_cache)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Embedding batch size (default: 16)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache file already exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    label = _model_label(args.model)
    out_path = args.out_dir / f"{label}.pkl"

    if not args.force and out_path.exists():
        print(f"Cache exists, skipping: {out_path}")
        return 0

    if not args.qrels.exists():
        print(f"ERROR: qrels not found: {args.qrels}", file=sys.stderr)
        return 1

    # Load all unique query texts from qrels
    qrels = load_qrels(args.qrels)
    queries = [entry["query"] for entry in qrels.values()]
    print(f"Embedding {len(queries)} queries with {args.model} ...")

    # Import sentence-transformers here — FAISS must NOT be imported in this process
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(args.model, cache_folder=str(args.cache_dir))
    vecs = model.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=True)

    # Map query text → embedding vector
    cache = {text: vecs[i] for i, text in enumerate(queries)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"  ✓ {len(cache)} query embeddings saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
