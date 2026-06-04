"""
Pre-build FAISS indices for all unique (chunk, embed) pairs in a grid config.

Run this before evaluate.py to separate the slow embedding step from retrieval
evaluation. Once indices are on disk, evaluate.py loads them without re-ingesting.

Usage:
    python scripts/ingest_grid.py data/papers/ \\
        --config config/experiments/baseline.yaml \\
        --index-dir data/indices \\
        --limit 5          # optional: ingest only first N PDFs
        --force            # optional: rebuild even if index already exists
        --batch-size 16    # optional: override embedding batch size
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ChunkConfig, EmbedConfig, build_experiment_grid, build_grid_from_yaml

console = Console()


def _unique_chunk_embed_pairs(configs) -> list[tuple[ChunkConfig, EmbedConfig]]:
    seen: dict[str, tuple[ChunkConfig, EmbedConfig]] = {}
    for cfg in configs:
        key = f"{cfg.chunk.label()}__{cfg.embed.label()}"
        if key not in seen:
            seen[key] = (cfg.chunk, cfg.embed)
    return list(seen.values())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest_grid",
        description="Pre-build FAISS indices for all (chunk, embed) pairs in a grid config.",
    )
    p.add_argument("papers_dir", type=Path, help="Directory containing PDF files")
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="YAML experiment config (default: built-in grid)",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indices"),
        help="Root directory for FAISS indices (default: data/indices)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Max PDFs to ingest per index (default: all)"
    )
    p.add_argument(
        "--force", action="store_true", help="Rebuild indices even if they already exist on disk"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size (default: from config, typically 16)",
    )
    # Internal flag: build exactly one index then exit. Used by subprocess isolation.
    p.add_argument("--single-index", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _build_one(args: argparse.Namespace, index_key: str) -> int:
    """Build a single index in-process. Called only when --single-index is set."""
    import gc

    from src.experiment import build_chunker, build_embedder
    from src.pipeline import RAGPipeline

    pdf_paths = sorted(args.papers_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    configs = build_grid_from_yaml(args.config) if args.config else build_experiment_grid()
    pairs = {f"{c.chunk.label()}__{c.embed.label()}": (c.chunk, c.embed) for c in configs}

    if index_key not in pairs:
        print(f"ERROR: unknown index key '{index_key}'", file=sys.stderr)
        return 1

    chunk_cfg, embed_cfg = pairs[index_key]
    index_dir = args.index_dir / index_key

    chunker = build_chunker(chunk_cfg)
    embedder = build_embedder(embed_cfg)
    if args.batch_size is not None:
        embedder._batch_size = args.batch_size

    pipeline = RAGPipeline(chunker=chunker, embedder=embedder)
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = pipeline.ingest(
        pdf_paths=pdf_paths,
        index_dir=index_dir,
        chunk_label=chunk_cfg.label(),
    )
    print(f"  ✓ {len(chunks)} chunks from {len(pipeline.documents)} docs → {index_dir}")

    del chunks, pipeline, embedder, chunker
    gc.collect()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- Subprocess worker mode: build one index and exit ---
    if args.single_index:
        return _build_one(args, args.single_index)

    # --- Orchestrator mode: spawn one subprocess per index ---
    if not args.papers_dir.exists():
        console.print(f"[red]Error:[/] papers directory not found: {args.papers_dir}")
        return 1

    pdf_paths = sorted(args.papers_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        console.print(f"[red]Error:[/] no PDF files found in {args.papers_dir}")
        return 1

    configs = build_grid_from_yaml(args.config) if args.config else build_experiment_grid()
    pairs = _unique_chunk_embed_pairs(configs)

    console.print(
        Panel(
            f"[bold cyan]RAG Pipeline — Index Pre-build[/]\n\n"
            f"  PDFs          : [green]{len(pdf_paths)}[/]\n"
            f"  Index configs : [yellow]{len(pairs)}[/] unique (chunk, embed) pairs\n"
            f"  Index dir     : [dim]{args.index_dir}[/]",
            title="[bold]P4[/]",
            expand=False,
        )
    )

    args.index_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    skipped = 0
    failed = 0

    for chunk_cfg, embed_cfg in pairs:
        index_key = f"{chunk_cfg.label()}__{embed_cfg.label()}"
        index_dir = args.index_dir / index_key
        faiss_path = index_dir / "faiss_index" / "index.faiss"

        if not args.force and faiss_path.exists():
            console.print(f"[dim]skip[/]  {index_key}  (index exists)")
            skipped += 1
            continue

        console.print(f"[cyan]build[/] {index_key} ...")

        # Build subprocess command, forwarding all relevant args
        cmd = [
            sys.executable,
            __file__,
            str(args.papers_dir),
            "--index-dir",
            str(args.index_dir),
            "--single-index",
            index_key,
        ]
        if args.config:
            cmd += ["--config", str(args.config)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.force:
            cmd += ["--force"]

        # Each subprocess runs in isolation — OS reclaims all memory on exit
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            console.print(f"  [green]✓[/] {index_key} complete")
            built += 1
        else:
            console.print(
                f"  [red]✗[/] {index_key} failed (exit {result.returncode}) — "
                f"{'OOM' if result.returncode in (137, 139) else 'error'}"
            )
            failed += 1

    status = f"[green]Done.[/] Built [bold]{built}[/], skipped [dim]{skipped}[/]"
    if failed:
        status += f", [red]failed {failed}[/]"
    console.print(f"\n{status}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
