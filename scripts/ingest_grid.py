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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import build_grid_from_yaml, build_experiment_grid, ChunkConfig, EmbedConfig
from src.experiment import build_chunker, build_embedder
from src.pipeline import RAGPipeline

console = Console()


def _unique_chunk_embed_pairs(
    configs,
) -> list[tuple[ChunkConfig, EmbedConfig]]:
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
    p.add_argument("--config", "-c", type=Path, default=None,
                   help="YAML experiment config (default: built-in grid)")
    p.add_argument("--index-dir", type=Path, default=Path("data/indices"),
                   help="Root directory for FAISS indices (default: data/indices)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max PDFs to ingest per index (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild indices even if they already exist on disk")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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

    console.print(Panel(
        f"[bold cyan]RAG Pipeline — Index Pre-build[/]\n\n"
        f"  PDFs          : [green]{len(pdf_paths)}[/]\n"
        f"  Index configs : [yellow]{len(pairs)}[/] unique (chunk, embed) pairs\n"
        f"  Index dir     : [dim]{args.index_dir}[/]",
        title="[bold]P4[/]", expand=False,
    ))

    args.index_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    skipped = 0

    for chunk_cfg, embed_cfg in pairs:
        index_key = f"{chunk_cfg.label()}__{embed_cfg.label()}"
        index_dir = args.index_dir / index_key
        faiss_path = index_dir / "faiss_index" / "index.faiss"

        if not args.force and faiss_path.exists():
            console.print(f"[dim]skip[/]  {index_key}  (index exists)")
            skipped += 1
            continue

        console.print(f"[cyan]build[/] {index_key} ...")

        chunker = build_chunker(chunk_cfg)
        embedder = build_embedder(embed_cfg)
        pipeline = RAGPipeline(chunker=chunker, embedder=embedder)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Ingesting ({index_key})", total=len(pdf_paths))
            chunks = pipeline.ingest(
                pdf_paths=pdf_paths,
                index_dir=index_dir,
                chunk_label=chunk_cfg.label(),
            )

        console.print(
            f"  [green]✓[/] {len(chunks)} chunks from {len(pipeline.documents)} docs → [dim]{index_dir}[/]"
        )
        built += 1

    console.print(
        f"\n[green]Done.[/] Built [bold]{built}[/] indices, skipped [dim]{skipped}[/]."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
