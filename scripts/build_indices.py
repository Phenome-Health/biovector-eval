#!/usr/bin/env python
"""Build FAISS indices for embedding files.

This script builds multiple index types for each embedding file:
- Flat: Exact search baseline
- HNSW: Graph-based ANN (no quantization)
- SQ8: HNSW with 8-bit scalar quantization
- SQ4: HNSW with 4-bit scalar quantization
- PQ: HNSW with product quantization
- IVF Flat: Clustering-based search
- IVF PQ: Clustering + product quantization

Usage:
    uv run python scripts/build_indices.py \
        --embeddings data/embeddings \
        --output-dir data/indices \
        --index-types all

    # Build specific types only
    uv run python scripts/build_indices.py \
        --embeddings data/embeddings \
        --output-dir data/indices \
        --index-types flat,hnsw,sq8

    # Skip existing indices
    uv run python scripts/build_indices.py \
        --embeddings data/embeddings \
        --output-dir data/indices \
        --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from biovector_eval.utils.persistence import (
    build_flat_index,
    build_hnsw_index,
    build_hnsw_pq_index,
    build_hnsw_sq4_index,
    build_hnsw_sq8_index,
    build_ivf_flat_index,
    build_ivf_pq_index,
    build_ivf_sq4_index,
    build_ivf_sq8_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Index type configurations
INDEX_TYPES: dict[str, dict[str, Any]] = {
    "flat": {
        "builder": build_flat_index,
        "description": "Exact search baseline (IndexFlatIP)",
        "params": {},
    },
    "hnsw": {
        "builder": build_hnsw_index,
        "description": "Graph-based ANN (IndexHNSWFlat)",
        "params": {"M": 32, "ef_construction": 200},
    },
    "hnsw_sq8": {
        "builder": build_hnsw_sq8_index,
        "description": "HNSW + 8-bit scalar quantization",
        "params": {},
    },
    "hnsw_sq4": {
        "builder": build_hnsw_sq4_index,
        "description": "HNSW + 4-bit scalar quantization",
        "params": {},
    },
    "hnsw_pq": {
        "builder": build_hnsw_pq_index,
        "description": "HNSW + product quantization",
        "params": {},
    },
    "ivf_flat": {
        "builder": build_ivf_flat_index,
        "description": "Clustering-based search (IndexIVFFlat)",
        "params": {},
    },
    "ivf_pq": {
        "builder": build_ivf_pq_index,
        "description": "Clustering + product quantization (IndexIVFPQ)",
        "params": {},
    },
    "ivf_sq8": {
        "builder": build_ivf_sq8_index,
        "description": "Clustering + 8-bit scalar quantization (IndexIVFScalarQuantizer)",
        "params": {},
    },
    "ivf_sq4": {
        "builder": build_ivf_sq4_index,
        "description": "Clustering + 4-bit scalar quantization (IndexIVFScalarQuantizer)",
        "params": {},
    },
}


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (for testing). If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for embedding files"
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Directory containing embedding files (.npy) or multi-vector directories",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for FAISS index files",
    )
    parser.add_argument(
        "--index-types",
        default="all",
        help="Comma-separated index types or 'all'. Options: flat,hnsw,hnsw_sq8,hnsw_sq4,hnsw_pq,ivf_flat,ivf_sq8,ivf_sq4,ivf_pq",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip building indices that already exist",
    )
    return parser.parse_args(args)


def get_index_output_path(
    output_dir: Path,
    embedding_name: str,
    index_type: str,
) -> Path:
    """Get the output path for an index file.

    Args:
        output_dir: Base output directory.
        embedding_name: Name of the embedding file (without extension).
        index_type: Type of index (flat, hnsw, etc.).

    Returns:
        Path for the index file.
    """
    return output_dir / f"{embedding_name}_{index_type}.faiss"


def should_skip_index(index_path: Path, skip_existing: bool) -> bool:
    """Check if an index should be skipped.

    Args:
        index_path: Path where the index would be saved.
        skip_existing: Whether to skip existing indices.

    Returns:
        True if the index should be skipped.
    """
    return skip_existing and index_path.exists()


def build_index_for_embedding_file(
    embedding_path: Path,
    output_dir: Path,
    index_type: str,
    skip_existing: bool = False,
) -> Path:
    """Build a single index for an embedding file.

    Args:
        embedding_path: Path to embeddings.npy file.
        output_dir: Directory to save the index.
        index_type: Type of index to build.
        skip_existing: Whether to skip if index exists.

    Returns:
        Path to the created index file.
    """
    # Determine embedding name
    # For multi-vector: parent dir is "model_multi-vector", file is "embeddings.npy"
    # For single-vector: file is "model_strategy.npy"
    if embedding_path.name == "embeddings.npy":
        # Multi-vector directory structure
        embedding_name = embedding_path.parent.name
        metadata_path = embedding_path.parent / "metadata.json"
    else:
        # Single-vector flat file
        embedding_name = embedding_path.stem
        metadata_path = None

    # Get output path
    index_path = get_index_output_path(output_dir, embedding_name, index_type)

    # Check if should skip
    if should_skip_index(index_path, skip_existing):
        logger.info(f"Skipping existing: {index_path.name}")
        return index_path

    # Load embeddings
    logger.info(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)

    # Ensure float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Build index
    logger.info(f"Building {index_type} index for {embedding_name}...")
    start = time.time()

    builder = INDEX_TYPES[index_type]["builder"]
    index = builder(embeddings)

    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    elapsed = time.time() - start
    size_mb = index_path.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {index_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    # Copy metadata for multi-vector indices
    if metadata_path and metadata_path.exists():
        metadata_dest = index_path.with_suffix(".metadata.json")
        shutil.copy(metadata_path, metadata_dest)
        logger.info(f"  Copied metadata: {metadata_dest.name}")

    return index_path


def build_all_indices(
    embeddings_dir: Path,
    output_dir: Path,
    index_types: list[str],
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Build all indices for all embedding files.

    Args:
        embeddings_dir: Directory containing embedding files.
        output_dir: Directory to save indices.
        index_types: List of index types to build, or ["all"].
        skip_existing: Whether to skip existing indices.

    Returns:
        List of result dictionaries with build status.
    """
    # Expand "all" to all index types
    if "all" in index_types:
        index_types = list(INDEX_TYPES.keys())

    # Find all embedding files
    embedding_files: list[Path] = []

    # Find single-vector .npy files
    for npy_file in embeddings_dir.glob("*.npy"):
        embedding_files.append(npy_file)

    # Find multi-vector directories
    for subdir in embeddings_dir.iterdir():
        if subdir.is_dir() and (subdir / "embeddings.npy").exists():
            embedding_files.append(subdir / "embeddings.npy")

    logger.info(f"Found {len(embedding_files)} embedding file(s)")
    logger.info(f"Building {len(index_types)} index type(s): {', '.join(index_types)}")

    results: list[dict[str, Any]] = []

    for emb_path in sorted(embedding_files):
        for index_type in index_types:
            # Determine embedding name for result tracking
            if emb_path.name == "embeddings.npy":
                embedding_name = emb_path.parent.name
            else:
                embedding_name = emb_path.stem

            index_path = get_index_output_path(output_dir, embedding_name, index_type)

            result: dict[str, Any] = {
                "embedding": embedding_name,
                "index_type": index_type,
                "path": str(index_path),
                "skipped": False,
                "success": False,
                "error": None,
            }

            # Check if should skip
            if should_skip_index(index_path, skip_existing):
                result["skipped"] = True
                result["success"] = True
                results.append(result)
                continue

            # Build the index
            try:
                build_index_for_embedding_file(
                    embedding_path=emb_path,
                    output_dir=output_dir,
                    index_type=index_type,
                    skip_existing=False,  # Already checked above
                )
                result["success"] = True
            except Exception as e:
                logger.error(f"Error building {index_type} for {embedding_name}: {e}")
                result["error"] = str(e)

            results.append(result)

    return results


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Building FAISS indices")
    logger.info("=" * 60)
    logger.info(f"Embeddings: {args.embeddings}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Index types: {args.index_types}")
    logger.info(f"Skip existing: {args.skip_existing}")

    # Parse index types
    if args.index_types == "all":
        index_types = ["all"]
    else:
        index_types = [t.strip() for t in args.index_types.split(",")]
        # Validate
        for t in index_types:
            if t not in INDEX_TYPES:
                raise ValueError(
                    f"Unknown index type: {t}. Valid: {list(INDEX_TYPES.keys())}"
                )

    # Build indices
    start = time.time()
    results = build_all_indices(
        embeddings_dir=args.embeddings,
        output_dir=args.output_dir,
        index_types=index_types,
        skip_existing=args.skip_existing,
    )
    total_time = time.time() - start

    # Summary
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)

    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r["skipped"])
    failed = sum(1 for r in results if not r["success"])

    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.error("Failures:")
        for r in results:
            if not r["success"] and not r["skipped"]:
                logger.error(f"  {r['embedding']} / {r['index_type']}: {r['error']}")


if __name__ == "__main__":
    main()
