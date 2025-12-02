#!/usr/bin/env python
"""Build Flat (exact) indices from existing embeddings.

This script loads the pre-computed embeddings and builds Flat indices
for baseline comparison in evaluation.

Usage:
    uv run python scripts/build_flat_indices.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from biovector_eval.utils.persistence import (
    build_flat_index,
    load_embeddings,
    save_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


EMBEDDINGS_DIR = Path("data/embeddings")
INDICES_DIR = Path("data/indices")

MODELS = [
    "bge-small-en-v1.5",
    "bge-m3",
    "chemberta-77m-mtr",
]


def main() -> None:
    """Build Flat indices for all models."""
    logger.info("Building Flat indices from existing embeddings...")

    total_start = time.perf_counter()

    for model_slug in MODELS:
        embedding_path = EMBEDDINGS_DIR / f"{model_slug}.npy"

        if not embedding_path.exists():
            logger.warning(f"Embeddings not found: {embedding_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_slug}")
        logger.info(f"{'='*60}")

        # Load embeddings
        start = time.perf_counter()
        embeddings = load_embeddings(embedding_path)
        logger.info(f"Loaded embeddings in {time.perf_counter() - start:.1f}s")

        # Build Flat index
        start = time.perf_counter()
        logger.info(f"Building Flat index ({embeddings.shape[1]}d)...")
        index = build_flat_index(embeddings)
        build_time = time.perf_counter() - start

        # Save index
        output_path = INDICES_DIR / f"{model_slug}_flat.faiss"
        save_index(index, output_path)

        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"  Flat: {output_path} ({size_mb:.1f} MB, {build_time:.1f}s)")

        del embeddings, index

    total_time = time.perf_counter() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE - Total time: {total_time:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
