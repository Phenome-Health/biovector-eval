#!/usr/bin/env python
"""Generate embeddings for HMDB metabolites using all synonyms, no variations.

This script generates embeddings for all supported models and strategies,
using ALL synonyms (no cutoff) and NO synthetic variations.

Features:
- Supports all 4 models: minilm, bge-m3, sapbert, biolord
- Supports all 3 strategies: single-primary, single-pooled, multi-vector
- Incremental execution (skips existing embeddings)
- Memory efficient batch processing
- GPU memory clearing between models

Usage:
    # Generate all models, all strategies
    uv run python scripts/generate_all_embeddings.py \
        --metabolites data/hmdb/metabolites.json \
        --output-dir data/embeddings/full_synonyms \
        --device cuda

    # Generate specific model and strategy
    uv run python scripts/generate_all_embeddings.py \
        --metabolites data/hmdb/metabolites.json \
        --output-dir data/embeddings/full_synonyms \
        --models bge-m3 \
        --strategies multi-vector \
        --device cuda

    # Skip existing (incremental)
    uv run python scripts/generate_all_embeddings.py \
        --metabolites data/hmdb/metabolites.json \
        --output-dir data/embeddings/full_synonyms \
        --skip-existing \
        --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Configurations
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    dimension: int


MODELS: dict[str, ModelConfig] = {
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
        dimension=1024,
    ),
    "sapbert": ModelConfig(
        name="SapBERT",
        model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        dimension=768,
    ),
    "biolord": ModelConfig(
        name="BioLORD-2023",
        model_id="FremyCompany/BioLORD-2023",
        dimension=768,
    ),
}

STRATEGIES = ["single-primary", "single-pooled", "multi-vector"]

# Weights for multi-vector strategy
PRIMARY_WEIGHT = 1.0
SYNONYM_WEIGHT = 0.9


# =============================================================================
# Utility Functions
# =============================================================================


def clear_gpu_memory() -> None:
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug.

    Args:
        model_id: Full model ID (e.g., "BAAI/bge-m3")

    Returns:
        Lowercase slug (e.g., "bge-m3")
    """
    return model_id.split("/")[-1].lower()


def get_output_paths(
    output_dir: Path,
    model_slug: str,
    strategy: str,
) -> dict[str, Path | None]:
    """Get output file paths for embeddings and metadata.

    Args:
        output_dir: Base output directory
        model_slug: Model name slug (e.g., "bge-m3")
        strategy: Strategy name (e.g., "single-primary", "multi-vector")

    Returns:
        Dict with 'embeddings', 'metadata', and 'directory' paths.
    """
    if strategy == "multi-vector":
        strategy_dir = output_dir / f"{model_slug}_{strategy}"
        return {
            "embeddings": strategy_dir / "embeddings.npy",
            "metadata": strategy_dir / "metadata.json",
            "directory": strategy_dir,
        }
    else:
        return {
            "embeddings": output_dir / f"{model_slug}_{strategy}.npy",
            "metadata": None,
            "directory": None,
        }


def check_existing(
    paths: dict[str, Path | None],
    skip_existing: bool,
) -> bool:
    """Check if embeddings already exist and should be skipped.

    Args:
        paths: Output paths from get_output_paths()
        skip_existing: Whether to skip existing files

    Returns:
        True if should skip, False if should generate
    """
    if not skip_existing:
        return False

    embeddings_path = paths["embeddings"]
    if embeddings_path.exists():
        # For multi-vector, also check metadata exists
        if paths["metadata"] is not None:
            return paths["metadata"].exists()
        return True
    return False


# =============================================================================
# Embedding Generation Functions
# =============================================================================


def generate_single_primary(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
) -> np.ndarray:
    """Generate single-primary embeddings (one per metabolite, name only).

    Args:
        metabolites: List of metabolite dicts with 'hmdb_id', 'name', 'synonyms'
        model: Loaded SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Normalized float32 embeddings array (N x D) where N = len(metabolites)
    """
    logger.info(
        f"Generating single-primary embeddings for {len(metabolites):,} metabolites..."
    )

    texts = [m["name"] for m in metabolites]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def generate_single_pooled(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
    metabolite_batch_size: int = 5000,
) -> np.ndarray:
    """Generate single-pooled embeddings (average of name + ALL synonyms).

    Unlike the original script, this uses ALL synonyms (no 15-synonym limit).
    This creates a semantic centroid for each metabolite.

    Args:
        metabolites: List of metabolite dicts
        model: Loaded SentenceTransformer model
        batch_size: Batch size for encoding
        metabolite_batch_size: Number of metabolites to process at once

    Returns:
        Normalized float32 embeddings array (N x D) where N = len(metabolites)
    """
    logger.info(
        f"Generating single-pooled embeddings (ALL synonyms) for {len(metabolites):,} metabolites..."
    )

    # Get embedding dimension from test encode
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output array
    pooled_embeddings = np.zeros((len(metabolites), embed_dim), dtype=np.float32)

    num_batches = (
        len(metabolites) + metabolite_batch_size - 1
    ) // metabolite_batch_size
    encode_batch_size = min(batch_size, 64)  # Smaller for memory efficiency

    logger.info(f"Processing in {num_batches} metabolite batches...")

    for batch_idx in range(num_batches):
        start_m = batch_idx * metabolite_batch_size
        end_m = min(start_m + metabolite_batch_size, len(metabolites))
        batch_metabolites = metabolites[start_m:end_m]

        # Collect all texts for this batch with their ranges
        batch_texts: list[str] = []
        batch_ranges: list[tuple[int, int]] = []

        for m in batch_metabolites:
            text_start = len(batch_texts)
            # Name + ALL synonyms (no limit!)
            texts_for_m = [m["name"]] + m.get("synonyms", [])
            batch_texts.extend(texts_for_m)
            batch_ranges.append((text_start, len(batch_texts)))

        logger.info(
            f"  Batch {batch_idx + 1}/{num_batches}: "
            f"{len(batch_metabolites):,} metabolites, {len(batch_texts):,} texts"
        )

        # Encode (NOT normalized - we'll normalize after averaging)
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

        # Average per metabolite and store
        for i, (text_start, text_end) in enumerate(batch_ranges):
            m_embeddings = batch_embeddings[text_start:text_end]
            mean_emb = np.mean(m_embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            pooled_embeddings[start_m + i] = mean_emb

        # Clear memory
        del batch_embeddings, batch_texts, batch_ranges
        clear_gpu_memory()

    logger.info(f"Generated {len(pooled_embeddings):,} pooled embeddings")
    return pooled_embeddings


def generate_multi_vector(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Generate multi-vector embeddings (primary + ALL synonyms, NO variations).

    Unlike the original script:
    - Uses ALL synonyms (no 15-synonym limit)
    - Does NOT generate synthetic variations
    - Weights: primary=1.0, synonyms=0.9

    Args:
        metabolites: List of metabolite dicts
        model: Loaded SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings array, metadata dict).
        Metadata format: {"0": {"hmdb_id": "...", "tier": "...", "weight": ..., "text": "..."}, ...}
    """
    logger.info("Generating multi-vector embeddings (ALL synonyms, NO variations)...")

    # Phase 1: Collect all texts and metadata
    all_texts: list[str] = []
    all_metadata: list[dict[str, Any]] = []

    for m in tqdm(metabolites, desc="Collecting texts"):
        hmdb_id = m["hmdb_id"]
        name = m["name"]
        synonyms = m.get("synonyms", [])

        # Primary name
        all_texts.append(name)
        all_metadata.append(
            {
                "hmdb_id": hmdb_id,
                "tier": "primary",
                "weight": PRIMARY_WEIGHT,
                "text": name,
            }
        )

        # ALL synonyms (no limit!) - deduplicate by lowercase
        seen_lower = {name.lower()}
        for syn in synonyms:
            if syn.lower() not in seen_lower:
                all_texts.append(syn)
                all_metadata.append(
                    {
                        "hmdb_id": hmdb_id,
                        "tier": "synonym",
                        "weight": SYNONYM_WEIGHT,
                        "text": syn,
                    }
                )
                seen_lower.add(syn.lower())

    logger.info(
        f"Collected {len(all_texts):,} texts from {len(metabolites):,} metabolites"
    )

    # Phase 2: Encode in chunks
    chunk_size = 20000  # 20K texts per chunk
    encode_batch_size = min(batch_size, 32)  # Small for memory
    num_chunks = (len(all_texts) + chunk_size - 1) // chunk_size

    # Get embedding dimension
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output
    all_embeddings = np.zeros((len(all_texts), embed_dim), dtype=np.float32)

    logger.info(f"Encoding in {num_chunks} chunks...")

    for i in range(0, len(all_texts), chunk_size):
        chunk_texts = all_texts[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        logger.info(f"  Chunk {chunk_num}/{num_chunks}: {len(chunk_texts):,} texts")

        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        all_embeddings[i : i + len(chunk_texts)] = chunk_embeddings

        del chunk_embeddings
        clear_gpu_memory()

    # Phase 3: Build metadata dict (index -> metadata)
    metadata = {str(i): meta for i, meta in enumerate(all_metadata)}

    logger.info(f"Generated {len(all_embeddings):,} multi-vector embeddings")
    return all_embeddings, metadata


# =============================================================================
# Saving Functions
# =============================================================================


def save_embeddings(
    embeddings: np.ndarray,
    metadata: dict[str, dict[str, Any]] | None,
    paths: dict[str, Path | None],
) -> None:
    """Save embeddings and optional metadata to disk.

    Args:
        embeddings: Embedding array (N x D)
        metadata: Metadata dict for multi-vector, or None for single-vector
        paths: Output paths from get_output_paths()
    """
    # Create directories
    if paths["directory"]:
        paths["directory"].mkdir(parents=True, exist_ok=True)
    else:
        paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(paths["embeddings"], embeddings)
    size_mb = embeddings.nbytes / 1024 / 1024
    logger.info(
        f"Saved embeddings: {paths['embeddings']} ({embeddings.shape}, {size_mb:.1f} MB)"
    )

    # Save metadata for multi-vector
    if paths["metadata"] and metadata:
        with open(paths["metadata"], "w") as f:
            json.dump(metadata, f)
        meta_size_mb = paths["metadata"].stat().st_size / 1024 / 1024
        logger.info(f"Saved metadata: {paths['metadata']} ({meta_size_mb:.1f} MB)")


# =============================================================================
# Main Processing Functions
# =============================================================================


def process_model_strategy(
    config: ModelConfig,
    metabolites: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    strategy: str,
    batch_size: int = 512,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Process a single model+strategy combination.

    Args:
        config: Model configuration
        metabolites: Metabolite data
        output_dir: Output directory
        device: Device for inference ('cuda' or 'cpu')
        strategy: Strategy name
        batch_size: Batch size for encoding
        skip_existing: Whether to skip if embeddings exist

    Returns:
        Result dict with timing and status information
    """
    model_slug = get_model_slug(config.model_id)
    paths = get_output_paths(output_dir, model_slug, strategy)

    result: dict[str, Any] = {
        "model": config.name,
        "model_slug": model_slug,
        "strategy": strategy,
        "skipped": False,
        "success": False,
        "error": None,
    }

    # Check if should skip
    if check_existing(paths, skip_existing):
        logger.info(f"Skipping existing: {model_slug}/{strategy}")
        result["skipped"] = True
        result["success"] = True
        return result

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing: {config.name} / {strategy}")
    logger.info(f"{'=' * 60}")

    total_start = time.time()

    try:
        # Load model
        logger.info(f"Loading model on {device}...")
        model_start = time.time()
        model = SentenceTransformer(config.model_id, device=device)
        result["model_load_time"] = time.time() - model_start
        logger.info(f"Model loaded in {result['model_load_time']:.1f}s")

        # Generate embeddings based on strategy
        embed_start = time.time()

        if strategy == "single-primary":
            embeddings = generate_single_primary(metabolites, model, batch_size)
            metadata = None
        elif strategy == "single-pooled":
            embeddings = generate_single_pooled(metabolites, model, batch_size)
            metadata = None
        else:  # multi-vector
            embeddings, metadata = generate_multi_vector(metabolites, model, batch_size)

        result["embedding_time"] = time.time() - embed_start
        result["embedding_shape"] = list(embeddings.shape)
        result["num_vectors"] = embeddings.shape[0]

        # Unload model
        del model
        clear_gpu_memory()

        # Save embeddings
        save_start = time.time()
        save_embeddings(embeddings, metadata, paths)
        result["save_time"] = time.time() - save_start
        result["embeddings_path"] = str(paths["embeddings"])

        # Clean up
        del embeddings
        if metadata:
            del metadata
        clear_gpu_memory()

        result["total_time"] = time.time() - total_start
        result["success"] = True

        logger.info(
            f"Completed {config.name}/{strategy} in {result['total_time']:.1f}s"
        )

    except Exception as e:
        logger.error(f"Error processing {config.name}/{strategy}: {e}")
        import traceback

        traceback.print_exc()
        result["error"] = str(e)
        clear_gpu_memory()

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for HMDB metabolites (all synonyms, no variations)"
    )
    parser.add_argument(
        "--metabolites",
        required=True,
        type=Path,
        help="Path to metabolites JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all' (options: minilm,bge-m3,sapbert,biolord)",
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help="Comma-separated strategies or 'all' (options: single-primary,single-pooled,multi-vector)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for encoding (default: 512)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip embeddings that already exist",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parsed_args = parse_args(args)

    # Setup
    output_dir = parsed_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if parsed_args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = parsed_args.device

    # Verify CUDA
    if device == "cuda":
        try:
            torch.cuda.init()
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {total_mem:.1f} GB")
        except RuntimeError as e:
            logger.warning(f"CUDA initialization failed: {e}")
            device = "cpu"

    logger.info(f"Using device: {device}")

    # Load metabolites
    logger.info(f"Loading metabolites from {parsed_args.metabolites}...")
    with open(parsed_args.metabolites) as f:
        metabolites = json.load(f)
    logger.info(f"Loaded {len(metabolites):,} metabolites")

    # Select models
    if parsed_args.models == "all":
        model_configs = list(MODELS.values())
    else:
        model_names = [m.strip() for m in parsed_args.models.split(",")]
        model_configs = [MODELS[m] for m in model_names if m in MODELS]

    # Select strategies
    if parsed_args.strategies == "all":
        strategies = STRATEGIES
    else:
        strategies = [s.strip() for s in parsed_args.strategies.split(",")]

    logger.info(f"Models: {[m.name for m in model_configs]}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Skip existing: {parsed_args.skip_existing}")

    # Process each model/strategy combination
    total_start = time.time()
    results = []

    for config in model_configs:
        for strategy in strategies:
            result = process_model_strategy(
                config=config,
                metabolites=metabolites,
                output_dir=output_dir,
                device=device,
                strategy=strategy,
                batch_size=parsed_args.batch_size,
                skip_existing=parsed_args.skip_existing,
            )
            results.append(result)

    # Save summary
    total_time = time.time() - total_start
    summary = {
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time / 60:.1f} minutes",
        "metabolite_count": len(metabolites),
        "device": device,
        "results": results,
    }

    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Summary saved: {summary_path}")

    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r["skipped"])
    failed = sum(1 for r in results if not r["success"])

    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")

    for r in results:
        status = (
            "SKIPPED"
            if r["skipped"]
            else ("OK" if r["success"] else f"FAILED: {r['error']}")
        )
        time_str = f"{r.get('total_time', 0):.1f}s" if not r["skipped"] else "-"
        logger.info(f"  {r['model']}/{r['strategy']}: {status} ({time_str})")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
