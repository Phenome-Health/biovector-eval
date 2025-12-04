#!/usr/bin/env python
"""Generate embeddings for HMDB metabolites using multiple models.

This script:
1. Loads metabolite data from JSON
2. For each embedding model:
   - Generates embeddings on GPU
   - Saves embeddings to .npy files
   - Builds and saves FAISS indices (HNSW, SQ8, SQ4, PQ)
3. Clears GPU memory between models

Run overnight with:
    nohup uv run python scripts/generate_embeddings.py \
        --metabolites data/hmdb/metabolites.json \
        --output-dir data/ \
        --device cuda \
        --models all \
        > logs/embeddings.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from biovector_eval.core.embedding_strategy import (
    EmbeddingStrategy,
    MultiVectorStrategy,
    StrategyType,
    get_strategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    dimension: int  # Expected embedding dimension


# Models to evaluate
# See docs/decisions.md for rationale on model selection
MODELS: dict[str, ModelConfig] = {
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
    "gte-qwen2": ModelConfig(
        name="GTE-Qwen2-1.5B",
        model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        dimension=1536,
    ),
}

# Legacy models (deprecated - see docs/decisions.md)
LEGACY_MODELS: dict[str, ModelConfig] = {
    "bge-small": ModelConfig(
        name="BGE-Small",
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
    ),
    "chemberta": ModelConfig(
        name="ChemBERTa",
        model_id="DeepChem/ChemBERTa-77M-MTR",
        dimension=384,
    ),
}


def get_model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return model_id.split("/")[-1].lower()


def clear_gpu_memory() -> None:
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 512,
) -> np.ndarray:
    """Generate embeddings for texts.

    Args:
        model: SentenceTransformer model.
        texts: List of text strings.
        batch_size: Batch size for encoding.

    Returns:
        Normalized float32 embeddings (N x D).
    """
    logger.info(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )

    embeddings = embeddings.astype("float32")
    logger.info(f"Generated embeddings: {embeddings.shape}")
    return embeddings


def parse_strategy_argument(strategy_name: str) -> StrategyType:
    """Parse strategy argument string to StrategyType enum.

    Args:
        strategy_name: Strategy name string (e.g., 'single-primary')

    Returns:
        StrategyType enum value

    Raises:
        ValueError: If unknown strategy name
    """
    try:
        return StrategyType(strategy_name)
    except ValueError as err:
        valid_strategies = [s.value for s in StrategyType]
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Valid strategies: {valid_strategies}"
        ) from err


def generate_embeddings_with_strategy(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    strategy: EmbeddingStrategy,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, dict[str, Any]] | None]:
    """Generate embeddings using specified strategy (batch optimized).

    Args:
        metabolites: List of metabolite dicts with 'hmdb_id', 'name', 'synonyms'
        model: SentenceTransformer model for encoding
        strategy: Embedding strategy to use
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings array, metadata dict or None).
        For single-vector strategies, metadata is None.
        For multi-vector, metadata maps vector index to {hmdb_id, tier, weight, text}.
    """
    logger.info(f"Generating embeddings with strategy: {strategy.name}")
    logger.info(f"Processing {len(metabolites)} metabolites...")

    if strategy.name == "single-primary":
        return _generate_single_primary_batch(metabolites, model, batch_size)
    elif strategy.name == "single-pooled":
        return _generate_single_pooled_batch(metabolites, model, batch_size)
    else:  # multi-vector
        return _generate_multi_vector_batch(metabolites, model, batch_size)


def _generate_single_primary_batch(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
) -> tuple[np.ndarray, None]:
    """Batch-optimized single-primary embedding generation."""
    texts = [m["name"] for m in metabolites]

    logger.info(f"Encoding {len(texts)} primary names...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32), None


def _generate_single_pooled_batch(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
    max_synonyms: int = 15,
) -> tuple[np.ndarray, None]:
    """Batch-optimized single-pooled embedding generation.

    Strategy: Process metabolites in batches to avoid OOM.
    For each batch:
    1. Collect texts for batch of metabolites
    2. Encode texts
    3. Average to get pooled embeddings
    4. Clear GPU memory before next batch
    """
    # Process metabolites in batches to avoid OOM
    metabolite_batch_size = 5000  # Process 5K metabolites at a time (~40K texts)
    encode_batch_size = min(batch_size, 64)  # Small batch for encoding
    num_batches = (len(metabolites) + metabolite_batch_size - 1) // metabolite_batch_size

    logger.info(
        f"Processing {len(metabolites)} metabolites in {num_batches} batches "
        f"(encode_batch_size={encode_batch_size})..."
    )

    # Get embedding dimension from a test encode
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output array
    pooled_embeddings = np.zeros((len(metabolites), embed_dim), dtype=np.float32)

    for batch_idx in range(num_batches):
        start_m = batch_idx * metabolite_batch_size
        end_m = min(start_m + metabolite_batch_size, len(metabolites))
        batch_metabolites = metabolites[start_m:end_m]

        # Collect texts for this batch
        batch_texts: list[str] = []
        batch_ranges: list[tuple[int, int]] = []

        for m in batch_metabolites:
            text_start = len(batch_texts)
            texts_for_m = [m["name"]]
            synonyms = m.get("synonyms", [])[:max_synonyms]
            texts_for_m.extend(synonyms)
            batch_texts.extend(texts_for_m)
            batch_ranges.append((text_start, len(batch_texts)))

        logger.info(
            f"  Batch {batch_idx + 1}/{num_batches}: "
            f"{len(batch_metabolites)} metabolites, {len(batch_texts)} texts"
        )

        # Encode this batch
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

        # Clear memory before next batch
        del batch_embeddings, batch_texts, batch_ranges
        clear_gpu_memory()

    logger.info(f"Generated {len(pooled_embeddings)} pooled embeddings")
    return pooled_embeddings, None


def _generate_multi_vector_batch(
    metabolites: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
    max_synonyms: int = 15,
    max_variations: int = 20,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Batch-optimized multi-vector embedding generation.

    Strategy: Process in small chunks to avoid OOM.
    1. Collect ALL texts and metadata (strings only, low memory)
    2. Encode in very small chunks, concatenating results
    3. Build metadata mapping
    """

    strategy = MultiVectorStrategy()

    # Phase 1: Collect all texts and metadata
    all_texts: list[str] = []
    all_metadata: list[dict[str, Any]] = []

    logger.info("Preparing multi-vector texts...")
    for m in metabolites:
        hmdb_id = m["hmdb_id"]
        name = m["name"]
        synonyms = m.get("synonyms", [])[:max_synonyms]

        # Primary name
        all_texts.append(name)
        all_metadata.append(
            {"hmdb_id": hmdb_id, "tier": "primary", "weight": 1.0, "text": name}
        )

        # Synonyms
        seen_lower = {name.lower()}
        for syn in synonyms:
            if syn.lower() not in seen_lower:
                all_texts.append(syn)
                all_metadata.append(
                    {"hmdb_id": hmdb_id, "tier": "synonym", "weight": 0.9, "text": syn}
                )
                seen_lower.add(syn.lower())

        # Variations
        variations = strategy._generate_variations(name, synonyms)
        for var in variations[:max_variations]:
            if var.lower() not in seen_lower:
                all_texts.append(var)
                all_metadata.append(
                    {
                        "hmdb_id": hmdb_id,
                        "tier": "variation",
                        "weight": 0.7,
                        "text": var,
                    }
                )
                seen_lower.add(var.lower())

    logger.info(f"Collected {len(all_texts)} texts from {len(metabolites)} metabolites")

    # Phase 2: Encode in small chunks to avoid OOM
    # Very small chunks since we keep all embeddings in memory
    chunk_size = 20000  # 20K texts per chunk
    encode_batch_size = min(batch_size, 32)  # Very small batch for encoding
    num_chunks = (len(all_texts) + chunk_size - 1) // chunk_size
    logger.info(f"Encoding texts in {num_chunks} chunks (batch_size={encode_batch_size})...")

    # Get embedding dimension
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output array to avoid memory fragmentation
    all_embeddings = np.zeros((len(all_texts), embed_dim), dtype=np.float32)

    for i in range(0, len(all_texts), chunk_size):
        chunk_texts = all_texts[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        logger.info(f"  Chunk {chunk_num}/{num_chunks}: {len(chunk_texts)} texts")

        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Store directly in pre-allocated array
        all_embeddings[i : i + len(chunk_texts)] = chunk_embeddings

        del chunk_embeddings
        clear_gpu_memory()

    # Phase 3: Build metadata mapping
    metadata = {str(i): meta for i, meta in enumerate(all_metadata)}

    logger.info(f"Generated {len(all_embeddings)} multi-vector embeddings")
    return all_embeddings.astype(np.float32), metadata


def get_output_paths(
    output_dir: Path,
    model_slug: str,
    strategy_name: str,
) -> dict[str, Path | None]:
    """Get output file paths for embeddings and metadata.

    Args:
        output_dir: Base output directory
        model_slug: Model name slug (e.g., 'bge-m3')
        strategy_name: Strategy name (e.g., 'single-primary')

    Returns:
        Dict with 'embeddings' and 'metadata' paths.
        For single-vector strategies, metadata is None.
        For multi-vector, both paths point to a subdirectory.
    """
    embeddings_dir = output_dir / "embeddings"

    if strategy_name == "multi-vector":
        # Multi-vector uses directory structure
        strategy_dir = embeddings_dir / f"{model_slug}_{strategy_name}"
        return {
            "embeddings": strategy_dir / "embeddings.npy",
            "metadata": strategy_dir / "metadata.json",
        }
    else:
        # Single-vector strategies use flat file
        return {
            "embeddings": embeddings_dir / f"{model_slug}_{strategy_name}.npy",
            "metadata": None,
        }


def save_multi_vector_output(
    embeddings: np.ndarray,
    metadata: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save multi-vector embeddings and metadata.

    Args:
        embeddings: Embedding array (N x D)
        metadata: Dict mapping index to {hmdb_id, tier, weight, text}
        output_dir: Directory to save embeddings.npy and metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings: {embeddings_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")


def build_and_save_indices(
    embeddings: np.ndarray,
    output_dir: Path,
    model_slug: str,
    hnsw_m: int = 32,
    ef_construction: int = 200,
) -> dict[str, Path]:
    """Build and save all index types.

    Args:
        embeddings: Normalized float32 embeddings.
        output_dir: Directory to save indices.
        model_slug: Model name for filenames.
        hnsw_m: HNSW M parameter.
        ef_construction: HNSW efConstruction.

    Returns:
        Dictionary mapping index type to file path.
    """
    indices_dir = output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]
    paths: dict[str, Path] = {}

    # HNSW (no quantization)
    logger.info(f"Building HNSW index ({dimension}d, M={hnsw_m})...")
    start = time.time()
    hnsw = faiss.IndexHNSWFlat(dimension, hnsw_m)
    hnsw.hnsw.efConstruction = ef_construction
    hnsw.add(embeddings)
    hnsw_path = indices_dir / f"{model_slug}_hnsw.faiss"
    faiss.write_index(hnsw, str(hnsw_path))
    logger.info(
        f"  HNSW: {hnsw_path} ({hnsw_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)"
    )
    paths["hnsw"] = hnsw_path
    del hnsw

    # SQ8 (8-bit scalar quantization)
    logger.info("Building SQ8 index...")
    start = time.time()
    sq8 = faiss.IndexHNSWSQ(dimension, faiss.ScalarQuantizer.QT_8bit, hnsw_m)
    sq8.train(embeddings)
    sq8.add(embeddings)
    sq8_path = indices_dir / f"{model_slug}_sq8.faiss"
    faiss.write_index(sq8, str(sq8_path))
    logger.info(
        f"  SQ8: {sq8_path} ({sq8_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)"
    )
    paths["sq8"] = sq8_path
    del sq8

    # SQ4 (4-bit scalar quantization)
    logger.info("Building SQ4 index...")
    start = time.time()
    sq4 = faiss.IndexHNSWSQ(dimension, faiss.ScalarQuantizer.QT_4bit, hnsw_m)
    sq4.train(embeddings)
    sq4.add(embeddings)
    sq4_path = indices_dir / f"{model_slug}_sq4.faiss"
    faiss.write_index(sq4, str(sq4_path))
    logger.info(
        f"  SQ4: {sq4_path} ({sq4_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)"
    )
    paths["sq4"] = sq4_path
    del sq4

    # PQ (product quantization)
    logger.info("Building PQ index...")
    start = time.time()
    # Adjust pq_m to divide dimension evenly
    pq_m = 32
    while dimension % pq_m != 0 and pq_m > 1:
        pq_m -= 1
    pq = faiss.IndexHNSWPQ(dimension, pq_m, hnsw_m)
    pq.train(embeddings)
    pq.add(embeddings)
    pq_path = indices_dir / f"{model_slug}_pq.faiss"
    faiss.write_index(pq, str(pq_path))
    logger.info(
        f"  PQ (m={pq_m}): {pq_path} ({pq_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)"
    )
    paths["pq"] = pq_path
    del pq

    return paths


def process_model(
    config: ModelConfig,
    texts: list[str],
    output_dir: Path,
    device: str,
    batch_size: int = 512,
) -> dict:
    """Process a single model: generate embeddings and build indices (legacy).

    Args:
        config: Model configuration.
        texts: List of metabolite names.
        output_dir: Output directory.
        device: Device for inference.
        batch_size: Batch size for encoding.

    Returns:
        Dictionary with timing and path information.
    """
    model_slug = get_model_slug(config.model_id)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {config.name} ({config.model_id})")
    logger.info(f"{'='*60}")

    total_start = time.time()
    result = {
        "model": config.name,
        "model_id": config.model_id,
        "model_slug": model_slug,
    }

    # Load model
    logger.info(f"Loading model on {device}...")
    model_start = time.time()
    model = SentenceTransformer(config.model_id, device=device)
    result["model_load_time"] = time.time() - model_start
    logger.info(f"Model loaded in {result['model_load_time']:.1f}s")

    # Generate embeddings
    embed_start = time.time()
    embeddings = generate_embeddings(model, texts, batch_size)
    result["embedding_time"] = time.time() - embed_start
    result["embedding_shape"] = list(embeddings.shape)

    # Verify dimension
    if embeddings.shape[1] != config.dimension:
        logger.warning(
            f"Dimension mismatch: expected {config.dimension}, got {embeddings.shape[1]}"
        )

    # Save embeddings
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / f"{model_slug}.npy"
    np.save(embeddings_path, embeddings)
    result["embeddings_path"] = str(embeddings_path)
    result["embeddings_size_mb"] = embeddings.nbytes / 1024 / 1024
    logger.info(
        f"Saved embeddings: {embeddings_path} ({result['embeddings_size_mb']:.1f} MB)"
    )

    # Unload model to free GPU memory
    del model
    clear_gpu_memory()

    # Build and save indices
    index_start = time.time()
    index_paths = build_and_save_indices(embeddings, output_dir, model_slug)
    result["index_time"] = time.time() - index_start
    result["index_paths"] = {k: str(v) for k, v in index_paths.items()}

    # Clean up
    del embeddings
    clear_gpu_memory()

    result["total_time"] = time.time() - total_start
    logger.info(f"Completed {config.name} in {result['total_time']:.1f}s")

    return result


def process_model_with_strategy(
    config: ModelConfig,
    metabolites: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    strategy_name: str,
    batch_size: int = 512,
) -> dict:
    """Process a model with specified embedding strategy.

    Args:
        config: Model configuration.
        metabolites: List of metabolite dicts with hmdb_id, name, synonyms.
        output_dir: Output directory.
        device: Device for inference.
        strategy_name: Strategy name (single-primary, single-pooled, multi-vector).
        batch_size: Batch size for encoding.

    Returns:
        Dictionary with timing and path information.
    """
    model_slug = get_model_slug(config.model_id)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {config.name} ({config.model_id})")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"{'='*60}")

    total_start = time.time()
    result = {
        "model": config.name,
        "model_id": config.model_id,
        "model_slug": model_slug,
        "strategy": strategy_name,
    }

    # Load model
    logger.info(f"Loading model on {device}...")
    model_start = time.time()
    model = SentenceTransformer(config.model_id, device=device)
    result["model_load_time"] = time.time() - model_start
    logger.info(f"Model loaded in {result['model_load_time']:.1f}s")

    # Get strategy
    strategy_type = parse_strategy_argument(strategy_name)
    strategy = get_strategy(strategy_type)

    # Generate embeddings with strategy
    embed_start = time.time()
    embeddings, metadata = generate_embeddings_with_strategy(
        metabolites, model, strategy
    )
    result["embedding_time"] = time.time() - embed_start
    result["embedding_shape"] = list(embeddings.shape)
    result["num_vectors"] = embeddings.shape[0]

    logger.info(f"Generated {embeddings.shape[0]} vectors ({embeddings.shape[1]}d)")

    # Get output paths
    paths = get_output_paths(output_dir, model_slug, strategy_name)

    # Save embeddings
    if strategy_name == "multi-vector":
        # Save to directory structure
        strategy_dir = output_dir / "embeddings" / f"{model_slug}_{strategy_name}"
        save_multi_vector_output(embeddings, metadata, strategy_dir)
        result["embeddings_path"] = str(paths["embeddings"])
        result["metadata_path"] = str(paths["metadata"])
    else:
        # Save flat file
        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(paths["embeddings"], embeddings)
        result["embeddings_path"] = str(paths["embeddings"])
        logger.info(f"Saved embeddings: {paths['embeddings']}")

    result["embeddings_size_mb"] = embeddings.nbytes / 1024 / 1024

    # Unload model to free GPU memory
    del model
    clear_gpu_memory()

    # Clean up embeddings
    del embeddings
    clear_gpu_memory()

    result["total_time"] = time.time() - total_start
    logger.info(
        f"Completed {config.name} ({strategy_name}) in {result['total_time']:.1f}s"
    )

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for HMDB metabolites"
    )
    parser.add_argument(
        "--metabolites",
        required=True,
        help="Path to metabolites JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for embeddings and indices",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--strategy",
        default="single-primary",
        choices=["single-primary", "single-pooled", "multi-vector"],
        help="Embedding strategy",
    )
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Verify CUDA actually works if requested
    if device == "cuda":
        try:
            torch.cuda.init()
            logger.info(f"Using device: {device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        except RuntimeError as e:
            logger.warning(f"CUDA initialization failed: {e}")
            logger.warning("Falling back to CPU. Consider rebooting to fix CUDA.")
            device = "cpu"
            logger.info(f"Using device: {device}")
    else:
        logger.info(f"Using device: {device}")

    # Load metabolites
    logger.info(f"Loading metabolites from {args.metabolites}...")
    with open(args.metabolites) as f:
        metabolites = json.load(f)

    texts = [m["name"] for m in metabolites]
    ids = [m["hmdb_id"] for m in metabolites]
    logger.info(f"Loaded {len(texts)} metabolites")
    logger.info(f"Strategy: {args.strategy}")

    # Save ID mapping
    id_mapping_path = output_dir / "hmdb" / "id_mapping.json"
    id_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(id_mapping_path, "w") as f:
        json.dump({"ids": ids, "count": len(ids)}, f)
    logger.info(f"Saved ID mapping: {id_mapping_path}")

    # Select models
    if args.models == "all":
        model_configs = list(MODELS.values())
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        model_configs = [MODELS[m] for m in model_names if m in MODELS]

    logger.info(f"Models to process: {[m.name for m in model_configs]}")

    # Process each model
    total_start = time.time()
    results = []

    for config in model_configs:
        try:
            result = process_model_with_strategy(
                config,
                metabolites,
                output_dir,
                device,
                args.strategy,
                args.batch_size,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {config.name}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "model": config.name,
                    "strategy": args.strategy,
                    "error": str(e),
                }
            )
            clear_gpu_memory()

    # Save summary
    total_time = time.time() - total_start
    summary = {
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time / 60:.1f} minutes",
        "metabolite_count": len(texts),
        "device": device,
        "models": results,
    }

    summary_path = output_dir / "embedding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Summary saved: {summary_path}")

    # Print summary
    for r in results:
        if "error" in r:
            logger.info(f"  {r['model']}: FAILED - {r['error']}")
        else:
            timing_parts = [f"embed: {r['embedding_time']:.1f}s"]
            if "index_time" in r:
                timing_parts.append(f"index: {r['index_time']:.1f}s")
            logger.info(
                f"  {r['model']}: {r['total_time']:.1f}s "
                f"({', '.join(timing_parts)})"
            )


if __name__ == "__main__":
    main()
