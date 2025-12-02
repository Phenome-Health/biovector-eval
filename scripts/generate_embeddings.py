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

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
MODELS: dict[str, ModelConfig] = {
    "bge-small": ModelConfig(
        name="BGE-Small",
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
        dimension=1024,
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
    logger.info(f"  HNSW: {hnsw_path} ({hnsw_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)")
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
    logger.info(f"  SQ8: {sq8_path} ({sq8_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)")
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
    logger.info(f"  SQ4: {sq4_path} ({sq4_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)")
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
    logger.info(f"  PQ (m={pq_m}): {pq_path} ({pq_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - start:.1f}s)")
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
    """Process a single model: generate embeddings and build indices.

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
    logger.info(f"Saved embeddings: {embeddings_path} ({result['embeddings_size_mb']:.1f} MB)")

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
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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
            result = process_model(
                config,
                texts,
                output_dir,
                device,
                args.batch_size,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {config.name}: {e}")
            results.append({
                "model": config.name,
                "error": str(e),
            })
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
            logger.info(
                f"  {r['model']}: {r['total_time']:.1f}s "
                f"(embed: {r['embedding_time']:.1f}s, index: {r['index_time']:.1f}s)"
            )


if __name__ == "__main__":
    main()
