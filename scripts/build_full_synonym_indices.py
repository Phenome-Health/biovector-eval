#!/usr/bin/env python3
"""Build vector indices with ALL HMDB synonyms (no variations).

Creates 6 indices:
- single-primary: 1 vector per metabolite (name only)
- single-pooled: 1 vector per metabolite (avg of name + ALL synonyms)
- multi-vector: N vectors per metabolite (1 primary + ALL synonyms)

Each strategy with two index types:
- hnsw_sq8: HNSW with scalar quantization (8-bit)
- ivf_sq8: IVF with scalar quantization (8-bit)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = Path("data/hmdb/metabolites.json")
OUTPUT_DIR = Path("data/indices/full_synonyms")
BATCH_SIZE = 512
DIMENSION = 384  # MiniLM dimension


def load_metabolites() -> list[dict[str, Any]]:
    """Load HMDB metabolites."""
    logger.info(f"Loading metabolites from {DATA_PATH}")
    with open(DATA_PATH) as f:
        metabolites = json.load(f)
    logger.info(f"Loaded {len(metabolites):,} metabolites")
    return metabolites


def generate_single_primary_embeddings(
    metabolites: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[str]]:
    """Generate one embedding per metabolite from primary name."""
    logger.info("Generating single-primary embeddings...")

    texts = [m["name"] for m in metabolites]
    ids = [m["hmdb_id"] for m in metabolites]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info(f"Generated {len(embeddings):,} embeddings")
    return embeddings.astype(np.float32), ids


def generate_single_pooled_embeddings(
    metabolites: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[str]]:
    """Generate one embedding per metabolite from averaged name + ALL synonyms."""
    logger.info("Generating single-pooled embeddings (ALL synonyms)...")

    pooled_embeddings = []
    ids = []

    for m in tqdm(metabolites, desc="Pooling"):
        texts = [m["name"]] + m.get("synonyms", [])

        # Encode all texts (not normalized)
        embeds = model.encode(texts, normalize_embeddings=False)

        # Average and normalize
        mean_embed = np.mean(embeds, axis=0)
        norm = np.linalg.norm(mean_embed)
        if norm > 0:
            mean_embed = mean_embed / norm

        pooled_embeddings.append(mean_embed)
        ids.append(m["hmdb_id"])

    embeddings = np.array(pooled_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings):,} pooled embeddings")
    return embeddings, ids


def generate_multi_vector_embeddings(
    metabolites: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[dict]]:
    """Generate multiple embeddings per metabolite (primary + ALL synonyms).

    NO synthetic variations - just raw HMDB data.
    """
    logger.info("Generating multi-vector embeddings (ALL synonyms, NO variations)...")

    # Collect all texts with metadata
    all_texts = []
    all_metadata = []

    for m in metabolites:
        hmdb_id = m["hmdb_id"]
        name = m["name"]
        synonyms = m.get("synonyms", [])

        # Primary name
        all_texts.append(name)
        all_metadata.append({
            "hmdb_id": hmdb_id,
            "tier": "primary",
            "weight": 1.0,
            "text": name,
        })

        # All synonyms (no limit!)
        seen = {name.lower()}
        for syn in synonyms:
            if syn.lower() not in seen:
                all_texts.append(syn)
                all_metadata.append({
                    "hmdb_id": hmdb_id,
                    "tier": "synonym",
                    "weight": 0.9,
                    "text": syn,
                })
                seen.add(syn.lower())

    logger.info(f"Encoding {len(all_texts):,} texts...")

    # Batch encode all
    embeddings = model.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info(f"Generated {len(embeddings):,} multi-vector embeddings")
    return embeddings.astype(np.float32), all_metadata


def build_hnsw_sq8_index(
    embeddings: np.ndarray,
    hnsw_m: int = 32,
    ef_construction: int = 200,
) -> faiss.Index:
    """Build HNSW index with scalar quantization (8-bit)."""
    logger.info(f"Building HNSW+SQ8 index: {embeddings.shape[0]:,} vectors, dim={embeddings.shape[1]}")

    dim = embeddings.shape[1]

    # Create HNSW index with SQ (pass quantizer type directly)
    index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, hnsw_m)
    index.hnsw.efConstruction = ef_construction

    # Train and add
    index.train(embeddings)
    index.add(embeddings)

    logger.info(f"Built HNSW+SQ8 index with {index.ntotal:,} vectors")
    return index


def build_ivf_sq8_index(
    embeddings: np.ndarray,
    nlist: int = 1024,
    nprobe: int = 32,
) -> faiss.Index:
    """Build IVF index with scalar quantization (8-bit)."""
    logger.info(f"Building IVF+SQ8 index: {embeddings.shape[0]:,} vectors, dim={embeddings.shape[1]}")

    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    # Adjust nlist for smaller datasets
    nlist = min(nlist, n_vectors // 39)  # Ensure at least 39 vectors per cluster

    # Create quantizer and IVF+SQ index
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit
    )

    # Train and add
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = nprobe

    logger.info(f"Built IVF+SQ8 index with {index.ntotal:,} vectors (nlist={nlist})")
    return index


def save_index(
    index: faiss.Index,
    output_path: Path,
    metadata: list[dict] | list[str] | None = None,
) -> None:
    """Save FAISS index and optional metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(output_path))
    logger.info(f"Saved index to {output_path}")

    # Save metadata if provided
    if metadata is not None:
        meta_path = output_path.with_suffix(".metadata.json")

        # Convert to index-keyed dict if it's a list of dicts
        if metadata and isinstance(metadata[0], dict):
            meta_dict = {str(i): m for i, m in enumerate(metadata)}
        else:
            # It's a list of IDs - create simple metadata
            meta_dict = {str(i): {"hmdb_id": id_} for i, id_ in enumerate(metadata)}

        with open(meta_path, "w") as f:
            json.dump(meta_dict, f)
        logger.info(f"Saved metadata to {meta_path}")


def main() -> int:
    """Build all 6 indices."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and model
    metabolites = load_metabolites()
    logger.info(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device="cuda")

    # =========================================================================
    # Single-Primary Strategy
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-PRIMARY STRATEGY")
    logger.info("=" * 60)

    sp_embeddings, sp_ids = generate_single_primary_embeddings(metabolites, model)

    # HNSW+SQ8
    sp_hnsw = build_hnsw_sq8_index(sp_embeddings)
    save_index(sp_hnsw, OUTPUT_DIR / "minilm_single-primary_hnsw_sq8.faiss", sp_ids)

    # IVF+SQ8
    sp_ivf = build_ivf_sq8_index(sp_embeddings)
    save_index(sp_ivf, OUTPUT_DIR / "minilm_single-primary_ivf_sq8.faiss", sp_ids)

    del sp_hnsw, sp_ivf  # Free memory

    # =========================================================================
    # Single-Pooled Strategy
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-POOLED STRATEGY (ALL SYNONYMS)")
    logger.info("=" * 60)

    pool_embeddings, pool_ids = generate_single_pooled_embeddings(metabolites, model)

    # HNSW+SQ8
    pool_hnsw = build_hnsw_sq8_index(pool_embeddings)
    save_index(pool_hnsw, OUTPUT_DIR / "minilm_single-pooled_hnsw_sq8.faiss", pool_ids)

    # IVF+SQ8
    pool_ivf = build_ivf_sq8_index(pool_embeddings)
    save_index(pool_ivf, OUTPUT_DIR / "minilm_single-pooled_ivf_sq8.faiss", pool_ids)

    del pool_embeddings, pool_hnsw, pool_ivf  # Free memory

    # =========================================================================
    # Multi-Vector Strategy
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-VECTOR STRATEGY (ALL SYNONYMS, NO VARIATIONS)")
    logger.info("=" * 60)

    mv_embeddings, mv_metadata = generate_multi_vector_embeddings(metabolites, model)

    # HNSW+SQ8
    mv_hnsw = build_hnsw_sq8_index(mv_embeddings)
    save_index(mv_hnsw, OUTPUT_DIR / "minilm_multi-vector_hnsw_sq8.faiss", mv_metadata)

    # IVF+SQ8
    mv_ivf = build_ivf_sq8_index(mv_embeddings)
    save_index(mv_ivf, OUTPUT_DIR / "minilm_multi-vector_ivf_sq8.faiss", mv_metadata)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)

    # List created files
    for f in sorted(OUTPUT_DIR.glob("*.faiss")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
