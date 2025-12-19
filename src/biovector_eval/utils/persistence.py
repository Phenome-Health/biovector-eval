"""Persistence utilities for embeddings and FAISS indices."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def get_model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug.

    Examples:
        "BAAI/bge-small-en-v1.5" -> "bge-small-en-v1.5"
        "DeepChem/ChemBERTa-77M-MTR" -> "chemberta-77m-mtr"
    """
    # Take the part after the last slash
    slug = model_id.split("/")[-1]
    # Convert to lowercase
    slug = slug.lower()
    return slug


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save embeddings to a .npy file.

    Args:
        embeddings: Numpy array of embeddings (N x D).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    size_mb = embeddings.nbytes / 1024 / 1024
    logger.info(f"Saved embeddings: {path} ({embeddings.shape}, {size_mb:.1f} MB)")


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load embeddings from a .npy file.

    Args:
        path: Path to the .npy file.

    Returns:
        Numpy array of embeddings.
    """
    path = Path(path)
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings: {path} ({embeddings.shape})")
    return embeddings


def save_index(index: faiss.Index, path: Path | str) -> None:
    """Save FAISS index to disk.

    Args:
        index: FAISS index to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Saved index: {path} ({size_mb:.1f} MB)")


def load_index(path: Path | str) -> faiss.Index:
    """Load FAISS index from disk.

    Args:
        path: Path to the FAISS index file.

    Returns:
        FAISS index.
    """
    path = Path(path)
    index = faiss.read_index(str(path))
    logger.info(f"Loaded index: {path} (ntotal={index.ntotal})")
    return index


def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """Build Flat (exact) index for baseline comparison.

    Args:
        embeddings: Normalized float32 embeddings (N x D).

    Returns:
        FAISS Flat index with embeddings added.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(embeddings)
    return index


def build_hnsw_index(
    embeddings: np.ndarray,
    hnsw_m: int = 32,
    ef_construction: int = 200,
) -> faiss.Index:
    """Build HNSW index from embeddings.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        hnsw_m: HNSW M parameter (connections per node).
        ef_construction: HNSW efConstruction parameter.

    Returns:
        FAISS HNSW index with embeddings added.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    return index


def build_hnsw_sq8_index(
    embeddings: np.ndarray,
    hnsw_m: int = 32,
) -> faiss.Index:
    """Build HNSW+SQ8 (8-bit scalar quantization) index.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        hnsw_m: HNSW M parameter.

    Returns:
        FAISS HNSW+SQ8 index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWSQ(dimension, faiss.ScalarQuantizer.QT_8bit, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index


def build_hnsw_sq4_index(
    embeddings: np.ndarray,
    hnsw_m: int = 32,
) -> faiss.Index:
    """Build HNSW+SQ4 (4-bit scalar quantization) index.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        hnsw_m: HNSW M parameter.

    Returns:
        FAISS HNSW+SQ4 index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWSQ(dimension, faiss.ScalarQuantizer.QT_4bit, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index


def build_hnsw_pq_index(
    embeddings: np.ndarray,
    pq_m: int = 32,
    hnsw_m: int = 32,
) -> faiss.Index:
    """Build HNSW+PQ (product quantization) index.

    Note: Uses L2 metric (default) because IndexHNSWPQ with METRIC_INNER_PRODUCT
    causes segfaults in FAISS. Since embeddings are L2-normalized, L2 distance
    is equivalent to inner product for ranking: d_L2 = 2 - 2*IP.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        pq_m: Number of subquantizers (must divide dimension evenly).
        hnsw_m: HNSW M parameter.

    Returns:
        FAISS HNSW+PQ index.
    """
    dimension = embeddings.shape[1]
    # Adjust pq_m if it doesn't divide evenly
    while dimension % pq_m != 0 and pq_m > 1:
        pq_m -= 1
    # Don't use METRIC_INNER_PRODUCT - causes segfault in IndexHNSWPQ
    index = faiss.IndexHNSWPQ(dimension, pq_m, hnsw_m)
    index.train(embeddings)
    index.add(embeddings)
    return index


def build_ivf_flat_index(
    embeddings: np.ndarray,
    nlist: int | None = None,
    nprobe: int = 32,
) -> faiss.Index:
    """Build IVFFlat index (clustering + exhaustive search within clusters).

    IVF (Inverted File) indexes partition vectors into clusters using k-means.
    At search time, only vectors in the nprobe nearest clusters are searched.

    This is useful for large-scale datasets (100M+) where HNSW becomes memory-
    intensive. The trade-off is accuracy vs speed (controlled by nprobe).

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        nlist: Number of clusters. If None, uses sqrt(N) as default.
        nprobe: Number of clusters to search (speed/accuracy trade-off).

    Returns:
        FAISS IVFFlat index (trained and populated).
    """
    n_vectors, dimension = embeddings.shape

    # Default nlist to sqrt(n) if not specified
    if nlist is None:
        nlist = max(1, int(np.sqrt(n_vectors)))

    # Ensure nlist doesn't exceed number of vectors
    nlist = min(nlist, n_vectors)

    # Create quantizer (coarse search structure)
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVFFlat index
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train on embeddings (learns cluster centroids)
    index.train(embeddings)

    # Add vectors
    index.add(embeddings)

    # Set nprobe for search
    index.nprobe = nprobe

    return index


def build_ivf_pq_index(
    embeddings: np.ndarray,
    nlist: int | None = None,
    pq_m: int = 32,
    pq_nbits: int = 8,
    nprobe: int = 32,
) -> faiss.Index:
    """Build IVFPQ index (clustering + product quantization).

    Combines IVF clustering with product quantization for maximum compression.
    This is the most memory-efficient index type, suitable for billion-scale
    datasets.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        nlist: Number of clusters. If None, uses sqrt(N) as default.
        pq_m: Number of subquantizers (must divide dimension evenly).
        pq_nbits: Bits per subquantizer (usually 8).
        nprobe: Number of clusters to search.

    Returns:
        FAISS IVFPQ index (trained and populated).
    """
    n_vectors, dimension = embeddings.shape

    # Default nlist to sqrt(n) if not specified
    if nlist is None:
        nlist = max(1, int(np.sqrt(n_vectors)))

    # Ensure nlist doesn't exceed number of vectors
    nlist = min(nlist, n_vectors)

    # Adjust pq_m to divide dimension evenly
    while dimension % pq_m != 0 and pq_m > 1:
        pq_m -= 1

    # Create quantizer (coarse search structure)
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVFPQ index
    index = faiss.IndexIVFPQ(
        quantizer, dimension, nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT
    )

    # Train on embeddings (learns cluster centroids and PQ codebook)
    index.train(embeddings)

    # Add vectors
    index.add(embeddings)

    # Set nprobe for search
    index.nprobe = nprobe

    return index


def build_ivf_sq8_index(
    embeddings: np.ndarray,
    nlist: int | None = None,
    nprobe: int = 32,
) -> faiss.Index:
    """Build IVF+SQ8 index (clustering + 8-bit scalar quantization).

    Combines IVF clustering with 8-bit scalar quantization. Provides ~4x
    compression compared to IVFFlat while maintaining good accuracy.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        nlist: Number of clusters. If None, uses sqrt(N) as default.
        nprobe: Number of clusters to search.

    Returns:
        FAISS IVF+SQ8 index (trained and populated).
    """
    n_vectors, dimension = embeddings.shape

    # Default nlist to sqrt(n) if not specified
    if nlist is None:
        nlist = max(1, int(np.sqrt(n_vectors)))

    # Ensure nlist doesn't exceed number of vectors
    nlist = min(nlist, n_vectors)

    # Create quantizer (coarse search structure)
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVF+SQ8 index
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dimension,
        nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )

    # Train on embeddings (learns cluster centroids and SQ parameters)
    index.train(embeddings)

    # Add vectors
    index.add(embeddings)

    # Set nprobe for search
    index.nprobe = nprobe

    return index


def build_ivf_sq4_index(
    embeddings: np.ndarray,
    nlist: int | None = None,
    nprobe: int = 32,
) -> faiss.Index:
    """Build IVF+SQ4 index (clustering + 4-bit scalar quantization).

    Combines IVF clustering with 4-bit scalar quantization. Provides ~8x
    compression compared to IVFFlat with some accuracy trade-off.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        nlist: Number of clusters. If None, uses sqrt(N) as default.
        nprobe: Number of clusters to search.

    Returns:
        FAISS IVF+SQ4 index (trained and populated).
    """
    n_vectors, dimension = embeddings.shape

    # Default nlist to sqrt(n) if not specified
    if nlist is None:
        nlist = max(1, int(np.sqrt(n_vectors)))

    # Ensure nlist doesn't exceed number of vectors
    nlist = min(nlist, n_vectors)

    # Create quantizer (coarse search structure)
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVF+SQ4 index
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dimension,
        nlist,
        faiss.ScalarQuantizer.QT_4bit,
        faiss.METRIC_INNER_PRODUCT,
    )

    # Train on embeddings (learns cluster centroids and SQ parameters)
    index.train(embeddings)

    # Add vectors
    index.add(embeddings)

    # Set nprobe for search
    index.nprobe = nprobe

    return index


def build_all_indices(
    embeddings: np.ndarray,
    output_dir: Path | str,
    model_slug: str,
    hnsw_m: int = 32,
    ef_construction: int = 200,
) -> dict[str, Path]:
    """Build and save all index types for a model.

    Args:
        embeddings: Normalized float32 embeddings (N x D).
        output_dir: Directory to save indices.
        model_slug: Model name slug for filenames.
        hnsw_m: HNSW M parameter.
        ef_construction: HNSW efConstruction parameter.

    Returns:
        Dictionary mapping index type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # HNSW (no quantization)
    logger.info(f"Building HNSW index for {model_slug}...")
    hnsw = build_hnsw_index(embeddings, hnsw_m, ef_construction)
    hnsw_path = output_dir / f"{model_slug}_hnsw.faiss"
    save_index(hnsw, hnsw_path)
    paths["hnsw"] = hnsw_path
    del hnsw

    # SQ8
    logger.info(f"Building SQ8 index for {model_slug}...")
    sq8 = build_sq8_index(embeddings, hnsw_m)
    sq8_path = output_dir / f"{model_slug}_sq8.faiss"
    save_index(sq8, sq8_path)
    paths["sq8"] = sq8_path
    del sq8

    # SQ4
    logger.info(f"Building SQ4 index for {model_slug}...")
    sq4 = build_sq4_index(embeddings, hnsw_m)
    sq4_path = output_dir / f"{model_slug}_sq4.faiss"
    save_index(sq4, sq4_path)
    paths["sq4"] = sq4_path
    del sq4

    # PQ
    logger.info(f"Building PQ index for {model_slug}...")
    pq = build_pq_index(embeddings, hnsw_m=hnsw_m)
    pq_path = output_dir / f"{model_slug}_pq.faiss"
    save_index(pq, pq_path)
    paths["pq"] = pq_path
    del pq

    return paths
