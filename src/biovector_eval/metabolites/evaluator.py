"""Embedding model evaluation for metabolite search."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from biovector_eval.core.metrics import SearchMetrics, evaluate_search_results
from biovector_eval.core.performance import (
    LatencyMetrics,
    MemoryMetrics,
    measure_latency,
    measure_memory,
)
from biovector_eval.metabolites.ground_truth import GroundTruthDataset

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    device: str = "cpu"
    batch_size: int = 512


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""

    model_name: str
    model_id: str
    accuracy: SearchMetrics
    latency: LatencyMetrics
    memory: MemoryMetrics
    index_config: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "model": {"name": self.model_name, "id": self.model_id},
            "accuracy": self.accuracy.to_dict(),
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "index_config": self.index_config,
        }

    def save(self, path: Path | str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Model configurations for Phase 1
METABOLITE_MODELS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="BGE-Small (Baseline)",
        model_id="BAAI/bge-small-en-v1.5",
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
    ),
    "chemberta-mtr": ModelConfig(
        name="ChemBERTa-MTR",
        model_id="DeepChem/ChemBERTa-77M-MTR",
    ),
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
}


class MetaboliteEvaluator:
    """Evaluate embedding models for metabolite search."""

    def __init__(
        self,
        texts: list[str],
        ids: list[str],
        ground_truth: GroundTruthDataset,
        device: str = "cpu",
    ):
        """Initialize evaluator.

        Args:
            texts: List of metabolite name strings to embed.
            ids: Corresponding HMDB IDs.
            ground_truth: Ground truth dataset for evaluation.
            device: Device for model inference ('cpu' or 'cuda').
        """
        self.texts = texts
        self.ids = ids
        self.ground_truth = ground_truth
        self.device = device
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._embeddings: np.ndarray | None = None

    def load_model(self, config: ModelConfig) -> SentenceTransformer:
        """Load embedding model."""
        logger.info(f"Loading model: {config.model_id}")
        self._model = SentenceTransformer(
            config.model_id, device=config.device or self.device
        )
        return self._model

    def generate_embeddings(self, batch_size: int = 512) -> np.ndarray:
        """Generate embeddings for all metabolites."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        logger.info(f"Generating embeddings for {len(self.texts)} metabolites")
        embeddings = self._model.encode(
            self.texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._embeddings = embeddings.astype("float32")
        return self._embeddings

    def build_index(
        self,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
    ) -> faiss.Index:
        """Build FAISS index from embeddings."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not generated")

        dimension = self._embeddings.shape[1]
        logger.info(
            f"Building {index_type} index: {dimension}d, {len(self._embeddings)} vectors"
        )

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, hnsw_m)
            index.hnsw.efConstruction = hnsw_ef_construction
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize for cosine similarity
        embeddings = self._embeddings.copy()
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self._index = index
        return index

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for similar metabolites."""
        if self._model is None or self._index is None:
            raise RuntimeError("Model and index must be loaded")

        query_vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        _, indices = self._index.search(query_vec, k)
        return [self.ids[i] for i in indices[0] if 0 <= i < len(self.ids)]

    def evaluate(
        self,
        config: ModelConfig,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
    ) -> EvaluationResult:
        """Run complete evaluation for a model.

        Args:
            config: Model configuration.
            hnsw_m: HNSW M parameter.
            hnsw_ef_construction: HNSW efConstruction parameter.
            hnsw_ef_search: HNSW efSearch parameter for queries.

        Returns:
            EvaluationResult with all metrics.
        """
        logger.info(f"Evaluating: {config.name}")

        # Load and embed
        self.load_model(config)
        self.generate_embeddings(config.batch_size)

        # Build index
        index = self.build_index("hnsw", hnsw_m, hnsw_ef_construction)
        index.hnsw.efSearch = hnsw_ef_search

        # Evaluate accuracy
        queries = [q.to_dict() for q in self.ground_truth.queries]
        accuracy = evaluate_search_results(queries, self.search, k=10)

        # Evaluate latency
        query_texts = [q.query for q in self.ground_truth.queries]
        latency = measure_latency(self.search, query_texts, k=10, warmup=20)

        # Evaluate memory
        assert (
            self._embeddings is not None
        ), "Embeddings must be loaded before evaluation"
        memory = measure_memory(
            len(self._embeddings), self._embeddings.shape[1], hnsw_m=hnsw_m
        )

        return EvaluationResult(
            model_name=config.name,
            model_id=config.model_id,
            accuracy=accuracy,
            latency=latency,
            memory=memory,
            index_config={
                "type": "hnsw",
                "hnsw_m": hnsw_m,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search": hnsw_ef_search,
            },
        )

    def compare_models(
        self,
        models: list[ModelConfig],
        output_dir: Path | str,
    ) -> dict[str, EvaluationResult]:
        """Compare multiple models and save results.

        Args:
            models: List of model configurations to compare.
            output_dir: Directory to save results.

        Returns:
            Dictionary mapping model names to results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, EvaluationResult] = {}
        for config in models:
            result = self.evaluate(config)
            results[config.name] = result
            result.save(output_dir / f"{config.name.lower().replace(' ', '_')}.json")

        # Save summary
        self._save_summary(results, output_dir / "comparison_summary.json")
        return results

    def _save_summary(self, results: dict[str, EvaluationResult], path: Path) -> None:
        """Save comparison summary."""
        baseline = results.get("BGE-Small (Baseline)")
        comparison = {}

        for name, result in results.items():
            entry = result.to_dict()
            if baseline and name != "BGE-Small (Baseline)":
                entry["vs_baseline"] = {
                    "recall@1_delta": result.accuracy.recall_at_1
                    - baseline.accuracy.recall_at_1,
                    "recall@5_delta": result.accuracy.recall_at_5
                    - baseline.accuracy.recall_at_5,
                    "mrr_delta": result.accuracy.mrr - baseline.accuracy.mrr,
                    "latency_ratio": result.latency.p95_ms / baseline.latency.p95_ms,
                }
            comparison[name] = entry

        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)


# =============================================================================
# Multi-Vector Search Support
# =============================================================================


@dataclass
class SearchResult:
    """A single search result with metadata."""

    hmdb_id: str
    score: float
    tier: str
    weight: float

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score (raw score * weight)."""
        return self.score * self.weight


def load_index_with_metadata(
    index_path: Path | str,
) -> tuple[faiss.Index, dict[str, dict[str, Any]] | None]:
    """Load a FAISS index and its associated metadata if present.

    Args:
        index_path: Path to the .faiss index file.

    Returns:
        Tuple of (index, metadata). Metadata is None if no metadata file exists.

    Raises:
        FileNotFoundError: If index file doesn't exist.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # Load index
    index = faiss.read_index(str(index_path))

    # Try to load metadata
    metadata_path = index_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return index, metadata

    return index, None


class MultiVectorSearcher:
    """Searcher that handles both single-vector and multi-vector indices.

    For multi-vector indices:
    - Oversamples search results to account for multiple vectors per metabolite
    - Deduplicates by HMDB ID, keeping the best weighted score
    - Returns results sorted by weighted score

    For single-vector indices:
    - Returns standard search results with weight 1.0
    """

    def __init__(
        self,
        index_path: Path | str,
        id_mapping: list[str] | None = None,
        index_type: str | None = None,
    ):
        """Initialize searcher.

        Args:
            index_path: Path to the .faiss index file.
            id_mapping: Optional list mapping index positions to HMDB IDs.
                       Required for single-vector indices without metadata.
            index_type: Index type string (e.g., "hnsw_pq", "ivf_pq", "flat").
                       Used to detect L2-metric indices that need score conversion.
        """
        index_path = Path(index_path)
        self.index, self.metadata = load_index_with_metadata(index_path)
        self._id_mapping = id_mapping
        self._index_type = index_type
        # HNSW+PQ uses L2 metric (lower=better) because IP causes FAISS segfaults
        # All other indices use inner product (higher=better)
        self._uses_l2_metric = index_type == "hnsw_pq"

    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector index."""
        return self.metadata is not None

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        oversample_factor: int = 50,
    ) -> list[SearchResult]:
        """Search for similar metabolites.

        Args:
            query_embedding: Query embedding vector (1 x dim).
            k: Number of results to return.
            oversample_factor: For multi-vector, multiply k by this to account
                              for duplicates before deduplication.

        Returns:
            List of SearchResult objects, sorted by weighted_score descending.
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if self.is_multi_vector:
            return self._search_multi_vector(query_embedding, k, oversample_factor)
        else:
            return self._search_single_vector(query_embedding, k)

    def _search_multi_vector(
        self,
        query_embedding: np.ndarray,
        k: int,
        oversample_factor: int,
    ) -> list[SearchResult]:
        """Search multi-vector index with deduplication."""
        assert self.metadata is not None

        # Oversample to account for multiple vectors per metabolite
        search_k = min(k * oversample_factor, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        # Deduplicate by HMDB ID, keeping best weighted score
        best_results: dict[str, SearchResult] = {}

        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:  # Invalid index
                continue

            idx_str = str(idx)
            if idx_str not in self.metadata:
                continue

            meta = self.metadata[idx_str]
            hmdb_id = meta["hmdb_id"]
            weight = 1.0  # Use equal weighting for all tiers (ignore stored 0.9 for synonyms)
            tier = meta["tier"]

            # Convert L2 distance to IP-equivalent score for PQ indices
            # For L2-normalized vectors: d_L2² = 2 - 2·IP, so IP = 1 - d_L2²/2
            # FAISS IndexHNSWPQ returns squared L2 distances
            if self._uses_l2_metric:
                score = 1.0 - (float(dist) / 2.0)
            else:
                score = float(dist)

            result = SearchResult(
                hmdb_id=hmdb_id,
                score=score,
                tier=tier,
                weight=weight,
            )

            # Keep best weighted score for each HMDB ID
            if (
                hmdb_id not in best_results
                or result.weighted_score > best_results[hmdb_id].weighted_score
            ):
                best_results[hmdb_id] = result

        # Sort by weighted score descending and return top k
        sorted_results = sorted(
            best_results.values(),
            key=lambda r: r.weighted_score,
            reverse=True,
        )
        return sorted_results[:k]

    def _search_single_vector(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> list[SearchResult]:
        """Search single-vector index."""
        distances, indices = self.index.search(query_embedding, k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:  # Invalid index
                continue

            # Get HMDB ID from mapping or metadata
            if self._id_mapping is not None:
                hmdb_id = self._id_mapping[idx]
            elif self.metadata is not None:
                hmdb_id = self.metadata[str(idx)]["hmdb_id"]
            else:
                hmdb_id = f"IDX_{idx}"  # Fallback

            results.append(
                SearchResult(
                    hmdb_id=hmdb_id,
                    score=float(dist),
                    tier="primary",
                    weight=1.0,
                )
            )

        return results

    def results_to_hmdb_ids(self, results: list[SearchResult]) -> list[str]:
        """Convert search results to list of HMDB IDs.

        Args:
            results: List of SearchResult objects.

        Returns:
            List of HMDB ID strings in the same order.
        """
        return [r.hmdb_id for r in results]
