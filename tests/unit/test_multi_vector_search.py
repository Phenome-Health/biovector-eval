"""Tests for multi-vector search support in evaluator.

TDD: Tests written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytest

# Import will fail until we add the functions - that's expected in TDD
from biovector_eval.metabolites.evaluator import (
    MultiVectorSearcher,
    SearchResult,
    load_index_with_metadata,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample normalized embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_multi_vector_metadata() -> dict[str, dict[str, Any]]:
    """Create sample multi-vector metadata.

    Structure: 10 metabolites, each with ~10 vectors (primary + synonyms + variations)
    """
    metadata = {}
    hmdb_ids = [f"HMDB{i:07d}" for i in range(10)]

    idx = 0
    for hmdb_id in hmdb_ids:
        # Primary (weight 1.0)
        metadata[str(idx)] = {
            "hmdb_id": hmdb_id,
            "tier": "primary",
            "weight": 1.0,
            "text": f"Primary name for {hmdb_id}",
        }
        idx += 1

        # Synonyms (weight 0.9)
        for syn_idx in range(5):
            metadata[str(idx)] = {
                "hmdb_id": hmdb_id,
                "tier": "synonym",
                "weight": 0.9,
                "text": f"Synonym {syn_idx} for {hmdb_id}",
            }
            idx += 1

        # Variations (weight 0.7)
        for var_idx in range(4):
            metadata[str(idx)] = {
                "hmdb_id": hmdb_id,
                "tier": "variation",
                "weight": 0.7,
                "text": f"Variation {var_idx} for {hmdb_id}",
            }
            idx += 1

    return metadata


@pytest.fixture
def temp_index_dir(tmp_path: Path, sample_embeddings: np.ndarray) -> Path:
    """Create temporary directory with index and metadata files."""
    index_dir = tmp_path / "indices"
    index_dir.mkdir()

    # Build and save flat index
    index = faiss.IndexFlatIP(384)
    index.add(sample_embeddings)
    faiss.write_index(index, str(index_dir / "test_multi-vector_flat.faiss"))

    return index_dir


@pytest.fixture
def temp_metadata_file(
    temp_index_dir: Path, sample_multi_vector_metadata: dict
) -> Path:
    """Create metadata file alongside index."""
    metadata_path = temp_index_dir / "test_multi-vector_flat.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(sample_multi_vector_metadata, f)
    return metadata_path


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Should create SearchResult with required fields."""
        result = SearchResult(
            hmdb_id="HMDB0000001",
            score=0.95,
            tier="primary",
            weight=1.0,
        )
        assert result.hmdb_id == "HMDB0000001"
        assert result.score == 0.95
        assert result.tier == "primary"
        assert result.weight == 1.0

    def test_search_result_weighted_score(self) -> None:
        """Should calculate weighted score correctly."""
        result = SearchResult(
            hmdb_id="HMDB0000001",
            score=0.8,
            tier="synonym",
            weight=0.9,
        )
        assert result.weighted_score == pytest.approx(0.72)  # 0.8 * 0.9

    def test_search_result_comparison(self) -> None:
        """SearchResults should be comparable by weighted_score."""
        result1 = SearchResult("HMDB1", score=0.9, tier="primary", weight=1.0)
        result2 = SearchResult("HMDB2", score=0.8, tier="synonym", weight=0.9)

        # result1.weighted_score = 0.9, result2.weighted_score = 0.72
        assert result1.weighted_score > result2.weighted_score


class TestLoadIndexWithMetadata:
    """Tests for loading index with metadata."""

    def test_load_index_with_metadata(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_multi_vector_metadata: dict,
    ) -> None:
        """Should load both index and metadata."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        index, metadata = load_index_with_metadata(index_path)

        assert index.ntotal == 100
        assert metadata == sample_multi_vector_metadata

    def test_load_index_without_metadata(
        self, temp_index_dir: Path, sample_embeddings: np.ndarray
    ) -> None:
        """Should return None metadata for single-vector index."""
        # Create index without metadata
        index_path = temp_index_dir / "test_single-primary_flat.faiss"
        index = faiss.IndexFlatIP(384)
        index.add(sample_embeddings)
        faiss.write_index(index, str(index_path))

        loaded_index, metadata = load_index_with_metadata(index_path)

        assert loaded_index.ntotal == 100
        assert metadata is None

    def test_load_index_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing index."""
        with pytest.raises(FileNotFoundError):
            load_index_with_metadata(tmp_path / "nonexistent.faiss")


class TestMultiVectorSearcher:
    """Tests for MultiVectorSearcher class."""

    def test_searcher_initialization(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
    ) -> None:
        """Should initialize searcher with index and metadata."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        assert searcher.index.ntotal == 100
        assert searcher.metadata is not None
        assert len(searcher.metadata) == 100

    def test_searcher_is_multi_vector(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
    ) -> None:
        """Should correctly identify multi-vector index."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        assert searcher.is_multi_vector is True

    def test_searcher_is_single_vector(
        self, temp_index_dir: Path, sample_embeddings: np.ndarray
    ) -> None:
        """Should correctly identify single-vector index."""
        index_path = temp_index_dir / "test_single-primary_flat.faiss"
        index = faiss.IndexFlatIP(384)
        index.add(sample_embeddings)
        faiss.write_index(index, str(index_path))

        searcher = MultiVectorSearcher(index_path)
        assert searcher.is_multi_vector is False


class TestMultiVectorSearch:
    """Tests for multi-vector search functionality."""

    def test_search_returns_search_results(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Search should return list of SearchResult objects."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        query = sample_embeddings[0:1]  # Use first embedding as query
        results = searcher.search(query, k=5)

        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_deduplicates_by_hmdb_id(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Search should deduplicate results by HMDB ID."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        query = sample_embeddings[0:1]
        results = searcher.search(query, k=10)

        # Check no duplicate HMDB IDs
        hmdb_ids = [r.hmdb_id for r in results]
        assert len(hmdb_ids) == len(set(hmdb_ids))

    def test_search_keeps_best_weighted_score(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Search should keep the best weighted score for each HMDB ID."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        query = sample_embeddings[0:1]
        results = searcher.search(query, k=5)

        # Results should be sorted by weighted_score descending
        weighted_scores = [r.weighted_score for r in results]
        assert weighted_scores == sorted(weighted_scores, reverse=True)

    def test_search_oversamples_for_dedup(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Multi-vector search should oversample to account for duplicates."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        query = sample_embeddings[0:1]

        # Request k=5, but with ~10 vectors per metabolite, need to oversample
        results = searcher.search(query, k=5)

        # Should get up to 5 unique HMDB IDs
        assert len(results) <= 5


class TestSingleVectorSearch:
    """Tests for single-vector search (no deduplication needed)."""

    def test_single_vector_search(
        self, temp_index_dir: Path, sample_embeddings: np.ndarray
    ) -> None:
        """Single-vector search should work without metadata."""
        # Create single-vector index (no metadata)
        index_path = temp_index_dir / "test_single-primary_flat.faiss"
        index = faiss.IndexFlatIP(384)
        index.add(sample_embeddings)
        faiss.write_index(index, str(index_path))

        # Create ID mapping (simulating single-vector case)
        id_mapping = [f"HMDB{i:07d}" for i in range(100)]

        searcher = MultiVectorSearcher(index_path, id_mapping=id_mapping)

        query = sample_embeddings[0:1]
        results = searcher.search(query, k=5)

        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)
        # Single-vector always has weight 1.0
        assert all(r.weight == 1.0 for r in results)


class TestSearchWithIdMapping:
    """Tests for search with explicit ID mapping."""

    def test_search_with_id_mapping(
        self, temp_index_dir: Path, sample_embeddings: np.ndarray
    ) -> None:
        """Should use ID mapping when provided."""
        index_path = temp_index_dir / "test_single-primary_flat.faiss"
        index = faiss.IndexFlatIP(384)
        index.add(sample_embeddings)
        faiss.write_index(index, str(index_path))

        id_mapping = [f"CUSTOM_ID_{i}" for i in range(100)]
        searcher = MultiVectorSearcher(index_path, id_mapping=id_mapping)

        query = sample_embeddings[0:1]
        results = searcher.search(query, k=5)

        assert all(r.hmdb_id.startswith("CUSTOM_ID_") for r in results)


class TestSearchResultsToHmdbIds:
    """Tests for converting search results to HMDB ID list."""

    def test_results_to_hmdb_ids(
        self,
        temp_index_dir: Path,
        temp_metadata_file: Path,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Should convert results to list of HMDB IDs."""
        index_path = temp_index_dir / "test_multi-vector_flat.faiss"
        searcher = MultiVectorSearcher(index_path)

        query = sample_embeddings[0:1]
        results = searcher.search(query, k=5)
        hmdb_ids = searcher.results_to_hmdb_ids(results)

        assert isinstance(hmdb_ids, list)
        assert all(isinstance(id_, str) for id_ in hmdb_ids)
        assert len(hmdb_ids) == len(results)


class TestWeightedScoring:
    """Tests for weighted scoring calculation."""

    def test_primary_tier_highest_weight(self) -> None:
        """Primary tier should have highest effective score."""
        # Same raw score, different tiers
        primary = SearchResult("HMDB1", score=0.8, tier="primary", weight=1.0)
        synonym = SearchResult("HMDB2", score=0.8, tier="synonym", weight=0.9)
        variation = SearchResult("HMDB3", score=0.8, tier="variation", weight=0.7)

        assert primary.weighted_score > synonym.weighted_score
        assert synonym.weighted_score > variation.weighted_score

    def test_high_synonym_beats_low_primary(self) -> None:
        """High-scoring synonym can beat low-scoring primary."""
        # Synonym with high raw score
        synonym = SearchResult("HMDB1", score=0.95, tier="synonym", weight=0.9)
        # Primary with low raw score
        primary = SearchResult("HMDB2", score=0.7, tier="primary", weight=1.0)

        # synonym: 0.95 * 0.9 = 0.855
        # primary: 0.7 * 1.0 = 0.7
        assert synonym.weighted_score > primary.weighted_score
