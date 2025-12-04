"""Tests for embedding strategy module.

TDD: Write tests first, then implement the strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from biovector_eval.core.embedding_strategy import (
    EmbeddingStrategy,
    MultiVectorStrategy,
    SinglePooledStrategy,
    SinglePrimaryStrategy,
    StrategyType,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_strategy_type_values(self) -> None:
        """StrategyType should have exactly 3 values."""
        assert StrategyType.SINGLE_PRIMARY.value == "single-primary"
        assert StrategyType.SINGLE_POOLED.value == "single-pooled"
        assert StrategyType.MULTI_VECTOR.value == "multi-vector"
        assert len(StrategyType) == 3


class MockModel:
    """Mock embedding model for testing."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._call_count = 0

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Return mock embeddings.

        Matches SentenceTransformer behavior:
        - Single string input -> 1D array (dim,)
        - List input -> 2D array (n, dim)
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        self._call_count += 1
        # Generate deterministic embeddings based on text hash
        embeddings = np.array(
            [
                np.array([hash(t) % 100 / 100.0] * self.dim, dtype=np.float32)
                for t in texts
            ]
        )

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        # Return 1D for single string input (matches SentenceTransformer)
        if is_single:
            return embeddings[0]

        return embeddings


@pytest.fixture
def sample_metabolite() -> dict:
    """Sample metabolite data for testing."""
    return {
        "hmdb_id": "HMDB0000122",
        "name": "D-Glucose",
        "synonyms": [
            "Grape sugar",
            "Dextrose",
            "Blood sugar",
            "Corn sugar",
            "D-Glc",
            "alpha-D-glucose",
            "beta-D-glucose",
            "D-(+)-Glucose",
            "Glucopyranose",
            "Glucose",
        ],
    }


@pytest.fixture
def sample_metabolite_no_synonyms() -> dict:
    """Metabolite with no synonyms."""
    return {
        "hmdb_id": "HMDB0001234",
        "name": "Unknown Compound",
        "synonyms": [],
    }


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock embedding model."""
    return MockModel(dim=384)


class TestSinglePrimaryStrategy:
    """Tests for SinglePrimaryStrategy."""

    def test_single_primary_one_vector(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePrimaryStrategy should return exactly 1 vector per metabolite."""
        strategy = SinglePrimaryStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        assert len(result) == 1
        assert result[0]["hmdb_id"] == "HMDB0000122"

    def test_single_primary_uses_name(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePrimaryStrategy should use only the primary name."""
        strategy = SinglePrimaryStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        assert result[0]["text"] == "D-Glucose"
        assert result[0]["tier"] == "primary"
        assert result[0]["weight"] == 1.0

    def test_single_primary_embedding_shape(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """Embedding should have correct dimension."""
        strategy = SinglePrimaryStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        assert result[0]["embedding"].shape == (384,)

    def test_single_primary_vectors_per_metabolite(self) -> None:
        """vectors_per_metabolite should return 1."""
        strategy = SinglePrimaryStrategy()
        assert strategy.vectors_per_metabolite() == 1


class TestSinglePooledStrategy:
    """Tests for SinglePooledStrategy."""

    def test_single_pooled_one_vector(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePooledStrategy should return exactly 1 vector per metabolite."""
        strategy = SinglePooledStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        assert len(result) == 1

    def test_single_pooled_averages_vectors(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePooledStrategy should average name + synonym embeddings."""
        strategy = SinglePooledStrategy()

        # Get individual embeddings to verify averaging
        texts = [sample_metabolite["name"]] + sample_metabolite["synonyms"][:15]
        individual_embeddings = mock_model.encode(texts, normalize_embeddings=False)
        expected_mean = np.mean(individual_embeddings, axis=0)
        expected_normalized = expected_mean / np.linalg.norm(expected_mean)

        result = strategy.prepare_vectors(sample_metabolite, mock_model)
        actual_embedding = result[0]["embedding"]

        # Should be close to normalized mean (allowing for float precision)
        np.testing.assert_array_almost_equal(
            actual_embedding, expected_normalized, decimal=5
        )

    def test_single_pooled_normalizes(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePooledStrategy output should be L2 normalized."""
        strategy = SinglePooledStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        embedding = result[0]["embedding"]
        norm = np.linalg.norm(embedding)

        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_single_pooled_handles_no_synonyms(
        self,
        sample_metabolite_no_synonyms: dict,
        mock_model: MockModel,
    ) -> None:
        """SinglePooledStrategy should work when synonyms list is empty."""
        strategy = SinglePooledStrategy()
        result = strategy.prepare_vectors(sample_metabolite_no_synonyms, mock_model)

        assert len(result) == 1
        # Should just use the primary name
        assert result[0]["text"] == "Unknown Compound"

    def test_single_pooled_limits_synonyms(
        self,
        mock_model: MockModel,
    ) -> None:
        """SinglePooledStrategy should limit to 15 synonyms."""
        metabolite = {
            "hmdb_id": "HMDB0000001",
            "name": "Test",
            "synonyms": [f"Synonym{i}" for i in range(50)],
        }
        strategy = SinglePooledStrategy(max_synonyms=15)
        result = strategy.prepare_vectors(metabolite, mock_model)

        # Should use 1 (name) + 15 (synonyms) = 16 texts max
        assert len(result) == 1
        # The text field should indicate pooling
        assert result[0]["tier"] == "pooled"

    def test_single_pooled_vectors_per_metabolite(self) -> None:
        """vectors_per_metabolite should return 1."""
        strategy = SinglePooledStrategy()
        assert strategy.vectors_per_metabolite() == 1


class TestMultiVectorStrategy:
    """Tests for MultiVectorStrategy."""

    def test_multi_vector_count(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """MultiVectorStrategy should return multiple vectors per metabolite."""
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        # Should have at least: 1 primary + some synonyms + some variations
        assert len(result) > 1
        # But not excessively many (upper bound)
        assert len(result) <= 50

    def test_multi_vector_primary_weight(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """Primary name vector should have weight 1.0."""
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        primary_vectors = [v for v in result if v["tier"] == "primary"]
        assert len(primary_vectors) == 1
        assert primary_vectors[0]["weight"] == 1.0
        assert primary_vectors[0]["text"] == "D-Glucose"

    def test_multi_vector_synonym_weight(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """Synonym vectors should have weight 0.9."""
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        synonym_vectors = [v for v in result if v["tier"] == "synonym"]
        assert len(synonym_vectors) > 0
        for v in synonym_vectors:
            assert v["weight"] == 0.9

    def test_multi_vector_variation_weight(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """Variation vectors should have weight 0.7."""
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        variation_vectors = [v for v in result if v["tier"] == "variation"]
        for v in variation_vectors:
            assert v["weight"] == 0.7

    def test_multi_vector_generates_variations(
        self,
        mock_model: MockModel,
    ) -> None:
        """MultiVectorStrategy should generate text variations."""
        metabolite = {
            "hmdb_id": "HMDB0000001",
            "name": "alpha-ketoglutarate",
            "synonyms": ["2-Oxoglutarate", "alpha-KG"],
        }
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(metabolite, mock_model)

        texts = [v["text"] for v in result]

        # Should generate variations like:
        # - Greek letter conversion: α-ketoglutarate
        # - Hyphen removal: alphaketoglutarate
        assert any("α" in t or "alphaketoglutarate" in t.lower() for t in texts)

    def test_multi_vector_hmdb_id_on_all(
        self,
        sample_metabolite: dict,
        mock_model: MockModel,
    ) -> None:
        """All vectors should have the same hmdb_id."""
        strategy = MultiVectorStrategy()
        result = strategy.prepare_vectors(sample_metabolite, mock_model)

        for v in result:
            assert v["hmdb_id"] == "HMDB0000122"

    def test_multi_vector_vectors_per_metabolite(self) -> None:
        """vectors_per_metabolite should return 'variable'."""
        strategy = MultiVectorStrategy()
        assert strategy.vectors_per_metabolite() == "variable"


class TestEmbeddingStrategyInterface:
    """Tests for the abstract EmbeddingStrategy interface."""

    def test_all_strategies_inherit_from_base(self) -> None:
        """All strategy classes should inherit from EmbeddingStrategy."""
        assert issubclass(SinglePrimaryStrategy, EmbeddingStrategy)
        assert issubclass(SinglePooledStrategy, EmbeddingStrategy)
        assert issubclass(MultiVectorStrategy, EmbeddingStrategy)

    def test_strategy_name_attribute(self) -> None:
        """All strategies should have a name attribute."""
        assert SinglePrimaryStrategy().name == "single-primary"
        assert SinglePooledStrategy().name == "single-pooled"
        assert MultiVectorStrategy().name == "multi-vector"
