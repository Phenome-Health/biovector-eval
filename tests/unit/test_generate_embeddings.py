"""Tests for generate_embeddings.py strategy support.

TDD: Tests written before implementation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from biovector_eval.core.embedding_strategy import (
    MultiVectorStrategy,
    SinglePooledStrategy,
    SinglePrimaryStrategy,
    StrategyType,
    get_strategy,
)

# Import will fail until we add the functions - that's expected in TDD
from scripts.generate_embeddings import (
    MODELS,
    generate_embeddings_with_strategy,
    get_output_paths,
    parse_strategy_argument,
    save_multi_vector_output,
)


class MockModel:
    """Mock embedding model for testing."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 512,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Return mock embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = np.random.randn(len(texts), self.dim).astype(np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings


@pytest.fixture
def sample_metabolites() -> list[dict]:
    """Sample metabolite data for testing."""
    return [
        {
            "hmdb_id": "HMDB0000122",
            "name": "D-Glucose",
            "synonyms": ["Grape sugar", "Dextrose", "Blood sugar"],
        },
        {
            "hmdb_id": "HMDB0000161",
            "name": "L-Alanine",
            "synonyms": ["Alanine", "2-Aminopropanoic acid"],
        },
        {
            "hmdb_id": "HMDB0000067",
            "name": "Cholesterol",
            "synonyms": [],
        },
    ]


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock embedding model."""
    return MockModel(dim=384)


@pytest.fixture
def temp_output_dir() -> Path:
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestParseStrategyArgument:
    """Tests for strategy argument parsing."""

    def test_strategy_argument_parsing_single_primary(self) -> None:
        """Should accept 'single-primary' strategy."""
        strategy = parse_strategy_argument("single-primary")
        assert strategy == StrategyType.SINGLE_PRIMARY

    def test_strategy_argument_parsing_single_pooled(self) -> None:
        """Should accept 'single-pooled' strategy."""
        strategy = parse_strategy_argument("single-pooled")
        assert strategy == StrategyType.SINGLE_POOLED

    def test_strategy_argument_parsing_multi_vector(self) -> None:
        """Should accept 'multi-vector' strategy."""
        strategy = parse_strategy_argument("multi-vector")
        assert strategy == StrategyType.MULTI_VECTOR

    def test_strategy_argument_invalid(self) -> None:
        """Should raise ValueError for invalid strategy."""
        with pytest.raises(ValueError):
            parse_strategy_argument("invalid-strategy")


class TestLoadStrategyByName:
    """Tests for loading strategy class by name."""

    def test_load_strategy_by_name_single_primary(self) -> None:
        """Should load SinglePrimaryStrategy."""
        strategy = get_strategy("single-primary")
        assert isinstance(strategy, SinglePrimaryStrategy)
        assert strategy.name == "single-primary"

    def test_load_strategy_by_name_single_pooled(self) -> None:
        """Should load SinglePooledStrategy."""
        strategy = get_strategy("single-pooled")
        assert isinstance(strategy, SinglePooledStrategy)
        assert strategy.name == "single-pooled"

    def test_load_strategy_by_name_multi_vector(self) -> None:
        """Should load MultiVectorStrategy."""
        strategy = get_strategy("multi-vector")
        assert isinstance(strategy, MultiVectorStrategy)
        assert strategy.name == "multi-vector"


class TestSinglePrimaryOutputShape:
    """Tests for single-primary strategy output."""

    def test_single_primary_output_shape(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Single-primary should produce (N, dim) embeddings."""
        strategy = SinglePrimaryStrategy()
        embeddings, metadata = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        # Should have one embedding per metabolite
        assert embeddings.shape == (3, 384)
        # Metadata should be None for single-vector strategies
        assert metadata is None

    def test_single_primary_embeddings_normalized(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Single-primary embeddings should be L2 normalized."""
        strategy = SinglePrimaryStrategy()
        embeddings, _ = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


class TestSinglePooledOutputShape:
    """Tests for single-pooled strategy output."""

    def test_single_pooled_output_shape(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Single-pooled should produce (N, dim) embeddings."""
        strategy = SinglePooledStrategy()
        embeddings, metadata = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        # Should have one embedding per metabolite
        assert embeddings.shape == (3, 384)
        # Metadata should be None for single-vector strategies
        assert metadata is None

    def test_single_pooled_embeddings_normalized(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Single-pooled embeddings should be L2 normalized."""
        strategy = SinglePooledStrategy()
        embeddings, _ = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


class TestMultiVectorOutputStructure:
    """Tests for multi-vector strategy output structure."""

    def test_multi_vector_output_structure(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Multi-vector should produce embeddings + metadata."""
        strategy = MultiVectorStrategy()
        embeddings, metadata = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        # Should have more vectors than metabolites
        assert embeddings.shape[0] > len(sample_metabolites)
        assert embeddings.shape[1] == 384

        # Metadata should be present
        assert metadata is not None
        assert len(metadata) == embeddings.shape[0]

    def test_multi_vector_embeddings_normalized(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Multi-vector embeddings should be L2 normalized."""
        strategy = MultiVectorStrategy()
        embeddings, _ = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(
            norms, np.ones(embeddings.shape[0]), decimal=5
        )


class TestMultiVectorMetadataSchema:
    """Tests for multi-vector metadata schema."""

    def test_multi_vector_metadata_schema(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Multi-vector metadata should have correct schema."""
        strategy = MultiVectorStrategy()
        _, metadata = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        assert metadata is not None

        # Each entry should have required fields
        for _idx, entry in metadata.items():
            assert "hmdb_id" in entry
            assert "tier" in entry
            assert "weight" in entry
            assert "text" in entry

            # Validate types
            assert isinstance(entry["hmdb_id"], str)
            assert entry["tier"] in ["primary", "synonym", "variation", "pooled"]
            assert isinstance(entry["weight"], float)
            assert 0.0 < entry["weight"] <= 1.0
            assert isinstance(entry["text"], str)

    def test_multi_vector_metadata_hmdb_ids(
        self,
        sample_metabolites: list[dict],
        mock_model: MockModel,
    ) -> None:
        """Multi-vector metadata should contain all original HMDB IDs."""
        strategy = MultiVectorStrategy()
        _, metadata = generate_embeddings_with_strategy(
            metabolites=sample_metabolites,
            model=mock_model,
            strategy=strategy,
        )

        assert metadata is not None

        hmdb_ids_in_metadata = {entry["hmdb_id"] for entry in metadata.values()}
        expected_hmdb_ids = {m["hmdb_id"] for m in sample_metabolites}

        assert expected_hmdb_ids == hmdb_ids_in_metadata


class TestOutputFileNaming:
    """Tests for output file naming conventions."""

    def test_output_file_naming_single_primary(self, temp_output_dir: Path) -> None:
        """Single-primary should use {model}_{strategy}.npy naming."""
        paths = get_output_paths(
            output_dir=temp_output_dir,
            model_slug="bge-m3",
            strategy_name="single-primary",
        )

        assert (
            paths["embeddings"]
            == temp_output_dir / "embeddings" / "bge-m3_single-primary.npy"
        )
        assert "metadata" not in paths or paths["metadata"] is None

    def test_output_file_naming_single_pooled(self, temp_output_dir: Path) -> None:
        """Single-pooled should use {model}_{strategy}.npy naming."""
        paths = get_output_paths(
            output_dir=temp_output_dir,
            model_slug="sapbert",
            strategy_name="single-pooled",
        )

        assert (
            paths["embeddings"]
            == temp_output_dir / "embeddings" / "sapbert_single-pooled.npy"
        )

    def test_output_file_naming_multi_vector(self, temp_output_dir: Path) -> None:
        """Multi-vector should use directory structure with embeddings.npy + metadata.json."""
        paths = get_output_paths(
            output_dir=temp_output_dir,
            model_slug="biolord",
            strategy_name="multi-vector",
        )

        expected_dir = temp_output_dir / "embeddings" / "biolord_multi-vector"
        assert paths["embeddings"] == expected_dir / "embeddings.npy"
        assert paths["metadata"] == expected_dir / "metadata.json"


class TestSaveMultiVectorOutput:
    """Tests for saving multi-vector output."""

    def test_save_multi_vector_output(self, temp_output_dir: Path) -> None:
        """Should save embeddings.npy and metadata.json correctly."""
        embeddings = np.random.randn(10, 384).astype(np.float32)
        metadata = {
            str(i): {
                "hmdb_id": f"HMDB{i:07d}",
                "tier": "primary",
                "weight": 1.0,
                "text": f"Metabolite {i}",
            }
            for i in range(10)
        }

        output_dir = temp_output_dir / "embeddings" / "test_multi-vector"
        save_multi_vector_output(
            embeddings=embeddings,
            metadata=metadata,
            output_dir=output_dir,
        )

        # Verify files exist
        assert (output_dir / "embeddings.npy").exists()
        assert (output_dir / "metadata.json").exists()

        # Verify content
        loaded_embeddings = np.load(output_dir / "embeddings.npy")
        np.testing.assert_array_equal(loaded_embeddings, embeddings)

        with open(output_dir / "metadata.json") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata


class TestIntegrationWithExistingFunctions:
    """Integration tests with existing generate_embeddings.py functions."""

    def test_models_dict_unchanged(self) -> None:
        """MODELS dict should still contain the expected models."""
        assert "bge-m3" in MODELS
        assert "sapbert" in MODELS
        assert "biolord" in MODELS
        assert "gte-qwen2" in MODELS

        # Verify dimensions
        assert MODELS["bge-m3"].dimension == 1024
        assert MODELS["sapbert"].dimension == 768
        assert MODELS["biolord"].dimension == 768
        assert MODELS["gte-qwen2"].dimension == 1536
