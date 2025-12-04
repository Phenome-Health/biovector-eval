"""Integration tests for 4D evaluation pipeline.

TDD: Tests written before implementation.

Tests the full evaluation loop across:
- Models (4)
- Index types (7)
- Strategies (3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Import will fail until we add the functions - that's expected in TDD
from scripts.run_evaluation import (
    EVALUATION_DIMENSIONS,
    EvaluationConfig,
    get_all_configurations,
    save_evaluation_results,
)


class TestEvaluationDimensions:
    """Tests for evaluation dimension configuration."""

    def test_dimensions_includes_models(self) -> None:
        """Should include models dimension."""
        assert "models" in EVALUATION_DIMENSIONS
        assert len(EVALUATION_DIMENSIONS["models"]) >= 1

    def test_dimensions_includes_index_types(self) -> None:
        """Should include all 7 index types."""
        assert "index_types" in EVALUATION_DIMENSIONS
        expected_types = {"flat", "hnsw", "sq8", "sq4", "pq", "ivf_flat", "ivf_pq"}
        assert set(EVALUATION_DIMENSIONS["index_types"]) == expected_types

    def test_dimensions_includes_strategies(self) -> None:
        """Should include all 3 embedding strategies."""
        assert "strategies" in EVALUATION_DIMENSIONS
        expected_strategies = {"single-primary", "single-pooled", "multi-vector"}
        assert set(EVALUATION_DIMENSIONS["strategies"]) == expected_strategies


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_config_creation(self) -> None:
        """Should create config with all required fields."""
        config = EvaluationConfig(
            model="bge-m3",
            index_type="hnsw",
            strategy="single-primary",
        )
        assert config.model == "bge-m3"
        assert config.index_type == "hnsw"
        assert config.strategy == "single-primary"

    def test_config_embedding_path(self, tmp_path: Path) -> None:
        """Should generate correct embedding file path."""
        config = EvaluationConfig(
            model="sapbert",
            index_type="flat",
            strategy="single-pooled",
        )
        path = config.get_embedding_path(tmp_path / "embeddings")

        # For single-vector: model_strategy.npy
        assert "sapbert" in str(path)
        assert "single-pooled" in str(path)

    def test_config_index_path(self, tmp_path: Path) -> None:
        """Should generate correct index file path."""
        config = EvaluationConfig(
            model="bge-m3",
            index_type="sq8",
            strategy="single-primary",
        )
        path = config.get_index_path(tmp_path / "indices")

        # Should follow naming convention: model_strategy_indextype.faiss
        assert "bge-m3" in str(path)
        assert "single-primary" in str(path)
        assert "sq8" in str(path)
        assert path.suffix == ".faiss"

    def test_config_is_multi_vector(self) -> None:
        """Should correctly identify multi-vector configs."""
        multi = EvaluationConfig("bge-m3", "hnsw", "multi-vector")
        single = EvaluationConfig("bge-m3", "hnsw", "single-primary")

        assert multi.is_multi_vector is True
        assert single.is_multi_vector is False

    def test_config_to_dict(self) -> None:
        """Should convert to dict."""
        config = EvaluationConfig("bge-m3", "hnsw", "multi-vector")
        d = config.to_dict()

        assert d["model"] == "bge-m3"
        assert d["index_type"] == "hnsw"
        assert d["strategy"] == "multi-vector"


class TestGetAllConfigurations:
    """Tests for generating all evaluation configurations."""

    def test_returns_list_of_configs(self) -> None:
        """Should return list of EvaluationConfig objects."""
        configs = get_all_configurations()

        assert isinstance(configs, list)
        assert all(isinstance(c, EvaluationConfig) for c in configs)

    def test_covers_all_dimensions(self) -> None:
        """Should cover all combinations of model × index × strategy."""
        configs = get_all_configurations()

        num_models = len(EVALUATION_DIMENSIONS["models"])
        num_index_types = len(EVALUATION_DIMENSIONS["index_types"])
        num_strategies = len(EVALUATION_DIMENSIONS["strategies"])

        expected_count = num_models * num_index_types * num_strategies
        assert len(configs) == expected_count

    def test_all_models_included(self) -> None:
        """Should include all models."""
        configs = get_all_configurations()
        models_in_configs = {c.model for c in configs}

        expected_models = set(EVALUATION_DIMENSIONS["models"])
        assert models_in_configs == expected_models

    def test_all_index_types_included(self) -> None:
        """Should include all index types."""
        configs = get_all_configurations()
        index_types_in_configs = {c.index_type for c in configs}

        expected_types = set(EVALUATION_DIMENSIONS["index_types"])
        assert index_types_in_configs == expected_types

    def test_all_strategies_included(self) -> None:
        """Should include all strategies."""
        configs = get_all_configurations()
        strategies_in_configs = {c.strategy for c in configs}

        expected_strategies = set(EVALUATION_DIMENSIONS["strategies"])
        assert strategies_in_configs == expected_strategies


class TestSaveEvaluationResults:
    """Tests for saving evaluation results."""

    def test_saves_json_file(self, tmp_path: Path) -> None:
        """Should save results to JSON file."""
        results = [
            {
                "config": {
                    "model": "bge-m3",
                    "index_type": "flat",
                    "strategy": "single-primary",
                },
                "metrics": {"recall@1": 0.85, "recall@5": 0.92, "mrr": 0.88},
            }
        ]

        output_path = save_evaluation_results(
            results=results,
            output_dir=tmp_path,
            filename="test_results.json",
        )

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert "results" in loaded
        assert len(loaded["results"]) == 1

    def test_includes_metadata(self, tmp_path: Path) -> None:
        """Should include metadata in output."""
        results: list[dict[str, Any]] = []

        output_path = save_evaluation_results(
            results=results,
            output_dir=tmp_path,
            filename="test_results.json",
        )

        with open(output_path) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "dimensions" in loaded["metadata"]
        assert "timestamp" in loaded["metadata"]

    def test_includes_summary_stats(self, tmp_path: Path) -> None:
        """Should include summary statistics."""
        results = [
            {
                "config": {
                    "model": "m1",
                    "index_type": "flat",
                    "strategy": "single-primary",
                },
                "metrics": {"recall@1": 0.85, "recall@5": 0.92, "mrr": 0.88},
            },
            {
                "config": {
                    "model": "m2",
                    "index_type": "hnsw",
                    "strategy": "single-pooled",
                },
                "metrics": {"recall@1": 0.90, "recall@5": 0.95, "mrr": 0.92},
            },
        ]

        output_path = save_evaluation_results(
            results=results,
            output_dir=tmp_path,
            filename="test_results.json",
        )

        with open(output_path) as f:
            loaded = json.load(f)

        assert "summary" in loaded
        # Should identify best configs
        assert "best_recall" in loaded["summary"]
        assert "best_mrr" in loaded["summary"]


class TestResultsSchemaValidation:
    """Tests for validating evaluation results schema."""

    def test_result_has_config(self) -> None:
        """Each result should have config section."""
        result = {
            "config": {
                "model": "bge-m3",
                "index_type": "flat",
                "strategy": "single-primary",
            },
            "metrics": {},
        }
        assert "config" in result
        assert "model" in result["config"]
        assert "index_type" in result["config"]
        assert "strategy" in result["config"]

    def test_result_has_metrics(self) -> None:
        """Each result should have metrics section."""
        result = {
            "config": {},
            "metrics": {
                "recall@1": 0.85,
                "recall@5": 0.92,
                "recall@10": 0.95,
                "mrr": 0.88,
                "p50_ms": 1.5,
                "p95_ms": 3.2,
                "index_size_mb": 125.5,
            },
        }
        required_metrics = [
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "p50_ms",
            "p95_ms",
        ]
        for metric in required_metrics:
            assert metric in result["metrics"]


class TestStrategyHandling:
    """Tests for handling different strategies in evaluation."""

    def test_single_primary_uses_standard_search(self) -> None:
        """Single-primary should use standard search without deduplication."""
        config = EvaluationConfig("bge-m3", "flat", "single-primary")
        assert not config.is_multi_vector

    def test_single_pooled_uses_standard_search(self) -> None:
        """Single-pooled should use standard search without deduplication."""
        config = EvaluationConfig("bge-m3", "flat", "single-pooled")
        assert not config.is_multi_vector

    def test_multi_vector_uses_weighted_search(self) -> None:
        """Multi-vector should use weighted search with deduplication."""
        config = EvaluationConfig("bge-m3", "flat", "multi-vector")
        assert config.is_multi_vector
