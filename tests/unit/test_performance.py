"""Tests for performance profiling module."""

import pytest

from biovector_eval.core.performance import (
    LatencyMetrics,
    MemoryMetrics,
    measure_latency,
    measure_memory,
)


class TestLatencyMetrics:
    """Tests for LatencyMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Metrics convert to dictionary correctly."""
        metrics = LatencyMetrics(
            p50_ms=1.0,
            p95_ms=2.0,
            p99_ms=3.0,
            mean_ms=1.5,
            num_queries=100,
        )
        d = metrics.to_dict()
        assert d["p50_ms"] == 1.0
        assert d["p95_ms"] == 2.0
        assert d["p99_ms"] == 3.0
        assert d["mean_ms"] == 1.5
        assert d["num_queries"] == 100


class TestMemoryMetrics:
    """Tests for MemoryMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Metrics convert to dictionary correctly."""
        metrics = MemoryMetrics(
            index_size_mb=100.0,
            peak_rss_mb=500.0,
            vectors_count=10000,
            bytes_per_vector=1024.0,
        )
        d = metrics.to_dict()
        assert d["index_size_mb"] == 100.0
        assert d["peak_rss_mb"] == 500.0
        assert d["vectors_count"] == 10000
        assert d["bytes_per_vector"] == 1024.0


class TestMeasureLatency:
    """Tests for measure_latency function."""

    def test_basic_measurement(self) -> None:
        """Basic latency measurement works."""
        call_count = 0

        def mock_search(query: str, k: int) -> list[str]:
            nonlocal call_count
            call_count += 1
            return [f"result_{i}" for i in range(k)]

        queries = ["query1", "query2", "query3", "query4", "query5"]
        metrics = measure_latency(mock_search, queries, k=10, warmup=2)

        # Should have made warmup + len(queries) calls
        assert call_count == 2 + 5

        assert metrics.num_queries == 5
        assert metrics.p50_ms >= 0
        assert metrics.p95_ms >= metrics.p50_ms
        assert metrics.p99_ms >= metrics.p95_ms
        assert metrics.mean_ms >= 0


class TestMeasureMemory:
    """Tests for measure_memory function."""

    def test_float32_unquantized(self) -> None:
        """Unquantized (float32) memory calculation."""
        metrics = measure_memory(num_vectors=10000, dimension=384, quantization="none")

        # float32: 10000 * 384 * 4 = 15,360,000 bytes
        # HNSW graph: 10000 * 32 * 8 = 2,560,000 bytes
        # Total: 17,920,000 bytes = ~17.09 MB
        expected_size = (10000 * 384 * 4 + 10000 * 32 * 8) / 1024 / 1024
        assert metrics.index_size_mb == pytest.approx(expected_size, rel=0.01)
        assert metrics.vectors_count == 10000

    def test_sq8_quantized(self) -> None:
        """SQ8 quantized memory calculation."""
        metrics = measure_memory(num_vectors=10000, dimension=384, quantization="sq8")

        # SQ8: 10000 * 384 * 1 = 3,840,000 bytes
        # HNSW graph: 10000 * 32 * 8 = 2,560,000 bytes
        # Total: 6,400,000 bytes = ~6.1 MB
        expected_size = (10000 * 384 * 1 + 10000 * 32 * 8) / 1024 / 1024
        assert metrics.index_size_mb == pytest.approx(expected_size, rel=0.01)

    def test_sq4_quantized(self) -> None:
        """SQ4 quantized memory calculation."""
        metrics = measure_memory(num_vectors=10000, dimension=384, quantization="sq4")

        # SQ4: 10000 * 384 * 0.5 = 1,920,000 bytes
        # HNSW graph: 10000 * 32 * 8 = 2,560,000 bytes
        # Total: 4,480,000 bytes = ~4.27 MB
        expected_size = (10000 * 384 * 0.5 + 10000 * 32 * 8) / 1024 / 1024
        assert metrics.index_size_mb == pytest.approx(expected_size, rel=0.01)

    def test_custom_hnsw_m(self) -> None:
        """Custom HNSW M parameter affects graph overhead."""
        metrics_m16 = measure_memory(num_vectors=10000, dimension=384, hnsw_m=16)
        metrics_m64 = measure_memory(num_vectors=10000, dimension=384, hnsw_m=64)

        # Higher M should result in larger index
        assert metrics_m64.index_size_mb > metrics_m16.index_size_mb
