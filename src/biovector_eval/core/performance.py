"""Performance profiling for vector search operations."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import psutil


@dataclass
class LatencyMetrics:
    """Latency distribution metrics."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    num_queries: int

    def to_dict(self) -> dict[str, float]:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "num_queries": self.num_queries,
        }


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    index_size_mb: float
    peak_rss_mb: float
    vectors_count: int
    bytes_per_vector: float

    def to_dict(self) -> dict[str, float]:
        return {
            "index_size_mb": self.index_size_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "vectors_count": self.vectors_count,
            "bytes_per_vector": self.bytes_per_vector,
        }


def measure_latency(
    search_fn: Callable[[str, int], list],
    queries: Sequence[str],
    k: int = 10,
    warmup: int = 10,
) -> LatencyMetrics:
    """Measure search latency distribution.

    Args:
        search_fn: Search function taking (query, k) and returning results.
        queries: List of query strings.
        k: Number of results to retrieve.
        warmup: Number of warmup queries before measurement.

    Returns:
        LatencyMetrics with percentile distribution.
    """
    # Warmup
    for q in queries[:warmup]:
        _ = search_fn(q, k)

    # Measure
    latencies: list[float] = []
    for q in queries:
        start = time.perf_counter()
        _ = search_fn(q, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return LatencyMetrics(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        num_queries=len(queries),
    )


def measure_memory(
    num_vectors: int,
    dimension: int,
    quantization: str = "none",
    hnsw_m: int = 32,
) -> MemoryMetrics:
    """Estimate index memory usage.

    Args:
        num_vectors: Number of vectors in the index.
        dimension: Vector dimension.
        quantization: Quantization type ('none', 'sq8', 'sq4', 'pq').
        hnsw_m: HNSW M parameter for graph overhead calculation.

    Returns:
        MemoryMetrics with size information.
    """
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / 1024 / 1024

    # Calculate based on quantization
    if quantization == "none":
        bytes_per_dim = 4  # float32
    elif quantization == "sq8":
        bytes_per_dim = 1
    elif quantization == "sq4":
        bytes_per_dim = 0.5
    elif quantization == "pq":
        bytes_per_dim = 1  # Approximate for PQ
    else:
        bytes_per_dim = 4

    vector_bytes = num_vectors * dimension * bytes_per_dim
    # Add HNSW graph overhead (~M * 8 bytes per vector)
    graph_bytes = num_vectors * hnsw_m * 8
    total_bytes = vector_bytes + graph_bytes
    index_size_mb = total_bytes / 1024 / 1024

    return MemoryMetrics(
        index_size_mb=index_size_mb,
        peak_rss_mb=peak_rss_mb,
        vectors_count=num_vectors,
        bytes_per_vector=total_bytes / num_vectors,
    )
