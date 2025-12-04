"""Evaluation metrics for biological entity vector search."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyMetrics:
    """Container for latency measurements."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
        }


def measure_latency(
    search_fn: Callable[[str, int], list],
    queries: list[str],
    k: int = 10,
    warmup: int = 10,
) -> LatencyMetrics:
    """Measure search latency percentiles.

    Args:
        search_fn: Function that takes query string and k, returns results.
        queries: List of query strings to benchmark.
        k: Number of results to retrieve.
        warmup: Number of warmup queries to run before timing.

    Returns:
        LatencyMetrics with p50, p95, p99, and mean latencies in milliseconds.
    """
    # Warmup runs
    for query in queries[:warmup]:
        search_fn(query, k)

    # Timed runs
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        search_fn(query, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    return LatencyMetrics(
        p50_ms=float(np.percentile(latencies_arr, 50)),
        p95_ms=float(np.percentile(latencies_arr, 95)),
        p99_ms=float(np.percentile(latencies_arr, 99)),
        mean_ms=float(np.mean(latencies_arr)),
    )


@dataclass
class SearchMetrics:
    """Container for search evaluation metrics."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_queries: int

    def to_dict(self) -> dict[str, float]:
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "num_queries": self.num_queries,
        }


def recall_at_k(
    ground_truth: set[str],
    predictions: Sequence[str],
    k: int,
) -> float:
    """Calculate recall at k.

    Args:
        ground_truth: Set of correct entity IDs.
        predictions: Ranked list of predicted entity IDs.
        k: Number of top predictions to consider.

    Returns:
        Proportion of ground truth found in top-k predictions.
    """
    if not ground_truth:
        return 0.0
    top_k = set(predictions[:k])
    return len(ground_truth & top_k) / len(ground_truth)


def mean_reciprocal_rank(
    ground_truth: set[str],
    predictions: Sequence[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        ground_truth: Set of correct entity IDs.
        predictions: Ranked list of predicted entity IDs.

    Returns:
        Reciprocal of the rank of the first correct prediction.
    """
    for i, pred in enumerate(predictions, start=1):
        if pred in ground_truth:
            return 1.0 / i
    return 0.0


def evaluate_search_results(
    queries: list[dict],
    search_fn: Callable[[str, int], list[str]],
    k: int = 10,
) -> SearchMetrics:
    """Evaluate search function against ground truth queries.

    Args:
        queries: List of query dicts with 'query' and 'expected' keys.
        search_fn: Function that takes query string and k, returns list of IDs.
        k: Maximum k for evaluation.

    Returns:
        SearchMetrics with aggregated results.
    """
    recall_1: list[float] = []
    recall_5: list[float] = []
    recall_10: list[float] = []
    mrr_scores: list[float] = []

    for q in queries:
        query_text = q["query"]
        expected = set(q["expected"])
        predictions = search_fn(query_text, k)

        recall_1.append(recall_at_k(expected, predictions, 1))
        recall_5.append(recall_at_k(expected, predictions, 5))
        recall_10.append(recall_at_k(expected, predictions, 10))
        mrr_scores.append(mean_reciprocal_rank(expected, predictions))

    return SearchMetrics(
        recall_at_1=float(np.mean(recall_1)),
        recall_at_5=float(np.mean(recall_5)),
        recall_at_10=float(np.mean(recall_10)),
        mrr=float(np.mean(mrr_scores)),
        num_queries=len(queries),
    )
