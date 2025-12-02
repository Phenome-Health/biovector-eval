#!/usr/bin/env python
"""Run full evaluation of all model/index configurations.

Compares 15 configurations (5 index types × 3 models) on:
- Recall@1, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- Latency (P50, P95, P99)
- Index size

Usage:
    uv run python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from biovector_eval.core.metrics import (
    LatencyMetrics,
    SearchMetrics,
    measure_latency,
    mean_reciprocal_rank,
    recall_at_k,
)
from biovector_eval.utils.persistence import load_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration
MODELS = {
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "slug": "bge-small-en-v1.5",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "slug": "bge-m3",
    },
    "chemberta": {
        "model_id": "DeepChem/ChemBERTa-77M-MTR",
        "slug": "chemberta-77m-mtr",
    },
}

INDEX_TYPES = ["flat", "hnsw", "sq8", "sq4", "pq"]

DATA_DIR = Path("data")
INDICES_DIR = DATA_DIR / "indices"
RESULTS_DIR = Path("results/hmdb")


@dataclass
class EvaluationResult:
    """Results for a single model/index configuration."""

    model: str
    index_type: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    index_size_mb: float
    num_queries: int

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "index_type": self.index_type,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "index_size_mb": round(self.index_size_mb, 1),
            "num_queries": self.num_queries,
        }


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth queries."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def load_metabolites(path: Path) -> list[dict]:
    """Load metabolite data."""
    with open(path) as f:
        return json.load(f)


def create_id_mapping(metabolites: list[dict]) -> dict[int, str]:
    """Create mapping from index position to HMDB ID."""
    return {i: m["hmdb_id"] for i, m in enumerate(metabolites)}


def evaluate_index(
    index: faiss.Index,
    model: SentenceTransformer,
    queries: list[dict],
    id_mapping: dict[int, str],
    index_path: Path,
    k: int = 10,
) -> EvaluationResult:
    """Evaluate a single index configuration."""

    def search_fn(query_text: str, k: int) -> list[str]:
        """Search function for evaluation."""
        # Encode query
        query_embedding = model.encode(
            query_text, normalize_embeddings=True, show_progress_bar=False
        )
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        distances, indices = index.search(query_embedding, k)

        # Map indices to HMDB IDs
        return [id_mapping[int(idx)] for idx in indices[0] if idx >= 0]

    # Evaluate recall and MRR
    recall_1_scores: list[float] = []
    recall_5_scores: list[float] = []
    recall_10_scores: list[float] = []
    mrr_scores: list[float] = []

    for q in tqdm(queries, desc="Evaluating", leave=False):
        expected = set(q["expected"])
        predictions = search_fn(q["query"], k)

        recall_1_scores.append(recall_at_k(expected, predictions, 1))
        recall_5_scores.append(recall_at_k(expected, predictions, 5))
        recall_10_scores.append(recall_at_k(expected, predictions, 10))
        mrr_scores.append(mean_reciprocal_rank(expected, predictions))

    # Measure latency
    query_texts = [q["query"] for q in queries]
    latency = measure_latency(search_fn, query_texts, k=k, warmup=10)

    # Get index size
    index_size_mb = index_path.stat().st_size / 1024 / 1024

    return EvaluationResult(
        model="",  # Filled in by caller
        index_type="",  # Filled in by caller
        recall_at_1=float(np.mean(recall_1_scores)),
        recall_at_5=float(np.mean(recall_5_scores)),
        recall_at_10=float(np.mean(recall_10_scores)),
        mrr=float(np.mean(mrr_scores)),
        p50_ms=latency.p50_ms,
        p95_ms=latency.p95_ms,
        p99_ms=latency.p99_ms,
        mean_ms=latency.mean_ms,
        index_size_mb=index_size_mb,
        num_queries=len(queries),
    )


def main() -> None:
    """Run full evaluation."""
    logger.info("Starting HMDB metabolite evaluation")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading ground truth and metabolites...")
    ground_truth = load_ground_truth(DATA_DIR / "hmdb/ground_truth.json")
    metabolites = load_metabolites(DATA_DIR / "hmdb/metabolites.json")
    id_mapping = create_id_mapping(metabolites)
    logger.info(f"  Loaded {len(ground_truth)} queries, {len(metabolites)} metabolites")

    # Results storage
    all_results: list[EvaluationResult] = []
    results_by_model: dict[str, dict[str, dict]] = {}

    # Evaluate each model
    for model_name, model_config in MODELS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name} ({model_config['model_id']})")
        logger.info("=" * 60)

        # Load embedding model
        logger.info("Loading embedding model...")
        start = time.perf_counter()
        model = SentenceTransformer(model_config["model_id"])
        logger.info(f"  Model loaded in {time.perf_counter() - start:.1f}s")

        results_by_model[model_name] = {}

        # Evaluate each index type
        for index_type in INDEX_TYPES:
            index_path = INDICES_DIR / f"{model_config['slug']}_{index_type}.faiss"

            if not index_path.exists():
                logger.warning(f"  Index not found: {index_path}")
                continue

            logger.info(f"\n  Evaluating: {index_type}")

            # Load index
            index = load_index(index_path)

            # Set efSearch for HNSW-based indices
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = 128

            # Evaluate
            result = evaluate_index(
                index=index,
                model=model,
                queries=ground_truth,
                id_mapping=id_mapping,
                index_path=index_path,
            )
            result.model = model_name
            result.index_type = index_type

            all_results.append(result)
            results_by_model[model_name][index_type] = result.to_dict()

            logger.info(f"    Recall@1: {result.recall_at_1:.4f}")
            logger.info(f"    Recall@5: {result.recall_at_5:.4f}")
            logger.info(f"    MRR: {result.mrr:.4f}")
            logger.info(f"    P95 latency: {result.p95_ms:.2f}ms")
            logger.info(f"    Index size: {result.index_size_mb:.1f}MB")

            del index

        del model

    # Generate summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    # Find best configurations
    best_recall = max(all_results, key=lambda r: r.recall_at_1)
    best_latency = min(all_results, key=lambda r: r.p95_ms)
    best_memory = min(all_results, key=lambda r: r.index_size_mb)

    recommendations = {
        "best_accuracy": f"{best_recall.model} + {best_recall.index_type} (Recall@1: {best_recall.recall_at_1:.4f})",
        "best_latency": f"{best_latency.model} + {best_latency.index_type} (P95: {best_latency.p95_ms:.2f}ms)",
        "best_memory": f"{best_memory.model} + {best_memory.index_type} ({best_memory.index_size_mb:.1f}MB)",
    }

    logger.info("\nRecommendations:")
    for key, value in recommendations.items():
        logger.info(f"  {key}: {value}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "evaluation_results.json"

    output = {
        "metadata": {
            "num_queries": len(ground_truth),
            "num_metabolites": len(metabolites),
            "models": list(MODELS.keys()),
            "index_types": INDEX_TYPES,
        },
        "models": results_by_model,
        "recommendations": recommendations,
        "all_results": [r.to_dict() for r in all_results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
