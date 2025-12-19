#!/usr/bin/env python
"""Run 4D evaluation of embedding model/index/strategy configurations.

Evaluates all combinations of:
- Models (4): BGE-M3, SapBERT, BioLORD, GTE-Qwen2
- Index types (7): flat, hnsw, sq8, sq4, pq, ivf_flat, ivf_pq
- Strategies (3): single-primary, single-pooled, multi-vector

Total: 4 × 7 × 3 = 84 configurations

Metrics:
- Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100
- MRR (Mean Reciprocal Rank)
- Latency (P50, P95, P99)
- Index size

Usage:
    uv run python scripts/run_evaluation.py \
        --data-dir data/ \
        --output-dir results/

    # Evaluate specific configurations
    uv run python scripts/run_evaluation.py \
        --models bge-m3,sapbert \
        --strategies single-primary,multi-vector \
        --index-types flat,hnsw
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from biovector_eval.core.metrics import mean_reciprocal_rank, recall_at_k
from biovector_eval.core.performance import measure_latency
from biovector_eval.metabolites.evaluator import MultiVectorSearcher
from biovector_eval.utils.persistence import load_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Dimensions
# =============================================================================

EVALUATION_DIMENSIONS: dict[str, list[str]] = {
    "models": ["bge-m3", "sapbert", "biolord", "gte-qwen2", "minilm"],
    "index_types": ["flat", "hnsw", "hnsw_sq8", "hnsw_sq4", "hnsw_pq", "ivf_flat", "ivf_sq8", "ivf_sq4", "ivf_pq"],
    "strategies": ["single-primary", "single-pooled", "multi-vector"],
}

# Query categories for per-category metrics
QUERY_CATEGORIES = [
    "exact_match", "synonym_match", "fuzzy_match",
    "greek_letter", "numeric_prefix", "special_prefix",
    "arivale",  # Real-world metabolite names from Arivale data
]

# Recall@k values to compute per category
K_VALUES = [1, 5, 10, 20, 50, 100]

# Model slug mapping (for file paths)
MODEL_SLUGS: dict[str, str] = {
    "bge-m3": "bge-m3",
    "sapbert": "sapbert-from-pubmedbert-fulltext",
    "biolord": "biolord-2023",
    "gte-qwen2": "gte-qwen2-1.5b-instruct",
    "minilm": "all-minilm-l6-v2",
}

# Model HuggingFace IDs
MODEL_IDS: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "biolord": "FremyCompany/BioLORD-2023",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation."""

    model: str
    index_type: str
    strategy: str

    @property
    def is_multi_vector(self) -> bool:
        """Check if this uses multi-vector strategy."""
        return self.strategy == "multi-vector"

    def get_model_slug(self) -> str:
        """Get filesystem slug for model."""
        return MODEL_SLUGS.get(self.model, self.model)

    def get_embedding_path(self, embeddings_dir: Path) -> Path:
        """Get path to embeddings file."""
        slug = self.get_model_slug()
        if self.is_multi_vector:
            return embeddings_dir / f"{slug}_{self.strategy}" / "embeddings.npy"
        return embeddings_dir / f"{slug}_{self.strategy}.npy"

    def get_index_path(self, indices_dir: Path) -> Path:
        """Get path to index file."""
        slug = self.get_model_slug()
        return indices_dir / f"{slug}_{self.strategy}_{self.index_type}.faiss"

    def get_metadata_path(self, indices_dir: Path) -> Path | None:
        """Get path to metadata file (for multi-vector only)."""
        if not self.is_multi_vector:
            return None
        index_path = self.get_index_path(indices_dir)
        return index_path.with_suffix(".metadata.json")

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "index_type": self.index_type,
            "strategy": self.strategy,
        }


def get_all_configurations(
    models: list[str] | None = None,
    index_types: list[str] | None = None,
    strategies: list[str] | None = None,
) -> list[EvaluationConfig]:
    """Generate all evaluation configurations.

    Args:
        models: List of models to evaluate (None = all)
        index_types: List of index types to evaluate (None = all)
        strategies: List of strategies to evaluate (None = all)

    Returns:
        List of EvaluationConfig objects
    """
    models = models or EVALUATION_DIMENSIONS["models"]
    index_types = index_types or EVALUATION_DIMENSIONS["index_types"]
    strategies = strategies or EVALUATION_DIMENSIONS["strategies"]

    configs = []
    for model, index_type, strategy in product(models, index_types, strategies):
        configs.append(EvaluationConfig(model, index_type, strategy))

    return configs


# =============================================================================
# Evaluation Functions
# =============================================================================


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth queries."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def load_metabolites(path: Path) -> list[dict]:
    """Load metabolite data."""
    with open(path) as f:
        return json.load(f)


def create_id_mapping(metabolites: list[dict]) -> list[str]:
    """Create mapping from index position to HMDB ID."""
    return [m["hmdb_id"] for m in metabolites]


def evaluate_configuration(
    config: EvaluationConfig,
    model: SentenceTransformer,
    queries: list[dict],
    indices_dir: Path,
    id_mapping: list[str],
    k: int = 100,
) -> dict[str, Any] | None:
    """Evaluate a single configuration.

    Args:
        config: Evaluation configuration
        model: Loaded SentenceTransformer model
        queries: Ground truth queries
        indices_dir: Directory containing indices
        id_mapping: List mapping index positions to HMDB IDs
        k: Number of results to retrieve

    Returns:
        Dictionary with config and metrics, or None if index not found
    """
    index_path = config.get_index_path(indices_dir)

    if not index_path.exists():
        logger.warning(f"Index not found: {index_path}")
        return None

    # Load index
    if config.is_multi_vector:
        searcher = MultiVectorSearcher(index_path, index_type=config.index_type)
    else:
        index = load_index(index_path)
        # Set efSearch for HNSW-based indices
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 128
        # Set nprobe for IVF indices
        if hasattr(index, "nprobe"):
            index.nprobe = 32
        searcher = None

    # Create search function
    def search_fn(query_text: str, k: int) -> list[str]:
        # Encode query
        query_embedding = model.encode(
            query_text, normalize_embeddings=True, show_progress_bar=False
        )
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        if config.is_multi_vector:
            assert searcher is not None
            results = searcher.search(query_embedding, k)
            return searcher.results_to_hmdb_ids(results)
        else:
            assert index is not None
            distances, indices_result = index.search(query_embedding, k)
            return [id_mapping[int(idx)] for idx in indices_result[0] if idx >= 0]

    # Evaluate metrics
    recall_1_scores: list[float] = []
    recall_5_scores: list[float] = []
    recall_10_scores: list[float] = []
    recall_20_scores: list[float] = []
    recall_50_scores: list[float] = []
    recall_100_scores: list[float] = []
    mrr_scores: list[float] = []

    # Per-category score tracking (all k values)
    category_scores: dict[str, dict[str, list[float]]] = {
        cat: {f"recall@{k}": [] for k in K_VALUES} for cat in QUERY_CATEGORIES
    }

    for q in queries:
        expected = set(q["expected"])
        predictions = search_fn(q["query"], k)

        # Aggregate scores
        recall_1_scores.append(recall_at_k(expected, predictions, 1))
        recall_5_scores.append(recall_at_k(expected, predictions, 5))
        recall_10_scores.append(recall_at_k(expected, predictions, 10))
        recall_20_scores.append(recall_at_k(expected, predictions, 20))
        recall_50_scores.append(recall_at_k(expected, predictions, 50))
        recall_100_scores.append(recall_at_k(expected, predictions, 100))
        mrr_scores.append(mean_reciprocal_rank(expected, predictions))

        # Per-category scores (all k values)
        category = q.get("category", "exact_match")
        if category in category_scores:
            for k_val in K_VALUES:
                category_scores[category][f"recall@{k_val}"].append(
                    recall_at_k(expected, predictions, k_val)
                )

    # Measure latency
    query_texts = [q["query"] for q in queries]
    latency = measure_latency(search_fn, query_texts, k=k, warmup=10)

    # Get index size
    index_size_mb = index_path.stat().st_size / 1024 / 1024

    # Build per-category metrics
    per_category_metrics: dict[str, float] = {}
    for cat in QUERY_CATEGORIES:
        for k_val in K_VALUES:
            key = f"recall@{k_val}_{cat}"
            scores = category_scores[cat][f"recall@{k_val}"]
            if scores:
                per_category_metrics[key] = round(float(np.mean(scores)), 4)
            else:
                per_category_metrics[key] = 0.0

    return {
        "config": config.to_dict(),
        "metrics": {
            "recall@1": round(float(np.mean(recall_1_scores)), 4),
            "recall@5": round(float(np.mean(recall_5_scores)), 4),
            "recall@10": round(float(np.mean(recall_10_scores)), 4),
            "recall@20": round(float(np.mean(recall_20_scores)), 4),
            "recall@50": round(float(np.mean(recall_50_scores)), 4),
            "recall@100": round(float(np.mean(recall_100_scores)), 4),
            "mrr": round(float(np.mean(mrr_scores)), 4),
            "p50_ms": round(latency.p50_ms, 3),
            "p95_ms": round(latency.p95_ms, 3),
            "p99_ms": round(latency.p99_ms, 3),
            "mean_ms": round(latency.mean_ms, 3),
            "index_size_mb": round(index_size_mb, 1),
            "num_queries": len(queries),
            # Per-category metrics
            **per_category_metrics,
        },
    }


def save_evaluation_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    filename: str = "evaluation_results.json",
) -> Path:
    """Save evaluation results to JSON file.

    Args:
        results: List of evaluation results
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Compute summary
    summary = {}
    if results:
        # Best recall@1
        best_recall = max(results, key=lambda r: r["metrics"]["recall@1"])
        summary["best_recall"] = {
            "config": best_recall["config"],
            "recall@1": best_recall["metrics"]["recall@1"],
        }

        # Best MRR
        best_mrr = max(results, key=lambda r: r["metrics"]["mrr"])
        summary["best_mrr"] = {
            "config": best_mrr["config"],
            "mrr": best_mrr["metrics"]["mrr"],
        }

        # Best latency (only if p95_ms present)
        results_with_latency = [r for r in results if "p95_ms" in r["metrics"]]
        if results_with_latency:
            best_latency = min(
                results_with_latency, key=lambda r: r["metrics"]["p95_ms"]
            )
            summary["best_latency"] = {
                "config": best_latency["config"],
                "p95_ms": best_latency["metrics"]["p95_ms"],
            }

        # Smallest index (only if index_size_mb present)
        results_with_size = [r for r in results if "index_size_mb" in r["metrics"]]
        if results_with_size:
            smallest = min(
                results_with_size, key=lambda r: r["metrics"]["index_size_mb"]
            )
            summary["smallest_index"] = {
                "config": smallest["config"],
                "index_size_mb": smallest["metrics"]["index_size_mb"],
            }

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dimensions": EVALUATION_DIMENSIONS,
            "num_configurations": len(results),
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def run_4d_evaluation(
    data_dir: Path,
    output_dir: Path,
    models: list[str] | None = None,
    index_types: list[str] | None = None,
    strategies: list[str] | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run full 4D evaluation.

    Args:
        data_dir: Directory containing data (embeddings, indices, ground truth)
        output_dir: Directory to save results
        models: List of models to evaluate (None = all)
        index_types: List of index types to evaluate (None = all)
        strategies: List of strategies to evaluate (None = all)
        device: Device for model inference

    Returns:
        List of evaluation results
    """
    # Paths
    indices_dir = data_dir / "indices"
    ground_truth_path = data_dir / "hmdb" / "ground_truth.json"
    metabolites_path = data_dir / "hmdb" / "metabolites.json"

    # Load data
    logger.info("Loading ground truth and metabolites...")
    queries = load_ground_truth(ground_truth_path)
    metabolites = load_metabolites(metabolites_path)
    id_mapping = create_id_mapping(metabolites)
    logger.info(f"  Loaded {len(queries)} queries, {len(metabolites)} metabolites")

    # Get configurations
    configs = get_all_configurations(models, index_types, strategies)
    logger.info(f"Evaluating {len(configs)} configurations")

    # Group by model (to load model once per group)
    configs_by_model: dict[str, list[EvaluationConfig]] = {}
    for config in configs:
        if config.model not in configs_by_model:
            configs_by_model[config.model] = []
        configs_by_model[config.model].append(config)

    # Evaluate
    all_results: list[dict[str, Any]] = []

    for model_name, model_configs in configs_by_model.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info("=" * 60)

        # Load model
        model_id = MODEL_IDS.get(model_name)
        if not model_id:
            logger.warning(f"Unknown model: {model_name}")
            continue

        logger.info(f"Loading model: {model_id}")
        start = time.perf_counter()
        model = SentenceTransformer(model_id, device=device)
        logger.info(f"  Loaded in {time.perf_counter() - start:.1f}s")

        # Evaluate each configuration for this model
        for config in tqdm(model_configs, desc=f"Evaluating {model_name}"):
            result = evaluate_configuration(
                config=config,
                model=model,
                queries=queries,
                indices_dir=indices_dir,
                id_mapping=id_mapping,
            )

            if result:
                all_results.append(result)
                logger.debug(
                    f"  {config.strategy}/{config.index_type}: "
                    f"R@1={result['metrics']['recall@1']:.3f}"
                )

        del model

    # Save results
    output_path = save_evaluation_results(all_results, output_dir)
    logger.info(f"\nResults saved to: {output_path}")

    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 4D evaluation of embedding models"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hmdb"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models (default: all)",
    )
    parser.add_argument(
        "--index-types",
        type=str,
        default=None,
        help="Comma-separated list of index types (default: all)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("4D Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Parse dimension filters
    models = args.models.split(",") if args.models else None
    index_types = args.index_types.split(",") if args.index_types else None
    strategies = args.strategies.split(",") if args.strategies else None

    # Run evaluation
    results = run_4d_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=models,
        index_types=index_types,
        strategies=strategies,
        device=args.device,
    )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Evaluated {len(results)} configurations")

    if results:
        best = max(results, key=lambda r: r["metrics"]["recall@1"])
        logger.info(f"\nBest Recall@1: {best['metrics']['recall@1']:.4f}")
        logger.info(f"  Config: {best['config']}")


if __name__ == "__main__":
    main()
