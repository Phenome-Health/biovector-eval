#!/usr/bin/env python
"""Merge multiple evaluation result files into a single combined file.

This script merges evaluation results from different runs, such as when
adding new models or index types to existing evaluation results.

Usage:
    python scripts/merge_evaluation_results.py \
        --base results/hmdb/evaluation_results.json \
        --new results/hmdb_minilm/evaluation_results.json \
        --new results/hmdb_pq/evaluation_results.json \
        --output results/hmdb/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def config_key(config: dict) -> str:
    """Generate a unique key for a configuration."""
    return f"{config['model']}_{config['index_type']}_{config['strategy']}"


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from results."""
    if not results:
        return {}

    summary = {}

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
        best_latency = min(results_with_latency, key=lambda r: r["metrics"]["p95_ms"])
        summary["best_latency"] = {
            "config": best_latency["config"],
            "p95_ms": best_latency["metrics"]["p95_ms"],
        }

    # Smallest index (only if index_size_mb present)
    results_with_size = [r for r in results if "index_size_mb" in r["metrics"]]
    if results_with_size:
        smallest = min(results_with_size, key=lambda r: r["metrics"]["index_size_mb"])
        summary["smallest_index"] = {
            "config": smallest["config"],
            "index_size_mb": smallest["metrics"]["index_size_mb"],
        }

    return summary


def extract_dimensions(results: list[dict]) -> dict:
    """Extract unique dimensions from results."""
    models = set()
    index_types = set()
    strategies = set()

    for r in results:
        config = r["config"]
        models.add(config["model"])
        index_types.add(config["index_type"])
        strategies.add(config["strategy"])

    return {
        "models": sorted(models),
        "index_types": sorted(index_types),
        "strategies": sorted(strategies),
    }


def merge_results(base_results: list[dict], new_results_list: list[list[dict]]) -> list[dict]:
    """Merge multiple result lists, preferring newer results for duplicates."""
    # Create a dictionary keyed by config
    merged: dict[str, dict] = {}

    # Add base results
    for r in base_results:
        key = config_key(r["config"])
        merged[key] = r

    # Add new results (overwrites duplicates)
    for new_results in new_results_list:
        for r in new_results:
            key = config_key(r["config"])
            if key in merged:
                logger.info(f"  Updating existing config: {key}")
            else:
                logger.info(f"  Adding new config: {key}")
            merged[key] = r

    return list(merged.values())


def load_evaluation_file(path: Path) -> dict:
    """Load evaluation results file."""
    with open(path) as f:
        return json.load(f)


def save_merged_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """Save merged results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions = extract_dimensions(results)
    summary = compute_summary(results)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dimensions": dimensions,
            "num_configurations": len(results),
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved merged results to {output_path}")
    logger.info(f"  Total configurations: {len(results)}")
    logger.info(f"  Models: {dimensions['models']}")
    logger.info(f"  Index types: {dimensions['index_types']}")
    logger.info(f"  Strategies: {dimensions['strategies']}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge evaluation results from multiple files"
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base evaluation results file",
    )
    parser.add_argument(
        "--new",
        type=Path,
        action="append",
        dest="new_files",
        help="New evaluation results file(s) to merge (can specify multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for merged results",
    )

    args = parser.parse_args()

    # Load base results
    logger.info(f"Loading base results from {args.base}...")
    base_data = load_evaluation_file(args.base)
    base_results = base_data.get("results", [])
    logger.info(f"  Found {len(base_results)} configurations")

    # Load new results
    new_results_list = []
    if args.new_files:
        for new_path in args.new_files:
            logger.info(f"Loading new results from {new_path}...")
            new_data = load_evaluation_file(new_path)
            new_results = new_data.get("results", [])
            logger.info(f"  Found {len(new_results)} configurations")
            new_results_list.append(new_results)

    # Merge
    logger.info("Merging results...")
    merged_results = merge_results(base_results, new_results_list)

    # Save
    save_merged_results(merged_results, args.output)

    # Show summary
    if merged_results:
        summary = compute_summary(merged_results)
        print("\nMerged Results Summary:")
        print("-" * 40)
        if "best_recall" in summary:
            cfg = summary["best_recall"]["config"]
            print(f"Best Recall@1: {summary['best_recall']['recall@1']:.4f}")
            print(f"  Config: {cfg['model']} + {cfg['index_type']} + {cfg['strategy']}")
        if "best_mrr" in summary:
            cfg = summary["best_mrr"]["config"]
            print(f"Best MRR: {summary['best_mrr']['mrr']:.4f}")
            print(f"  Config: {cfg['model']} + {cfg['index_type']} + {cfg['strategy']}")
        if "best_latency" in summary:
            cfg = summary["best_latency"]["config"]
            print(f"Best P95 Latency: {summary['best_latency']['p95_ms']:.3f}ms")
            print(f"  Config: {cfg['model']} + {cfg['index_type']} + {cfg['strategy']}")


if __name__ == "__main__":
    main()
