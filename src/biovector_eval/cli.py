"""Command-line interface for biovector evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_check_gpu(args: argparse.Namespace) -> int:
    """Check GPU status and availability."""
    from biovector_eval.utils.device import check_gpu_status, get_best_device

    status = check_gpu_status()

    print("\n" + "=" * 60)
    print("GPU STATUS CHECK")
    print("=" * 60)

    print(f"\nCUDA Available: {status['cuda_available']}")

    if status["cuda_available"]:
        print(f"CUDA Version: {status['cuda_version']}")
        print(f"Device Count: {status['device_count']}")

        for device in status["devices"]:
            print(f"\nGPU {device['index']}: {device['name']}")
            print(f"  Memory: {device['total_memory_gb']:.1f} GB")
            print(f"  Compute Capability: {device['compute_capability']}")

        if "error" in status:
            print(f"\nWarning: {status['error']}")

    print(f"\nRecommended Device: {status['recommended_device']}")

    # Test actual device selection
    best = get_best_device()
    print(f"Auto-detected Device: {best}")

    if best == "cuda":
        print("\nGPU is ready for embedding models!")
    else:
        print("\nWill use CPU for embedding models.")
        if status["cuda_available"]:
            print("(GPU detected but not usable - may need system restart)")

    return 0


def cmd_generate_ground_truth(args: argparse.Namespace) -> int:
    """Generate ground truth dataset."""
    from biovector_eval.metabolites.ground_truth import HMDBGroundTruthGenerator

    logger.info(f"Loading metabolites from {args.metabolites}")
    with open(args.metabolites) as f:
        metabolites = json.load(f)

    generator = HMDBGroundTruthGenerator(metabolites, seed=args.seed)
    dataset = generator.generate_dataset(
        exact=args.exact,
        synonym=args.synonym,
        fuzzy=args.fuzzy,
        greek=args.greek,
        numeric=args.numeric,
        special=args.special,
    )
    dataset.save(args.output)
    logger.info(f"Generated {len(dataset.queries)} queries -> {args.output}")
    return 0


def cmd_evaluate_models(args: argparse.Namespace) -> int:
    """Evaluate embedding models."""
    from biovector_eval.metabolites.evaluator import (
        METABOLITE_MODELS,
        MetaboliteEvaluator,
    )
    from biovector_eval.metabolites.ground_truth import GroundTruthDataset
    from biovector_eval.utils.device import get_best_device

    # Auto-detect device if requested
    device = args.device
    if device == "auto":
        device = get_best_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Loading metabolites from {args.metabolites}")
    with open(args.metabolites) as f:
        metabolites = json.load(f)

    texts = [m["name"] for m in metabolites]
    ids = [m["hmdb_id"] for m in metabolites]

    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = GroundTruthDataset.load(args.ground_truth)

    if args.models == "all":
        models = list(METABOLITE_MODELS.values())
    else:
        models = [
            METABOLITE_MODELS[m]
            for m in args.models.split(",")
            if m in METABOLITE_MODELS
        ]

    for m in models:
        m.device = device

    evaluator = MetaboliteEvaluator(texts, ids, ground_truth, device)
    results = evaluator.compare_models(models, args.output)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Recall@1: {r.accuracy.recall_at_1:.3f}")
        print(f"  Recall@5: {r.accuracy.recall_at_5:.3f}")
        print(f"  MRR:      {r.accuracy.mrr:.3f}")
        print(f"  P95 Lat:  {r.latency.p95_ms:.1f}ms")

    return 0


def cmd_evaluate_quantization(args: argparse.Namespace) -> int:
    """Evaluate quantization strategies."""
    from sentence_transformers import SentenceTransformer

    from biovector_eval.core.quantization import QuantizationEvaluator
    from biovector_eval.metabolites.ground_truth import GroundTruthDataset
    from biovector_eval.utils.device import get_best_device

    # Auto-detect device if requested
    device = args.device
    if device == "auto":
        device = get_best_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Loading metabolites from {args.metabolites}")
    with open(args.metabolites) as f:
        metabolites = json.load(f)

    texts = [m["name"] for m in metabolites]
    ids = [m["hmdb_id"] for m in metabolites]

    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = GroundTruthDataset.load(args.ground_truth)
    queries = [q.to_dict() for q in ground_truth.queries]

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)

    evaluator = QuantizationEvaluator(embeddings, ids)
    results = evaluator.compare_all(queries, model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPARISON")
    print("=" * 60)
    for r in results:
        print(f"\n{r.config.qtype.value.upper()}:")
        print(f"  Compression: {r.compression_ratio:.1f}x")
        print(f"  Accuracy Loss: {r.accuracy_loss * 100:.1f}%")
        print(f"  Recall@5: {r.accuracy.recall_at_5:.3f}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Biological entity vector evaluation toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check-gpu
    cg = subparsers.add_parser("check-gpu", help="Check GPU status and availability")
    cg.set_defaults(func=cmd_check_gpu)

    # generate-ground-truth
    gt = subparsers.add_parser(
        "generate-ground-truth", help="Generate ground truth dataset"
    )
    gt.add_argument(
        "--metabolites", required=True, help="Path to metabolites JSON file"
    )
    gt.add_argument("--output", required=True, help="Output path for ground truth JSON")
    gt.add_argument(
        "--exact", type=int, default=150, help="Number of exact match queries"
    )
    gt.add_argument(
        "--synonym", type=int, default=150, help="Number of synonym match queries"
    )
    gt.add_argument(
        "--fuzzy", type=int, default=100, help="Number of fuzzy match queries"
    )
    gt.add_argument(
        "--greek", type=int, default=100, help="Number of greek letter edge case queries"
    )
    gt.add_argument(
        "--numeric", type=int, default=100, help="Number of numeric prefix edge case queries"
    )
    gt.add_argument(
        "--special", type=int, default=100, help="Number of special prefix edge case queries"
    )
    gt.add_argument("--seed", type=int, default=42, help="Random seed")
    gt.set_defaults(func=cmd_generate_ground_truth)

    # evaluate-models
    em = subparsers.add_parser("evaluate-models", help="Evaluate embedding models")
    em.add_argument(
        "--metabolites", required=True, help="Path to metabolites JSON file"
    )
    em.add_argument(
        "--ground-truth", required=True, help="Path to ground truth JSON file"
    )
    em.add_argument("--output", required=True, help="Output directory for results")
    em.add_argument(
        "--models", default="all", help="Comma-separated model names or 'all'"
    )
    em.add_argument(
        "--device", default="auto", help="Device for inference (auto/cpu/cuda)"
    )
    em.set_defaults(func=cmd_evaluate_models)

    # evaluate-quantization
    eq = subparsers.add_parser(
        "evaluate-quantization", help="Evaluate quantization strategies"
    )
    eq.add_argument(
        "--metabolites", required=True, help="Path to metabolites JSON file"
    )
    eq.add_argument(
        "--ground-truth", required=True, help="Path to ground truth JSON file"
    )
    eq.add_argument(
        "--model", default="BAAI/bge-small-en-v1.5", help="Embedding model to use"
    )
    eq.add_argument("--output", required=True, help="Output path for results JSON")
    eq.add_argument(
        "--device", default="auto", help="Device for inference (auto/cpu/cuda)"
    )
    eq.set_defaults(func=cmd_evaluate_quantization)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
