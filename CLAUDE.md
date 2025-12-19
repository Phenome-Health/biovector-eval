# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

biovector-eval is a framework for evaluating and optimizing vector databases for biological/biomedical domains. It benchmarks embedding models, quantization strategies, and index configurations for metabolite search (with plans to expand to proteins, clinical tests, and disease ontologies).

## Commands

```bash
# Run all checks (ruff, black, pyright, pytest)
./scripts/check.sh

# Auto-fix formatting/linting issues
./scripts/fix.sh

# Run tests
uv run pytest -v

# Run single test file
uv run pytest tests/unit/test_metrics.py -v

# Run specific test
uv run pytest tests/unit/test_metrics.py::test_recall_at_k -v

# Type checking
uv run pyright src/

# CLI commands
uv run biovector-eval check-gpu
uv run biovector-eval generate-ground-truth --metabolites <path> --output <path>
uv run biovector-eval evaluate-models --metabolites <path> --ground-truth <path> --output <dir>
uv run biovector-eval evaluate-quantization --metabolites <path> --ground-truth <path> --output <path>
```

## Architecture

### Core Module (`src/biovector_eval/core/`)

- **embedding_strategy.py**: Three embedding approaches for converting metabolites to vectors:
  - `SinglePrimaryStrategy`: One vector from primary name (minimal storage)
  - `SinglePooledStrategy`: One vector averaged from name + synonyms (balance)
  - `MultiVectorStrategy`: Multiple vectors per metabolite with tiered weights (highest recall, ~36 vectors/metabolite)

- **metrics.py**: Evaluation metrics (Recall@k, MRR) and latency measurement
- **quantization.py**: Quantization strategies (None, SQ8, SQ4, PQ) for compression vs accuracy tradeoffs
- **performance.py**: Memory and latency benchmarking

### Metabolites Module (`src/biovector_eval/metabolites/`)

- **evaluator.py**: Main evaluation pipeline - loads models, generates embeddings, builds FAISS indices, runs benchmarks. Contains `MultiVectorSearcher` for handling both single and multi-vector indices with deduplication.
- **ground_truth.py**: Generates test queries (exact, synonym, fuzzy, edge cases) from HMDB metabolite data

### Key Evaluation Flow

1. Parse metabolite data (HMDB XML → JSON via `scripts/parse_hmdb_xml.py`)
2. Generate ground truth queries (`generate-ground-truth` CLI)
3. Generate embeddings with chosen strategy (`scripts/generate_embeddings.py`)
4. Build FAISS indices (`scripts/build_indices.py`)
5. Run evaluation comparing models/strategies/quantization

### Index Types

Uses FAISS with these index types:
- `IndexFlatIP`: Exact search baseline
- `IndexHNSWFlat`: Fast approximate search (default for production)
- IVF variants for scaling beyond 10M vectors

### Multi-Vector Search

When using `MultiVectorStrategy`, the `MultiVectorSearcher`:
1. Oversamples results (k × 50) to account for duplicate entities
2. Deduplicates by HMDB ID, keeping best weighted score
3. Weights: primary (1.0), synonym (0.9), variation (0.7)

## Code Style

- Use type hints
- Follow Conventional Commits (feat:, fix:, docs:, test:, refactor:, chore:)
- Add dependencies with `uv add <package>` and commit both pyproject.toml and uv.lock
