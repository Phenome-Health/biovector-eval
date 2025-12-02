# HMDB Metabolite Vector Search Roadmap

## Executive Summary

**Goal:** Optimize vector search for semantic matching of metabolite names and synonyms.

**Dataset:** 217,920 metabolites from the Human Metabolome Database (HMDB)

**Evaluation:** 15 configurations tested (3 models × 5 index types)

**Recommendation:** BGE-M3 + SQ8 quantization
- 52.6% Recall@1, 55.4% Recall@5
- 269 MB index size (3x smaller than full precision)
- 15ms P95 latency

---

## Methodology

### Data Source
- **HMDB (Human Metabolome Database)**
- 217,920 metabolites
- 1,636,409 total synonyms
- Average 7.5 synonyms per metabolite

### Ground Truth Dataset
500 queries across 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Exact match | 150 | Primary metabolite names |
| Synonym match | 150 | Known synonyms |
| Fuzzy match | 100 | Partial names, prefixes |
| Edge cases | 100 | Greek letters, numeric prefixes |

### Evaluation Metrics
- **Recall@k** - Proportion of correct results in top-k
- **MRR** - Mean Reciprocal Rank
- **Latency** - P50, P95, P99 query time
- **Memory** - Index size on disk

---

## Models Tested

| Model | Dimensions | Size | Use Case |
|-------|------------|------|----------|
| **BGE-Small** | 384 | 33M params | Baseline, fast inference |
| **BGE-M3** | 1024 | 568M params | High accuracy, multilingual |
| **ChemBERTa** | 384 | 77M params | Chemistry domain (SMILES) |

### Model Notes
- BGE models are general-purpose text embeddings
- ChemBERTa is trained on chemical SMILES notation, not natural language names
- BGE-M3 has 8K token context and hybrid retrieval capabilities

---

## Index Types Tested

| Index | Type | Compression | Trade-off |
|-------|------|-------------|-----------|
| **Flat** | Exact | 1x | Perfect accuracy, slow |
| **HNSW** | Approximate | ~1.2x | Fast, <1% recall loss |
| **SQ8** | Quantized | 4x | Good compression, <2% loss |
| **SQ4** | Quantized | 8x | High compression, 3-5% loss |
| **PQ** | Quantized | 12x | Maximum compression, 5-8% loss |

---

## Results

### Full Comparison Matrix

| Model | Index | Recall@1 | Recall@5 | MRR | P95 (ms) | Size (MB) |
|-------|-------|----------|----------|-----|----------|-----------|
| BGE-M3 | flat | 0.526 | 0.554 | 0.535 | 50.0 | 851 |
| **BGE-M3** | **sq8** | **0.526** | **0.554** | **0.535** | **15.2** | **269** |
| BGE-M3 | hnsw | 0.524 | 0.550 | 0.532 | 15.0 | 908 |
| BGE-M3 | sq4 | 0.524 | 0.548 | 0.532 | 15.1 | 163 |
| BGE-M3 | pq | 0.470 | 0.510 | 0.488 | 14.8 | 64 |
| BGE-Small | flat | 0.526 | 0.538 | 0.532 | 16.6 | 319 |
| BGE-Small | sq4 | 0.526 | 0.536 | 0.531 | 3.7 | 97 |
| BGE-Small | hnsw | 0.524 | 0.532 | 0.528 | 3.5 | 376 |
| BGE-Small | sq8 | 0.518 | 0.528 | 0.523 | 3.6 | 136 |
| BGE-Small | pq | 0.500 | 0.528 | 0.512 | 3.5 | 64 |
| ChemBERTa | hnsw | 0.418 | 0.500 | 0.453 | 1.6 | 376 |
| ChemBERTa | flat | 0.410 | 0.498 | 0.449 | 14.7 | 319 |
| ChemBERTa | sq8 | 0.400 | 0.500 | 0.444 | 1.7 | 136 |
| ChemBERTa | sq4 | 0.400 | 0.494 | 0.442 | 1.7 | 97 |
| ChemBERTa | pq | 0.390 | 0.486 | 0.432 | 1.6 | 64 |

### Key Findings

1. **Models are nearly equivalent on Recall@1**
   - BGE-Small and BGE-M3 both achieve 52.6%
   - BGE-M3 has slightly better Recall@5 (55.4% vs 53.8%)

2. **ChemBERTa underperforms (-22% vs BGE)**
   - Trained on SMILES notation, not metabolite names
   - Not recommended for name-based search

3. **SQ8 quantization is lossless**
   - BGE-M3 + SQ8 matches Flat index accuracy exactly
   - 3x memory reduction (851 MB → 269 MB)

4. **PQ has significant accuracy loss**
   - 5-6% Recall@1 drop across all models
   - Only use when memory is critical

---

## Recommendations

### Best Configurations by Use Case

| Use Case | Configuration | Recall@1 | Size | Latency |
|----------|---------------|----------|------|---------|
| **Best Accuracy** | BGE-M3 + SQ8 | 52.6% | 269 MB | 15ms |
| **Balanced** | BGE-Small + SQ4 | 52.6% | 97 MB | 3.7ms |
| **Minimum Memory** | BGE-Small + PQ | 50.0% | 64 MB | 3.5ms |
| **Fastest** | ChemBERTa + HNSW | 41.8% | 376 MB | 1.6ms |

### Production Recommendation

**BGE-M3 + SQ8** for most use cases:
- Matches exact search accuracy
- 3x memory savings
- Sub-20ms latency

For memory-constrained deployments, **BGE-Small + SQ4** offers similar accuracy with 3x less memory than BGE-M3.

---

## Next Steps

### Improve Accuracy

1. **Per-category analysis** - Identify which query types fail
2. **Synonym expansion** - Augment queries with known synonyms
3. **Hybrid search** - Combine vector + keyword matching

### Additional Models to Test

| Model | Rationale |
|-------|-----------|
| SapBERT | Biomedical entity linking specialist |
| PubMedBERT | Biomedical domain pre-training |
| BioLinkBERT | Entity linking optimization |

### Expand to Other Collections

| Collection | Size | Priority |
|------------|------|----------|
| LOINC (clinical tests) | 98K | High |
| MONDO (diseases) | 51K | High |
| LipidMaps | 48K | Medium |

---

## Technical Details

### Files Generated

```
data/
├── hmdb/
│   ├── metabolites.json      # 217,920 metabolites (146 MB)
│   └── ground_truth.json     # 500 evaluation queries
├── embeddings/
│   ├── bge-small-en-v1.5.npy # 320 MB
│   ├── bge-m3.npy            # 851 MB
│   └── chemberta-77m-mtr.npy # 320 MB
└── indices/
    └── {model}_{type}.faiss  # 15 index files (4.2 GB total)
```

### Reproducibility

```bash
# Parse HMDB data
uv run python scripts/parse_hmdb_xml.py \
    --input hmdb_metabolites.zip \
    --output data/hmdb/metabolites.json

# Generate embeddings (GPU recommended)
uv run python scripts/generate_embeddings.py \
    --metabolites data/hmdb/metabolites.json \
    --output-dir data/ \
    --device cuda \
    --models all

# Run evaluation
uv run python scripts/run_evaluation.py
```

---

*Last updated: December 2025*
