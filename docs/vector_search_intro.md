# Vector Search for Biological Entities

A 10-slide introduction to embedding models, quantization, and indexing

---

## Slide 1: Title

# Vector Search for Biological Entities

### Finding metabolites, diseases, and clinical tests by meaning, not just keywords

**The Challenge:** How do you find "Vitamin E" when the database says "alpha-tocopherol"?

---

## Slide 2: The Problem with Text Search

### Traditional keyword search fails for biological entities

| Query | Expected Match | Problem |
|-------|----------------|---------|
| "glucose" | D-Glucose | Different capitalization, prefix |
| "Vitamin E" | alpha-tocopherol | Common name vs scientific name |
| "adrenaline" | Epinephrine | Regional naming difference |
| "3-hydroxybutyrate" | beta-Hydroxybutyric acid | Greek letter vs number |

### What we need:
- Match by **meaning**, not exact text
- Handle synonyms, abbreviations, typos
- Scale to 200K+ entities

---

## Slide 3: What are Embeddings?

### Converting text to numbers that capture meaning

```
"glucose" → [0.12, -0.45, 0.78, ..., 0.23]  (384 numbers)
```

**Key insight:** Similar concepts have similar numbers.

```
"glucose"      → [0.12, -0.45, 0.78, ...]
"D-glucose"    → [0.11, -0.44, 0.79, ...]  ← Very similar!
"cholesterol"  → [0.89, 0.23, -0.56, ...]  ← Very different
```

### The embedding model learns these representations from millions of text examples.

---

## Slide 4: Embedding Models

### General vs Domain-Specific Models

| Model Type | Example | Strength |
|------------|---------|----------|
| **General** | BGE-M3 | Works well across all text |
| **Domain-specific** | ChemBERTa | Trained on chemistry data |

### Our finding:
General models (BGE) performed **better** than chemistry-specific models for metabolite name matching.

**Why?** ChemBERTa was trained on chemical formulas (SMILES), not natural language names.

*Lesson: Domain models aren't always better—test before assuming!*

---

## Slide 5: How Vector Similarity Works

### Finding the closest match

1. **Convert query to vector:** "glucose" → [0.12, -0.45, ...]
2. **Compare to all stored vectors:** Calculate similarity scores
3. **Return top matches:** Rank by similarity

### Cosine Similarity
Measures angle between vectors (ignoring length):
- **1.0** = Identical meaning
- **0.0** = Unrelated
- **-1.0** = Opposite meaning

```
similarity("glucose", "D-glucose") = 0.97  ← Match!
similarity("glucose", "cholesterol") = 0.23  ← Different
```

---

## Slide 6: The Scale Problem

### 217,000 metabolites × 1024 dimensions = 850 MB

| Metric | Value |
|--------|-------|
| Metabolites | 217,920 |
| Vector dimensions | 1024 (BGE-M3) |
| Storage per vector | 4 KB |
| **Total storage** | **851 MB** |

### Problems at scale:
1. **Memory:** Entire index must fit in RAM
2. **Latency:** Comparing to 217K vectors takes time
3. **Cost:** Larger vectors = more compute

### Solutions: **Quantization** and **Indexing**

---

## Slide 7: Quantization - Compressing Vectors

### Trade storage for (small) accuracy loss

| Method | Compression | How it works |
|--------|-------------|--------------|
| **SQ8** | 4x smaller | 32-bit → 8-bit per number |
| **SQ4** | 8x smaller | 32-bit → 4-bit per number |
| **PQ** | 12x smaller | Group numbers, use codebook |

### Real results from our evaluation:

| Method | Index Size | Recall@1 | Accuracy Loss |
|--------|------------|----------|---------------|
| Full precision | 851 MB | 52.6% | - |
| **SQ8** | **269 MB** | **52.6%** | **0%** |
| SQ4 | 163 MB | 52.4% | 0.4% |
| PQ | 64 MB | 47.0% | 10.6% |

**SQ8 is the sweet spot:** 3x smaller with zero accuracy loss!

---

## Slide 8: Indexing with HNSW

### Approximate search: Don't compare everything

**Brute force:** Compare query to all 217,920 vectors (slow)

**HNSW (Hierarchical Navigable Small World):** Build a graph, navigate to answer (fast)

### How HNSW works:
1. Vectors connected in a multi-layer graph
2. Start at top layer, navigate down
3. Find approximate nearest neighbors quickly

### Trade-off:

| Method | Latency | Accuracy |
|--------|---------|----------|
| Exact (Flat) | 50 ms | 100% |
| **HNSW** | **15 ms** | **99.6%** |

3x faster with <1% accuracy loss.

---

## Slide 9: Our Results

### 15 configurations tested on 217,920 HMDB metabolites

**Best configurations:**

| Use Case | Model + Index | Recall@1 | Size | Speed |
|----------|---------------|----------|------|-------|
| **Recommended** | BGE-M3 + SQ8 | 52.6% | 269 MB | 15 ms |
| Memory-constrained | BGE-Small + SQ4 | 52.6% | 97 MB | 4 ms |
| Minimum size | BGE-Small + PQ | 50.0% | 64 MB | 3 ms |

### Key findings:
1. **SQ8 quantization is lossless** - Use it always
2. **BGE-M3 slightly better than BGE-Small** for recall@5
3. **ChemBERTa underperforms** - Wrong training data

---

## Slide 10: Key Takeaways

### What we learned

1. **Embeddings enable semantic search**
   - Find matches by meaning, not keywords
   - Handle synonyms, abbreviations, variations

2. **General models can beat domain-specific**
   - Test assumptions, don't assume domain = better
   - ChemBERTa trained on SMILES, not names

3. **Quantization is (almost) free**
   - SQ8 gives 4x compression with 0% loss
   - Always use at least SQ8 in production

4. **HNSW indexing is fast**
   - 3x speedup with <1% accuracy loss
   - Essential for real-time applications

### Recommended stack:
**BGE-M3 + SQ8 + HNSW** for balanced accuracy, memory, and speed.

---

## Appendix: Technical Reference

### Embedding Models

| Model | HuggingFace ID | Dimensions |
|-------|----------------|------------|
| BGE-Small | BAAI/bge-small-en-v1.5 | 384 |
| BGE-M3 | BAAI/bge-m3 | 1024 |
| ChemBERTa | DeepChem/ChemBERTa-77M-MTR | 384 |

### Index Types (FAISS)

| Type | Description |
|------|-------------|
| IndexFlatIP | Exact inner product search |
| IndexHNSWFlat | HNSW with full precision |
| IndexHNSWSQ | HNSW with scalar quantization |
| IndexHNSWPQ | HNSW with product quantization |

### Metrics Explained

| Metric | Definition |
|--------|------------|
| Recall@k | % of correct results in top-k |
| MRR | 1/rank of first correct result |
| P95 Latency | 95th percentile query time |

---

*Created for the BioVector Evaluation Project | December 2025*
