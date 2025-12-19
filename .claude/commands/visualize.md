# Visualize Evaluation Results

Analyze and visualize biovector-eval results from a JSON file.

## Arguments
- `$ARGUMENTS` - Path to evaluation results JSON file (default: `results/evaluation_results.json`)

## Instructions

1. **Verify the results file exists** at the specified path (or default path if none provided)

2. **Read and analyze the results JSON** to provide comprehensive insights:

### Summary Statistics
- Report total number of configurations evaluated
- List dimensions: models, index types, strategies
- Show best configuration for each key metric:
  - Best Recall@1 (and its config)
  - Best MRR (and its config)
  - Best Latency P95 (and its config)
  - Smallest Index Size (and its config)

### Recall@k Analysis (LLM Re-ranking Potential)
- Check if extended k values are available (recall@20, @50, @100)
- Compute LLM benefit score: `(recall@max_k - recall@1) / recall@max_k`
  - Use recall@100 if available, otherwise recall@10
- High score (>0.3) = correct answers often in top-k but not top-1 → ideal for LLM re-ranking
- List top 10 configurations by LLM benefit score
- Identify best configuration at each k value - these may differ!

### Plateau Analysis (Extended k only)
If recall@20, @50, @100 are available:
- Compute gains between k intervals: 1→5, 5→10, 10→20, 20→50, 50→100
- Identify where recall gains drop below 1% (plateau point)
- Report "optimal k" for each config (smallest k where subsequent gain < 1%)
- Show LLM headroom: `recall@100 - recall@1` (total potential improvement)
- Recommend k value for LLM re-ranking based on plateau analysis

### Category Breakdown (if available)
If per-category metrics exist (recall@1_exact_match, recall@1_synonym_match, etc.):
- Show recall@1 and recall@10 for all 6 categories:
  - **Core query types**: exact_match, synonym_match, fuzzy_match
  - **Special character handling**: greek_letter, numeric_prefix, special_prefix
- Identify which strategies excel at each query type
- Difficulty hierarchy: exact (easy) → synonym/numeric/special (medium) → fuzzy/greek (hard)
- Highlight configs where harder categories lag significantly (improvement opportunities)
- Show LLM re-ranking potential per category (recall@10 - recall@1)
- Identify best configuration for each category type

### Trade-off Analysis
- Identify Pareto-optimal configurations for:
  - Recall@1 vs Latency (P95)
  - Recall@1 vs Index Size
- Highlight configs that offer best balance of accuracy and efficiency

### Anomaly Detection
- Flag multi-vector + ANN configurations with near-zero recall (<0.01)
- These indicate potential bugs: multi-vector + (hnsw|sq8|sq4|pq) often fails
- Compare anomalous vs working multi-vector configurations
- Note: multi-vector + (flat|ivf_flat|ivf_pq) typically works correctly

### Deployment Recommendations
Generate recommendations for these scenarios:
| Scenario | Criteria |
|----------|----------|
| Maximum Accuracy | Highest recall@1, regardless of cost |
| Low Latency (<20ms) | Best recall@1 where p95_ms < 20 |
| Compact (<100MB) | Best recall@1 where index_size_mb < 100 |
| Best for LLM Re-ranking | Highest LLM benefit score with recall@10 > 0.4 |

### Key Takeaways
Summarize findings including:
- Which strategy performs best overall
- Model performance comparison (often similar across models)
- Compression vs accuracy trade-offs
- Latency characteristics by index type

3. **Offer to launch the notebook** for interactive exploration:
   ```bash
   uv sync --extra notebooks
   uv run jupyter lab notebooks/analysis.ipynb
   ```

4. If the results file path differs from default, remind user to update `RESULTS_PATH` in the notebook's first cell.

## Data Structure Expected

```json
{
  "metadata": {
    "dimensions": { "models": [...], "index_types": [...], "strategies": [...] },
    "num_configurations": int
  },
  "summary": {
    "best_recall": { "config": {...}, "recall@1": float },
    "best_mrr": { "config": {...}, "mrr": float },
    "best_latency": { "config": {...}, "p95_ms": float },
    "smallest_index": { "config": {...}, "index_size_mb": float }
  },
  "results": [
    {
      "config": { "model": str, "index_type": str, "strategy": str },
      "metrics": {
        "recall@1": float, "recall@5": float, "recall@10": float,
        "recall@20": float, "recall@50": float, "recall@100": float,  // Extended k (optional)
        "mrr": float,
        "p50_ms": float, "p95_ms": float, "p99_ms": float, "mean_ms": float,
        "index_size_mb": float, "num_queries": int,
        // Per-category metrics (optional - 6 categories × 2 k values = 12 fields)
        "recall@1_exact_match": float, "recall@10_exact_match": float,
        "recall@1_synonym_match": float, "recall@10_synonym_match": float,
        "recall@1_fuzzy_match": float, "recall@10_fuzzy_match": float,
        "recall@1_greek_letter": float, "recall@10_greek_letter": float,
        "recall@1_numeric_prefix": float, "recall@10_numeric_prefix": float,
        "recall@1_special_prefix": float, "recall@10_special_prefix": float
      }
    }
  ]
}
```
