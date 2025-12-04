#!/bin/bash
# Run all embedding generation combinations in sequence
# Usage: nohup ./scripts/run_all_embeddings.sh > logs/all_embeddings.log 2>&1 &

set -e  # Exit on error

METABOLITES="data/hmdb/metabolites.json"
OUTPUT_DIR="data/"
DEVICE="cuda"
BATCH_SIZE=64

MODELS=("bge-m3" "sapbert" "biolord" "gte-qwen2")
STRATEGIES=("single-primary" "single-pooled" "multi-vector")

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Starting all embeddings generation"
echo "$(date)"
echo "=========================================="

total_start=$(date +%s)

for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        # Skip if embedding already exists
        if [[ "$strategy" == "multi-vector" ]]; then
            embedding_path="${OUTPUT_DIR}embeddings/${model}_${strategy}/embeddings.npy"
        else
            # Handle model slug conversion
            case $model in
                "sapbert") slug="sapbert-from-pubmedbert-fulltext" ;;
                "biolord") slug="biolord-2023" ;;
                "gte-qwen2") slug="gte-qwen2-1.5b-instruct" ;;
                *) slug="$model" ;;
            esac
            embedding_path="${OUTPUT_DIR}embeddings/${slug}_${strategy}.npy"
        fi

        if [[ -f "$embedding_path" ]]; then
            echo ""
            echo "[SKIP] $model / $strategy - already exists: $embedding_path"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Processing: $model / $strategy"
        echo "$(date)"
        echo "=========================================="

        start=$(date +%s)

        uv run python scripts/generate_embeddings.py \
            --metabolites "$METABOLITES" \
            --output-dir "$OUTPUT_DIR" \
            --device "$DEVICE" \
            --models "$model" \
            --strategy "$strategy" \
            --batch-size "$BATCH_SIZE"

        end=$(date +%s)
        duration=$((end - start))
        echo "Completed $model / $strategy in $((duration / 60)) minutes"
    done
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))

echo ""
echo "=========================================="
echo "ALL EMBEDDINGS COMPLETE"
echo "$(date)"
echo "Total time: $((total_duration / 3600)) hours $((total_duration % 3600 / 60)) minutes"
echo "=========================================="
