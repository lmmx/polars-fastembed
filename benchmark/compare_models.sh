#!/bin/bash

# Benchmark all fastembed models - survives crashes/segfaults

MODELS=(
    "Qdrant/all-MiniLM-L6-v2-onnx"
    "Xenova/all-MiniLM-L6-v2"
    "Xenova/all-MiniLM-L12-v2"
    "Xenova/all-mpnet-base-v2"
    "Xenova/bge-base-en-v1.5"
    "Qdrant/bge-base-en-v1.5-onnx-Q"
    "Xenova/bge-large-en-v1.5"
    "Qdrant/bge-large-en-v1.5-onnx-Q"
    "Xenova/bge-small-en-v1.5"
    "Qdrant/bge-small-en-v1.5-onnx-Q"
    "nomic-ai/nomic-embed-text-v1"
    "nomic-ai/nomic-embed-text-v1.5"
    "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
    "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
    "Xenova/paraphrase-multilingual-mpnet-base-v2"
    "Xenova/bge-small-zh-v1.5"
    "Xenova/bge-large-zh-v1.5"
    "lightonai/modernbert-embed-large"
    "intfloat/multilingual-e5-small"
    "intfloat/multilingual-e5-base"
    "Qdrant/multilingual-e5-large-onnx"
    "mixedbread-ai/mxbai-embed-large-v1"
    "Alibaba-NLP/gte-base-en-v1.5"
    "Alibaba-NLP/gte-large-en-v1.5"
    "Qdrant/clip-ViT-B-32-text"
    "jinaai/jina-embeddings-v2-base-code"
    "onnx-community/embeddinggemma-300m-ONNX"
)

PASSED=()
FAILED=()

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Testing: $model"
    echo "============================================================"

    just bench -m "$model"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ PASSED: $model"
        PASSED+=("$model")
    elif [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 136 ]; then
        echo "✗ CRASHED (signal $((exit_code - 128))): $model"
        FAILED+=("$model (CRASH)")
    else
        echo "✗ FAILED (exit $exit_code): $model"
        FAILED+=("$model (exit $exit_code)")
    fi
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Passed: ${#PASSED[@]}/${#MODELS[@]}"
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed models:"
    for m in "${FAILED[@]}"; do
        echo "  - $m"
    done
fi

echo ""
echo "Passed models:"
for m in "${PASSED[@]}"; do
    echo "  - $m"
done
