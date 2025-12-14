#!/bin/bash

# Time benchmark for passing fastembed models

MODELS=(
    "Qdrant/all-MiniLM-L6-v2-onnx"
    "Xenova/all-MiniLM-L6-v2"
    "Xenova/all-MiniLM-L12-v2"
    "Xenova/bge-base-en-v1.5"
    "Xenova/bge-large-en-v1.5"
    "Xenova/bge-small-en-v1.5"
    "nomic-ai/nomic-embed-text-v1"
    "nomic-ai/nomic-embed-text-v1.5"
    "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
    "Xenova/paraphrase-multilingual-mpnet-base-v2"
    "Xenova/bge-small-zh-v1.5"
    "lightonai/modernbert-embed-large"
    "intfloat/multilingual-e5-small"
    "intfloat/multilingual-e5-base"
    "Qdrant/multilingual-e5-large-onnx"
    "mixedbread-ai/mxbai-embed-large-v1"
    "Alibaba-NLP/gte-base-en-v1.5"
    "Alibaba-NLP/gte-large-en-v1.5"
    "Qdrant/clip-ViT-B-32-text"
    "jinaai/jina-embeddings-v2-base-code"
)

echo "model,time_seconds"

for model in "${MODELS[@]}"; do
    start=$(date +%s.%N)
    just bench -m "$model" > /dev/null 2>&1
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    echo "$model,$elapsed"
done
