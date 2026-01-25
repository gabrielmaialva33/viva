#!/bin/bash
set -e

# Model: Meta-Llama-3-8B-Instruct (GGUF Quantum)
# Quantization: Q4_K_M (Balanced Speed/Quality)
# Url: https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

mkdir -p models
OUTPUT="models/Llama-3-8B-Instruct.Q4_K_M.gguf"

if [ -f "$OUTPUT" ]; then
    echo "Model already exists at $OUTPUT"
    exit 0
fi

echo "Downloading Llama-3-8B-Instruct (Q4_K_M)..."
wget -O "$OUTPUT" "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

echo "Download complete: $OUTPUT"
