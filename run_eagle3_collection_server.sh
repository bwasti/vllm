#!/bin/bash

# Run vLLM server with EAGLE3 data collection enabled

VLLM_EAGLE3_DATA_COLLECTION_TOPK=256 \
VLLM_ENABLE_EAGLE3_DATA_COLLECTION=1 \
VLLM_EAGLE3_DATA_COLLECTION_DIR="./eagle_data_qwen3_1.7b" \
VLLM_USE_V1=1 \
VLLM_SERVER_DEV_MODE=1 \
LD_PRELOAD=/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8/libcublas.so.12:/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8/libcublasLt.so.12 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization 0.7

# Data Collection Parameters:
# VLLM_EAGLE3_DATA_COLLECTION_TOPK: Store only top-k logits to save disk space (256 is good)
# VLLM_ENABLE_EAGLE3_DATA_COLLECTION: Enable data collection (must be 1)
# VLLM_EAGLE3_DATA_COLLECTION_DIR: Directory to save collected data
# VLLM_USE_V1: Use vLLM V1 engine (required for EAGLE3)
# VLLM_SERVER_DEV_MODE: Enable development mode
#
# The server will automatically collect hidden states from layers (2, 14, 25) for Qwen3-1.7B
# Each inference request will save training data to the specified directory
#
# After collecting data, use: python collect_spec_decode_data.py to send requests
# Then train with: python train_eagle3_qwen3_vllm.py --target-layers 2,14,25
