#!/bin/bash

# Run vLLM server with EAGLE3 speculative decoding

#LD_PRELOAD=/usr/local/cuda-12.4/lib64/libcublas.so \
#VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
LD_PRELOAD=/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8/libcublas.so.12:/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8/libcublasLt.so.12 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --speculative-config '{"model": "./eagle3_qwen3_model/best_model/", "method": "eagle3", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1}' \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7

# Parameters explained:
# --model: The target (verifier) model
# --speculative-config: JSON configuration for speculative decoding
#   - model: Your trained EAGLE3 draft model path
#   - method: "eagle3" for EAGLE3 speculative decoding
#   - num_speculative_tokens: How many tokens to generate speculatively (5-10 typical for EAGLE)
#   - draft_tensor_parallel_size: TP size for draft model (usually 1, since it's small)
# --tensor-parallel-size: TP size for the main model
