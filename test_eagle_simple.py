#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple EAGLE test that works with FLASHINFER + prefix caching.
This is the minimal test that showed 40% acceptance rate.
"""

import os

# Set FLASHINFER before importing vLLM
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

print("Testing EAGLE with FLASHINFER + enable_prefix_caching=True")
print("=" * 80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/users/bwasti/wearable_maverick_vllm/")

# Create LLM with BOTH fixes
llm = LLM(
    model="/data/users/bwasti/wearable_maverick_vllm/",
    speculative_config={
        "model": "/data/users/bwasti/wearable_maverick_vllm/draft/",
        "method": "eagle",
        "num_speculative_tokens": 4,
        "max_model_len": 1536,
        "draft_model_config": {
            "config_format": "hf",
        },
    },
    tensor_parallel_size=8,
    gpu_memory_utilization=0.7,
    max_model_len=1536,
    max_num_seqs=12,
    dtype="bfloat16",
    trust_remote_code=True,
    enable_prefix_caching=True,  # FIX 1: Enable prefix caching
    kv_cache_dtype="auto",
    config_format="hf",
    disable_log_stats=False,  # Enable stats to see acceptance rate
)

print("\nRunning 10 warmup + 20 benchmark requests...")

# Generate test prompts
prompts = [
    f"Tell me a story about {topic}"
    for topic in [
        "a brave knight",
        "a curious scientist",
        "a wise owl",
        "an adventurous explorer",
        "a talented musician",
        "a clever detective",
        "a kind teacher",
        "a skilled chef",
        "a determined athlete",
        "a creative artist",
        "a loyal friend",
        "a fearless pilot",
        "a patient gardener",
        "a resourceful engineer",
        "a compassionate doctor",
        "a witty comedian",
        "a mysterious stranger",
        "a rebellious teenager",
        "a philosophical monk",
        "a charismatic leader",
        "a dedicated researcher",
        "a talented dancer",
        "a brave firefighter",
        "a curious child",
        "a wise elder",
        "a skilled architect",
        "a determined inventor",
        "a creative writer",
        "a loyal dog",
        "a fearless astronaut",
    ]
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
    ignore_eos=False,
)

# Warmup
print("Warmup...")
_ = llm.generate(prompts[:10], sampling_params)

# Benchmark
print("Benchmarking...")
outputs = llm.generate(prompts[10:30], sampling_params, use_tqdm=True)

print(f"\nCompleted {len(outputs)} requests")
print("=" * 80)
print('Look for "SpecDecoding metrics" lines above showing acceptance rate ~40%')
print("=" * 80)
