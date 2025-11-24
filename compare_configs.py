#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Debug script to compare Server API vs LLM API engine configurations.

This script adds logging/assertions to dump the exact configurations
and execution paths for both APIs to find the difference causing
the acceptance rate discrepancy.
"""

import json
import sys

from vllm.engine.arg_utils import EngineArgs


def dump_engine_config(engine_args: EngineArgs, label: str):
    """Dump all engine configuration to compare."""
    print(f"\n{'=' * 80}")
    print(f"{label} ENGINE CONFIGURATION")
    print(f"{'=' * 80}")

    # Get the vllm_config
    vllm_config = engine_args.create_engine_config()

    print("\n### ModelConfig ###")
    print(f"  model: {vllm_config.model_config.model}")
    print(f"  dtype: {vllm_config.model_config.dtype}")
    print(f"  max_model_len: {vllm_config.model_config.max_model_len}")
    print(f"  tokenizer_mode: {vllm_config.model_config.tokenizer_mode}")

    print("\n### CacheConfig ###")
    print(f"  block_size: {vllm_config.cache_config.block_size}")
    print(
        f"  gpu_memory_utilization: {vllm_config.cache_config.gpu_memory_utilization}"
    )
    print(f"  enable_prefix_caching: {vllm_config.cache_config.enable_prefix_caching}")

    print("\n### ParallelConfig ###")
    print(f"  tensor_parallel_size: {vllm_config.parallel_config.tensor_parallel_size}")
    print(
        f"  pipeline_parallel_size: "
        f"{vllm_config.parallel_config.pipeline_parallel_size}"
    )
    print(
        f"  distributed_executor_backend: "
        f"{vllm_config.parallel_config.distributed_executor_backend}"
    )

    print("\n### SchedulerConfig ###")
    print(f"  max_num_seqs: {vllm_config.scheduler_config.max_num_seqs}")
    print(
        f"  max_num_batched_tokens: "
        f"{vllm_config.scheduler_config.max_num_batched_tokens}"
    )
    print(
        f"  enable_chunked_prefill: "
        f"{vllm_config.scheduler_config.enable_chunked_prefill}"
    )
    print(f"  async_scheduling: {vllm_config.scheduler_config.async_scheduling}")
    print(f"  policy: {vllm_config.scheduler_config.policy}")

    if vllm_config.speculative_config:
        print("\n### SpeculativeConfig ###")
        print(f"  method: {vllm_config.speculative_config.method}")
        print(f"  model: {vllm_config.speculative_config.model}")
        print(f"  num_spec_tokens: {vllm_config.speculative_config.num_spec_tokens}")
        print(
            f"  draft_model_config: {vllm_config.speculative_config.draft_model_config}"
        )
        print(
            f"  speculative_max_model_len: "
            f"{vllm_config.speculative_config.speculative_max_model_len}"
        )

    print(f"\n{'=' * 80}\n")

    return vllm_config


def main():
    """Compare LLM API configuration."""

    # Create engine args matching our benchmark_eagle.py
    llm_config = {
        "model": "/data/users/bwasti/wearable_maverick_vllm/",
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 1536,
        "max_num_seqs": 12,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "auto",
        "config_format": "hf",
        "disable_log_stats": False,  # Match server API
        "speculative_config": {
            "model": "/data/users/bwasti/wearable_maverick_vllm/draft/",
            "method": "eagle",
            "num_speculative_tokens": 4,
            "max_model_len": 1536,
            "draft_model_config": {
                "config_format": "hf",
            },
        },
    }

    print("Creating LLM with configuration...")
    print(json.dumps(llm_config, indent=2))

    # Create EngineArgs to dump config (without initializing full engine)
    engine_args = EngineArgs(
        model=llm_config["model"],
        tensor_parallel_size=llm_config["tensor_parallel_size"],
        gpu_memory_utilization=llm_config["gpu_memory_utilization"],
        max_model_len=llm_config["max_model_len"],
        max_num_seqs=llm_config["max_num_seqs"],
        dtype=llm_config["dtype"],
        trust_remote_code=llm_config["trust_remote_code"],
        enable_prefix_caching=llm_config["enable_prefix_caching"],
        kv_cache_dtype=llm_config["kv_cache_dtype"],
        config_format=llm_config["config_format"],
        disable_log_stats=llm_config["disable_log_stats"],
        speculative_config=llm_config["speculative_config"],
    )

    dump_engine_config(engine_args, "LLM() API")

    print("\n\nNOTE: To compare with server API, check the server logs during startup.")
    print("Look for 'Initializing a V1 LLM engine' message which dumps the config.")
    print("\nKey things to compare:")
    print("1. async_scheduling (should be False for both)")
    print("2. max_num_batched_tokens")
    print("3. enable_chunked_prefill")
    print("4. Speculative config parameters")

    return 0


if __name__ == "__main__":
    sys.exit(main())
