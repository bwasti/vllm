#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU Profiling script for batch invariance testing across 8 GPUs.

This script profiles vLLM without EAGLE to test batch invariance behavior.
It uses real workloads from ShareGPT (lmsys chat) dataset to ensure realistic
performance profiling across multiple GPUs.

The script can profile with and without batch invariance mode enabled to
compare performance characteristics.

Environment variables:
    VLLM_TORCH_PROFILER_DIR: Directory to save profiler traces (default: ./vllm_profile_batch_invariance)
    VLLM_PROFILER_DELAY_ITERS: Number of iterations to wait before profiling (default: 2)
    VLLM_PROFILER_MAX_ITERS: Maximum iterations to profile (default: 10)
    VLLM_BATCH_INVARIANT: Enable batch invariance mode (0 or 1, default: 0)
    PROFILE_NUM_REQUESTS: Number of requests to profile (default: 100)
    PROFILE_DATASET_PATH: Path to ShareGPT dataset JSON (optional)
    PROFILE_MAX_TOKENS: Max tokens to generate per request (default: 512)
    PROFILE_TP_SIZE: Tensor parallel size (default: 8)
    PROFILE_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    PROFILE_MAX_MODEL_LEN: Max model length (default: 8192)
    PROFILE_BATCH_SIZE: Max batch size (default: 256)

Example usage:
    # Basic profiling with defaults (8 GPUs, ShareGPT dataset)
    python profiling_scripts/profile_batch_invariance.py

    # Profile WITH batch invariance enabled
    VLLM_BATCH_INVARIANT=1 python profiling_scripts/profile_batch_invariance.py

    # Compare both modes (run twice with different VLLM_BATCH_INVARIANT)
    VLLM_BATCH_INVARIANT=0 VLLM_TORCH_PROFILER_DIR=./profile_baseline \\
        python profiling_scripts/profile_batch_invariance.py
    VLLM_BATCH_INVARIANT=1 VLLM_TORCH_PROFILER_DIR=./profile_batch_inv \\
        python profiling_scripts/profile_batch_invariance.py

    # Custom dataset and settings
    PROFILE_DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        PROFILE_NUM_REQUESTS=200 PROFILE_BATCH_SIZE=512 \\
        python profiling_scripts/profile_batch_invariance.py

    # Smaller workload for quick testing
    PROFILE_NUM_REQUESTS=20 PROFILE_MAX_TOKENS=128 PROFILE_TP_SIZE=1 \\
        python profiling_scripts/profile_batch_invariance.py

    # Profile with specific model (e.g., DeepSeek V3)
    python profiling_scripts/profile_batch_invariance.py --model deepseek-ai/DeepSeek-V3
"""

import argparse
import os
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import RandomDataset, SampleRequest


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile batch invariance behavior on 8 GPUs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/",
        help="Model to profile",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=int(os.getenv("PROFILE_NUM_REQUESTS", "100")),
        help="Number of requests to profile",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("PROFILE_DATASET_PATH", ""),
        help="Path to ShareGPT dataset JSON (optional, will use HF if not provided)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("PROFILE_MAX_TOKENS", "512")),
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.getenv("PROFILE_TP_SIZE", "8")),
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("PROFILE_GPU_MEM_UTIL", "0.9")),
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("PROFILE_MAX_MODEL_LEN", "8192")),
        help="Max model length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=int(os.getenv("PROFILE_BATCH_SIZE", "256")),
        help="Max batch size (max_num_seqs)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="FLASH_ATTN",
        help="Attention backend to use",
    )
    return parser.parse_args()


def load_random_requests(
    tokenizer: Any, num_requests: int, input_len: int = 2048, output_len: int = 512
) -> list[SampleRequest]:
    """Load synthetic random requests for profiling."""
    print("Generating random synthetic requests for profiling...")
    print(f"  Num requests: {num_requests}")

    dataset = RandomDataset()
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        request_id_prefix="profile_",
        input_len=input_len,
        output_len=output_len,
    )

    print(f"  Generated {len(requests)} requests")
    return requests


def main():
    """Main profiling function."""
    args = parse_args()

    # Set up batch invariance mode
    batch_invariant = os.getenv("VLLM_BATCH_INVARIANT", "0")
    os.environ["VLLM_BATCH_INVARIANT"] = batch_invariant

    # Set attention backend
    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    # Set up profiler environment variables if not already set
    profiler_dir = os.getenv(
        "VLLM_TORCH_PROFILER_DIR", "./vllm_profile_batch_invariance"
    )
    os.environ["VLLM_TORCH_PROFILER_DIR"] = profiler_dir

    # Set reasonable defaults for profiling iterations
    if "VLLM_PROFILER_DELAY_ITERS" not in os.environ:
        os.environ["VLLM_PROFILER_DELAY_ITERS"] = "2"
    if "VLLM_PROFILER_MAX_ITERS" not in os.environ:
        os.environ["VLLM_PROFILER_MAX_ITERS"] = "10"

    print("=" * 80)
    print("BATCH INVARIANCE PROFILING CONFIGURATION")
    print("=" * 80)
    print(f"Model:                 {args.model}")
    print(f"Tensor Parallel Size:  {args.tp_size}")
    print(f"Max Batch Size:        {args.max_num_seqs}")
    print(f"Max Model Length:      {args.max_model_len}")
    print(f"Attention Backend:     {args.attention_backend}")
    print(
        f"Batch Invariant Mode:  {'ENABLED' if batch_invariant == '1' else 'DISABLED'}"
    )
    print(f"GPU Memory Util:       {args.gpu_memory_utilization}")
    print(f"Num Requests:          {args.num_requests}")
    print(f"Max Tokens:            {args.max_tokens}")
    print(f"Profiler Output Dir:   {profiler_dir}")
    print(f"Profiler Delay Iters:  {os.environ['VLLM_PROFILER_DELAY_ITERS']}")
    print(f"Profiler Max Iters:    {os.environ['VLLM_PROFILER_MAX_ITERS']}")
    print("=" * 80)
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    if num_gpus < args.tp_size:
        print(f"WARNING: Need {args.tp_size} GPUs but only {num_gpus} available")
        print(f"Continuing with TP size = {num_gpus}")
        args.tp_size = num_gpus
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print()

    # Load dataset
    requests = load_random_requests(
        tokenizer, args.num_requests, input_len=2048, output_len=args.max_tokens
    )
    print()

    # Initialize LLM
    print("Initializing vLLM...")
    start_init = time.perf_counter()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
    )

    init_time = time.perf_counter() - start_init
    print(f"Initialization took {init_time:.2f}s")
    print()

    # Prepare prompts and sampling params
    print("Preparing prompts...")
    prompts = [req.prompt for req in requests]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    print()

    # Warmup
    print("Running warmup (3 requests)...")
    warmup_prompts = prompts[:3]
    _ = llm.generate(warmup_prompts, sampling_params)
    print("Warmup complete")
    print()

    # Start profiling
    print("=" * 80)
    print("STARTING PROFILING")
    print("=" * 80)
    print(f"This will profile {len(prompts)} requests")
    print(f"Traces will be saved to: {profiler_dir}")
    print()

    start_time = time.perf_counter()

    # Enable profiling and run inference
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    llm.stop_profile()

    elapsed_time = time.perf_counter() - start_time

    # Calculate statistics
    total_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs if output.outputs
    )
    throughput = total_tokens / elapsed_time

    print()
    print("=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(
        f"Mode:                 {'BATCH INVARIANT' if batch_invariant == '1' else 'BASELINE'}"
    )
    print(f"Total time:           {elapsed_time:.2f}s")
    print(f"Total tokens:         {total_tokens}")
    print(f"Throughput:           {throughput:.2f} tokens/s")
    print(f"Requests processed:   {len(outputs)}")
    print(f"Avg time per request: {elapsed_time / len(outputs):.2f}s")
    print()
    print(f"Profiler traces saved to: {profiler_dir}")
    print()
    print("To view the traces:")
    print(f"  tensorboard --logdir {profiler_dir}")
    print("=" * 80)
    print()
    print("TIP: To compare batch invariance vs baseline, run this script twice:")
    print("  1. VLLM_BATCH_INVARIANT=0 VLLM_TORCH_PROFILER_DIR=./profile_baseline \\")
    print("       python profiling_scripts/profile_batch_invariance.py")
    print("  2. VLLM_BATCH_INVARIANT=1 VLLM_TORCH_PROFILER_DIR=./profile_batch_inv \\")
    print("       python profiling_scripts/profile_batch_invariance.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
