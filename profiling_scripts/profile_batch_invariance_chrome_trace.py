#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU Profiling script for batch invariance - Chrome Trace Format

This script profiles vLLM batch invariance and outputs traces in Chrome
trace format (JSON) that can be viewed in:
  - Perfetto UI (https://ui.perfetto.dev/)
  - chrome://tracing
  - Speedscope (https://www.speedscope.app/)

Environment variables:
    VLLM_BATCH_INVARIANT: Enable batch invariance (0 or 1, default: 0)
    PROFILE_OUTPUT_FILE: Output trace file (default: ./batch_invariance_profile.json)
    PROFILE_NUM_REQUESTS: Number of requests to profile (default: 100)
    PROFILE_DATASET_PATH: Path to ShareGPT dataset JSON (optional)
    PROFILE_MAX_TOKENS: Max tokens to generate per request (default: 512)
    PROFILE_TP_SIZE: Tensor parallel size (default: 8)
    PROFILE_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    PROFILE_MAX_MODEL_LEN: Max model length (default: 8192)
    PROFILE_BATCH_SIZE: Max batch size (default: 256)

Example usage:
    # Profile baseline
    VLLM_BATCH_INVARIANT=0 PROFILE_OUTPUT_FILE=baseline.json \\
        python profiling_scripts/profile_batch_invariance_chrome_trace.py

    # Profile with batch invariance
    VLLM_BATCH_INVARIANT=1 PROFILE_OUTPUT_FILE=batch_inv.json \\
        python profiling_scripts/profile_batch_invariance_chrome_trace.py

    # View in Perfetto
    # Upload the .json files to https://ui.perfetto.dev/

    # Or view in Chrome
    # Open chrome://tracing and load the .json files
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
        description="Profile batch invariance with Chrome trace output"
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
        help="Path to ShareGPT dataset JSON",
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
        help="Max batch size",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.getenv("PROFILE_OUTPUT_FILE", "./batch_invariance_profile.json"),
        help="Output trace file path",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes (increases trace size)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="FLASH_ATTN",
        help="Attention backend",
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

    # Set batch invariance mode
    batch_invariant = os.getenv("VLLM_BATCH_INVARIANT", "0")
    os.environ["VLLM_BATCH_INVARIANT"] = batch_invariant
    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    print("=" * 80)
    print("BATCH INVARIANCE CHROME TRACE PROFILING")
    print("=" * 80)
    print(f"Model:                 {args.model}")
    print(f"TP Size:               {args.tp_size}")
    print(
        f"Batch Invariant Mode:  {'ENABLED' if batch_invariant == '1' else 'DISABLED'}"
    )
    print(f"Num Requests:          {args.num_requests}")
    print(f"Output File:           {args.output_file}")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    if num_gpus < args.tp_size:
        print(f"WARNING: Adjusting TP size from {args.tp_size} to {num_gpus}")
        args.tp_size = num_gpus
    print()

    # Load tokenizer and dataset
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    # Prepare prompts
    prompts = [req.prompt for req in requests]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )

    # Warmup
    print("Running warmup...")
    _ = llm.generate(prompts[:3], sampling_params)
    print("Warmup complete")
    print()

    # Profile with PyTorch profiler in Chrome trace format
    print("=" * 80)
    print("STARTING PROFILING")
    print("=" * 80)
    print()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=args.record_shapes,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        elapsed_time = time.perf_counter() - start_time

    # Export to Chrome trace format
    print()
    print(f"Exporting trace to {args.output_file}...")
    prof.export_chrome_trace(args.output_file)

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
    print()
    print(f"Chrome trace saved to: {args.output_file}")
    print()
    print("To view the trace:")
    print("  1. Upload to Perfetto: https://ui.perfetto.dev/")
    print(f"  2. Or open chrome://tracing and load: {args.output_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
