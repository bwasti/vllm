#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU Profiling script for EAGLE with DeepSeek V3 - Chrome Trace Format

This script profiles EAGLE speculative decoding and outputs traces in Chrome
trace format (JSON) that can be viewed in:
  - Perfetto UI (https://ui.perfetto.dev/)
  - chrome://tracing
  - Speedscope (https://www.speedscope.app/)

The Chrome trace format is more compact and easier to share than TensorBoard
traces.

Environment variables:
    PROFILE_OUTPUT_FILE: Output trace file (default: ./eagle_profile.json)
    PROFILE_NUM_REQUESTS: Number of requests to profile (default: 100)
    PROFILE_DATASET_PATH: Path to ShareGPT dataset JSON (optional)
    PROFILE_MAX_TOKENS: Max tokens to generate per request (default: 512)
    PROFILE_TP_SIZE: Tensor parallel size (default: 8)
    PROFILE_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    PROFILE_MAX_MODEL_LEN: Max model length (default: 8192)

Example usage:
    # Basic profiling (outputs eagle_profile.json)
    python profiling_scripts/profile_eagle_chrome_trace.py

    # Custom output file
    PROFILE_OUTPUT_FILE=my_eagle_trace.json \\
        python profiling_scripts/profile_eagle_chrome_trace.py

    # View in Perfetto
    # Upload the .json file to https://ui.perfetto.dev/

    # Or view in Chrome
    # Open chrome://tracing and load the .json file
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
        description="Profile EAGLE with Chrome trace output"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/",
        help="Target model for EAGLE",
    )
    parser.add_argument(
        "--speculative-model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/draft/",
        help="EAGLE speculative model (custom EAGLE model for Llama 4 Maverick)",
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
        default=float(os.getenv("PROFILE_GPU_MEM_UTIL", "0.7")),
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("PROFILE_MAX_MODEL_LEN", "1536")),
        help="Max model length",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of speculative tokens",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.getenv("PROFILE_OUTPUT_FILE", "./eagle_profile.json"),
        help="Output trace file path",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes (increases trace size)",
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

    # Set attention backend to FLASHINFER (same as launch.sh)
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    print("=" * 80)
    print("EAGLE CHROME TRACE PROFILING")
    print("=" * 80)
    print(f"Target Model:          {args.model}")
    print(f"Speculative Model:     {args.speculative_model}")
    print(f"TP Size:               {args.tp_size}")
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
        print(f"ERROR: Need {args.tp_size} GPUs but only {num_gpus} available")
        return 1
    print()

    # Load tokenizer and dataset
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Use input_len that fits within max_model_len (1536) with room for output
    input_len = min(1024, args.max_model_len - args.max_tokens - 100)
    requests = load_random_requests(
        tokenizer, args.num_requests, input_len=input_len, output_len=args.max_tokens
    )
    print()

    # Initialize LLM with EAGLE
    print("Initializing vLLM with EAGLE...")
    start_init = time.perf_counter()

    llm = LLM(
        model=args.model,
        speculative_config={
            "model": args.speculative_model,
            "method": "eagle",
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,  # Draft model max length
        },
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=12,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
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
