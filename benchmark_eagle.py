#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Self-contained EAGLE benchmarking script with Perfetto trace support.

This script uses the vLLM LLM() API to benchmark EAGLE speculative decoding
and generates Chrome/Perfetto traces for performance analysis.

Features:
- Uses LLM() API directly (no server needed)
- Generates random dataset for consistent benchmarking
- Outputs Chrome trace format compatible with Perfetto UI
- Supports concurrent request batching
- Configurable via command-line arguments

Usage:
    # Basic benchmark with defaults
    python benchmark_eagle.py

    # Custom configuration
    python benchmark_eagle.py --num-requests 200 --max-tokens 256 --tp-size 4

    # View trace in Perfetto
    # Upload output file to https://ui.perfetto.dev/

Environment variables:
    VLLM_ATTENTION_BACKEND: Attention backend (default: FLASHINFER)
    LD_PRELOAD: Set to use specific CUDA libraries (see launch.sh)
"""

import argparse
import json
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
        description="Benchmark EAGLE with Perfetto trace support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/",
        help="Target model path",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/draft/",
        help="EAGLE draft model path",
    )

    # EAGLE configuration
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=4,
        help="Number of speculative tokens for EAGLE",
    )
    parser.add_argument(
        "--disable-eagle",
        action="store_true",
        help="Disable EAGLE (baseline comparison)",
    )

    # Hardware configuration
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--disable-cudagraph",
        action="store_true",
        help="Disable CUDA graph optimization (may help with memory issues)",
    )
    parser.add_argument(
        "--disable-async-scheduling",
        action="store_true",
        help="Disable async scheduling (test without async optimizations)",
    )
    parser.add_argument(
        "--enable-async-scheduling",
        action="store_true",
        help="Enable async scheduling (improves latency and throughput)",
    )

    # Workload configuration
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to benchmark",
    )
    parser.add_argument(
        "--use-random-prompts",
        action="store_true",
        help="Use random token sequences instead of text prompts (worse for EAGLE)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input length in tokens (only for --use-random-prompts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max output tokens to generate per request",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1536,
        help="Max model sequence length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=12,
        help="Max number of sequences in a batch",
    )

    # Sampling configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )

    # Profiling configuration
    parser.add_argument(
        "--output-trace",
        type=str,
        default="./eagle_benchmark_trace.json",
        help="Output Chrome trace file path",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable PyTorch profiler (increases overhead)",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes in trace (increases size)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=3,
        help="Number of warmup requests before benchmarking",
    )

    # Output configuration
    parser.add_argument(
        "--stats-file",
        type=str,
        default="",
        help="Optional JSON file to save benchmark statistics",
    )

    return parser.parse_args()


def load_random_requests(
    tokenizer: Any,
    num_requests: int,
    input_len: int = 1024,
    output_len: int = 512,
) -> list[SampleRequest]:
    """Generate random synthetic requests for benchmarking."""
    print(f"Generating {num_requests} random requests...")
    print(f"  Input length:  {input_len} tokens")
    print(f"  Output length: {output_len} tokens")

    dataset = RandomDataset()
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        request_id_prefix="bench_",
        input_len=input_len,
        output_len=output_len,
    )

    print(f"  Generated {len(requests)} requests")
    return requests


def load_text_requests(
    num_requests: int,
    output_len: int = 512,
) -> list[SampleRequest]:
    """Generate real text requests for benchmarking (better for EAGLE)."""
    print(f"Generating {num_requests} text requests...")
    print(f"  Output length: {output_len} tokens")

    # Base prompts that will be repeated/cycled
    base_prompts = [
        "Tell me a story about a brave knight",
        "Tell me a story about a curious scientist",
        "Tell me a story about a wise owl",
        "Tell me a story about an adventurous explorer",
        "Tell me a story about a talented musician",
        "Tell me a story about a clever detective",
        "Tell me a story about a kind teacher",
        "Tell me a story about a skilled chef",
        "Tell me a story about a determined athlete",
        "Tell me a story about a creative artist",
        "Tell me a story about a loyal friend",
        "Tell me a story about a fearless pilot",
        "Tell me a story about a patient gardener",
        "Tell me a story about a resourceful engineer",
        "Tell me a story about a compassionate doctor",
        "Tell me a story about a witty comedian",
        "Tell me a story about a mysterious stranger",
        "Tell me a story about a rebellious teenager",
        "Tell me a story about a philosophical monk",
        "Tell me a story about a charismatic leader",
    ]

    # Cycle through base prompts to generate requested number
    requests = []
    for i in range(num_requests):
        prompt = base_prompts[i % len(base_prompts)]
        requests.append(
            SampleRequest(
                request_id=f"bench_{i}",
                prompt=prompt,
                prompt_len=len(prompt.split()),  # Approximate
                expected_output_len=output_len,
            )
        )

    print(f"  Generated {len(requests)} text requests")
    return requests


def print_config(args: argparse.Namespace, num_gpus: int):
    """Print benchmark configuration."""
    print("=" * 80)
    print("EAGLE BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Target Model:          {args.model}")
    if not args.disable_eagle:
        print(f"Draft Model:           {args.draft_model}")
        print(f"Speculative Tokens:    {args.num_speculative_tokens}")
    else:
        print("Mode:                  BASELINE (EAGLE disabled)")
    print(f"Tensor Parallel Size:  {args.tp_size}")
    print(f"Available GPUs:        {num_gpus}")
    print(f"GPU Memory Util:       {args.gpu_memory_utilization}")
    print(f"Max Model Length:      {args.max_model_len}")
    print(f"Max Batch Size:        {args.max_num_seqs}")

    # Show async scheduling status
    if args.enable_async_scheduling:
        print("Async Scheduling:      ENABLED")
    elif args.disable_async_scheduling:
        print("Async Scheduling:      DISABLED")
    else:
        print("Async Scheduling:      DEFAULT (False)")

    print()
    print(f"Num Requests:          {args.num_requests}")
    print(f"Input Length:          {args.input_len} tokens")
    print(f"Max Output Tokens:     {args.max_tokens}")
    print(f"Temperature:           {args.temperature}")
    print(f"Top-p:                 {args.top_p}")
    print()
    print(f"Profiling Enabled:     {args.enable_profiling}")
    print(f"Output Trace:          {args.output_trace}")
    if args.stats_file:
        print(f"Stats File:            {args.stats_file}")
    print("=" * 80)
    print()


def main():
    """Main benchmark function."""
    args = parse_args()

    # Set environment variables (same as launch.sh)
    os.environ["VLLM_ATTENTION_BACKEND"] = os.getenv(
        "VLLM_ATTENTION_BACKEND", "FLASHINFER"
    )

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    num_gpus = torch.cuda.device_count()
    if num_gpus < args.tp_size:
        print(f"ERROR: Need {args.tp_size} GPUs but only {num_gpus} available")
        return 1

    print_config(args, num_gpus)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Tokenizer loaded")
    print()

    # Generate requests
    # Generate prompts
    if args.use_random_prompts:
        # Use random token sequences (worse for EAGLE)
        # Ensure input_len fits within max_model_len
        actual_input_len = min(
            args.input_len, args.max_model_len - args.max_tokens - 100
        )
        if actual_input_len != args.input_len:
            print(
                f"Warning: Adjusted input length from {args.input_len} to "
                f"{actual_input_len} to fit within "
                f"max_model_len={args.max_model_len}"
            )
            print()

        requests = load_random_requests(
            tokenizer,
            args.num_requests + args.warmup_requests,
            input_len=actual_input_len,
            output_len=args.max_tokens,
        )
    else:
        # Use real text prompts (better for EAGLE)
        requests = load_text_requests(
            args.num_requests + args.warmup_requests,
            output_len=args.max_tokens,
        )
    print()

    # Initialize LLM
    print("Initializing vLLM...")
    start_init = time.perf_counter()

    llm_config = {
        "model": args.model,
        "tensor_parallel_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enable_prefix_caching": True,  # CRITICAL: Required for EAGLE!
        "kv_cache_dtype": "auto",
        "config_format": "hf",  # Force HF format to avoid Mistral auto-detection
        "disable_log_stats": False,  # Enable stats logging including acceptance rate
    }

    # Disable CUDA graph if requested (helps with debugging)
    if args.disable_cudagraph:
        llm_config["enforce_eager"] = True

    # Handle async scheduling flags
    if args.enable_async_scheduling and args.disable_async_scheduling:
        print("ERROR: Cannot both enable and disable async scheduling!")
        return 1
    elif args.enable_async_scheduling:
        llm_config["async_scheduling"] = True
    elif args.disable_async_scheduling:
        llm_config["async_scheduling"] = False

    # Add EAGLE configuration if enabled
    if not args.disable_eagle:
        llm_config["speculative_config"] = {
            "model": args.draft_model,
            "method": "eagle",
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "draft_model_config": {
                "config_format": "hf",  # Force HF format for draft model too
            },
        }

    llm = LLM(**llm_config)

    init_time = time.perf_counter() - start_init
    print(f"Initialization took {init_time:.2f}s")
    print()

    # Prepare prompts and sampling parameters
    all_prompts = [req.prompt for req in requests]
    warmup_prompts = all_prompts[: args.warmup_requests]
    bench_prompts = all_prompts[args.warmup_requests :]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )

    # Warmup
    print(f"Running warmup with {args.warmup_requests} requests...")
    _ = llm.generate(warmup_prompts, sampling_params)
    print("Warmup complete")
    print()

    # Benchmark
    print("=" * 80)
    print("STARTING BENCHMARK")
    print("=" * 80)
    print(f"Benchmarking {len(bench_prompts)} requests...")
    print()

    if args.enable_profiling:
        # Profile with PyTorch profiler
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
            outputs = llm.generate(bench_prompts, sampling_params, use_tqdm=True)
            elapsed_time = time.perf_counter() - start_time

        # Export trace
        print()
        print(f"Exporting trace to {args.output_trace}...")
        prof.export_chrome_trace(args.output_trace)
        print("Trace exported")
    else:
        # Run without profiling overhead
        start_time = time.perf_counter()
        outputs = llm.generate(bench_prompts, sampling_params, use_tqdm=True)
        elapsed_time = time.perf_counter() - start_time
        print()
        print("Note: Profiling disabled. Use --enable-profiling to generate traces.")

    # Calculate statistics
    total_input_tokens = sum(req.prompt_len for req in requests[args.warmup_requests :])
    total_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs if output.outputs
    )
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_output_tokens / elapsed_time
    requests_per_sec = len(outputs) / elapsed_time

    # Print results
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    mode = (
        "BASELINE"
        if args.disable_eagle
        else f"EAGLE (spec_tokens={args.num_speculative_tokens})"
    )
    print(f"Mode:                  {mode}")
    print(f"Total time:            {elapsed_time:.2f}s")
    print(f"Requests processed:    {len(outputs)}")
    print(f"Requests/sec:          {requests_per_sec:.2f}")
    print()
    print(f"Total input tokens:    {total_input_tokens}")
    print(f"Total output tokens:   {total_output_tokens}")
    print(f"Total tokens:          {total_tokens}")
    print(f"Output throughput:     {throughput:.2f} tokens/s")
    print(f"Avg time per request:  {elapsed_time / len(outputs):.3f}s")
    print(f"Avg output per req:    {total_output_tokens / len(outputs):.1f} tokens")
    print("=" * 80)

    if args.enable_profiling:
        print()
        print("To view the trace:")
        print("  1. Upload to Perfetto: https://ui.perfetto.dev/")
        print(f"  2. Or open chrome://tracing and load: {args.output_trace}")
        print("=" * 80)

    # Save statistics to JSON if requested
    if args.stats_file:
        stats = {
            "mode": mode,
            "config": {
                "model": args.model,
                "draft_model": args.draft_model if not args.disable_eagle else None,
                "num_speculative_tokens": args.num_speculative_tokens
                if not args.disable_eagle
                else 0,
                "tp_size": args.tp_size,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "workload": {
                "num_requests": len(outputs),
                "input_len": actual_input_len,
                "max_tokens": args.max_tokens,
            },
            "results": {
                "elapsed_time": elapsed_time,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "throughput_tokens_per_sec": throughput,
                "requests_per_sec": requests_per_sec,
                "avg_time_per_request": elapsed_time / len(outputs),
                "avg_output_per_request": total_output_tokens / len(outputs),
            },
        }

        print()
        print(f"Saving statistics to {args.stats_file}...")
        with open(args.stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print("Statistics saved")

    return 0


if __name__ == "__main__":
    sys.exit(main())
