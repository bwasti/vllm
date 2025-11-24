#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for testing EAGLE via the OpenAI-compatible server API.

This script sends requests to a running vLLM server and monitors acceptance rates.
Use this to compare server API behavior vs LLM() API behavior.

Usage:
    # Start server first:
    ./launch.sh --mode eagle --port 8000

    # Then run this benchmark:
    python benchmark_server_api.py --port 8000 --num-requests 20
"""

import argparse
import json
import sys
import time
from typing import Any

import requests
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE via server API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/users/bwasti/wearable_maverick_vllm/",
        help="Model path (for tokenizer)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Number of requests to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per request",
    )
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
        help="Top-p sampling",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a story about a brave knight.",
        help="Prompt to use for all requests",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests (1=sequential)",
    )

    return parser.parse_args()


def check_server_health(base_url: str) -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def send_completion_request(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool = False,
) -> dict[str, Any] | None:
    """Send a completion request to the server."""
    url = f"{base_url}/v1/completions"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    try:
        if stream:
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            full_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data["choices"][0]["text"]:
                                full_text += data["choices"][0]["text"]
                        except json.JSONDecodeError:
                            continue

            return {
                "text": full_text,
                "prompt_tokens": None,  # Not available in streaming
                "completion_tokens": None,
            }
        else:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            return {
                "text": data["choices"][0]["text"],
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "completion_tokens": data["usage"]["completion_tokens"],
            }
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def main():
    """Main benchmark function."""
    args = parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 80)
    print("SERVER API BENCHMARK")
    print("=" * 80)
    print(f"Server:              {base_url}")
    print(f"Num Requests:        {args.num_requests}")
    print(f"Max Tokens:          {args.max_tokens}")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Streaming:           {args.stream}")
    print(f"Concurrent:          {args.concurrent}")
    print("=" * 80)
    print()

    # Check server health
    print("Checking server health...")
    if not check_server_health(base_url):
        print("ERROR: Server is not healthy!")
        print("Make sure to start the server first:")
        print(f"  ./launch.sh --mode eagle --port {args.port}")
        return 1
    print("✓ Server is healthy")
    print()

    # Get model name from server
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        models_response.raise_for_status()
        models_data = models_response.json()
        model_name = models_data["data"][0]["id"]
        print(f"Model name: {model_name}")
    except Exception as e:
        print(f"Warning: Could not get model name: {e}")
        model_name = args.model
    print()

    # Load tokenizer for token counting
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_tokens = len(tokenizer.encode(args.prompt))
    print(f"Prompt tokens: {prompt_tokens}")
    print()

    # Run benchmark
    print("=" * 80)
    print("STARTING BENCHMARK")
    print("=" * 80)
    print()

    results = []
    start_time = time.time()

    for i in range(args.num_requests):
        print(f"Request {i + 1}/{args.num_requests}...", end=" ", flush=True)

        req_start = time.time()
        result = send_completion_request(
            base_url,
            model_name,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.stream,
        )
        req_time = time.time() - req_start

        if result:
            print(f"✓ ({req_time:.2f}s, {len(result['text'])} chars)")
            results.append(result)
        else:
            print("✗ Failed")

    total_time = time.time() - start_time
    print()

    # Calculate statistics
    successful_requests = len(results)
    if successful_requests == 0:
        print("ERROR: No successful requests!")
        return 1

    total_completion_tokens = sum(
        r["completion_tokens"] for r in results if r["completion_tokens"] is not None
    )

    # Estimate tokens if streaming (count chars / 4 as rough approximation)
    if args.stream:
        total_completion_tokens = sum(len(r["text"]) for r in results) // 4
        print("Note: Token counts are estimated for streaming mode")

    throughput = total_completion_tokens / total_time
    requests_per_sec = successful_requests / total_time

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Total time:            {total_time:.2f}s")
    print(f"Successful requests:   {successful_requests}/{args.num_requests}")
    print(f"Requests/sec:          {requests_per_sec:.2f}")
    print()
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Output throughput:     {throughput:.2f} tokens/s")
    print(f"Avg time per request:  {total_time / successful_requests:.3f}s")
    print(f"Avg tokens per req:    {total_completion_tokens / successful_requests:.1f}")
    print("=" * 80)
    print()
    print("To see acceptance rates, check the server logs for 'SpecDecoding metrics'")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
