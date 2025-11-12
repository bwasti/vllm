#!/bin/bash

# Stress test script for vLLM EAGLE server
# Usage: ./stress_test.sh [OPTIONS]
#
# Options:
#   --workload {code|chat|debug}  Workload type (default: chat)
#   --host HOST                   Server host (default: localhost)
#   --port PORT                   Server port (default: 8000)
#   --num-requests N              Number of requests (default: 100)
#   --qps QPS                     Requests per second (default: 10)
#   --duration SECONDS            Run for N seconds (overrides num-requests)
#   --output FILE                 Output metrics file (default: metrics.json)
#   --dataset DATASET             HuggingFace dataset to use (optional)
#   --max-tokens N                Max output tokens per request (default: 256)
#   --temperature T               Temperature (default: 1.0 for chat/code, 0.0 for debug)
#   --help                        Show this help message

set -e

# Default configuration
WORKLOAD="chat"
HOST="localhost"
PORT=8000
NUM_REQUESTS=100
QPS=10
DURATION=""
OUTPUT_FILE="metrics.json"
DATASET=""
MAX_TOKENS=256
TEMPERATURE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --qps)
            QPS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            echo "Stress test script for vLLM EAGLE server"
            echo ""
            echo "Usage: ./stress_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workload {code|chat|debug}  Workload type (default: chat)"
            echo "  --host HOST                   Server host (default: localhost)"
            echo "  --port PORT                   Server port (default: 8000)"
            echo "  --num-requests N              Number of requests (default: 100)"
            echo "  --qps QPS                     Requests per second (default: 10)"
            echo "  --duration SECONDS            Run for N seconds (overrides num-requests)"
            echo "  --output FILE                 Output metrics file (default: metrics.json)"
            echo "  --dataset DATASET             HuggingFace dataset to use (optional)"
            echo "  --max-tokens N                Max output tokens per request (default: 256)"
            echo "  --temperature T               Temperature (default: 1.0 for chat/code, 0.0 for debug)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Workload types:"
            echo "  code   - Programming questions from bigcode/the-stack-dedup"
            echo "  chat   - Conversational prompts from lmsys/lmsys-chat-1m"
            echo "  debug  - Same query repeated (for debugging, uses temp=0)"
            echo ""
            echo "Examples:"
            echo "  ./stress_test.sh --workload chat --num-requests 500"
            echo "  ./stress_test.sh --workload code --qps 5 --duration 300"
            echo "  ./stress_test.sh --workload debug --num-requests 100 --temperature 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate workload type
if [[ "$WORKLOAD" != "code" && "$WORKLOAD" != "chat" && "$WORKLOAD" != "debug" ]]; then
    echo "Error: Invalid workload '$WORKLOAD'. Must be 'code', 'chat', or 'debug'"
    exit 1
fi

# Set default temperature based on workload if not specified
if [[ -z "$TEMPERATURE" ]]; then
    if [[ "$WORKLOAD" == "debug" ]]; then
        TEMPERATURE=0.0
    else
        TEMPERATURE=1.0
    fi
fi

# Print configuration
echo "========================================"
echo "vLLM EAGLE Stress Test Configuration"
echo "========================================"
echo "Workload:            $WORKLOAD"
echo "Server:              http://$HOST:$PORT"
if [[ -n "$DURATION" ]]; then
    echo "Duration:            ${DURATION}s (QPS: $QPS)"
else
    echo "Requests:            $NUM_REQUESTS (QPS: $QPS)"
fi
echo "Max Tokens:          $MAX_TOKENS"
echo "Temperature:         $TEMPERATURE"
echo "Output File:         $OUTPUT_FILE"
if [[ -n "$DATASET" ]]; then
    echo "Dataset:             $DATASET"
fi
echo "========================================"
echo ""

# Create Python stress test script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/stress_test.py"

cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
import asyncio
import aiohttp
import argparse
import json
import time
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

@dataclass
class RequestMetrics:
    request_id: int
    workload_type: str
    prompt_length: int
    output_length: int
    ttft: float  # Time to first token
    total_latency: float
    success: bool
    error: Optional[str] = None

class StressTest:
    def __init__(self, host: str, port: int, workload: str, max_tokens: int, temperature: float):
        self.base_url = f"http://{host}:{port}"
        self.workload = workload
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.metrics: List[RequestMetrics] = []
        self.prompts: List[str] = []

    async def load_prompts(self, dataset: Optional[str] = None, num_prompts: int = 1000):
        """Load prompts from HuggingFace dataset or use defaults."""
        if self.workload == "debug":
            # Debug mode: same prompt repeated
            self.prompts = [
                "Write a Python function to compute the Fibonacci sequence using dynamic programming."
            ] * num_prompts
            print(f"✓ Loaded {len(self.prompts)} debug prompts (same query)")
            return

        # Try to load from HuggingFace
        try:
            from datasets import load_dataset

            if self.workload == "code":
                dataset_name = dataset or "bigcode/the-stack-dedup"
                print(f"Loading code dataset: {dataset_name}...")
                ds = load_dataset(dataset_name, split="train", streaming=True)
                self.prompts = []
                for i, item in enumerate(ds):
                    if i >= num_prompts:
                        break
                    # Use first few lines as prompt
                    code = item.get('content', '')
                    if code:
                        lines = code.split('\n')[:5]
                        prompt = "Complete this code:\n" + '\n'.join(lines)
                        self.prompts.append(prompt)

            elif self.workload == "chat":
                dataset_name = dataset or "lmsys/lmsys-chat-1m"
                print(f"Loading chat dataset: {dataset_name}...")
                ds = load_dataset(dataset_name, split="train", streaming=True)
                self.prompts = []
                for i, item in enumerate(ds):
                    if i >= num_prompts:
                        break
                    # Use first user message as prompt
                    conv = item.get('conversation', [])
                    if conv and len(conv) > 0:
                        user_msg = conv[0].get('content', '')
                        if user_msg:
                            self.prompts.append(user_msg)

            print(f"✓ Loaded {len(self.prompts)} prompts from HuggingFace")

        except Exception as e:
            print(f"Warning: Could not load from HuggingFace ({e})")
            print(f"Using fallback prompts for {self.workload} workload")
            self._load_fallback_prompts(num_prompts)

    def _load_fallback_prompts(self, num_prompts: int):
        """Load fallback prompts when HuggingFace is not available."""
        if self.workload == "code":
            base_prompts = [
                "Write a Python function to sort a list using quicksort.",
                "Implement a binary search tree in C++.",
                "Create a REST API endpoint in Node.js for user authentication.",
                "Write a SQL query to find the top 10 customers by revenue.",
                "Implement a depth-first search algorithm in Python.",
                "Create a React component for a todo list.",
                "Write a function to validate email addresses using regex.",
                "Implement a LRU cache in Python.",
                "Create a Docker file for a Flask application.",
                "Write a function to find the longest palindromic substring.",
            ]
        else:  # chat
            base_prompts = [
                "What are the main differences between Python and JavaScript?",
                "Explain the concept of machine learning to a beginner.",
                "What are some tips for staying productive while working from home?",
                "How does blockchain technology work?",
                "What are the benefits of meditation?",
                "Explain the theory of relativity in simple terms.",
                "What are some healthy breakfast ideas?",
                "How can I improve my writing skills?",
                "What is the difference between AI and machine learning?",
                "Give me advice on learning a new language.",
            ]

        # Repeat prompts to reach desired count
        self.prompts = (base_prompts * (num_prompts // len(base_prompts) + 1))[:num_prompts]

    async def send_request(self, session: aiohttp.ClientSession, request_id: int, prompt: str) -> RequestMetrics:
        """Send a single request and measure metrics."""
        start_time = time.time()

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True  # Use streaming to measure TTFT
        }

        try:
            ttft = None
            output_length = 0

            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestMetrics(
                        request_id=request_id,
                        workload_type=self.workload,
                        prompt_length=len(prompt.split()),
                        output_length=0,
                        ttft=0,
                        total_latency=time.time() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )

                async for line in response.content:
                    if ttft is None:
                        ttft = time.time() - start_time

                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    output_length += len(delta['content'].split())
                        except json.JSONDecodeError:
                            pass

            total_latency = time.time() - start_time

            return RequestMetrics(
                request_id=request_id,
                workload_type=self.workload,
                prompt_length=len(prompt.split()),
                output_length=output_length,
                ttft=ttft or total_latency,
                total_latency=total_latency,
                success=True
            )

        except Exception as e:
            return RequestMetrics(
                request_id=request_id,
                workload_type=self.workload,
                prompt_length=len(prompt.split()),
                output_length=0,
                ttft=0,
                total_latency=time.time() - start_time,
                success=False,
                error=str(e)
            )

    async def run_test(self, num_requests: int, qps: float, duration: Optional[int] = None):
        """Run stress test with specified QPS."""
        print(f"Starting stress test...")
        print(f"Target: {qps} requests/second")
        print("")

        async with aiohttp.ClientSession() as session:
            request_id = 0
            start_time = time.time()
            request_interval = 1.0 / qps

            # Determine when to stop
            if duration:
                end_time = start_time + duration
                total_requests = int(duration * qps)
            else:
                end_time = float('inf')
                total_requests = num_requests

            print(f"Sending up to {total_requests} requests...")

            while request_id < total_requests and time.time() < end_time:
                # Send request
                prompt = self.prompts[request_id % len(self.prompts)]
                task = asyncio.create_task(self.send_request(session, request_id, prompt))

                # Don't await - let it run in background
                task.add_done_callback(lambda t: self.metrics.append(t.result()))

                request_id += 1

                # Print progress
                if request_id % 10 == 0:
                    elapsed = time.time() - start_time
                    actual_qps = request_id / elapsed if elapsed > 0 else 0
                    print(f"  Sent: {request_id}/{total_requests} | Elapsed: {elapsed:.1f}s | Actual QPS: {actual_qps:.1f}")

                # Wait for next request slot
                next_request_time = start_time + request_id * request_interval
                sleep_time = max(0, next_request_time - time.time())
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Wait for all requests to complete
            print("")
            print("Waiting for pending requests to complete...")
            while len(self.metrics) < request_id:
                await asyncio.sleep(0.1)
                print(f"  Completed: {len(self.metrics)}/{request_id}", end='\r')

            print("")
            print(f"✓ All requests completed")

    def print_summary(self):
        """Print summary statistics."""
        if not self.metrics:
            print("No metrics collected")
            return

        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]

        print("")
        print("=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Total Requests:      {len(self.metrics)}")
        print(f"Successful:          {len(successful)} ({100*len(successful)/len(self.metrics):.1f}%)")
        print(f"Failed:              {len(failed)} ({100*len(failed)/len(self.metrics):.1f}%)")
        print("")

        if successful:
            ttfts = [m.ttft for m in successful]
            latencies = [m.total_latency for m in successful]
            output_lengths = [m.output_length for m in successful]

            print("Latency Statistics (seconds):")
            print(f"  TTFT (Time to First Token):")
            print(f"    Mean:    {statistics.mean(ttfts):.3f}")
            print(f"    Median:  {statistics.median(ttfts):.3f}")
            print(f"    P95:     {sorted(ttfts)[int(len(ttfts)*0.95)]:.3f}")
            print(f"    P99:     {sorted(ttfts)[int(len(ttfts)*0.99)]:.3f}")
            print("")
            print(f"  Total Latency:")
            print(f"    Mean:    {statistics.mean(latencies):.3f}")
            print(f"    Median:  {statistics.median(latencies):.3f}")
            print(f"    P95:     {sorted(latencies)[int(len(latencies)*0.95)]:.3f}")
            print(f"    P99:     {sorted(latencies)[int(len(latencies)*0.99)]:.3f}")
            print("")
            print(f"Output Statistics:")
            print(f"  Mean tokens:     {statistics.mean(output_lengths):.1f}")
            print(f"  Median tokens:   {statistics.median(output_lengths):.1f}")

            # Throughput
            total_time = max(m.total_latency for m in successful)
            total_tokens = sum(output_lengths)
            print("")
            print(f"Throughput:")
            print(f"  Tokens/second:   {total_tokens/total_time:.1f}")

        if failed:
            print("")
            print(f"Errors:")
            error_counts = defaultdict(int)
            for m in failed:
                error_counts[m.error or "Unknown"] += 1
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"  {error}: {count}")

        print("=" * 60)

    def save_metrics(self, output_file: str):
        """Save detailed metrics to JSON file."""
        output = {
            "workload": self.workload,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "total_requests": len(self.metrics),
            "successful_requests": sum(1 for m in self.metrics if m.success),
            "failed_requests": sum(1 for m in self.metrics if not m.success),
            "metrics": [asdict(m) for m in self.metrics]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Detailed metrics saved to: {output_file}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workload", choices=["code", "chat", "debug"], default="chat")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--qps", type=float, default=10)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--dataset", default=None)

    args = parser.parse_args()

    # Set default temperature if not specified
    if args.temperature is None:
        args.temperature = 0.0 if args.workload == "debug" else 1.0

    test = StressTest(
        host=args.host,
        port=args.port,
        workload=args.workload,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Load prompts
    max_prompts = args.duration * int(args.qps) if args.duration else args.num_requests
    await test.load_prompts(dataset=args.dataset, num_prompts=max(max_prompts, 1000))

    # Run test
    await test.run_test(args.num_requests, args.qps, args.duration)

    # Print results
    test.print_summary()
    test.save_metrics(args.output)

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_EOF

# Run the Python script
python "$PYTHON_SCRIPT" \
    --host "$HOST" \
    --port "$PORT" \
    --workload "$WORKLOAD" \
    --num-requests "$NUM_REQUESTS" \
    --qps "$QPS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output "$OUTPUT_FILE" \
    ${DURATION:+--duration "$DURATION"} \
    ${DATASET:+--dataset "$DATASET"}

exit_code=$?

# Clean up Python script
rm -f "$PYTHON_SCRIPT"

exit $exit_code
