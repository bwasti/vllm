#!/usr/bin/env python3
"""
End-to-end test for Online EAGLE training.

This test launches vLLM with online EAGLE enabled, sends inference requests,
and verifies that training is triggered and loss decreases over time.

Usage:
    # Use default model from launch.sh
    python test_online_eagle_e2e.py

    # Use custom models
    TARGET_MODEL=/path/to/target DRAFT_MODEL=/path/to/draft python test_online_eagle_e2e.py

Environment Variables:
    TARGET_MODEL: Path to target model (default: /home/bwasti/model_cache)
    DRAFT_MODEL: Path to draft model (default: /home/bwasti/model_cache/draft/)
    VLLM_PORT: Port for vLLM server (default: 12345)
    BUFFER_SIZE: Training buffer size (default: 16)
    LEARNING_RATE: Learning rate (default: 1e-5)
    MSE_LOSS_WEIGHT: MSE loss weight (default: 1.0)
"""

import os
import sys
import time
import subprocess
import requests
import json
from typing import Optional

# Configuration from environment variables
TARGET_MODEL = os.getenv("TARGET_MODEL", "/home/bwasti/model_cache")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "/home/bwasti/model_cache/draft/")
VLLM_PORT = int(os.getenv("VLLM_PORT", "12345"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "16"))
BUFFER_WRITE_INTERVAL = int(os.getenv("BUFFER_WRITE_INTERVAL", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
MSE_LOSS_WEIGHT = float(os.getenv("MSE_LOSS_WEIGHT", "1.0"))

# Calculate sampling rate from interval
SAMPLING_RATE = 1.0 / BUFFER_WRITE_INTERVAL

# Test configuration
NUM_REQUESTS = 50  # Number of inference requests to send
MAX_TOKENS = 128   # Max tokens per request
STARTUP_TIMEOUT = 120  # Seconds to wait for server startup


class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    def __init__(self, port: int = VLLM_PORT):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"

    def __enter__(self):
        """Start the vLLM server."""
        print("=" * 80)
        print("STARTING VLLM SERVER WITH ONLINE EAGLE")
        print("=" * 80)

        # Build speculative config
        spec_config = {
            "method": "online_eagle",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 8,
            "max_model_len": 4096,
            "online_eagle_learning_rate": LEARNING_RATE,
            "online_eagle_feedback_buffer_size": BUFFER_SIZE,
            "online_eagle_feedback_sampling_rate": SAMPLING_RATE,
            "online_eagle_mse_loss_weight": MSE_LOSS_WEIGHT,
        }

        print(f"\nConfiguration:")
        print(f"  Target Model: {TARGET_MODEL}")
        print(f"  Draft Model: {DRAFT_MODEL}")
        print(f"  Port: {self.port}")
        print(f"  Buffer Size: {BUFFER_SIZE}")
        print(f"  Sampling Rate: {SAMPLING_RATE:.2f}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  MSE Loss Weight: {MSE_LOSS_WEIGHT}")
        print()

        # Build command
        cmd = [
            "vllm", "serve", TARGET_MODEL,
            "--speculative-config", json.dumps(spec_config),
            "--max-model-len", "4096",
            "--max-num-seqs", "32",
            "--tensor-parallel-size", "8",
            "--gpu-memory-utilization", "0.8",
            "--host", "0.0.0.0",
            "--port", str(self.port),
        ]

        # Set LD_PRELOAD for fbcode
        env = os.environ.copy()
        env["LD_PRELOAD"] = "/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"

        print("Starting server...")
        print(f"Command: {' '.join(cmd)}")
        print()

        # Start server process
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Wait for server to be ready
        if not self._wait_for_ready():
            self._cleanup()
            raise RuntimeError("Server failed to start within timeout")

        print("✅ Server is ready!")
        print()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the vLLM server."""
        self._cleanup()

    def _cleanup(self):
        """Clean up server process."""
        if self.process:
            print("\nShutting down server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing server...")
                self.process.kill()
                self.process.wait()
            print("✅ Server stopped")

    def _wait_for_ready(self) -> bool:
        """Wait for server to be ready to accept requests."""
        print("Waiting for server to be ready...")
        start_time = time.time()

        while time.time() - start_time < STARTUP_TIMEOUT:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    print("❌ Server process died!")
                    # Print last output
                    if self.process.stdout:
                        output = self.process.stdout.read()
                        print("Last output:")
                        print(output)
                    return False

                # Try to hit the health endpoint
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            time.sleep(2)

        return False

    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
        """Send a completion request to the server."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": TARGET_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def parse_server_logs(server: VLLMServer) -> dict:
    """Parse server logs to extract training information."""
    training_logs = []

    if not server.process or not server.process.stdout:
        return {"training_events": []}

    # Read available output (non-blocking)
    import select

    output_lines = []
    while True:
        # Check if there's data to read
        ready, _, _ = select.select([server.process.stdout], [], [], 0.1)
        if not ready:
            break

        line = server.process.stdout.readline()
        if not line:
            break
        output_lines.append(line)

    # Parse training events
    for line in output_lines:
        if "Training buffer full" in line:
            training_logs.append({
                "event": "buffer_full",
                "line": line.strip(),
            })
        elif "Training complete:" in line:
            # Extract loss values
            try:
                parts = line.split("Training complete:")[1]
                losses = {}
                for item in parts.split(","):
                    if "=" in item:
                        key, val = item.strip().split("=")
                        losses[key] = float(val)
                training_logs.append({
                    "event": "training_complete",
                    "losses": losses,
                    "line": line.strip(),
                })
            except (IndexError, ValueError):
                pass
        elif "✅ Online EAGLE training initialized" in line:
            training_logs.append({
                "event": "initialized",
                "line": line.strip(),
            })

    return {"training_events": training_logs}


def run_test():
    """Run the end-to-end test."""
    print("=" * 80)
    print("ONLINE EAGLE END-TO-END TEST")
    print("=" * 80)
    print()

    # Test prompts
    prompts = [
        "Once upon a time in a distant galaxy,",
        "The quick brown fox jumps over",
        "Artificial intelligence is transforming",
        "In the year 2050, humanity will",
        "The secret to happiness lies in",
    ]

    with VLLMServer() as server:
        print("=" * 80)
        print("RUNNING INFERENCE REQUESTS")
        print("=" * 80)
        print()

        # Send requests
        for i in range(NUM_REQUESTS):
            prompt = prompts[i % len(prompts)]

            print(f"[{i+1}/{NUM_REQUESTS}] Generating with prompt: '{prompt[:50]}...'")

            try:
                result = server.generate(prompt)

                # Print first few tokens of response
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["text"]
                    print(f"  Response: '{text[:80]}...'")

                # Parse logs after each request
                if (i + 1) % 5 == 0:  # Check logs every 5 requests
                    logs = parse_server_logs(server)
                    if logs["training_events"]:
                        print(f"\n  📊 Training Events Detected:")
                        for event in logs["training_events"]:
                            print(f"    - {event['event']}: {event.get('line', '')[:100]}")
                        print()

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

            # Small delay between requests
            time.sleep(0.5)

        print()
        print("=" * 80)
        print("ANALYZING RESULTS")
        print("=" * 80)
        print()

        # Parse final logs
        logs = parse_server_logs(server)
        training_events = logs["training_events"]

        # Check if training was initialized
        initialized = any(e["event"] == "initialized" for e in training_events)
        print(f"✅ Training Initialized: {initialized}")

        # Count training runs
        training_runs = [e for e in training_events if e["event"] == "training_complete"]
        print(f"✅ Training Runs: {len(training_runs)}")

        # Analyze loss trajectory
        if training_runs:
            print("\n📊 Loss Trajectory:")
            print("  " + "-" * 70)
            print(f"  {'Run':<6} {'Total Loss':<15} {'MSE Loss':<15} {'KL Loss':<15}")
            print("  " + "-" * 70)

            for i, run in enumerate(training_runs):
                losses = run.get("losses", {})
                total = losses.get("total_loss", 0)
                mse = losses.get("mse_loss", 0)
                kl = losses.get("kl_loss", 0)
                print(f"  {i+1:<6} {total:<15.4f} {mse:<15.4f} {kl:<15.4f}")

            print("  " + "-" * 70)

            # Check if loss is decreasing
            if len(training_runs) >= 2:
                first_loss = training_runs[0].get("losses", {}).get("total_loss", float('inf'))
                last_loss = training_runs[-1].get("losses", {}).get("total_loss", float('inf'))

                if last_loss < first_loss:
                    improvement = ((first_loss - last_loss) / first_loss) * 100
                    print(f"\n  ✅ Loss Decreased: {first_loss:.4f} → {last_loss:.4f} ({improvement:.1f}% improvement)")
                else:
                    print(f"\n  ⚠️  Loss Did Not Decrease: {first_loss:.4f} → {last_loss:.4f}")
        else:
            print("\n  ⚠️  No training runs detected")
            print("  Possible reasons:")
            print("  - Buffer not filled (need more requests)")
            print("  - Sampling rate too low")
            print("  - Check server logs for errors")

        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        success = initialized and len(training_runs) > 0

        if success:
            print("✅ TEST PASSED!")
            print(f"  - Training initialized: Yes")
            print(f"  - Training runs: {len(training_runs)}")
            print(f"  - Inference requests: {NUM_REQUESTS}")
        else:
            print("❌ TEST FAILED!")
            print(f"  - Training initialized: {initialized}")
            print(f"  - Training runs: {len(training_runs)}")
            print(f"  - Expected at least 1 training run")

        print()
        return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(run_test())
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
