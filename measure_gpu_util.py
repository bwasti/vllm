#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple GPU utilization measurement tool.

Run this in the background while running benchmark_eagle.py to measure
average GPU utilization. This helps determine if host overhead (like
shm_broadcast blocking) is actually causing GPU idle time.

Usage:
    # Terminal 1: Start GPU monitoring
    python measure_gpu_util.py --output gpu_util.json

    # Terminal 2: Run benchmark
    python benchmark_eagle.py ...

    # Check results
    cat gpu_util.json
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict


def parse_nvidia_smi_output(output: str) -> dict[int, float]:
    """Parse nvidia-smi output to get GPU utilization per GPU."""
    utils = {}
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        try:
            util = float(line.strip())
            utils[i] = util
        except ValueError:
            continue
    return utils


def monitor_gpu_utilization(interval_ms: int = 100, duration_s: float | None = None):
    """
    Monitor GPU utilization at regular intervals.

    Args:
        interval_ms: Sampling interval in milliseconds
        duration_s: Duration to monitor in seconds (None = indefinite)

    Returns:
        dict with statistics per GPU
    """
    start_time = time.time()
    samples_per_gpu = defaultdict(list)

    try:
        while True:
            # Get GPU utilization from nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            utils = parse_nvidia_smi_output(result.stdout)
            for gpu_id, util in utils.items():
                samples_per_gpu[gpu_id].append(util)

            # Check if we should stop
            if duration_s is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration_s:
                    break

            # Sleep for interval
            time.sleep(interval_ms / 1000.0)

    except KeyboardInterrupt:
        print("\nStopping GPU monitoring...")

    # Compute statistics
    stats = {}
    for gpu_id, samples in samples_per_gpu.items():
        if not samples:
            continue

        stats[f"gpu_{gpu_id}"] = {
            "mean": sum(samples) / len(samples),
            "min": min(samples),
            "max": max(samples),
            "num_samples": len(samples),
            "samples_below_50": sum(1 for s in samples if s < 50),
            "samples_below_80": sum(1 for s in samples if s < 80),
            "percent_below_50": 100 * sum(1 for s in samples if s < 50) / len(samples),
            "percent_below_80": 100 * sum(1 for s in samples if s < 80) / len(samples),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization to detect host overhead issues"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Sampling interval in milliseconds (default: 100ms)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to monitor in seconds (default: indefinite)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )

    args = parser.parse_args()

    print("Starting GPU utilization monitoring...")
    print(f"Interval: {args.interval}ms")
    if args.duration:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: indefinite (Ctrl+C to stop)")
    print()

    stats = monitor_gpu_utilization(interval_ms=args.interval, duration_s=args.duration)

    # Print results
    print("\n" + "=" * 80)
    print("GPU Utilization Statistics")
    print("=" * 80)

    for gpu_id, gpu_stats in sorted(stats.items()):
        print(f"\n{gpu_id.upper()}:")
        print(f"  Mean utilization:     {gpu_stats['mean']:.1f}%")
        print(f"  Min utilization:      {gpu_stats['min']:.1f}%")
        print(f"  Max utilization:      {gpu_stats['max']:.1f}%")
        print(f"  Samples < 50%:        {gpu_stats['percent_below_50']:.1f}%")
        print(f"  Samples < 80%:        {gpu_stats['percent_below_80']:.1f}%")
        print(f"  Total samples:        {gpu_stats['num_samples']}")

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)

    all_means = [s["mean"] for s in stats.values()]
    if all_means:
        overall_mean = sum(all_means) / len(all_means)

        if overall_mean > 90:
            print("✓ High GPU utilization - host overhead is NOT a bottleneck")
            print("  The shm_broadcast blocking is likely hidden by GPU compute.")
        elif overall_mean > 70:
            print("~ Moderate GPU utilization - some overhead may be present")
            print("  Check profiling traces to see if optimization is worthwhile.")
        else:
            print("✗ Low GPU utilization - host overhead IS likely a bottleneck")
            print("  The shm_broadcast blocking (or other host code) is limiting GPU.")
            print("  Optimization would likely help!")


if __name__ == "__main__":
    main()
