#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quick summary of GPU gaps across all traces - aggregates data efficiently.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def quick_parse_events(trace_data: dict) -> tuple[list, list]:
    """Quick parse to get just GPU kernels and CPU events we care about."""
    gpu_kernels = []
    cpu_events = []

    raw_events = trace_data.get(
        "traceEvents", trace_data if isinstance(trace_data, list) else []
    )

    for event in raw_events:
        if event.get("ph") != "X":
            continue

        cat = event.get("cat", "")
        name = event.get("name", "")
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if cat == "kernel":
            gpu_kernels.append((ts, dur, name))
        elif (
            cat not in ["cuda_runtime", "kernel", "gpu_memcpy", "gpu_memset"]
            and dur > 0
        ):
            # Check if it's one of our patterns of interest
            name_lower = name.lower()
            if any(
                p in name_lower
                for p in ["shm_broadcast", "dequeue", "prepare_input", "broadcast"]
            ):
                cpu_events.append((ts, dur, name))

    return gpu_kernels, cpu_events


def analyze_one_trace(filepath: Path, min_gap_ms: float = 0.05):
    """Quick analysis of one trace file."""
    # Load trace
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rt") as f:
            trace_data = json.load(f)
    else:
        with open(filepath) as f:
            trace_data = json.load(f)

    gpu_kernels, cpu_events = quick_parse_events(trace_data)

    if len(gpu_kernels) < 2:
        return None

    # Sort kernels by time
    gpu_kernels.sort(key=lambda x: x[0])

    # Find gaps
    gaps_with_patterns = defaultdict(list)
    total_gap_time = 0
    num_gaps = 0

    for i in range(len(gpu_kernels) - 1):
        prev_end = gpu_kernels[i][0] + gpu_kernels[i][1]
        next_start = gpu_kernels[i + 1][0]
        gap_dur_us = next_start - prev_end

        if gap_dur_us < min_gap_ms * 1000:
            continue

        gap_dur_ms = gap_dur_us / 1000.0
        num_gaps += 1
        total_gap_time += gap_dur_ms

        # Check for CPU events in this gap
        for cpu_ts, cpu_dur, cpu_name in cpu_events:
            cpu_end = cpu_ts + cpu_dur
            # Check overlap
            if cpu_ts < next_start and cpu_end > prev_end:
                # Categorize by pattern
                name_lower = cpu_name.lower()
                if "shm_broadcast" in name_lower or "broadcast" in name_lower:
                    gaps_with_patterns["shm_broadcast"].append(gap_dur_ms)
                if "dequeue" in name_lower:
                    gaps_with_patterns["dequeue"].append(gap_dur_ms)
                if "prepare_input" in name_lower:
                    gaps_with_patterns["prepare_input"].append(gap_dur_ms)

    trace_dur_ms = (
        gpu_kernels[-1][0] + gpu_kernels[-1][1] - gpu_kernels[0][0]
    ) / 1000.0
    kernel_time_ms = sum(dur for _, dur, _ in gpu_kernels) / 1000.0

    return {
        "num_kernels": len(gpu_kernels),
        "num_gaps": num_gaps,
        "trace_dur_ms": trace_dur_ms,
        "kernel_time_ms": kernel_time_ms,
        "gap_time_ms": total_gap_time,
        "gpu_util": (kernel_time_ms / trace_dur_ms * 100) if trace_dur_ms > 0 else 0,
        "pattern_gaps": dict(gaps_with_patterns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=str, default="./traces")
    parser.add_argument("--min-gap-ms", type=float, default=0.05)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_files = list(trace_dir.glob("*.pt.trace.json.gz")) + list(
        trace_dir.glob("*.pt.trace.json")
    )

    if not trace_files:
        print(f"No traces found in {trace_dir}")
        return

    print(f"Analyzing {len(trace_files)} trace files...\n")

    # Aggregate results
    all_results = []

    for i, trace_file in enumerate(trace_files, 1):
        print(f"[{i}/{len(trace_files)}] {trace_file.name}...", end=" ", flush=True)
        result = analyze_one_trace(trace_file, min_gap_ms=args.min_gap_ms)
        if result:
            all_results.append(result)
            print(f"✓ ({result['num_gaps']} gaps, {result['gpu_util']:.1f}% util)")
        else:
            print("✗ (insufficient data)")

    if not all_results:
        print("\nNo valid results!")
        return

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS ACROSS ALL TRACES")
    print("=" * 80)

    total_kernels = sum(r["num_kernels"] for r in all_results)
    total_gaps = sum(r["num_gaps"] for r in all_results)
    total_trace_dur = sum(r["trace_dur_ms"] for r in all_results)
    total_kernel_time = sum(r["kernel_time_ms"] for r in all_results)
    total_gap_time = sum(r["gap_time_ms"] for r in all_results)
    avg_gpu_util = sum(r["gpu_util"] for r in all_results) / len(all_results)

    print("\nOverall:")
    print(f"  Total GPU kernels:     {total_kernels:,}")
    print(f"  Total gaps:            {total_gaps:,}")
    print(f"  Total trace duration:  {total_trace_dur:.2f} ms")
    print(f"  Total kernel time:     {total_kernel_time:.2f} ms")
    print(f"  Total gap time:        {total_gap_time:.2f} ms")
    print(f"  Average GPU util:      {avg_gpu_util:.1f}%")
    print(f"  Average idle:          {100 - avg_gpu_util:.1f}%")

    # Pattern statistics
    print("\n" + "=" * 80)
    print("PATTERN BREAKDOWN")
    print("=" * 80)

    all_patterns = defaultdict(list)
    for result in all_results:
        for pattern, gaps in result["pattern_gaps"].items():
            all_patterns[pattern].extend(gaps)

    print(
        f"\n{'Pattern':<25} {'# Gaps':>10} {'Total Time':>15} {'% of Gaps':>12} {'Avg Duration':>15}"
    )
    print("-" * 80)

    for pattern in ["shm_broadcast", "dequeue", "prepare_input"]:
        if pattern in all_patterns:
            gaps = all_patterns[pattern]
            total_time = sum(gaps)
            pct = (total_time / total_gap_time * 100) if total_gap_time > 0 else 0
            avg_dur = total_time / len(gaps) if gaps else 0
            print(
                f"{pattern:<25} {len(gaps):>10} {total_time:>13.2f} ms {pct:>11.1f}% {avg_dur:>13.3f} ms"
            )

    # Unaccounted time
    accounted_time = sum(sum(gaps) for gaps in all_patterns.values())
    # Note: Some gaps might have multiple patterns, so this is approximate

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    idle_pct = 100 - avg_gpu_util

    if avg_gpu_util > 90:
        print("\n✓ GPU utilization is excellent (>90%)")
        print("  Host overhead is minimal and likely not worth optimizing.")
    elif avg_gpu_util > 80:
        print("\n~ GPU utilization is good (80-90%)")
        print(f"  {idle_pct:.1f}% idle time could potentially be improved.")
    else:
        print("\n✗ GPU utilization could be improved (<80%)")
        print(f"  {idle_pct:.1f}% idle time - optimization likely worthwhile!")

    # Specific recommendations
    print("\nSpecific findings:")

    for pattern in ["shm_broadcast", "dequeue", "prepare_input"]:
        if pattern in all_patterns:
            gaps = all_patterns[pattern]
            total_time = sum(gaps)
            pct_of_idle = (
                (total_time / total_gap_time * 100) if total_gap_time > 0 else 0
            )

            if pct_of_idle > 20:
                print(
                    f"  ⚠ '{pattern}' accounts for {pct_of_idle:.1f}% of idle time - significant!"
                )
            elif pct_of_idle > 10:
                print(
                    f"  ~ '{pattern}' accounts for {pct_of_idle:.1f}% of idle time - moderate"
                )
            else:
                print(
                    f"  ✓ '{pattern}' accounts for {pct_of_idle:.1f}% of idle time - minor"
                )


if __name__ == "__main__":
    main()
