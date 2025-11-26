#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Analyze GPU idle gaps in PyTorch profiler traces to identify CPU overhead.

This script focuses on finding "whitespace" between GPU kernel invocations
and identifying what CPU operations are happening during those gaps.

Usage:
    python analyze_gpu_gaps.py --trace-dir ./traces
    python analyze_gpu_gaps.py --trace-dir ./traces --min-gap-ms 0.1
    python analyze_gpu_gaps.py --trace-dir ./traces --output gaps_report.txt
"""

import argparse
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Event:
    """Represents a profiling event."""

    name: str
    cat: str  # category
    pid: int  # process id
    tid: int  # thread id
    ts: float  # timestamp (microseconds)
    dur: float  # duration (microseconds)
    args: dict

    @property
    def ts_ms(self) -> float:
        """Timestamp in milliseconds."""
        return self.ts / 1000.0

    @property
    def dur_ms(self) -> float:
        """Duration in milliseconds."""
        return self.dur / 1000.0

    @property
    def end_ts(self) -> float:
        """End timestamp in microseconds."""
        return self.ts + self.dur

    @property
    def is_gpu_kernel(self) -> bool:
        """Is this a GPU kernel?"""
        return self.cat == "kernel"

    @property
    def is_cpu(self) -> bool:
        """Is this a CPU event?"""
        return self.cat not in ["cuda_runtime", "kernel", "gpu_memcpy", "gpu_memset"]


@dataclass
class GpuGap:
    """Represents a gap between GPU kernels."""

    start_ts: float  # microseconds
    end_ts: float  # microseconds
    duration_ms: float
    prev_kernel: str
    next_kernel: str
    cpu_events: list[Event]

    def overlapping_cpu_time_ms(self) -> float:
        """Total time spent in CPU operations during this gap."""
        return sum(e.dur_ms for e in self.cpu_events)

    def get_dominant_cpu_operation(self) -> tuple[str, float] | None:
        """Get the CPU operation taking the most time in this gap."""
        if not self.cpu_events:
            return None
        longest = max(self.cpu_events, key=lambda e: e.dur)
        return (longest.name, longest.dur_ms)

    def __str__(self) -> str:
        return f"Gap({self.duration_ms:.3f}ms, {len(self.cpu_events)} CPU ops)"


def load_trace_file(filepath: Path) -> dict:
    """Load a trace file (supports .json and .json.gz)."""
    print(f"  Loading {filepath.name}...", end=" ", flush=True)
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rt") as f:
            data = json.load(f)
    else:
        with open(filepath) as f:
            data = json.load(f)
    print("✓")
    return data


def parse_events(trace_data: dict) -> list[Event]:
    """Parse trace events from the JSON data."""
    events = []

    # Handle different trace formats
    if "traceEvents" in trace_data:
        raw_events = trace_data["traceEvents"]
    elif isinstance(trace_data, list):
        raw_events = trace_data
    else:
        print(f"Warning: Unknown trace format, keys: {trace_data.keys()}")
        return events

    for event in raw_events:
        # Only process complete events with duration
        if event.get("ph") != "X":
            continue

        try:
            events.append(
                Event(
                    name=event.get("name", ""),
                    cat=event.get("cat", ""),
                    pid=event.get("pid", 0),
                    tid=event.get("tid", 0),
                    ts=event.get("ts", 0),
                    dur=event.get("dur", 0),
                    args=event.get("args", {}),
                )
            )
        except Exception:
            continue

    return events


def find_gpu_gaps(
    events: list[Event], min_gap_ms: float = 0.05
) -> tuple[list[GpuGap], dict]:
    """
    Find all gaps between GPU kernel executions.

    Returns:
        (gaps, stats) where gaps is a list of GpuGap objects and stats contains
        overall statistics about GPU utilization.
    """
    # Separate GPU and CPU events
    gpu_kernels = sorted([e for e in events if e.is_gpu_kernel], key=lambda x: x.ts)
    cpu_events = [e for e in events if e.is_cpu and e.dur > 0]

    if len(gpu_kernels) < 2:
        print("  Warning: Less than 2 GPU kernels found, cannot analyze gaps")
        return [], {}

    # Find gaps between consecutive kernels
    gaps = []
    total_gap_time = 0.0

    for i in range(len(gpu_kernels) - 1):
        prev_kernel = gpu_kernels[i]
        next_kernel = gpu_kernels[i + 1]

        gap_start = prev_kernel.end_ts
        gap_end = next_kernel.ts
        gap_duration_us = gap_end - gap_start

        # Skip negative gaps (overlapping kernels on different streams)
        if gap_duration_us < 0:
            continue

        gap_duration_ms = gap_duration_us / 1000.0

        # Skip gaps below threshold
        if gap_duration_ms < min_gap_ms:
            continue

        # Find CPU events that overlap with this gap
        overlapping_cpu = []
        for cpu_event in cpu_events:
            # Check if CPU event overlaps with gap
            cpu_start = cpu_event.ts
            cpu_end = cpu_event.end_ts

            # Event overlaps if it starts before gap ends and ends after gap starts
            if cpu_start < gap_end and cpu_end > gap_start:
                overlapping_cpu.append(cpu_event)

        gap = GpuGap(
            start_ts=gap_start,
            end_ts=gap_end,
            duration_ms=gap_duration_ms,
            prev_kernel=prev_kernel.name,
            next_kernel=next_kernel.name,
            cpu_events=overlapping_cpu,
        )

        gaps.append(gap)
        total_gap_time += gap_duration_ms

    # Calculate statistics
    trace_duration_us = gpu_kernels[-1].end_ts - gpu_kernels[0].ts
    trace_duration_ms = trace_duration_us / 1000.0
    total_kernel_time_ms = sum(k.dur_ms for k in gpu_kernels)

    stats = {
        "num_kernels": len(gpu_kernels),
        "num_gaps": len(gaps),
        "trace_duration_ms": trace_duration_ms,
        "total_kernel_time_ms": total_kernel_time_ms,
        "total_gap_time_ms": total_gap_time,
        "gpu_utilization": (total_kernel_time_ms / trace_duration_ms * 100)
        if trace_duration_ms > 0
        else 0,
    }

    # Sort gaps by duration (longest first)
    gaps.sort(key=lambda g: g.duration_ms, reverse=True)

    return gaps, stats


def analyze_gap_patterns(gaps: list[GpuGap]) -> dict:
    """
    Analyze patterns in GPU gaps to identify common CPU bottlenecks.
    """
    # Group gaps by dominant CPU operation
    gaps_by_cpu_op = defaultdict(list)
    gaps_with_no_cpu = []

    for gap in gaps:
        dominant = gap.get_dominant_cpu_operation()
        if dominant:
            op_name, _ = dominant
            gaps_by_cpu_op[op_name].append(gap)
        else:
            gaps_with_no_cpu.append(gap)

    # Calculate statistics for each CPU operation type
    cpu_op_stats = {}
    for op_name, op_gaps in gaps_by_cpu_op.items():
        total_gap_time = sum(g.duration_ms for g in op_gaps)
        total_cpu_time = sum(g.overlapping_cpu_time_ms() for g in op_gaps)

        cpu_op_stats[op_name] = {
            "num_gaps": len(op_gaps),
            "total_gap_time_ms": total_gap_time,
            "total_cpu_time_ms": total_cpu_time,
            "mean_gap_ms": total_gap_time / len(op_gaps),
            "max_gap_ms": max(g.duration_ms for g in op_gaps),
        }

    # Sort by total gap time
    sorted_ops = sorted(
        cpu_op_stats.items(), key=lambda x: x[1]["total_gap_time_ms"], reverse=True
    )

    return {
        "cpu_operations": dict(sorted_ops),
        "gaps_with_no_cpu": len(gaps_with_no_cpu),
        "no_cpu_gap_time_ms": sum(g.duration_ms for g in gaps_with_no_cpu),
    }


def print_gap_analysis(
    gaps: list[GpuGap], stats: dict, patterns: dict, top_n: int = 20
):
    """Print detailed gap analysis."""
    print("\n" + "=" * 80)
    print("GPU GAP ANALYSIS - Overall Statistics")
    print("=" * 80)

    print("\nTrace Overview:")
    print(f"  Total GPU kernels:     {stats['num_kernels']}")
    print(f"  Total gaps found:      {stats['num_gaps']}")
    print(f"  Trace duration:        {stats['trace_duration_ms']:.2f} ms")
    print(f"  Total kernel time:     {stats['total_kernel_time_ms']:.2f} ms")
    print(f"  Total gap time:        {stats['total_gap_time_ms']:.2f} ms")
    print(f"  GPU utilization:       {stats['gpu_utilization']:.1f}%")

    idle_percentage = (
        stats["total_gap_time_ms"] / stats["trace_duration_ms"] * 100
        if stats["trace_duration_ms"] > 0
        else 0
    )
    print(f"  GPU idle percentage:   {idle_percentage:.1f}%")

    # Pattern analysis
    print("\n" + "=" * 80)
    print("CPU Operations During GPU Gaps")
    print("=" * 80)

    if not patterns["cpu_operations"]:
        print("\n✓ No CPU operations captured during GPU gaps")
        print(
            "  (This might mean gaps are too small or CPU events aren't instrumented)"
        )
    else:
        print(
            f"\n{'CPU Operation':<50} {'#Gaps':>8} {'Total Gap':>12} {'Avg Gap':>12} {'Max Gap':>12}"
        )
        print("-" * 100)

        for op_name, op_stats in list(patterns["cpu_operations"].items())[:top_n]:
            display_name = op_name[:47] + "..." if len(op_name) > 50 else op_name
            print(
                f"{display_name:<50} {op_stats['num_gaps']:>8} "
                f"{op_stats['total_gap_time_ms']:>11.2f}ms {op_stats['mean_gap_ms']:>11.3f}ms "
                f"{op_stats['max_gap_ms']:>11.3f}ms"
            )

    if patterns["gaps_with_no_cpu"] > 0:
        print(f"\nGaps with no CPU operations captured: {patterns['gaps_with_no_cpu']}")
        print(f"  Total time: {patterns['no_cpu_gap_time_ms']:.2f} ms")

    # Detailed gap breakdown
    print("\n" + "=" * 80)
    print(f"Top {min(top_n, len(gaps))} Longest GPU Gaps")
    print("=" * 80)

    for i, gap in enumerate(gaps[:top_n], 1):
        print(f"\n#{i}: {gap.duration_ms:.3f} ms gap")
        print(f"  After:  {gap.prev_kernel[:70]}")
        print(f"  Before: {gap.next_kernel[:70]}")

        if gap.cpu_events:
            # Group by operation name
            cpu_by_name = defaultdict(list)
            for e in gap.cpu_events:
                cpu_by_name[e.name].append(e)

            print(f"  CPU operations during gap ({len(gap.cpu_events)} events):")

            # Sort by total time
            sorted_cpu = sorted(
                cpu_by_name.items(),
                key=lambda x: sum(e.dur for e in x[1]),
                reverse=True,
            )

            for op_name, evts in sorted_cpu[:5]:  # Show top 5
                total_ms = sum(e.dur_ms for e in evts)
                count = len(evts)
                display_name = (
                    op_name[:60] if len(op_name) <= 60 else op_name[:57] + "..."
                )
                if count > 1:
                    print(f"    - {display_name}: {total_ms:.3f} ms ({count} calls)")
                else:
                    print(f"    - {display_name}: {total_ms:.3f} ms")

            total_cpu_ms = gap.overlapping_cpu_time_ms()
            coverage = (
                (total_cpu_ms / gap.duration_ms * 100) if gap.duration_ms > 0 else 0
            )
            print(f"  Total CPU time: {total_cpu_ms:.3f} ms ({coverage:.0f}% of gap)")
        else:
            print(
                "  ⚠ No CPU operations captured (gap may be too short or CPU not instrumented)"
            )


def search_gaps_for_pattern(gaps: list[GpuGap], pattern: str) -> list[GpuGap]:
    """Find gaps where CPU operations match a pattern."""
    pattern_lower = pattern.lower()
    matching_gaps = []

    for gap in gaps:
        for cpu_event in gap.cpu_events:
            if pattern_lower in cpu_event.name.lower():
                matching_gaps.append(gap)
                break  # Only add gap once

    return matching_gaps


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU idle gaps to identify CPU overhead"
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="./traces",
        help="Directory containing trace files",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=float,
        default=0.05,
        help="Minimum gap duration to analyze in ms (default: 0.05)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top gaps/operations to show (default: 20)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for report"
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for gaps with specific CPU operation pattern",
    )

    args = parser.parse_args()

    # Find trace files
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        return

    trace_files = list(trace_dir.glob("*.pt.trace.json.gz")) + list(
        trace_dir.glob("*.pt.trace.json")
    )

    if not trace_files:
        print(f"No trace files found in {trace_dir}")
        return

    print(f"Found {len(trace_files)} trace file(s)\n")

    # Analyze each trace
    for trace_file in trace_files:
        print(f"\nAnalyzing: {trace_file.name}")

        try:
            # Load and parse
            trace_data = load_trace_file(trace_file)
            events = parse_events(trace_data)
            print(f"  Parsed {len(events)} events")

            # Find gaps
            print(
                f"  Finding GPU gaps (min {args.min_gap_ms}ms)...", end=" ", flush=True
            )
            gaps, stats = find_gpu_gaps(events, min_gap_ms=args.min_gap_ms)
            print(f"✓ Found {len(gaps)} gaps")

            if not gaps:
                print("  No significant gaps found!")
                continue

            # Analyze patterns
            print("  Analyzing gap patterns...", end=" ", flush=True)
            patterns = analyze_gap_patterns(gaps)
            print("✓")

            # Search mode
            if args.search:
                matching = search_gaps_for_pattern(gaps, args.search)
                print(f"\n{'=' * 80}")
                print(f"SEARCH: Gaps with '{args.search}' operations")
                print(f"{'=' * 80}")
                print(f"Found {len(matching)} gaps (out of {len(gaps)} total)")

                if matching:
                    total_time = sum(g.duration_ms for g in matching)
                    print(f"Total gap time: {total_time:.2f} ms")
                    print(f"\nTop {min(10, len(matching))} matching gaps:")

                    for i, gap in enumerate(matching[:10], 1):
                        print(f"\n  #{i}: {gap.duration_ms:.3f} ms")
                        # Show matching CPU operations
                        for cpu_event in gap.cpu_events:
                            if args.search.lower() in cpu_event.name.lower():
                                print(
                                    f"      {cpu_event.name}: {cpu_event.dur_ms:.3f} ms"
                                )
                continue

            # Print analysis
            print_gap_analysis(gaps, stats, patterns, top_n=args.top_n)

            # Specific pattern checks
            print("\n" + "=" * 80)
            print("SPECIFIC PATTERN CHECKS")
            print("=" * 80)

            for pattern in [
                "shm_broadcast",
                "dequeue",
                "prepare_input",
                "broadcast_object",
            ]:
                matching = search_gaps_for_pattern(gaps, pattern)
                if matching:
                    total_time = sum(g.duration_ms for g in matching)
                    print(
                        f"\n✗ Found '{pattern}' in {len(matching)} gaps "
                        f"(total gap time: {total_time:.2f} ms)"
                    )
                else:
                    print(f"\n✓ No '{pattern}' operations found in gaps")

        except Exception as e:
            print(f"\nError processing {trace_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
