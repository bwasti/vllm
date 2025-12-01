#!/bin/bash
# Sweep script to test different MoE optimization configurations
# Tests various combinations of expert placement and all2all backends
# Usage: ./sweep_moe_configs.sh [--eagle]

set -e

# Parse arguments
ENABLE_EAGLE=false
if [[ "$1" == "--eagle" ]]; then
    ENABLE_EAGLE=true
fi

# Configuration
MODEL="/data/users/bwasti/wearable_maverick_vllm"
TP=8
NUM_REQUESTS=100
SWEEP_DIR="sweep_traces"
if [[ "$ENABLE_EAGLE" == "true" ]]; then
    SWEEP_DIR="sweep_traces_eagle"
fi
RESULTS_FILE="${SWEEP_DIR}/results.txt"
LD_PRELOAD_LIBS="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"

# Create sweep directory
mkdir -p "${SWEEP_DIR}"

# Clear results file
echo "MoE Configuration Sweep Results" > "${RESULTS_FILE}"
echo "================================" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"
echo "Model: ${MODEL}" >> "${RESULTS_FILE}"
echo "Tensor Parallel: ${TP}" >> "${RESULTS_FILE}"
echo "Num Requests: ${NUM_REQUESTS}" >> "${RESULTS_FILE}"
if [[ "$ENABLE_EAGLE" == "true" ]]; then
    echo "EAGLE: ENABLED" >> "${RESULTS_FILE}"
else
    echo "EAGLE: DISABLED" >> "${RESULTS_FILE}"
fi
echo "Date: $(date)" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

# Function to run a single configuration
run_config() {
    local name=$1
    local flags=$2
    local trace_dir="${SWEEP_DIR}/${name}"

    echo ""
    echo "=========================================="
    echo "Running: ${name}"
    echo "Flags: ${flags}"
    echo "=========================================="

    # Run benchmark and capture output
    local disable_eagle_flag=""
    if [[ "$ENABLE_EAGLE" == "false" ]]; then
        disable_eagle_flag="--disable-eagle"
    fi

    local output=$(LD_PRELOAD="${LD_PRELOAD_LIBS}" python benchmark_eagle.py \
        --model "${MODEL}" \
        --tp "${TP}" \
        --num-requests "${NUM_REQUESTS}" \
        --enable-profiling \
        --output-trace "${trace_dir}" \
        ${disable_eagle_flag} \
        ${flags} 2>&1)

    # Extract throughput from output
    local throughput=$(echo "${output}" | grep -oP "Output throughput:\s+\K[\d.]+")

    echo "${output}" | tail -20

    # Analyze traces
    echo ""
    echo "Analyzing traces for ${name}..."
    local analysis=$(python quick_gap_summary.py --trace-dir "${trace_dir}" 2>&1)

    # Extract GPU utilization
    local avg_util=$(echo "${analysis}" | grep -oP "Average GPU util:\s+\K[\d.]+")
    local rank_utils=$(echo "${analysis}" | grep "rank-" | grep -oP "\d+\.\d+% util" | grep -oP "\d+\.\d+")

    # Calculate min/max utilization
    local min_util=$(echo "${rank_utils}" | sort -n | head -1)
    local max_util=$(echo "${rank_utils}" | sort -n | tail -1)

    # Write to results file
    echo "" >> "${RESULTS_FILE}"
    echo "Configuration: ${name}" >> "${RESULTS_FILE}"
    echo "  Flags: ${flags}" >> "${RESULTS_FILE}"
    echo "  Throughput: ${throughput} tokens/s" >> "${RESULTS_FILE}"
    echo "  Avg GPU Util: ${avg_util}%" >> "${RESULTS_FILE}"
    echo "  GPU Util Range: ${min_util}% - ${max_util}%" >> "${RESULTS_FILE}"
    echo "  Trace Directory: ${trace_dir}" >> "${RESULTS_FILE}"

    echo ""
    echo "Results: throughput=${throughput} t/s, avg_util=${avg_util}%, range=${min_util}%-${max_util}%"
    echo ""
}

# Test configurations
echo "Starting MoE configuration sweep..."

# 1. Baseline (linear placement, custom all-reduce)
run_config "baseline_linear" ""

# 2. Round-robin placement
run_config "round_robin" "--expert-placement-strategy round_robin"

# 3. NCCL all-reduce
run_config "nccl_allreduce" "--disable-custom-all-reduce"

# 4. Round-robin + NCCL
run_config "round_robin_nccl" "--expert-placement-strategy round_robin --disable-custom-all-reduce"

# 5. DeepEP high throughput
run_config "deepep_ht" "--all2all-backend deepep_high_throughput"

# 6. DeepEP high throughput + round-robin
run_config "deepep_ht_roundrobin" "--all2all-backend deepep_high_throughput --expert-placement-strategy round_robin"

# 7. DeepEP low latency
run_config "deepep_ll" "--all2all-backend deepep_low_latency"

# 8. DeepEP low latency + round-robin
run_config "deepep_ll_roundrobin" "--all2all-backend deepep_low_latency --expert-placement-strategy round_robin"

# Generate summary table
echo ""
echo "=========================================="
echo "Generating summary table..."
echo "=========================================="

export SWEEP_DIR
python << EOF
import re
import sys
import os

# Get sweep directory from environment
sweep_dir = os.environ.get("SWEEP_DIR", "sweep_traces")

# Read results file
with open(f"{sweep_dir}/results.txt", "r") as f:
    content = f.read()

# Parse configurations
configs = []
for block in content.split("Configuration: ")[1:]:
    lines = block.strip().split("\n")
    name = lines[0]

    throughput = re.search(r"Throughput: ([\d.]+)", block)
    avg_util = re.search(r"Avg GPU Util: ([\d.]+)", block)
    util_range = re.search(r"GPU Util Range: ([\d.]+)% - ([\d.]+)%", block)

    if throughput and avg_util and util_range:
        configs.append({
            "name": name,
            "throughput": float(throughput.group(1)),
            "avg_util": float(avg_util.group(1)),
            "min_util": float(util_range.group(1)),
            "max_util": float(util_range.group(2)),
        })

# Sort by avg utilization
configs.sort(key=lambda x: x["avg_util"], reverse=True)

# Print table
print("\n" + "="*100)
print("SUMMARY TABLE (sorted by GPU utilization)")
print("="*100)
print(f"{'Configuration':<30} {'Throughput':<15} {'Avg Util':<12} {'Util Range':<20} {'Imbalance':<12}")
print("-"*100)

for cfg in configs:
    imbalance = cfg["max_util"] - cfg["min_util"]
    print(f"{cfg['name']:<30} {cfg['throughput']:>10.2f} t/s   {cfg['avg_util']:>6.1f}%      "
          f"{cfg['min_util']:>5.1f}% - {cfg['max_util']:>5.1f}%      {imbalance:>6.1f}%")

print("="*100)

# Write table to file
with open(f"{sweep_dir}/summary_table.txt", "w") as f:
    f.write("="*100 + "\n")
    f.write("SUMMARY TABLE (sorted by GPU utilization)\n")
    f.write("="*100 + "\n")
    f.write(f"{'Configuration':<30} {'Throughput':<15} {'Avg Util':<12} {'Util Range':<20} {'Imbalance':<12}\n")
    f.write("-"*100 + "\n")

    for cfg in configs:
        imbalance = cfg["max_util"] - cfg["min_util"]
        f.write(f"{cfg['name']:<30} {cfg['throughput']:>10.2f} t/s   {cfg['avg_util']:>6.1f}%      "
                f"{cfg['min_util']:>5.1f}% - {cfg['max_util']:>5.1f}%      {imbalance:>6.1f}%\n")

    f.write("="*100 + "\n")

print(f"\nFull results saved to: {sweep_dir}/results.txt")
print(f"Summary table saved to: {sweep_dir}/summary_table.txt")
print(f"Trace files saved to: {sweep_dir}/<config_name>/")
EOF

echo ""
echo "Sweep complete!"
echo "Results: ${SWEEP_DIR}/results.txt"
echo "Summary: ${SWEEP_DIR}/summary_table.txt"
