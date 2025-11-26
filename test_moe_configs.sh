#!/bin/bash
# Test different MoE configurations to address load imbalance

MODEL="/data/users/bwasti/wearable_maverick_vllm"
DRAFT="/data/users/bwasti/eagle-llama-3.1-8b"
COMMON_ARGS="--model $MODEL --draft-model $DRAFT --num-requests 30 --enable-profiling"

echo "=========================================="
echo "Testing MoE Configurations for Load Balance"
echo "=========================================="
echo ""

# Baseline (current - no special MoE flags)
echo "1. BASELINE (TP=8, no expert parallelism)"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --output-trace traces_baseline"
echo ""

# Test 1: Enable Expert Parallelism
echo "2. EXPERT PARALLELISM (--enable-expert-parallel)"
echo "   This uses expert parallelism instead of tensor parallelism for MoE layers"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --enable-expert-parallel --output-trace traces_ep"
echo ""

# Test 2: Enable EPLB (Expert Parallelism Load Balancing)
echo "3. EXPERT PARALLELISM + LOAD BALANCING (--enable-eplb)"
echo "   Dynamically rebalances experts across GPUs based on load"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --enable-expert-parallel --enable-eplb --output-trace traces_eplb"
echo ""

# Test 3: Different all2all backend - DeepEP High Throughput
echo "4. DeepEP HIGH THROUGHPUT BACKEND"
echo "   Uses optimized all2all communication for expert parallel"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --enable-expert-parallel --all2all-backend deepep_high_throughput --output-trace traces_deepep_ht"
echo ""

# Test 4: Different all2all backend - DeepEP Low Latency
echo "5. DeepEP LOW LATENCY BACKEND"
echo "   Uses low-latency all2all communication for expert parallel"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --enable-expert-parallel --all2all-backend deepep_low_latency --output-trace traces_deepep_ll"
echo ""

# Test 5: Try TP=4 (less ranks = potentially better balance)
echo "6. REDUCE TP SIZE (TP=4 instead of TP=8)"
echo "   Fewer ranks may have better load balance"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 4 --output-trace traces_tp4"
echo ""

# Test 6: Combination - TP=4 + EP
echo "7. TP=4 + EXPERT PARALLELISM"
echo "   Command: python benchmark_eagle.py $COMMON_ARGS --tp-size 4 --enable-expert-parallel --output-trace traces_tp4_ep"
echo ""

echo "=========================================="
echo "Quick Test Commands (pick one to start)"
echo "=========================================="
echo ""
echo "# Quick test - Expert Parallelism:"
echo "python benchmark_eagle.py $COMMON_ARGS --tp-size 8 --enable-expert-parallel --output-trace traces_ep"
echo ""
echo "# Quick test - TP=4:"
echo "python benchmark_eagle.py $COMMON_ARGS --tp-size 4 --output-trace traces_tp4"
echo ""
echo "# After each test, analyze with:"
echo "python quick_gap_summary.py --trace-dir <trace_dir>"
echo ""
