#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Helper script to run batch invariance comparison profiling
# This script runs profiling twice - once with batch invariance disabled,
# and once with it enabled, saving traces to separate directories for comparison.

set -e

# Configuration
OUTPUT_BASE_DIR="${PROFILE_OUTPUT_DIR:-./profiles}"
NUM_REQUESTS="${PROFILE_NUM_REQUESTS:-100}"
MAX_TOKENS="${PROFILE_MAX_TOKENS:-512}"
TP_SIZE="${PROFILE_TP_SIZE:-8}"
MODEL="${PROFILE_MODEL:-/data/users/bwasti/wearable_maverick_vllm/}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Batch Invariance Comparison Profiling"
echo "========================================================================"
echo "This script will profile vLLM twice:"
echo "  1. Baseline (VLLM_BATCH_INVARIANT=0)"
echo "  2. Batch Invariant (VLLM_BATCH_INVARIANT=1)"
echo ""
echo "Configuration:"
echo "  Model:        $MODEL"
echo "  TP Size:      $TP_SIZE"
echo "  Requests:     $NUM_REQUESTS"
echo "  Max Tokens:   $MAX_TOKENS"
echo "  Output Dir:   $OUTPUT_BASE_DIR"
echo "========================================================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR/baseline"
mkdir -p "$OUTPUT_BASE_DIR/batch_invariant"

# Run baseline profiling
echo -e "${GREEN}[1/2] Running baseline profiling (VLLM_BATCH_INVARIANT=0)...${NC}"
echo ""
VLLM_BATCH_INVARIANT=0 \
    VLLM_TORCH_PROFILER_DIR="$OUTPUT_BASE_DIR/baseline" \
    PROFILE_NUM_REQUESTS="$NUM_REQUESTS" \
    PROFILE_MAX_TOKENS="$MAX_TOKENS" \
    PROFILE_TP_SIZE="$TP_SIZE" \
    python profiling_scripts/profile_batch_invariance.py \
        --model "$MODEL"

echo ""
echo -e "${GREEN}Baseline profiling complete!${NC}"
echo ""

# Run batch invariant profiling
echo -e "${GREEN}[2/2] Running batch invariant profiling (VLLM_BATCH_INVARIANT=1)...${NC}"
echo ""
VLLM_BATCH_INVARIANT=1 \
    VLLM_TORCH_PROFILER_DIR="$OUTPUT_BASE_DIR/batch_invariant" \
    PROFILE_NUM_REQUESTS="$NUM_REQUESTS" \
    PROFILE_MAX_TOKENS="$MAX_TOKENS" \
    PROFILE_TP_SIZE="$TP_SIZE" \
    python profiling_scripts/profile_batch_invariance.py \
        --model "$MODEL"

echo ""
echo -e "${GREEN}Batch invariant profiling complete!${NC}"
echo ""

# Summary
echo "========================================================================"
echo -e "${BLUE}Profiling Complete!${NC}"
echo "========================================================================"
echo ""
echo "Traces saved to:"
echo "  Baseline:        $OUTPUT_BASE_DIR/baseline"
echo "  Batch Invariant: $OUTPUT_BASE_DIR/batch_invariant"
echo ""
echo "To view and compare the traces, run:"
echo -e "${YELLOW}  tensorboard --logdir_spec=baseline:$OUTPUT_BASE_DIR/baseline,batch_inv:$OUTPUT_BASE_DIR/batch_invariant${NC}"
echo ""
echo "Then open http://localhost:6006 in your browser"
echo "========================================================================"
