#!/bin/bash
# Wrapper script to run the Online EAGLE end-to-end test with proper environment setup

# Set LD_PRELOAD for fbcode
export LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"

# Default configuration (can be overridden)
export TARGET_MODEL="${TARGET_MODEL:-/home/bwasti/model_cache}"
export DRAFT_MODEL="${DRAFT_MODEL:-/home/bwasti/model_cache/draft/}"
export VLLM_PORT="${VLLM_PORT:-12345}"
export BUFFER_SIZE="${BUFFER_SIZE:-16}"
export BUFFER_WRITE_INTERVAL="${BUFFER_WRITE_INTERVAL:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MSE_LOSS_WEIGHT="${MSE_LOSS_WEIGHT:-1.0}"

echo "Running Online EAGLE E2E Test"
echo "=============================="
echo "Configuration:"
echo "  TARGET_MODEL: $TARGET_MODEL"
echo "  DRAFT_MODEL: $DRAFT_MODEL"
echo "  VLLM_PORT: $VLLM_PORT"
echo "  BUFFER_SIZE: $BUFFER_SIZE"
echo "  BUFFER_WRITE_INTERVAL: $BUFFER_WRITE_INTERVAL"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  MSE_LOSS_WEIGHT: $MSE_LOSS_WEIGHT"
echo ""

# Run the test
python3 test_online_eagle_e2e.py
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ E2E Test PASSED!"
else
    echo ""
    echo "❌ E2E Test FAILED!"
fi

exit $exit_code
