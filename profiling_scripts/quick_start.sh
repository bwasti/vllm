#!/bin/bash
# Quick start script - demonstrates the most common profiling workflows

echo "========================================================================"
echo "vLLM Profiling Quick Start"
echo "========================================================================"
echo ""
echo "This script demonstrates common profiling workflows."
echo "Edit the variables below to customize for your setup."
echo ""

# Configuration (adjust these as needed)
NUM_REQUESTS=20  # Small number for quick testing
MAX_TOKENS=128
TP_SIZE=1        # Start with 1 GPU for testing

echo "Current configuration:"
echo "  NUM_REQUESTS: $NUM_REQUESTS"
echo "  MAX_TOKENS: $MAX_TOKENS"
echo "  TP_SIZE: $TP_SIZE"
echo ""
echo "To use different settings, edit this script or set environment variables."
echo ""
echo "========================================================================"
echo "Available Profiling Options:"
echo "========================================================================"
echo ""
echo "1. Profile EAGLE with Chrome Trace (Perfetto-compatible)"
echo "   Command: python profiling_scripts/profile_eagle_chrome_trace.py"
echo "   Output:  eagle_profile.json"
echo ""
echo "2. Profile Batch Invariance with Chrome Trace"
echo "   Command: python profiling_scripts/profile_batch_invariance_chrome_trace.py"
echo "   Output:  batch_invariance_profile.json"
echo ""
echo "3. Profile EAGLE with TensorBoard"
echo "   Command: python profiling_scripts/profile_eagle_llama4.py"
echo "   Output:  ./vllm_profile_eagle/ (directory)"
echo ""
echo "4. Profile Batch Invariance with TensorBoard"
echo "   Command: python profiling_scripts/profile_batch_invariance.py"
echo "   Output:  ./vllm_profile_batch_invariance/ (directory)"
echo ""
echo "========================================================================"
echo ""

# Prompt user
read -p "Would you like to run a quick test profile? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Run the scripts manually when ready."
    exit 0
fi

echo ""
echo "Select profiling option:"
echo "  1) EAGLE Chrome Trace (recommended for testing)"
echo "  2) Batch Invariance Chrome Trace"
echo "  3) EAGLE TensorBoard"
echo "  4) Batch Invariance TensorBoard"
echo ""
read -p "Enter option (1-4): " -n 1 -r option
echo ""
echo ""

case $option in
    1)
        echo "Running: EAGLE Chrome Trace Profile"
        PROFILE_NUM_REQUESTS=$NUM_REQUESTS \
        PROFILE_MAX_TOKENS=$MAX_TOKENS \
        PROFILE_TP_SIZE=$TP_SIZE \
        PROFILE_OUTPUT_FILE="eagle_profile_test.json" \
            python profiling_scripts/profile_eagle_chrome_trace.py

        echo ""
        echo "Profile complete! Upload eagle_profile_test.json to:"
        echo "  https://ui.perfetto.dev/"
        ;;
    2)
        echo "Running: Batch Invariance Chrome Trace Profile"
        VLLM_BATCH_INVARIANT=0 \
        PROFILE_NUM_REQUESTS=$NUM_REQUESTS \
        PROFILE_MAX_TOKENS=$MAX_TOKENS \
        PROFILE_TP_SIZE=$TP_SIZE \
        PROFILE_OUTPUT_FILE="batch_inv_test.json" \
            python profiling_scripts/profile_batch_invariance_chrome_trace.py

        echo ""
        echo "Profile complete! Upload batch_inv_test.json to:"
        echo "  https://ui.perfetto.dev/"
        ;;
    3)
        echo "Running: EAGLE TensorBoard Profile"
        VLLM_TORCH_PROFILER_DIR="./test_profile_eagle" \
        PROFILE_NUM_REQUESTS=$NUM_REQUESTS \
        PROFILE_MAX_TOKENS=$MAX_TOKENS \
        PROFILE_TP_SIZE=$TP_SIZE \
            python profiling_scripts/profile_eagle_llama4.py

        echo ""
        echo "Profile complete! View with:"
        echo "  tensorboard --logdir ./test_profile_eagle"
        ;;
    4)
        echo "Running: Batch Invariance TensorBoard Profile"
        VLLM_BATCH_INVARIANT=0 \
        VLLM_TORCH_PROFILER_DIR="./test_profile_batch" \
        PROFILE_NUM_REQUESTS=$NUM_REQUESTS \
        PROFILE_MAX_TOKENS=$MAX_TOKENS \
        PROFILE_TP_SIZE=$TP_SIZE \
            python profiling_scripts/profile_batch_invariance.py

        echo ""
        echo "Profile complete! View with:"
        echo "  tensorboard --logdir ./test_profile_batch"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "For more options, see: profiling_scripts/README.md"
echo "========================================================================"
