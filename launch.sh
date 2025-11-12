#!/bin/bash

# Launch script for vLLM server with EAGLE or Online EAGLE speculative decoding
# Usage: ./launch.sh [OPTIONS]
#
# Options:
#   --mode {eagle|online_eagle}  Speculative decoding mode (default: eagle)
#   --port PORT                  Server port (default: 8000)
#   --tp TP_SIZE                 Tensor parallel size (default: 1)
#   --host HOST                  Server host (default: 0.0.0.0)
#   --num-spec-tokens N          Number of speculative tokens (default: 4)
#   --training-config JSON       Training config for online_eagle (optional)
#   --help                       Show this help message

set -e  # Exit on error

# Default configuration
MODE="eagle"
PORT=8000
TP_SIZE=1
HOST="0.0.0.0"
NUM_SPEC_TOKENS=4
MODEL_PATH="/data/users/bwasti/wearable_maverick_vllm/"
DRAFT_PATH="/data/users/bwasti/wearable_maverick_vllm/draft/"
TRAINING_CONFIG=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --training-config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Launch script for vLLM server with EAGLE speculative decoding"
            echo ""
            echo "Usage: ./launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode {eagle|online_eagle}  Speculative decoding mode (default: eagle)"
            echo "  --port PORT                  Server port (default: 8000)"
            echo "  --tp TP_SIZE                 Tensor parallel size (default: 1)"
            echo "  --host HOST                  Server host (default: 0.0.0.0)"
            echo "  --num-spec-tokens N          Number of speculative tokens (default: 4)"
            echo "  --training-config JSON       Training config for online_eagle (optional)"
            echo "  --help                       Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./launch.sh --mode eagle"
            echo "  ./launch.sh --mode online_eagle --tp 2"
            echo "  ./launch.sh --mode online_eagle --training-config '{\"learning_rate\": 1e-4}'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "eagle" && "$MODE" != "online_eagle" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'eagle' or 'online_eagle'"
    exit 1
fi

# Validate paths exist
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [[ ! -d "$DRAFT_PATH" ]]; then
    echo "Error: Draft model path does not exist: $DRAFT_PATH"
    exit 1
fi

# Set up environment variables
export LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"
export VLLM_ATTENTION_BACKEND="FLASHINFER"

# Build speculative config
SPEC_CONFIG="{\"model\": \"$DRAFT_PATH\", \"method\": \"$MODE\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS"

# Add training config if provided and mode is online_eagle
if [[ "$MODE" == "online_eagle" && -n "$TRAINING_CONFIG" ]]; then
    SPEC_CONFIG="$SPEC_CONFIG, \"training_config\": $TRAINING_CONFIG"
fi

SPEC_CONFIG="$SPEC_CONFIG}"

# Print configuration
echo "========================================"
echo "vLLM EAGLE Server Configuration"
echo "========================================"
echo "Mode:                $MODE"
echo "Model Path:          $MODEL_PATH"
echo "Draft Path:          $DRAFT_PATH"
echo "Host:                $HOST"
echo "Port:                $PORT"
echo "Tensor Parallel:     $TP_SIZE"
echo "Spec Tokens:         $NUM_SPEC_TOKENS"
if [[ "$MODE" == "online_eagle" && -n "$TRAINING_CONFIG" ]]; then
    echo "Training Config:     $TRAINING_CONFIG"
fi
echo "========================================"
echo ""

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --speculative-config '$SPEC_CONFIG' \
    --tensor-parallel-size $TP_SIZE \
    --host $HOST \
    --port $PORT \
    --disable-log-requests \
    --kv-cache-dtype auto \
    --quantization compressed-tensors \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 128"

# Add online training flag if mode is online_eagle
if [[ "$MODE" == "online_eagle" ]]; then
    # Note: This flag doesn't exist yet, will be added in Phase 3
    # VLLM_CMD="$VLLM_CMD --enable-online-training"
    echo "Note: Online training mode will be enabled once implemented (Phase 3)"
fi

echo "Starting vLLM server..."
echo "Command: $VLLM_CMD"
echo ""

# Launch server
eval $VLLM_CMD &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_RETRIES=60
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "Server endpoints:"
        echo "  Health:  http://$HOST:$PORT/health"
        echo "  Models:  http://$HOST:$PORT/v1/models"
        echo "  Chat:    http://$HOST:$PORT/v1/chat/completions"
        echo ""
        echo "To stop the server, run: kill $SERVER_PID"
        echo ""

        # Wait for server process
        wait $SERVER_PID
        exit $?
    fi

    # Check if server process is still running
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "✗ Server process died unexpectedly"
        exit 1
    fi

    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
done

echo ""
echo "✗ Server failed to start within timeout"
kill $SERVER_PID 2>/dev/null || true
exit 1
