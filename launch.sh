#!/bin/bash
# EAGLE launch script with easy toggle between online and default modes

# === MODE SELECTION ===
# Set to "true" for online EAGLE (with training), "false" for default EAGLE
USE_ONLINE_EAGLE=true

# === Simple Training Configuration (only used if USE_ONLINE_EAGLE=true) ===
#
# How it works:
# - Samples are added to buffer based on BUFFER_WRITE_INTERVAL (lower = more frequent writes)
# - When buffer reaches BUFFER_SIZE, train once using ALL samples in buffer
# - After training, buffer is cleared and we start over
# - Trains all EAGLE model parameters (fc, attention, mlp layers)
#
BUFFER_WRITE_INTERVAL=1           # Add 1 sample every N spec iterations (lower = buffer fills faster)
                                  # Implemented via sampling_rate = 1/N, so 2.5 means ~40% of samples
BUFFER_SIZE=16 # Number of samples to collect before training
LEARNING_RATE=1e-5                # Learning rate for training step
MSE_LOSS_WEIGHT=1               # Weight for MSE hidden state loss in dual-loss training
                                  # Combined loss = (MSE_WEIGHT * MSE_loss) + KL_loss
                                  # Lower values (e.g., 0.01-0.1) prioritize logit matching over hidden state alignment
                                  # Set to 0.0 to disable MSE loss entirely (KL-only training)

# Build speculative config based on mode
if [ "$USE_ONLINE_EAGLE" = true ]; then
  # Calculate sampling rate from interval (sampling_rate = 1 / interval)
  SAMPLING_RATE=$(python3 -c "print(1.0 / ${BUFFER_WRITE_INTERVAL})")

  SPEC_CONFIG="{
    \"method\": \"online_eagle\",
    \"model\": \"/home/bwasti/model_cache/draft/\",
    \"num_speculative_tokens\": 3,
    \"draft_tensor_parallel_size\": 8,
    \"max_model_len\": 4096,
    \"online_eagle_learning_rate\": ${LEARNING_RATE},
    \"online_eagle_feedback_buffer_size\": ${BUFFER_SIZE},
    \"online_eagle_feedback_sampling_rate\": ${SAMPLING_RATE},
    \"online_eagle_mse_loss_weight\": ${MSE_LOSS_WEIGHT}
  }"
else
  SPEC_CONFIG="{
    \"method\": \"eagle\",
    \"model\": \"/home/bwasti/model_cache/draft/\",
    \"num_speculative_tokens\": 3,
    \"draft_tensor_parallel_size\": 8,
    \"max_model_len\": 4096
  }"
fi

LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" \
vllm serve /home/bwasti/model_cache \
     --speculative-config "${SPEC_CONFIG}" \
     --max-model-len 4096 \
     --max-num-seqs 32 \
     --tensor-parallel-size 8 \
     --gpu-memory-utilization 0.8 \
     --host 0.0.0.0 \
     --port 12345
