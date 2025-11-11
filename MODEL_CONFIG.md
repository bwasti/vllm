# Model Configuration for Online EAGLE Training

## Target Model (Llama 4 17x16 MoE)

**Path**: `/home/bwasti/model_cache/`

### Architecture
- **Model Type**: `Llama4ForConditionalGeneration`
- **Architecture**: 17-expert MoE with 16 active experts per token
- **Hidden Size**: 5120
- **Num Layers**: 48
- **Attention Heads**: 40
- **KV Heads**: 8 (GQA)
- **Head Dim**: 128
- **Vocab Size**: 202048
- **Intermediate Size**: 16384 (MLP), 8192 (base)
- **Num Local Experts**: 16
- **Num Experts Per Token**: 1
- **Tensor Parallel Size**: 8 (from launch.sh)

### Key Features
- **MoE Layers**: All 48 layers are MoE layers
- **No RoPE Layers**: Selective RoPE application (pattern: [1,1,1,0] repeating)
- **Attention**: Uses QK normalization, temperature tuning
- **Quantization**: FP8 compressed-tensors format
- **Dtype**: bfloat16

## Draft Model (EAGLE for Llama 4)

**Path**: `/home/bwasti/model_cache/draft/`

### Architecture
- **Model Type**: `EagleLlama4ForCausalLM`
- **Hidden Size**: 5120 (matches target)
- **Num Layers**: 3 (only 3 layers!)
- **Attention Heads**: 40
- **KV Heads**: 8 (GQA)
- **Head Dim**: 128
- **Vocab Size**: 202048 (matches target)
- **Intermediate Size**: 16384
- **Draft Vocab Size**: 202048

### Key Features
- **No MoE**: Draft model is dense (no MoE layers)
- **No RoPE Layers**: Pattern [1,1,1] - no RoPE in any layer
- **Shares**: embed_tokens and lm_head with target model
- **FC Layer**: Concatenates input embeddings with target hidden states

## Launch Configuration

From `launch.sh`:

```bash
# Online EAGLE Mode
USE_ONLINE_EAGLE=true

# Training Parameters
BUFFER_WRITE_INTERVAL=1           # Sample every spec iteration
BUFFER_SIZE=16                     # Train after 16 samples
LEARNING_RATE=1e-5                # Very small LR
MSE_LOSS_WEIGHT=1                 # Equal weight for MSE and KL

# Server Configuration
--tensor-parallel-size 8           # Target model uses TP=8
--draft-tensor-parallel-size 8     # Draft ALSO uses TP=8 (simpler!)
--num-speculative-tokens 3         # Generate 3 draft tokens
--max-model-len 4096
--max-num-seqs 32
--gpu-memory-utilization 0.8
```

## Memory Calculations

### Per-Sample Buffer Memory

For this specific model:
- `input_id`: 4 bytes (int32)
- `position`: 8 bytes (int64)
- `hidden_state`: 5120 * 2 = 10,240 bytes (fp16)
- `target_hidden`: 5120 * 2 = 10,240 bytes (fp16)
- `target_logits`: 202048 * 2 = 404,096 bytes (fp16)

**Total per sample**: ~424.5 KB

**For BUFFER_SIZE=16**: ~6.8 MB (very manageable!)

### Draft Model Size
- 3 layers × (5120 hidden × 40 heads × 128 head_dim) = relatively small
- No MoE = much smaller than target
- Can run on single GPU (TP=1)

## Important Considerations

### 1. Tensor Parallelism - Simplified!
- **Target**: TP=8 (model is huge, needs 8 GPUs)
- **Draft**: TP=8 (same sharding as target!)
- **Benefit**: Hidden states are already sharded the same way - no gathering needed!
- **Training**: Each rank trains its shard independently, weights stay in sync via TP communication

### 2. MoE vs Dense
- **Target**: MoE with expert routing
- **Draft**: Dense model (no MoE)
- **No Issue**: EAGLE just needs hidden states, doesn't care about MoE routing

### 3. No RoPE Layers
- **Both models**: Have `no_rope_layers` configuration
- **Draft**: All 3 layers have no RoPE ([1,1,1])
- **Implication**: Position embeddings might be handled differently
- **Need to verify**: How positions are used in forward pass

### 4. Hidden State Source
- **Target model**: 48 layers, we need hidden states from layer 47 (second-to-top)
- **Where**: In `gpu_model_runner.py:2586`, `sample_hidden_states = hidden_states[logits_indices]`
- **These are**: Already the hidden states from the final layer before lm_head

### 5. Tensor Parallel Training
- **Both models**: Use TP=8 with same sharding
- **Hidden states**: Already sharded correctly, shape is `[batch_size, 5120 // 8]` per rank
- **Training**: Each rank trains on its shard independently
- **Gradient sync**: vLLM's TP infrastructure handles this automatically
- **Buffer storage**: Each rank stores its own shard of hidden states and logits

## Integration Points for This Model

### 1. Collecting Hidden States (TP=8, Simplified!)

In `gpu_model_runner.py`, no gathering needed:

```python
# After line 2586-2587
sample_hidden_states = hidden_states[logits_indices]  # Already sharded!
logits = self.model.compute_logits(sample_hidden_states)  # Also sharded!

# NEW: Store directly - each rank stores its shard
if self.eagle_proposer and self.eagle_proposer.training_buffer:
    # No gathering needed! Each rank stores its own shard
    # Hidden states shape: [batch_size, 5120 // 8] per rank
    # Logits shape: [batch_size, 202048 // 8] per rank
    self.eagle_proposer.training_buffer.complete_samples(
        hidden=sample_hidden_states,  # Local shard
        logits=logits,  # Local shard
        ...
    )
```

### 2. Draft Model Location

EAGLE proposer loads draft from: `/home/bwasti/model_cache/draft/`
- This is an `EagleLlama4ForCausalLM` model
- Already has the FC layer and 3 decoder layers
- Ready to use for training

### 3. Speculative Decoding Config

From launch.sh, the config passed to vLLM:

```json
{
  "method": "online_eagle",
  "model": "/home/bwasti/model_cache/draft/",
  "num_speculative_tokens": 3,
  "draft_tensor_parallel_size": 8,
  "max_model_len": 4096,
  "online_eagle_learning_rate": 1e-5,
  "online_eagle_feedback_buffer_size": 16,
  "online_eagle_feedback_sampling_rate": 1.0,
  "online_eagle_mse_loss_weight": 1.0
}
```

## Testing Strategy

### 1. Verify Hidden State Shapes

```python
# Target hidden states per rank: [batch_size, 5120 // 8 = 640]
assert sample_hidden_states.shape[-1] == 640  # Per rank!

# EAGLE hidden states should match (per rank)
eagle_hidden, _ = eagle_model(input_ids, positions, hidden_states)
assert eagle_hidden.shape[-1] == 640  # Per rank!
```

### 2. Verify Logit Shapes

```python
# Both should produce [batch_size, 202048 // 8 = 25256] per rank
target_logits = target_model.compute_logits(sample_hidden_states)
eagle_logits = eagle_model.compute_logits(eagle_hidden)
assert target_logits.shape[-1] == 25256  # Per rank!
assert eagle_logits.shape[-1] == 25256  # Per rank!
```

### 3. Verify Weight Sharing

```python
# embed_tokens should be shared
assert eagle_model.model.embed_tokens is target_model.language_model.model.embed_tokens

# lm_head should be shared
assert eagle_model.lm_head is target_model.language_model.lm_head
```

### 4. Test with Small Buffer First

Start with `BUFFER_SIZE=16` (as in launch.sh):
- Very fast iteration
- Low memory overhead (~6.8 MB)
- Can verify training loop works before scaling up

## Next Steps

1. ✅ Understand model architecture
2. ⏳ Implement buffer with TP-aware gathering
3. ⏳ Implement online EAGLE training module
4. ⏳ Add integration hooks in gpu_model_runner.py
5. ⏳ Test with this specific model setup
6. ⏳ Monitor training metrics and acceptance rate

## Model-Specific Challenges

### Challenge 1: Tensor Parallel Training (SOLVED!)
- ✅ Both target and draft use TP=8 with same sharding
- ✅ Each rank trains independently on its shard
- ✅ vLLM's TP infrastructure handles gradient synchronization
- ✅ No manual gathering/scattering needed!

### Challenge 2: Large Vocab Size
- 202K vocab is larger than typical (vs 128K for Llama 3.1)
- Logits per sample PER RANK: 25256 * 2 bytes = ~50 KB (very manageable!)
- Total across 8 ranks: 404 KB (same as before)

### Challenge 3: Quantization Handling
- Target is FP8 quantized
- Draft is BF16
- Hidden states from target will be in FP8
- Need to ensure proper dtype casting when storing in buffer

## Launch Script Integration

The launch script already has online EAGLE configuration ready:
- ✅ Method: `online_eagle`
- ✅ Buffer size: 16 samples
- ✅ Learning rate: 1e-5
- ✅ Sampling rate: 1.0 (every sample)
- ✅ MSE weight: 1.0

We just need to implement the backend to support these config options!
