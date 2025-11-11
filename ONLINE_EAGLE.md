# Online EAGLE Training Plan

## Overview

This document outlines the plan for implementing online training for the EAGLE speculative decoding module in vLLM. The goal is to create a trainable PyTorch version of EAGLE that can improve over time while maintaining compatibility with the production inference pipeline.

## Background: How EAGLE Works

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is a speculative decoding method that drafts multiple future tokens in parallel to speed up inference.

### Architecture
- **Input**: Takes hidden states from the target model + input token IDs
- **FC Layer**: Concatenates input embeddings with target hidden states (`fc` layer in `llama_eagle.py:84`)
- **Decoder Layers**: Processes through multiple transformer layers (with first layer having `input_layernorm` disabled)
- **Output**: Produces both hidden states and logits for draft tokens

### Key Components
1. **Fused QKV Projections**: In `LlamaDecoderLayer`, attention uses fused `.qkv_proj` weights
2. **Gate-Up Projections**: MLP uses fused `.gate_up_proj` weights
3. **Shared Embeddings**: EAGLE shares `embed_tokens` and `lm_head` with target model (when shapes match)
4. **Weight Referencing**: Located at `vllm/v1/spec_decode/eagle.py` (proposer) and `vllm/model_executor/models/llama_eagle.py` (model)

## Training Objective

EAGLE is trained to predict:
1. **Hidden States**: Match the target model's hidden states for the next token (MSE loss)
2. **Logits**: Match the target model's logits distribution (KL divergence loss)

### Original EAGLE Training Formula
```
Loss = α * MSE(hidden_pred, hidden_target) + β * KL(logits_pred || logits_target)
```

Where:
- `α` ≈ 1.0 (hidden state weight)
- `β` ≈ 0.5 (logit weight)
- MSE operates on normalized hidden states
- KL divergence uses softmax probabilities with temperature=1.0

## Implementation Plan

### Phase 1: Create Trainable EAGLE Module

**File**: `vllm/v1/spec_decode/online_eagle.py`

**Purpose**: A vanilla PyTorch version that:
- References the same weights as the production EAGLE module
- Can be used for gradient-based training
- Does NOT interfere with inference

**Key Design Decisions**:
1. **Weight Sharing Strategy**:
   - Use `nn.Parameter` with the same underlying tensors from the inference EAGLE model
   - Set `requires_grad=True` on these parameters
   - Updates will be reflected in the original EAGLE module

2. **Parallelism Handling**:
   - Must handle tensor parallelism (TP) correctly
   - Fused weights need to be un-fused for vanilla PyTorch
   - Need utilities to convert between fused (vLLM) and unfused (PyTorch) formats

3. **Architecture**:
```python
class OnlineEagleLlamaForCausalLM(nn.Module):
    """Trainable version of EAGLE that references production weights"""

    def __init__(self, eagle_model: EagleLlamaForCausalLM):
        # Reference (don't copy) weights from eagle_model
        # Set requires_grad=True
        # Create unfused versions of QKV and gate_up projections

    def forward(self, input_ids, positions, hidden_states):
        # Vanilla PyTorch forward pass
        # Returns: (hidden_states, logits)

    def compute_loss(self, pred_hidden, target_hidden, pred_logits, target_logits):
        # MSE loss for hidden states
        # KL divergence for logits
        # Returns: total_loss, loss_dict
```

### Phase 2: Reference Buffer System

**File**: `vllm/v1/spec_decode/eagle_training_buffer.py`

**Purpose**: Accumulate reference data for periodic training

**Design**:
```python
class EagleTrainingBuffer:
    """Circular buffer for storing training references"""

    def __init__(self, buffer_size=1024, device='cuda'):
        self.buffer_size = buffer_size
        self.input_ids_buffer = []
        self.positions_buffer = []
        self.hidden_states_buffer = []  # From target model
        self.target_hidden_states_buffer = []  # Ground truth next token hidden
        self.target_logits_buffer = []  # Ground truth next token logits
        self.write_index = 0

    def add_sample(self, input_ids, positions, hidden_states,
                   target_hidden, target_logits):
        """Add a training sample to the buffer"""

    def should_train(self, train_every_n=256):
        """Check if buffer is ready for training"""

    def get_batch(self, batch_size=32):
        """Sample a mini-batch for training"""

    def clear(self):
        """Clear buffer after training"""
```

**Integration Points**:
- Hook into `EagleProposer.propose()` to collect samples
- Store data AFTER target model verification (only store accepted tokens)
- Periodic trigger based on buffer fill level

### Phase 3: Training Loop

**File**: `vllm/v1/spec_decode/eagle_trainer.py`

**Purpose**: Handles the actual training updates

**Design**:
```python
class EagleTrainer:
    """Handles periodic EAGLE training updates"""

    def __init__(self, online_eagle_model, optimizer_config):
        self.model = online_eagle_model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,  # Very small LR to avoid disrupting inference
            weight_decay=0.01
        )
        self.alpha = 1.0  # Hidden state loss weight
        self.beta = 0.5   # Logit loss weight

    def train_step(self, batch):
        """Single training step"""
        input_ids, positions, hidden_states, target_hidden, target_logits = batch

        # Forward pass
        pred_hidden, pred_logits = self.model(input_ids, positions, hidden_states)

        # Compute losses
        mse_loss = F.mse_loss(pred_hidden, target_hidden)
        kl_loss = F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction='batchmean'
        )

        total_loss = self.alpha * mse_loss + self.beta * kl_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def train_epoch(self, buffer, num_steps=10, batch_size=32):
        """Train for multiple steps on buffered data"""
        losses = []
        for _ in range(num_steps):
            batch = buffer.get_batch(batch_size)
            loss_dict = self.train_step(batch)
            losses.append(loss_dict)
        return losses
```

### Phase 4: Integration with EagleProposer

**Modifications to**: `vllm/v1/spec_decode/eagle.py`

**Changes**:
1. Add training buffer as optional component
2. Add data collection logic in `propose()`
3. Add periodic training trigger
4. Add thread-safe training execution (training in background)

```python
class EagleProposer:
    def __init__(self, ...):
        # Existing initialization
        ...

        # NEW: Training components (optional)
        self.enable_online_training = vllm_config.get('enable_eagle_training', False)
        if self.enable_online_training:
            self.training_buffer = EagleTrainingBuffer(buffer_size=1024)
            self.online_eagle = OnlineEagleLlamaForCausalLM(self.model)
            self.eagle_trainer = EagleTrainer(self.online_eagle)
            self.training_executor = ThreadPoolExecutor(max_workers=1)
            self.training_lock = threading.Lock()

    def propose(self, ...):
        # Existing proposal logic
        draft_tokens = ...

        # NEW: Collect training data (if enabled)
        if self.enable_online_training:
            self._collect_training_sample(
                target_hidden_states,
                next_token_ids,
                ...
            )

            # Trigger training if buffer is ready
            if self.training_buffer.should_train():
                self._trigger_async_training()

        return draft_tokens

    def _collect_training_sample(self, ...):
        """Store reference data for training"""

    def _trigger_async_training(self):
        """Launch training in background thread"""
        if self.training_lock.acquire(blocking=False):
            self.training_executor.submit(self._run_training)

    def _run_training(self):
        """Background training execution"""
        try:
            losses = self.trainer.train_epoch(self.training_buffer)
            logger.info(f"EAGLE training completed: {losses[-1]}")
            self.training_buffer.clear()
        finally:
            self.training_lock.release()
```

### Phase 5: Testing and Validation

**File**: `tests/v1/spec_decode/test_online_eagle.py`

**Test Coverage**:

1. **Weight Reference Test**:
```python
def test_weight_sharing():
    """Verify trainable EAGLE references same weights as inference EAGLE"""
    eagle_model = load_eagle_model()
    online_eagle = OnlineEagleLlamaForCausalLM(eagle_model)

    # Check weight sharing
    assert online_eagle.fc.weight.data_ptr() == eagle_model.model.fc.weight.data_ptr()

    # Check gradients work
    assert online_eagle.fc.weight.requires_grad == True
```

2. **Parity Test**:
```python
def test_forward_parity():
    """Verify vanilla version produces same outputs as vLLM version"""
    eagle_model = load_eagle_model()
    online_eagle = OnlineEagleLlamaForCausalLM(eagle_model)

    # Create test inputs
    input_ids = torch.randint(0, 1000, (4, 10))
    positions = torch.arange(10).unsqueeze(0).repeat(4, 1)
    hidden_states = torch.randn(4, 10, 4096)

    # Disable dropout for deterministic comparison
    eagle_model.eval()
    online_eagle.eval()

    with torch.no_grad():
        vllm_hidden, vllm_logits = eagle_model(input_ids, positions, hidden_states)
        vanilla_hidden, vanilla_logits = online_eagle(input_ids, positions, hidden_states)

    torch.testing.assert_close(vllm_hidden, vanilla_hidden, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(vllm_logits, vanilla_logits, rtol=1e-4, atol=1e-4)
```

3. **Parallelism Test**:
```python
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tensor_parallel_correctness(tp_size):
    """Verify training works correctly with tensor parallelism"""
    # Test unfused weight conversion
    # Test gradient synchronization
    # Test weight updates
```

4. **Training Test**:
```python
def test_training_reduces_loss():
    """Verify training actually reduces loss"""
    online_eagle = create_online_eagle()
    trainer = EagleTrainer(online_eagle)

    # Create synthetic training data
    buffer = create_synthetic_buffer()

    # Train for a few steps
    initial_loss = trainer.train_step(buffer.get_batch(32))['total_loss']
    for _ in range(10):
        trainer.train_step(buffer.get_batch(32))
    final_loss = trainer.train_step(buffer.get_batch(32))['total_loss']

    assert final_loss < initial_loss
```

5. **Integration Test**:
```python
def test_online_training_integration():
    """End-to-end test of online training in proposer"""
    proposer = create_proposer_with_training_enabled()

    # Run inference multiple times
    for _ in range(300):  # Enough to trigger training
        proposer.propose(...)

    # Verify training was triggered
    assert proposer.training_buffer.write_index > 0
    # Verify model was updated (check some weights changed)
```

### Phase 6: Utilities and Weight Management

**File**: `vllm/v1/spec_decode/eagle_weight_utils.py`

**Purpose**: Handle weight conversions between vLLM and vanilla PyTorch

```python
def unfuse_qkv_weights(fused_qkv_weight, num_heads, num_kv_heads, head_dim):
    """Convert fused QKV weight to separate Q, K, V weights"""
    # Split the fused weight
    # Handle different num_heads vs num_kv_heads (GQA)

def unfuse_gate_up_weights(fused_gate_up_weight):
    """Convert fused gate_up weight to separate gate and up weights"""

def fuse_qkv_weights(q_weight, k_weight, v_weight):
    """Convert separate Q, K, V weights to fused QKV weight"""

def gather_tensor_parallel_weights(weight, tp_dim):
    """Gather weights across tensor parallel ranks"""

def scatter_tensor_parallel_weights(weight, tp_dim, world_size):
    """Scatter weights to tensor parallel ranks"""
```

## Critical Considerations

### 1. Thread Safety
- Training runs in background thread
- Use locks to prevent concurrent training
- Ensure weight updates are atomic from inference perspective

### 2. Performance Impact
- Training should NOT block inference
- Buffer collection overhead should be minimal (< 1% latency increase)
- Consider training frequency vs. inference throughput tradeoff

### 3. Numerical Stability
- Use gradient clipping (max_norm=1.0)
- Very small learning rate (1e-5 or smaller)
- Monitor loss for divergence
- Add checkpoint/rollback mechanism if loss explodes

### 4. Tensor Parallelism
- Must handle weight gathering/scattering correctly
- Fused weights complicate gradient computation
- Need to test with TP=1, 2, 4, 8

### 5. Memory Management
- Training buffer should not grow unbounded
- Clear buffer after training
- Consider mixed precision training (FP16/BF16) to save memory

### 6. Monitoring and Debugging
- Log training losses periodically
- Track acceptance rate over time (should improve)
- Add metrics for buffer fill rate, training frequency
- Checkpoint weights periodically

## Configuration

Add new config options:

```python
# In vllm_config or speculative_config
enable_eagle_training: bool = False
eagle_training_buffer_size: int = 1024
eagle_training_batch_size: int = 32
eagle_training_steps: int = 10
eagle_training_lr: float = 1e-5
eagle_training_frequency: int = 256  # Train every N samples
eagle_alpha: float = 1.0  # Hidden state loss weight
eagle_beta: float = 0.5   # Logit loss weight
```

## Success Metrics

1. **Correctness**: Vanilla EAGLE exactly matches vLLM EAGLE (bit-for-bit)
2. **Training**: Loss decreases over training steps
3. **Improvement**: Acceptance rate increases over time
4. **Performance**: < 1% inference latency overhead when training is disabled
5. **Stability**: No divergence or NaN losses over extended training

## Future Enhancements

1. **Adaptive Training**: Adjust training frequency based on acceptance rate
2. **Per-Request Training**: Train on specific types of requests (e.g., long conversations)
3. **Multi-Model Support**: Extend to other EAGLE variants (EAGLE3, DeepSeek EAGLE)
4. **Distributed Training**: Use model parallelism for larger EAGLE models
5. **Curriculum Learning**: Gradually increase draft token count during training

## Timeline Estimate

- **Phase 1** (Trainable Module): 2-3 days
- **Phase 2** (Buffer System): 1 day
- **Phase 3** (Training Loop): 1-2 days
- **Phase 4** (Integration): 2-3 days
- **Phase 5** (Testing): 3-4 days
- **Phase 6** (Utilities): 1-2 days

**Total**: ~2 weeks for full implementation and testing

## Open Questions

1. Should we train only on accepted draft tokens, or all draft tokens (including rejected)?
2. How to handle multi-token EAGLE (tree attention mode)?
3. Should we save/load training checkpoints?
4. What's the optimal buffer size vs. training frequency tradeoff?
5. Should we support multiple optimizers (Adam, SGD, etc.)?
6. How to handle warmup / learning rate scheduling?

## References

- EAGLE Paper: https://arxiv.org/abs/2401.15077
- vLLM EAGLE Implementation: `vllm/v1/spec_decode/eagle.py`
- EAGLE Model: `vllm/model_executor/models/llama_eagle.py`
- Original EAGLE Training Code: https://github.com/SafeAILab/EAGLE
