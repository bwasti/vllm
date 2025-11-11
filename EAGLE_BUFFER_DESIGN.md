# EAGLE Training Buffer: What Data to Store

## Understanding the EAGLE Training Flow

### What EAGLE Predicts
Looking at `llama_eagle.py:90-106`, EAGLE's forward pass:
```python
def forward(self, input_ids, positions, hidden_states):
    # 1. Get input embeddings for next token
    input_embeds = self.embed_tokens(input_ids)

    # 2. Concatenate with hidden states from target model
    hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    # 3. Process through decoder layers
    for layer in self.layers:
        hidden_states, residual = layer(positions, hidden_states, residual)
    hidden_states = hidden_states + residual

    # 4. Return hidden states (used twice - for next layer AND for logits)
    return hidden_states, hidden_states
```

### Ground Truth Sources

From `gpu_model_runner.py:2586-2587`:
```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

The target model produces:
1. **Hidden states** for ALL tokens in the sequence
2. **Logits** computed from sampled positions (last token in each sequence)

## What the Buffer Must Store

### For Each Training Sample

We need to capture data **BEFORE** EAGLE makes its prediction and **AFTER** the target model verifies:

#### INPUTS to EAGLE (what it sees):
1. **`input_ids`**: `[batch_size]` - The "next" token IDs that EAGLE will draft for
   - From `eagle.py:233`: `self.input_ids[last_token_indices] = next_token_ids`
   - These are the tokens EAGLE is trying to predict

2. **`positions`**: `[batch_size]` or `[3, batch_size]` (for M-RoPE)
   - From `eagle.py:317-318`: `positions = target_positions[last_token_indices]`
   - Position embeddings for RoPE

3. **`hidden_states`**: `[batch_size, hidden_size]` - Hidden states from target model
   - From `eagle.py:327`: `hidden_states = hidden_states[last_token_indices]`
   - This is what the target model produced for the **current** token
   - EAGLE uses this as context to predict the **next** token

#### GROUND TRUTH (what EAGLE should predict):

4. **`target_next_hidden_states`**: `[batch_size, hidden_size]`
   - **The hidden states (second-to-top layer features) the target model produces for position t+1**
   - This is NOT in the current code - we need to collect it!
   - When the target model processes `next_token_ids` (token at t+1), it produces hidden states
   - We want EAGLE's output hidden states to match these

5. **`target_next_logits`**: `[batch_size, vocab_size]`
   - **The logits the target model produces AT position t+1 (NOT t+2!)**
   - From `gpu_model_runner.py:2587`: `logits = self.model.compute_logits(sample_hidden_states)`
   - These logits are computed from the hidden states at position t+1
   - EAGLE predicts hidden states at t+1, passes through lm_head, should match these logits

### The Collection Challenge

**Problem**: We need to collect ground truth for token `t+1`, but during the forward pass for token `t`, we don't have it yet!

**Solution**: Delayed collection with buffering

```
Position:      t              t+1           t+2
              ↓               ↓             ↓
Features:    f_t    →    f_{t+1}    →   f_{t+2}
Tokens:      tok_t      tok_{t+1}     tok_{t+2}

Time t:
  - Target model processes tokens [..., t-1, t]
  - Target produces: hidden_states[t] = f_t, logits for predicting tok_{t+1}
  - EAGLE takes f_t and tok_{t+1} (advanced token) as input
  - EAGLE predicts: f_{t+1} (feature at next position)
  - Rejection sampling may accept tok_{t+1}

Time t+1:
  - Target model processes: [..., t, t+1]
  - Target produces: hidden_states[t+1] = f_{t+1}, logits for tok_{t+1}
  - NOW we have ground truth for EAGLE's prediction from time t!
  - Store training sample: {
      inputs: (tok_{t+1}, positions[t], f_t),
      targets: (f_{t+1}, logits_{t+1})
    }

Loss Computation:
  - Feature loss: MSE(EAGLE_predicted_f_{t+1}, target_f_{t+1})
  - Token loss: KL(lm_head(EAGLE_predicted_f_{t+1}), softmax(target_logits_{t+1}))
```

## Buffer Implementation

### Buffer Structure

```python
class EagleTrainingBuffer:
    """
    Stores training samples for online EAGLE training.
    Uses a two-stage approach:
    1. Pending buffer: Stores EAGLE inputs waiting for ground truth
    2. Ready buffer: Stores complete samples ready for training
    """

    def __init__(self, buffer_size=1024, device='cuda'):
        self.buffer_size = buffer_size
        self.device = device

        # Pending samples waiting for ground truth
        # Key: (req_id, position) -> stores EAGLE inputs
        self.pending_samples = {}

        # Ready samples for training
        self.input_ids_buffer = []           # [N, 1] tensor
        self.positions_buffer = []           # [N, 1] or [N, 3, 1] tensor
        self.hidden_states_buffer = []       # [N, hidden_size] tensor
        self.target_hidden_buffer = []       # [N, hidden_size] tensor
        self.target_logits_buffer = []       # [N, vocab_size] tensor

        self.write_index = 0

    def add_pending_sample(
        self,
        req_id: str,
        position: int,
        input_id: int,
        position_emb: torch.Tensor,
        hidden_state: torch.Tensor,
    ):
        """
        Store EAGLE inputs for a sample that doesn't have ground truth yet.
        Called during EAGLE's propose() step.
        """
        key = (req_id, position)
        self.pending_samples[key] = {
            'input_id': input_id,
            'position_emb': position_emb.cpu(),  # Save memory
            'hidden_state': hidden_state.cpu(),
        }

    def complete_samples(
        self,
        req_id: str,
        position: int,
        target_hidden: torch.Tensor,
        target_logits: torch.Tensor,
    ):
        """
        Complete a pending sample with ground truth from target model.
        Called after target model forward pass.
        """
        key = (req_id, position)
        if key not in self.pending_samples:
            return  # Sample was rejected or not tracked

        pending = self.pending_samples.pop(key)

        # Add to ready buffer
        if self.write_index < self.buffer_size:
            self.input_ids_buffer.append(pending['input_id'])
            self.positions_buffer.append(pending['position_emb'])
            self.hidden_states_buffer.append(pending['hidden_state'])
            self.target_hidden_buffer.append(target_hidden.cpu())
            self.target_logits_buffer.append(target_logits.cpu())
            self.write_index += 1
        else:
            # Circular buffer - overwrite oldest
            idx = self.write_index % self.buffer_size
            self.input_ids_buffer[idx] = pending['input_id']
            self.positions_buffer[idx] = pending['position_emb']
            self.hidden_states_buffer[idx] = pending['hidden_state']
            self.target_hidden_buffer[idx] = target_hidden.cpu()
            self.target_logits_buffer[idx] = target_logits.cpu()
            self.write_index += 1

    def should_train(self, min_samples=256):
        """Check if we have enough samples to start training"""
        ready_samples = min(self.write_index, self.buffer_size)
        return ready_samples >= min_samples

    def get_batch(self, batch_size=32):
        """Sample a random mini-batch for training"""
        ready_samples = min(self.write_index, self.buffer_size)
        indices = torch.randperm(ready_samples)[:batch_size]

        batch_input_ids = torch.tensor(
            [self.input_ids_buffer[i] for i in indices],
            device=self.device
        )
        batch_positions = torch.stack(
            [self.positions_buffer[i] for i in indices]
        ).to(self.device)
        batch_hidden_states = torch.stack(
            [self.hidden_states_buffer[i] for i in indices]
        ).to(self.device)
        batch_target_hidden = torch.stack(
            [self.target_hidden_buffer[i] for i in indices]
        ).to(self.device)
        batch_target_logits = torch.stack(
            [self.target_logits_buffer[i] for i in indices]
        ).to(self.device)

        return (
            batch_input_ids,
            batch_positions,
            batch_hidden_states,
            batch_target_hidden,
            batch_target_logits,
        )

    def cleanup_stale_pending(self, max_age_steps=100):
        """Remove pending samples that are too old (probably rejected)"""
        # Track which samples were added and when
        # Remove old ones that never got completed
        pass

    def clear(self):
        """Clear ready buffer after training (keep pending samples)"""
        self.input_ids_buffer.clear()
        self.positions_buffer.clear()
        self.hidden_states_buffer.clear()
        self.target_hidden_buffer.clear()
        self.target_logits_buffer.clear()
        self.write_index = 0
```

## Integration Points

### 1. During EAGLE Proposal (`eagle.py:propose`)

After EAGLE makes predictions but before returning:

```python
def propose(self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, ...):
    # ... existing EAGLE proposal logic ...

    draft_token_ids = ...  # EAGLE's predictions

    # NEW: Store pending samples for training
    if self.training_buffer is not None:
        # For each request in batch
        for i in range(batch_size):
            req_id = self.current_batch_req_ids[i]
            position = target_positions[last_token_indices[i]].item()

            self.training_buffer.add_pending_sample(
                req_id=req_id,
                position=position,
                input_id=next_token_ids[i].item(),
                position_emb=target_positions[last_token_indices[i]],
                hidden_state=target_hidden_states[last_token_indices[i]],
            )

    return draft_token_ids
```

### 2. After Target Model Forward (`gpu_model_runner.py`)

After computing logits from hidden states:

```python
# In execute_model, after line 2587
logits = self.model.compute_logits(sample_hidden_states)

# NEW: Complete training samples with ground truth
if self.eagle_proposer and self.eagle_proposer.training_buffer:
    # For each request that just got processed
    for i in range(batch_size):
        req_id = self.input_batch.req_ids[i]
        position = current_positions[i].item()

        self.eagle_proposer.training_buffer.complete_samples(
            req_id=req_id,
            position=position,
            target_hidden=sample_hidden_states[i],
            target_logits=logits[i],
        )
```

### 3. Only Store Accepted Tokens

We should only create training samples for tokens that were **accepted** by rejection sampling:

```python
# After rejection sampling in gpu_model_runner.py
sampler_output = self.rejection_sampler(...)

# For each accepted token
for req_idx, accepted_tokens in enumerate(sampler_output.sampled_token_ids):
    if len(accepted_tokens) > 0:  # At least one token accepted
        # Complete the training sample
        self.training_buffer.complete_samples(...)
```

## Memory Considerations

### Memory Usage per Sample

For a model with `hidden_size=4096` and `vocab_size=128000`:

- `input_id`: 4 bytes (int32)
- `position`: 8 bytes (int64) or 24 bytes (M-RoPE)
- `hidden_state`: 4096 * 2 = 8,192 bytes (fp16)
- `target_hidden`: 4096 * 2 = 8,192 bytes (fp16)
- `target_logits`: 128000 * 2 = 256,000 bytes (fp16)

**Total per sample**: ~272 KB

For buffer_size=1024: **~279 MB**

### Optimizations

1. **Store on CPU**: Move completed samples to CPU memory
2. **Quantize logits**: Use fp16 or even int8 for logits (less critical than hidden states)
3. **Top-K logits only**: Store only top 1000 logits instead of full vocab
4. **Lazy completion**: Only complete samples for requests that continue (not finished)

## Training Loss Computation

```python
def compute_loss(
    pred_hidden: torch.Tensor,     # [batch, hidden_size]
    target_hidden: torch.Tensor,   # [batch, hidden_size]
    pred_logits: torch.Tensor,     # [batch, vocab_size]
    target_logits: torch.Tensor,   # [batch, vocab_size]
    alpha=1.0,
    beta=0.5,
):
    # MSE loss for hidden states
    mse_loss = F.mse_loss(pred_hidden, target_hidden)

    # KL divergence for logits
    # Target logits -> soft labels (detached, no grad)
    target_probs = F.softmax(target_logits.detach(), dim=-1)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='batchmean')

    total_loss = alpha * mse_loss + beta * kl_loss

    return total_loss, {
        'mse_loss': mse_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item(),
    }
```

## Summary: Buffer Contents

**Stored per sample (at position t):**
1. ✅ `input_id` - The advanced token at t+1 (helps EAGLE resolve uncertainty)
2. ✅ `position` - Position embedding for position t (where EAGLE reads features from)
3. ✅ `hidden_state` - Target model's second-to-top-layer feature at position t: `f_t` (EAGLE's input)
4. ✅ `target_hidden` - Target model's feature at position t+1: `f_{t+1}` (EAGLE should predict this)
5. ✅ `target_logits` - Target model's logits at position t+1 (for predicting token at t+1, NOT t+2!)

**Index Clarification:**
```
Position:           t              t+1
                   ↓               ↓
Features:         f_t    →     f_{t+1}
Tokens:          tok_t        tok_{t+1}

EAGLE Input:  (f_t, tok_{t+1}, pos_t)
EAGLE Output: predicted_f_{t+1}

Losses:
- MSE(predicted_f_{t+1}, target_f_{t+1})
- KL(lm_head(predicted_f_{t+1}), softmax(logits_{t+1}))
```

**NOT stored:**
- ❌ `target_token_ids` - Not needed, we have the actual next token in step t+1
- ❌ Full sequence history - Only need current step's data
- ❌ Attention masks - EAGLE uses same positions as target
- ❌ Draft tokens - Not needed for training, only for evaluation
- ❌ Logits for t+2 - EAGLE predicts features at t+1, not two steps ahead

The key insight is that we're training EAGLE to **predict the target model's feature at the next position (t+1)** given:
- The feature at current position (t)
- The advanced token that will be at position t+1
- Position embeddings

This allows EAGLE to speculate what features the target model would produce, enabling speculative decoding.
