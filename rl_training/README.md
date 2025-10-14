# RL Training with GRPO: TorchTitan + vLLM

A clean, modular implementation of reinforcement learning for language models using:
- **vLLM** for efficient inference and rollout generation
- **TorchTitan** for distributed training and weight updates
- **GRPO** (Group Relative Policy Optimization) for advantage-based RL

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  RL Training Loop                    │
│                                                       │
│  ┌─────────────┐         ┌──────────────┐          │
│  │    vLLM     │────────▶│   Rollouts   │          │
│  │  Inference  │         │   + Rewards  │          │
│  └─────────────┘         └──────────────┘          │
│         │                        │                   │
│         │                        ▼                   │
│         │                ┌──────────────┐          │
│         │                │     GRPO     │          │
│         │                │  Algorithm   │          │
│         │                └──────────────┘          │
│         │                        │                   │
│         │                        ▼                   │
│         │                ┌──────────────┐          │
│         └────────────────│  TorchTitan  │          │
│         Weight Updates   │   Training   │          │
│                          └──────────────┘          │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
rl_training/
├── __init__.py              # Package exports
├── grpo.py                  # GRPO algorithm implementation
├── vllm_inference.py        # vLLM inference wrapper
├── torchtitan_trainer.py    # TorchTitan training wrapper
├── train.py                 # Main training loop
├── config.yaml              # Configuration file
└── README.md                # This file
```

## Quick Start

### Basic Usage

```python
from rl_training import RLTrainer

# Define your reward function
def reward_fn(prompt: str, completion: str) -> float:
    """Compute reward for a prompt-completion pair."""
    # Your reward logic here
    return score

# Configure training
config = {
    "model_path": "meta-llama/Llama-3.2-1B",
    "max_tokens": 128,
    "temperature": 1.0,
    "training": {
        "learning_rate": 1e-6,
        "max_grad_norm": 1.0,
    },
    "grpo": {
        "kl_coef": 0.1,
        "clip_range": 0.2,
    }
}

# Create trainer
trainer = RLTrainer(config=config, reward_fn=reward_fn)

# Run training
trainer.train(
    prompts=your_prompts,
    num_iterations=100,
    rollouts_per_iteration=16,
    checkpoint_dir="./checkpoints"
)
```

### Using Configuration File

```python
import yaml
from rl_training import RLTrainer

# Load config
with open("rl_training/config.yaml") as f:
    config = yaml.safe_load(f)

# Create and run trainer
trainer = RLTrainer(config=config, reward_fn=your_reward_fn)
trainer.train(prompts=prompts, **config)
```

### Command Line

```bash
cd rl_training
python train.py
```

## Components

### 1. GRPO Algorithm (`grpo.py`)

Implements Group Relative Policy Optimization:
- Advantage computation using group-relative rewards
- PPO-style clipped objective
- Optional KL divergence penalty
- Support for different baseline methods (mean, median)

**Key Classes:**
- `GRPOAlgorithm`: Main algorithm implementation
- `RLBatch`: Batch of rollout data
- `RLSample`: Single training sample

### 2. vLLM Inference (`vllm_inference.py`)

Efficient inference wrapper for rollout generation:
- Batched generation with log probability extraction
- Weight synchronization with training loop
- Rollout buffer for accumulating data

**Key Classes:**
- `VLLMInferenceEngine`: Inference wrapper
- `RolloutBuffer`: Buffer for storing rollouts

### 3. TorchTitan Training (`torchtitan_trainer.py`)

Training wrapper with distributed support:
- Gradient accumulation and clipping
- Learning rate warmup
- FSDP/DDP support for distributed training
- Checkpoint saving/loading

**Key Classes:**
- `TorchTitanTrainer`: Single-device trainer
- `DistributedTorchTitanTrainer`: Distributed trainer
- `create_trainer()`: Factory function

### 4. Training Loop (`train.py`)

Main orchestration:
- Rollout generation via vLLM
- Reward computation
- Training step with GRPO loss
- Weight synchronization every N steps
- Checkpoint saving

**Key Class:**
- `RLTrainer`: Main training coordinator

## Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model_path: "meta-llama/Llama-3.2-1B"
tensor_parallel_size: 1

# Generation settings
max_tokens: 128
temperature: 1.0

# Training settings
training:
  learning_rate: 1.0e-6
  weight_decay: 0.01
  max_grad_norm: 1.0

# GRPO settings
grpo:
  kl_coef: 0.1
  clip_range: 0.2
  normalize_advantages: true

# Loop settings
num_iterations: 100
rollouts_per_iteration: 16
train_steps_per_rollout: 1
save_every: 10
```

## Custom Reward Functions

The reward function is the key to training:

```python
def simple_length_reward(prompt: str, completion: str) -> float:
    """Reward based on response length."""
    return float(len(completion.split()))

def qa_accuracy_reward(prompt: str, completion: str) -> float:
    """Reward based on answer correctness."""
    correct_answer = extract_answer(prompt)
    predicted = extract_answer(completion)
    return 1.0 if predicted == correct_answer else 0.0

def human_preference_reward(prompt: str, completion: str) -> float:
    """Reward based on preference model."""
    return preference_model.score(prompt, completion)
```

## Distributed Training

To use distributed training:

```python
config = {
    "distributed": True,
    "use_fsdp": True,  # or False for DDP
    # ... other config
}

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

## Weight Synchronization

Weights are automatically synchronized between vLLM (inference) and TorchTitan (training):

```python
# After training step
updated_weights = trainer.get_state_dict()
inference_engine.update_weights(updated_weights["model"])
```

## Performance Tips

1. **Batch Size**: Increase `rollouts_per_iteration` for better GPU utilization
2. **Gradient Accumulation**: Set `train_steps_per_rollout > 1` for larger effective batch size
3. **KL Penalty**: Tune `kl_coef` to balance exploration vs. staying close to reference policy
4. **Tensor Parallelism**: Use `tensor_parallel_size > 1` for large models
5. **Mixed Precision**: Set `dtype: "float16"` or `"bfloat16"` for faster training

## Example Workflows

### 1. Simple Test Run

```python
from rl_training import RLTrainer

config = {"model_path": "meta-llama/Llama-3.2-1B", ...}
trainer = RLTrainer(config, lambda p, c: len(c.split()))
trainer.train(prompts=["Hello"], num_iterations=10)
```

### 2. Resume from Checkpoint

```python
trainer = RLTrainer(config, reward_fn)
trainer.trainer.load_checkpoint("checkpoints/checkpoint_iter_50.pt")
trainer.train(prompts=prompts, num_iterations=100)
```

### 3. Custom Training Loop

```python
trainer = RLTrainer(config, reward_fn)

for iteration in range(100):
    # Generate rollouts
    batch = trainer.generate_rollouts(prompts, max_tokens=128)

    # Custom reward processing
    batch.rewards = process_rewards(batch.rewards)

    # Training step
    metrics = trainer.train_step(batch)

    # Custom logging
    log_to_wandb(metrics)
```

## Troubleshooting

### Issue: Out of Memory
- Reduce `rollouts_per_iteration`
- Use smaller `max_tokens`
- Enable `use_fsdp: true` for distributed training

### Issue: Training is Slow
- Increase `tensor_parallel_size`
- Use `dtype: "float16"`
- Increase `rollouts_per_iteration` for better batching

### Issue: Rewards not Improving
- Check your reward function
- Tune `kl_coef` (lower = more exploration)
- Adjust `learning_rate`
- Increase `clip_range` for more aggressive updates

## Integration with Existing TorchTitan Code

This package integrates with the TorchTitan adapter added in the previous commit:
- Uses `vllm/model_executor/adapters/torchtitan_adapter.py`
- Compatible with batch-invariant attention layers
- Supports hybrid vLLM + TorchTitan workflows

## References

- **GRPO Paper**: Group Relative Policy Optimization
- **TorchTitan**: [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- **vLLM**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)

## License

Same as vLLM project license.
