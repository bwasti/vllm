#!/usr/bin/env python3
"""
Simple example script to run RL training with GRPO.

Usage:
    python rl_training/example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_training import RLTrainer


def simple_reward_fn(prompt: str, completion: str) -> float:
    """
    Simple reward function based on completion length.

    Replace this with your actual reward logic:
    - Correctness checking
    - Human preference model
    - Task-specific metrics
    """
    # Reward longer responses (up to a point)
    word_count = len(completion.split())

    # Bonus for responses between 10-50 words
    if 10 <= word_count <= 50:
        return float(word_count) * 1.5
    else:
        return float(word_count)


def main():
    """Run RL training example."""

    # Configuration
    config = {
        # Model settings
        "model_path": "facebook/opt-125m",  # Small model for testing
        "tensor_parallel_size": 1,
        "max_model_len": 512,
        "dtype": "float16",
        "runner": "generate",  # Use v0 engine for stability

        # Generation settings
        "max_tokens": 64,
        "temperature": 0.8,
        "top_p": 0.95,

        # Training settings
        "training": {
            "learning_rate": 5e-6,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_steps": 5,
        },

        # GRPO settings
        "grpo": {
            "kl_coef": 0.05,
            "clip_range": 0.2,
            "normalize_advantages": True,
        },

        # Distributed
        "distributed": False,
    }

    # Example prompts
    prompts = [
        "Write a short poem about:",
        "Explain the concept of:",
        "What are the benefits of:",
        "Describe how to:",
        "Tell me about:",
    ]

    print("=" * 60)
    print("RL Training with GRPO Example")
    print("=" * 60)
    print(f"\nModel: {config['model_path']}")
    print(f"Prompts: {len(prompts)}")
    print(f"\nInitializing trainer...\n")

    # Create trainer
    trainer = RLTrainer(
        config=config,
        reward_fn=simple_reward_fn,
    )

    print("Starting training...\n")

    # Run training
    trainer.train(
        prompts=prompts,
        num_iterations=10,  # Small number for testing
        rollouts_per_iteration=4,  # Small batch for testing
        train_steps_per_rollout=1,
        save_every=5,
        checkpoint_dir="./rl_checkpoints",
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
