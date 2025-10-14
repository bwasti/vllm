"""
Main RL Training Loop using GRPO with TorchTitan + vLLM.

This script orchestrates:
1. Inference/rollout generation using vLLM
2. Gradient computation and model updates using TorchTitan
3. Periodic weight synchronization between inference and training
"""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from .grpo import GRPOAlgorithm, RLBatch, compute_log_probs
from .torchtitan_trainer import create_trainer
from .vllm_inference import VLLMInferenceEngine, RolloutBuffer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Main RL training loop coordinator.
    """

    def __init__(
        self,
        config: dict,
        reward_fn: Callable[[str, str], float],
    ):
        """
        Initialize RL trainer.

        Args:
            config: Configuration dictionary
            reward_fn: Function that takes (prompt, completion) and returns reward
        """
        self.config = config
        self.reward_fn = reward_fn

        # Initialize vLLM for inference
        logger.info("Initializing vLLM inference engine...")
        self.inference_engine = VLLMInferenceEngine(
            model_path=config["model_path"],
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 2048),
            dtype=config.get("dtype", "auto"),
            runner=config.get("runner", "generate"),
        )

        # Get model for training
        logger.info("Setting up TorchTitan trainer...")

        # For v1 engine, we'll use collective_rpc with cloudpickle
        # to access the model on workers
        import cloudpickle

        def get_model_reference(model):
            return model

        serialized_get_model = cloudpickle.dumps(get_model_reference)
        models = self.inference_engine.llm.collective_rpc(
            serialized_get_model, args=()
        )

        # Use the first worker's model for training
        # Note: This shares memory with vLLM, so weight updates affect inference
        model = models[0] if models else None

        if model is None:
            raise RuntimeError("Failed to extract model from vLLM engine")

        self.trainer = create_trainer(
            model=model,
            config=config["training"],
            distributed=config.get("distributed", False),
        )

        # Initialize GRPO algorithm
        self.grpo = GRPOAlgorithm(
            kl_coef=config["grpo"].get("kl_coef", 0.1),
            clip_range=config["grpo"].get("clip_range", 0.2),
            normalize_advantages=config["grpo"].get("normalize_advantages", True),
        )

        # Training state
        self.global_step = 0
        self.rollout_buffer = RolloutBuffer()

    def generate_rollouts(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> RLBatch:
        """
        Generate rollouts using vLLM.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            RLBatch with rollout data
        """
        logger.info(f"Generating {len(prompts)} rollouts...")

        # Generate completions
        results = self.inference_engine.generate(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.get("top_p", 1.0),
            logprobs=1,
        )

        # Compute rewards
        completions = []
        log_probs_list = []
        rewards = []

        for prompt, (completion, log_probs) in zip(prompts, results):
            reward = self.reward_fn(prompt, completion)

            completions.append(completion)
            log_probs_list.append(log_probs)
            rewards.append(reward)

        # Create batch
        batch = RLBatch(
            prompts=prompts,
            responses=completions,
            rewards=torch.tensor(rewards),
            log_probs=log_probs_list,
        )

        logger.info(f"Mean reward: {batch.rewards.mean().item():.4f}")

        return batch

    def train_step(self, batch: RLBatch) -> dict:
        """
        Perform a single training step.

        Args:
            batch: Batch of rollout data

        Returns:
            Dictionary with training metrics
        """
        # TODO: Recompute log probs with current policy
        # For now, use stored log probs (assumes policy hasn't changed)
        new_log_probs = batch.log_probs

        # Compute loss
        loss, info = self.grpo.compute_loss(batch, new_log_probs)

        # Backward pass
        self.trainer.compute_loss_and_backward(loss)

        # Optimizer step
        train_metrics = self.trainer.step()

        # Combine metrics
        metrics = {**info, **train_metrics}

        return metrics

    def sync_weights_to_vllm(self):
        """
        Synchronize trained weights back to vLLM inference engine.

        This ensures the next rollout uses the updated policy.
        """
        logger.info("Syncing weights from trainer to vLLM...")

        # Get the updated state dict from the trainer
        updated_state_dict = self.trainer.get_state_dict()["model"]

        # Apply the weights to all vLLM workers using cloudpickle
        import cloudpickle

        def load_weights(model):
            model.load_state_dict(updated_state_dict, strict=False)
            return None

        serialized_load = cloudpickle.dumps(load_weights)
        self.inference_engine.llm.collective_rpc(serialized_load, args=())

        logger.info("Weight sync complete")

    def train(
        self,
        prompts: List[str],
        num_iterations: int = 100,
        rollouts_per_iteration: int = 32,
        train_steps_per_rollout: int = 1,
        save_every: int = 10,
        sync_weights_every: int = 1,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            prompts: Pool of prompts to sample from
            num_iterations: Number of training iterations
            rollouts_per_iteration: Number of rollouts to generate per iteration
            train_steps_per_rollout: Number of gradient steps per rollout batch
            save_every: Save checkpoint every N iterations
            sync_weights_every: Sync weights to vLLM every N iterations
            checkpoint_dir: Directory to save checkpoints
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting RL training...")
        logger.info(f"Total iterations: {num_iterations}")
        logger.info(f"Rollouts per iteration: {rollouts_per_iteration}")
        logger.info(f"Syncing weights every: {sync_weights_every} iterations")

        for iteration in tqdm(range(num_iterations), desc="Training"):
            # Sample prompts for this iteration
            import random
            sampled_prompts = random.choices(prompts, k=rollouts_per_iteration)

            # Generate rollouts with current policy
            batch = self.generate_rollouts(
                prompts=sampled_prompts,
                max_tokens=self.config.get("max_tokens", 128),
                temperature=self.config.get("temperature", 1.0),
            )

            # Training steps
            for step in range(train_steps_per_rollout):
                metrics = self.train_step(batch)

                self.global_step += 1

                # Log metrics
                if self.global_step % 10 == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={metrics['loss']:.4f}, "
                        f"reward={metrics['mean_reward']:.4f}, "
                        f"grad_norm={metrics['grad_norm']:.4f}"
                    )

            # Sync weights back to vLLM for next iteration
            if (iteration + 1) % sync_weights_every == 0:
                self.sync_weights_to_vllm()

            # Save checkpoint
            if checkpoint_dir and (iteration + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration + 1}.pt"
                self.trainer.save_checkpoint(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Save training state
                state_path = checkpoint_dir / f"training_state_iter_{iteration + 1}.json"
                with open(state_path, "w") as f:
                    json.dump({
                        "iteration": iteration + 1,
                        "global_step": self.global_step,
                        "metrics": metrics,
                    }, f, indent=2)

        logger.info("Training completed!")


def main():
    """
    Example usage of the RL trainer.
    """
    # Example configuration
    config = {
        "model_path": "meta-llama/Llama-3.2-1B",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "dtype": "float16",
        "use_v2": True,
        "enable_torchtitan": True,
        "max_tokens": 128,
        "temperature": 1.0,
        "top_p": 1.0,
        "training": {
            "learning_rate": 1e-6,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_steps": 10,
        },
        "grpo": {
            "kl_coef": 0.1,
            "clip_range": 0.2,
            "normalize_advantages": True,
        },
        "distributed": False,
    }

    # Example reward function
    def example_reward_fn(prompt: str, completion: str) -> float:
        """Simple reward based on completion length."""
        return float(len(completion.split()))

    # Example prompts
    prompts = [
        "Write a short story about a robot:",
        "Explain quantum computing:",
        "What is the meaning of life?",
        "Describe a beautiful sunset:",
    ]

    # Create trainer
    trainer = RLTrainer(config=config, reward_fn=example_reward_fn)

    # Run training
    trainer.train(
        prompts=prompts,
        num_iterations=100,
        rollouts_per_iteration=16,
        train_steps_per_rollout=1,
        save_every=10,
        checkpoint_dir="./checkpoints",
    )


if __name__ == "__main__":
    main()
