"""
RL Training with GRPO using TorchTitan + vLLM.

This package provides a clean interface for reinforcement learning training
of language models using:
- vLLM for efficient inference/rollout generation
- TorchTitan for distributed training
- GRPO (Group Relative Policy Optimization) algorithm
"""

from .grpo import GRPOAlgorithm, RLBatch, RLSample, compute_log_probs
from .torchtitan_trainer import TorchTitanTrainer, DistributedTorchTitanTrainer, create_trainer
from .vllm_inference import VLLMInferenceEngine, RolloutBuffer
from .train import RLTrainer

__all__ = [
    "GRPOAlgorithm",
    "RLBatch",
    "RLSample",
    "compute_log_probs",
    "TorchTitanTrainer",
    "DistributedTorchTitanTrainer",
    "create_trainer",
    "VLLMInferenceEngine",
    "RolloutBuffer",
    "RLTrainer",
]
