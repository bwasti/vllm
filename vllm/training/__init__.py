# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online training infrastructure for vLLM.

This package provides components for training models on-the-fly during inference,
starting with EAGLE speculative decoding models.
"""

from vllm.training.config import OnlineTrainingMetrics, TrainingConfig
from vllm.training.eagle_trainer import EagleTrainer
from vllm.training.training_manager import TrainingManager

__all__ = [
    "TrainingConfig",
    "OnlineTrainingMetrics",
    "EagleTrainer",
    "TrainingManager",
]
