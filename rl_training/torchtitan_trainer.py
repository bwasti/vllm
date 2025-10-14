"""
TorchTitan Training Wrapper for RL.

Provides interface for training model updates using TorchTitan.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW


class TorchTitanTrainer:
    """
    Wrapper for training model using TorchTitan-style distributed training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
    ):
        """
        Initialize TorchTitan trainer.

        Args:
            model: The model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for learning rate
        """
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.step_count = 0

    def compute_loss_and_backward(
        self,
        loss: torch.Tensor,
    ):
        """
        Compute gradients for the given loss.

        Args:
            loss: Scalar loss tensor
        """
        loss.backward()

    def step(self) -> dict:
        """
        Perform a single optimization step.

        Returns:
            Dictionary with training metrics
        """
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        # Update learning rate with warmup
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            lr_scale = (self.step_count + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_scale

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step_count += 1

        metrics = {
            "grad_norm": grad_norm.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "step": self.step_count,
        }

        return metrics

    def get_state_dict(self) -> dict:
        """
        Get model state dict for checkpointing.

        Returns:
            State dictionary
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step_count,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load model and optimizer state.

        Args:
            state_dict: State dictionary to load
        """
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_count = state_dict.get("step", 0)

    def save_checkpoint(self, path: str):
        """
        Save checkpoint to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save(self.get_state_dict(), path)

    def load_checkpoint(self, path: str):
        """
        Load checkpoint from disk.

        Args:
            path: Path to load checkpoint from
        """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class DistributedTorchTitanTrainer(TorchTitanTrainer):
    """
    Distributed version of TorchTitan trainer using FSDP/DDP.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        use_fsdp: bool = True,
    ):
        """
        Initialize distributed trainer.

        Args:
            model: The model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps
            use_fsdp: Whether to use FSDP (vs DDP)
        """
        # Initialize distributed if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.use_fsdp = use_fsdp
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Wrap model for distributed training
        if use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(model)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model)

        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
        )

    def all_reduce_metrics(self, metrics: dict) -> dict:
        """
        All-reduce metrics across ranks.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Averaged metrics across all ranks
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=torch.cuda.current_device())
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                metrics[key] = tensor.item()

        return metrics

    def step(self) -> dict:
        """
        Perform distributed optimization step.

        Returns:
            Dictionary with training metrics (averaged across ranks)
        """
        metrics = super().step()
        return self.all_reduce_metrics(metrics)


def create_trainer(
    model: torch.nn.Module,
    config: dict,
    distributed: bool = False,
) -> TorchTitanTrainer:
    """
    Factory function to create appropriate trainer.

    Args:
        model: Model to train
        config: Training configuration dictionary
        distributed: Whether to use distributed training

    Returns:
        TorchTitanTrainer instance
    """
    if distributed:
        return DistributedTorchTitanTrainer(
            model=model,
            learning_rate=config.get("learning_rate", 1e-6),
            weight_decay=config.get("weight_decay", 0.01),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            warmup_steps=config.get("warmup_steps", 0),
            use_fsdp=config.get("use_fsdp", True),
        )
    else:
        return TorchTitanTrainer(
            model=model,
            learning_rate=config.get("learning_rate", 1e-6),
            weight_decay=config.get("weight_decay", 0.01),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            warmup_steps=config.get("warmup_steps", 0),
        )
