# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EAGLE trainer for online training during inference.

This module provides the EagleTrainer class which manages:
- Training data buffer
- Optimizer and learning rate scheduler
- Training loop with gradient accumulation
- Checkpoint management
- Metrics tracking
- Async training coordination
"""

import asyncio
import random
import time
from collections import deque
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset

from vllm.logger import init_logger
from vllm.model_executor.models.trainable_eagle import TrainableEagleLlamaForCausalLM
from vllm.training.config import OnlineTrainingMetrics, TrainingConfig

logger = init_logger(__name__)


class TrainingSample:
    """A single training sample for EAGLE.

    EAGLE learns to predict the next token given:
    - input_ids: The draft tokens
    - positions: Position indices
    - hidden_states: Target model hidden states (from last layer)
    - labels: Ground truth next tokens
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.positions = positions
        self.hidden_states = hidden_states
        self.labels = labels

    def to(self, device: torch.device) -> "TrainingSample":
        """Move sample to device."""
        return TrainingSample(
            input_ids=self.input_ids.to(device),
            positions=self.positions.to(device),
            hidden_states=self.hidden_states.to(device),
            labels=self.labels.to(device),
        )


class TrainingBuffer(Dataset):
    """Circular buffer for storing training samples.

    This buffer stores a fixed number of recent samples and implements
    the PyTorch Dataset interface for DataLoader compatibility.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: deque[TrainingSample] = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def add_sample(self, sample: TrainingSample) -> None:
        """Add a sample to the buffer (thread-safe).

        Args:
            sample: Training sample to add
        """
        async with self.lock:
            self.buffer.append(sample)

    async def add_samples(self, samples: list[TrainingSample]) -> None:
        """Add multiple samples to the buffer (thread-safe).

        Args:
            samples: List of training samples to add
        """
        async with self.lock:
            for sample in samples:
                self.buffer.append(sample)

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.buffer[idx]

    def clear(self) -> None:
        """Clear all samples from buffer."""
        self.buffer.clear()

    def get_batch(self, batch_size: int) -> list[TrainingSample]:
        """Get a random batch of samples.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of training samples
        """
        if len(self.buffer) < batch_size:
            # Return all samples if buffer is smaller than batch size
            return list(self.buffer)

        # Random sampling with replacement
        return random.choices(list(self.buffer), k=batch_size)


def collate_training_samples(samples: list[TrainingSample]) -> dict[str, torch.Tensor]:
    """Collate training samples into batched tensors.

    Args:
        samples: List of training samples

    Returns:
        Dictionary with batched tensors
    """
    if not samples:
        raise ValueError("Cannot collate empty sample list")

    # Debug: Check tensor shapes and types
    logger.debug("Collating %d samples", len(samples))
    for i, s in enumerate(samples[:2]):  # Log first 2 samples
        input_shape = s.input_ids.shape if hasattr(s.input_ids, "shape") else "N/A"
        logger.debug(
            "Sample %d: input_ids type=%s, shape=%s",
            i,
            type(s.input_ids),
            input_shape,
        )
        pos_shape = s.positions.shape if hasattr(s.positions, "shape") else "N/A"
        logger.debug(
            "Sample %d: positions type=%s, shape=%s",
            i,
            type(s.positions),
            pos_shape,
        )
        hidden_shape = (
            s.hidden_states.shape if hasattr(s.hidden_states, "shape") else "N/A"
        )
        logger.debug(
            "Sample %d: hidden_states type=%s, shape=%s",
            i,
            type(s.hidden_states),
            hidden_shape,
        )
        label_shape = s.labels.shape if hasattr(s.labels, "shape") else "N/A"
        logger.debug(
            "Sample %d: labels type=%s, shape=%s",
            i,
            type(s.labels),
            label_shape,
        )

    # Stack all tensors - need to squeeze to ensure consistent dimensions
    try:
        input_ids = torch.stack([s.input_ids.squeeze() for s in samples])
        positions = torch.stack([s.positions.squeeze() for s in samples])
        hidden_states = torch.stack([s.hidden_states for s in samples])
        labels = torch.stack([s.labels.squeeze() for s in samples])
    except Exception as e:
        logger.error("Failed to stack tensors: %s", e)
        logger.error(
            "Tensor shapes: input_ids=%s",
            [s.input_ids.shape for s in samples[:3]],
        )
        logger.error(
            "Tensor shapes: positions=%s",
            [s.positions.shape for s in samples[:3]],
        )
        logger.error(
            "Tensor shapes: hidden_states=%s",
            [s.hidden_states.shape for s in samples[:3]],
        )
        logger.error(
            "Tensor shapes: labels=%s",
            [s.labels.shape for s in samples[:3]],
        )
        raise

    return {
        "input_ids": input_ids,
        "positions": positions,
        "hidden_states": hidden_states,
        "labels": labels,
    }


class EagleTrainer:
    """Trainer for online EAGLE training during inference.

    This trainer manages the entire training pipeline:
    - Collecting training samples from inference
    - Managing training data buffer
    - Running training steps with gradient accumulation
    - Checkpoint management
    - Metrics tracking
    - Async training coordination
    """

    def __init__(
        self,
        model: TrainableEagleLlamaForCausalLM,
        config: TrainingConfig,
    ):
        """Initialize EAGLE trainer.

        Args:
            model: Trainable EAGLE model
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.metrics = OnlineTrainingMetrics()

        # Training data buffer
        self.buffer = TrainingBuffer(max_size=config.buffer_size)

        # Device
        self.device = torch.device(config.training_device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = self._create_lr_scheduler()

        # Training state
        self.training_step = 0
        self.samples_collected = 0
        self.is_training = False
        self.training_lock = asyncio.Lock()

        # Checkpoint management
        self.checkpoint_dir = None
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized EagleTrainer with buffer_size=%d, lr=%e, batch_size=%d",
            config.buffer_size,
            config.learning_rate,
            config.batch_size,
        )

    def _create_lr_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler_type == "constant":
            # No scheduler - constant learning rate
            return None

        elif self.config.lr_scheduler_type == "linear":
            # Linear warmup + linear decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps,
            )
            return warmup_scheduler

        elif self.config.lr_scheduler_type == "cosine":
            # Linear warmup + cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps,
            )
            # Assume we'll train for a long time, use large T_max
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=self.config.learning_rate * 0.1
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps],
            )
            return scheduler

        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

    async def add_training_sample(self, sample: TrainingSample) -> None:
        """Add a training sample to the buffer.

        Args:
            sample: Training sample to add
        """
        # Sample with probability (for controlling collection rate)
        if random.random() > self.config.sample_collection_prob:
            return

        await self.buffer.add_sample(sample)
        self.samples_collected += 1
        self.metrics.total_samples_collected += 1
        self.metrics.buffer_size = len(self.buffer)

        # Check if buffer is full
        if len(self.buffer) >= self.config.buffer_size:
            self.metrics.buffer_full_count += 1

    def should_train(self) -> bool:
        """Check if training should be triggered.

        Returns:
            True if training should run, False otherwise
        """
        # Check if we have enough samples
        if len(self.buffer) < self.config.min_samples_for_training:
            return False

        # Check if already training (don't start concurrent trainings)
        if self.is_training and self.config.max_concurrent_trainings <= 1:
            return False

        # Check if we should train based on interval
        if self.config.train_interval_samples is not None:
            # Train every N samples collected
            return self.samples_collected >= self.config.train_interval_samples
        else:
            # Train every N requests (handled externally by caller)
            return True

    async def train_step(self) -> dict[str, float]:
        """Run a single training step with gradient accumulation.

        Returns:
            Dictionary with training metrics for this step
        """
        async with self.training_lock:
            self.is_training = True
            self.model.train()

            step_start = time.time()
            total_loss = 0.0
            total_samples = 0

            try:
                # Gradient accumulation loop
                for accum_step in range(self.config.gradient_accumulation_steps):
                    # Get batch from buffer
                    samples = self.buffer.get_batch(self.config.batch_size)
                    if not samples:
                        logger.warning("No samples in buffer for training")
                        break

                    # Collate and move to device
                    batch = collate_training_samples(samples)
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    loss, logits, hidden_states = self.model(
                        input_ids=batch["input_ids"],
                        positions=batch["positions"],
                        hidden_states=batch["hidden_states"],
                        labels=batch["labels"],
                    )

                    # Scale loss by accumulation steps
                    loss = loss / self.config.gradient_accumulation_steps

                    # Backward pass
                    grad_stats = self.model.backward_step(loss)

                    total_loss += loss.item() * self.config.gradient_accumulation_steps
                    total_samples += len(samples)

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # LR scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Compute metrics
                avg_loss = total_loss / max(1, self.config.gradient_accumulation_steps)
                current_lr = self.optimizer.param_groups[0]["lr"]
                grad_norm = grad_stats.get("grad_norm", 0.0)

                step_time_ms = (time.time() - step_start) * 1000

                # Update metrics
                self.metrics.update_from_training_step(
                    loss=avg_loss,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    batch_size=total_samples,
                    step_time_ms=step_time_ms,
                )

                self.training_step += 1

                # Reset samples collected counter
                if self.config.train_interval_samples is not None:
                    self.samples_collected = 0

                # Logging
                if self.training_step % self.config.log_interval_steps == 0:
                    logger.info(
                        "Training step %d: loss=%.4f, lr=%.2e, "
                        "grad_norm=%.4f, time=%.1fms, "
                        "throughput=%.1f samples/s",
                        self.training_step,
                        avg_loss,
                        current_lr,
                        grad_norm,
                        step_time_ms,
                        self.metrics.samples_per_second,
                    )

                # Checkpointing
                if (
                    self.checkpoint_dir is not None
                    and self.training_step % self.config.checkpoint_interval_steps == 0
                ):
                    await self.save_checkpoint()

                return {
                    "loss": avg_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                    "samples_per_second": self.metrics.samples_per_second,
                }

            except Exception as e:
                logger.exception("Training step failed: %s", e)
                self.metrics.training_errors += 1
                self.metrics.last_error = str(e)
                raise

            finally:
                self.is_training = False

    async def train_n_steps(self, n_steps: int) -> list[dict[str, float]]:
        """Run N training steps.

        Args:
            n_steps: Number of training steps to run

        Returns:
            List of metrics dictionaries for each step
        """
        metrics_list = []
        for _ in range(n_steps):
            try:
                step_metrics = await self.train_step()
                metrics_list.append(step_metrics)
            except Exception as e:
                logger.error("Training failed at step %d: %s", self.training_step, e)
                break
        return metrics_list

    async def train_async(self, n_steps: int = 1) -> asyncio.Task:
        """Run training asynchronously in the background.

        Args:
            n_steps: Number of training steps to run

        Returns:
            Async task handle
        """
        if self.config.async_training:
            task = asyncio.create_task(self.train_n_steps(n_steps))
            return task
        else:
            # Run synchronously
            await self.train_n_steps(n_steps)
            return None

    async def save_checkpoint(self, checkpoint_name: str | None = None) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_name: Name for checkpoint file. If None, uses step number.
        """
        if self.checkpoint_dir is None:
            logger.warning("Checkpoint directory not set, skipping checkpoint")
            return

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.training_step}.pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        self.model.save_checkpoint(str(checkpoint_path))

        logger.info("Saved checkpoint to %s", checkpoint_path)

        # Clean up old checkpoints
        if self.config.keep_last_n_checkpoints > 0:
            await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        if self.checkpoint_dir is None:
            return

        # Get all checkpoint files
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Remove old checkpoints
        num_to_remove = len(checkpoints) - self.config.keep_last_n_checkpoints
        for checkpoint in checkpoints[:num_to_remove]:
            checkpoint.unlink()
            logger.info("Removed old checkpoint: %s", checkpoint)

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.model.load_checkpoint(checkpoint_path)
        logger.info("Loaded checkpoint from %s", checkpoint_path)

    def get_metrics(self) -> dict:
        """Get current training metrics.

        Returns:
            Dictionary with current metrics
        """
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset training metrics."""
        self.metrics = OnlineTrainingMetrics()

    def clear_buffer(self) -> None:
        """Clear training data buffer."""
        self.buffer.clear()
        self.samples_collected = 0
        self.metrics.buffer_size = 0
        logger.info("Cleared training buffer")
