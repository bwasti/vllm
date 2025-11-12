# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Training configuration for online EAGLE training.

This module defines the configuration options for training EAGLE models
on-the-fly during inference.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for online EAGLE training.

    This config controls the training behavior for EAGLE draft models
    during online inference.
    """

    # --- Optimization Settings ---
    learning_rate: float = 1e-4
    """Learning rate for optimizer."""

    weight_decay: float = 0.01
    """Weight decay (L2 regularization) coefficient."""

    adam_beta1: float = 0.9
    """Adam beta1 parameter (first moment decay)."""

    adam_beta2: float = 0.999
    """Adam beta2 parameter (second moment decay)."""

    adam_epsilon: float = 1e-8
    """Adam epsilon for numerical stability."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""

    # --- Learning Rate Scheduler ---
    use_lr_scheduler: bool = True
    """Whether to use learning rate scheduler."""

    warmup_steps: int = 100
    """Number of warmup steps for LR scheduler."""

    lr_scheduler_type: str = "cosine"
    """Type of LR scheduler: 'cosine', 'linear', or 'constant'."""

    # --- Training Batch Configuration ---
    batch_size: int = 2
    """Batch size for training (number of sequences per batch)."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps before optimizer step."""

    max_seq_len: int = 256
    """Maximum sequence length for training samples."""

    # --- Data Collection ---
    buffer_size: int = 100
    """Size of the training data buffer (number of samples)."""

    min_samples_for_training: int = 8
    """Minimum number of samples required before starting training."""

    sample_collection_prob: float = 1.0
    """Probability of collecting a sample for training (0.0 to 1.0)."""

    # --- Training Schedule ---
    train_interval_requests: int = 10
    """Train after every N inference requests."""

    train_interval_samples: int | None = None
    """Train after collecting N new samples (overrides train_interval_requests)."""

    training_steps_per_trigger: int = 1
    """Number of training steps to perform per training trigger."""

    # --- Async Training Settings ---
    async_training: bool = True
    """Whether to run training asynchronously (in background)."""

    max_concurrent_trainings: int = 1
    """Maximum number of concurrent training jobs (if async_training=True)."""

    # --- Checkpoint Settings ---
    checkpoint_interval_steps: int = 100
    """Save checkpoint every N training steps."""

    checkpoint_dir: str | None = None
    """Directory to save checkpoints. If None, checkpointing is disabled."""

    keep_last_n_checkpoints: int = 3
    """Number of recent checkpoints to keep (delete older ones)."""

    # --- Logging and Monitoring ---
    log_interval_steps: int = 10
    """Log training metrics every N steps."""

    enable_tensorboard: bool = False
    """Enable TensorBoard logging."""

    tensorboard_dir: str | None = None
    """Directory for TensorBoard logs."""

    # --- Validation Settings ---
    validation_interval_steps: int | None = None
    """Run validation every N training steps. If None, no validation."""

    validation_samples: int = 100
    """Number of samples to use for validation."""

    # --- Resource Management ---
    training_device: str = "cuda"
    """Device for training ('cuda' or 'cpu')."""

    pin_memory: bool = True
    """Pin memory for DataLoader (faster GPU transfer)."""

    num_workers: int = 2
    """Number of DataLoader worker processes."""

    # --- Advanced Settings ---
    compile_model: bool = False
    """Use torch.compile for training (experimental)."""

    use_mixed_precision: bool = False
    """Use automatic mixed precision (AMP) for training."""

    grad_checkpoint: bool = False
    """Use gradient checkpointing to save memory."""

    # --- Debug Settings ---
    debug_mode: bool = False
    """Enable debug mode with additional logging and checks."""

    validate_gradients: bool = False
    """Validate gradients for NaN/Inf after each step."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )

        if self.buffer_size < self.min_samples_for_training:
            raise ValueError(
                f"buffer_size ({self.buffer_size}) must be >= "
                f"min_samples_for_training ({self.min_samples_for_training})"
            )

        if not 0.0 <= self.sample_collection_prob <= 1.0:
            raise ValueError(
                f"sample_collection_prob must be in [0.0, 1.0], "
                f"got {self.sample_collection_prob}"
            )

        if self.lr_scheduler_type not in ["cosine", "linear", "constant"]:
            raise ValueError(
                f"lr_scheduler_type must be 'cosine', 'linear', or 'constant', "
                f"got {self.lr_scheduler_type}"
            )

        if self.enable_tensorboard and self.tensorboard_dir is None:
            raise ValueError("tensorboard_dir must be set when enable_tensorboard=True")

        if self.checkpoint_dir is not None and self.checkpoint_interval_steps <= 0:
            raise ValueError(
                "checkpoint_interval_steps must be positive when "
                f"checkpoint_dir is set, got {self.checkpoint_interval_steps}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create TrainingConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            TrainingConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert TrainingConfig to a dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def effective_batch_size(self) -> int:
        """Compute effective batch size accounting for gradient accumulation.

        Returns:
            Effective batch size = batch_size * gradient_accumulation_steps
        """
        return self.batch_size * self.gradient_accumulation_steps

    def steps_per_epoch(self, dataset_size: int) -> int:
        """Compute number of training steps per epoch.

        Args:
            dataset_size: Number of samples in the dataset

        Returns:
            Number of steps per epoch
        """
        return max(1, dataset_size // self.effective_batch_size())

    def total_training_steps(self, num_epochs: int, dataset_size: int) -> int:
        """Compute total number of training steps.

        Args:
            num_epochs: Number of training epochs
            dataset_size: Number of samples in the dataset

        Returns:
            Total number of training steps
        """
        return num_epochs * self.steps_per_epoch(dataset_size)


@dataclass
class OnlineTrainingMetrics:
    """Metrics for monitoring online training progress.

    This class tracks various statistics about the training process.
    """

    # Training statistics
    total_training_steps: int = 0
    """Total number of training steps completed."""

    total_samples_collected: int = 0
    """Total number of training samples collected."""

    total_samples_used: int = 0
    """Total number of samples used for training (may count duplicates)."""

    current_loss: float = 0.0
    """Most recent training loss."""

    average_loss: float = 0.0
    """Running average of training loss."""

    current_lr: float = 0.0
    """Current learning rate."""

    grad_norm: float = 0.0
    """Most recent gradient norm."""

    # Performance statistics
    training_time_ms: float = 0.0
    """Time spent in most recent training step (milliseconds)."""

    samples_per_second: float = 0.0
    """Training throughput (samples/second)."""

    # Validation statistics
    validation_loss: float | None = None
    """Most recent validation loss."""

    validation_accuracy: float | None = None
    """Most recent validation accuracy (if applicable)."""

    # Buffer statistics
    buffer_size: int = 0
    """Current number of samples in buffer."""

    buffer_full_count: int = 0
    """Number of times buffer became full."""

    # Error tracking
    training_errors: int = 0
    """Number of training errors encountered."""

    last_error: str | None = None
    """Most recent error message."""

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging.

        Returns:
            Dictionary representation of metrics
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def update_from_training_step(
        self,
        loss: float,
        lr: float,
        grad_norm: float,
        batch_size: int,
        step_time_ms: float,
    ) -> None:
        """Update metrics after a training step.

        Args:
            loss: Training loss for this step
            lr: Current learning rate
            grad_norm: Gradient norm for this step
            batch_size: Number of samples in this batch
            step_time_ms: Time taken for this step (milliseconds)
        """
        self.total_training_steps += 1
        self.total_samples_used += batch_size
        self.current_loss = loss
        self.current_lr = lr
        self.grad_norm = grad_norm
        self.training_time_ms = step_time_ms

        # Update running average loss (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.total_training_steps == 1:
            self.average_loss = loss
        else:
            self.average_loss = alpha * loss + (1 - alpha) * self.average_loss

        # Compute throughput
        if step_time_ms > 0:
            self.samples_per_second = (batch_size * 1000.0) / step_time_ms
