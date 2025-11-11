# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Training buffer for Online EAGLE.

This buffer implements a two-stage design:
1. Pending buffer: Stores EAGLE inputs waiting for ground truth
2. Ready buffer: Stores complete samples ready for training

The key insight is that at time t, EAGLE predicts features at t+1,
but we only get ground truth when the target model processes t+1.
So we need to match them up asynchronously.
"""

import torch
from typing import Dict, Tuple, Optional
from collections import defaultdict

from vllm.logger import init_logger

logger = init_logger(__name__)


class EagleTrainingBuffer:
    """
    Two-stage buffer for Online EAGLE training.

    Workflow:
    1. During EAGLE propose(): Store pending samples with inputs but no targets
    2. During target forward(): Complete pending samples with ground truth targets
    3. When buffer is full: Return batches for training
    """

    def __init__(
        self,
        buffer_size: int = 256,
        device: str = "cuda",
        sampling_rate: float = 1.0,
    ):
        """
        Args:
            buffer_size: Maximum number of samples to store
            device: Device to store tensors on
            sampling_rate: Fraction of samples to actually store (for controlling overhead)
        """
        self.buffer_size = buffer_size
        self.device = device
        self.sampling_rate = sampling_rate

        # Pending samples: key = (iteration, token_idx) -> sample dict
        self.pending_samples: Dict[Tuple[int, int], Dict] = {}

        # Ready samples (lists of tensors)
        self.input_ids_buffer = []
        self.positions_buffer = []
        self.hidden_states_buffer = []
        self.target_hidden_buffer = []
        self.target_logits_buffer = []

        self.write_index = 0
        self.sample_count = 0  # Total samples processed (for sampling rate)

        logger.info(
            f"EagleTrainingBuffer initialized: "
            f"buffer_size={buffer_size}, sampling_rate={sampling_rate}"
        )

    def should_collect_sample(self) -> bool:
        """
        Determine if we should collect this sample based on sampling rate.

        Returns:
            True if we should collect this sample
        """
        self.sample_count += 1

        # Simple deterministic sampling
        if self.sampling_rate >= 1.0:
            return True

        # Sample based on count
        return (self.sample_count % int(1.0 / self.sampling_rate)) == 0

    def add_pending_sample(
        self,
        iteration: int,
        token_idx: int,
        input_id: int,
        position_emb: torch.Tensor,
        hidden_state: torch.Tensor,
    ):
        """
        Store EAGLE inputs for a sample that doesn't have ground truth yet.

        Called during EAGLE's propose() step.

        Args:
            iteration: Iteration number
            token_idx: Token index in the batch
            input_id: The advanced token ID (input to EAGLE)
            position_emb: Position embeddings
            hidden_state: Target model's hidden states at position t
        """
        if not self.should_collect_sample():
            return

        key = (iteration, token_idx)

        # Store on CPU to save GPU memory
        self.pending_samples[key] = {
            "input_id": input_id,
            "position_emb": position_emb.detach().cpu(),
            "hidden_state": hidden_state.detach().cpu(),
        }

    def complete_samples(
        self,
        iteration: int,
        token_idx: int,
        target_hidden: torch.Tensor,
        target_logits: torch.Tensor,
    ):
        """
        Complete a pending sample with ground truth from target model.

        Called after target model forward pass produces hidden states and logits.

        Args:
            iteration: Iteration number (iteration when sample was created)
            token_idx: Token index
            target_hidden: Target's hidden states
            target_logits: Target's logits
        """
        key = (iteration, token_idx)

        if key not in self.pending_samples:
            # Sample wasn't collected (due to sampling rate) or already completed
            return

        pending = self.pending_samples.pop(key)

        # Add to ready buffer
        if self.write_index < self.buffer_size:
            # Still filling buffer
            self.input_ids_buffer.append(pending["input_id"])
            self.positions_buffer.append(pending["position_emb"])
            self.hidden_states_buffer.append(pending["hidden_state"])
            self.target_hidden_buffer.append(target_hidden.detach().cpu())
            self.target_logits_buffer.append(target_logits.detach().cpu())
            self.write_index += 1
        else:
            # Buffer full - overwrite oldest (circular buffer)
            idx = self.write_index % self.buffer_size
            self.input_ids_buffer[idx] = pending["input_id"]
            self.positions_buffer[idx] = pending["position_emb"]
            self.hidden_states_buffer[idx] = pending["hidden_state"]
            self.target_hidden_buffer[idx] = target_hidden.detach().cpu()
            self.target_logits_buffer[idx] = target_logits.detach().cpu()
            self.write_index += 1

    def should_train(self, min_samples: Optional[int] = None) -> bool:
        """
        Check if we have enough samples to start training.

        Args:
            min_samples: Minimum number of samples required (defaults to buffer_size)

        Returns:
            True if ready to train
        """
        if min_samples is None:
            min_samples = self.buffer_size

        ready_samples = min(self.write_index, self.buffer_size)
        return ready_samples >= min_samples

    def get_batch(self, batch_size: int = 32) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a random mini-batch for training.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (input_ids, positions, hidden_states, target_hidden, target_logits)
            or None if not enough samples
        """
        ready_samples = min(self.write_index, self.buffer_size)

        if ready_samples == 0:
            return None

        # Sample random indices
        actual_batch_size = min(batch_size, ready_samples)
        indices = torch.randperm(ready_samples)[:actual_batch_size]

        # Gather batch
        batch_input_ids = torch.tensor(
            [self.input_ids_buffer[i] for i in indices], device=self.device
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

    def get_all_samples(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get all samples in the buffer (for training on full buffer).

        Returns:
            Tuple of all samples or None if buffer is empty
        """
        ready_samples = min(self.write_index, self.buffer_size)
        if ready_samples == 0:
            return None

        return self.get_batch(batch_size=ready_samples)

    def cleanup_stale_pending(self, max_age_steps: int = 100):
        """
        Remove pending samples that are too old (probably rejected drafts).

        Args:
            max_age_steps: Maximum age in steps before removal
        """
        # TODO: Track sample age and remove old ones
        # For now, we just keep all pending samples
        # In practice, rejected drafts won't get completed and will stay pending
        pass

    def clear(self):
        """
        Clear ready buffer after training (keep pending samples).

        This is called after training completes to make room for new samples.
        """
        self.input_ids_buffer.clear()
        self.positions_buffer.clear()
        self.hidden_states_buffer.clear()
        self.target_hidden_buffer.clear()
        self.target_logits_buffer.clear()
        self.write_index = 0

        logger.info(
            f"Training buffer cleared. Pending samples: {len(self.pending_samples)}"
        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        ready_samples = min(self.write_index, self.buffer_size)
        return {
            "ready_samples": ready_samples,
            "pending_samples": len(self.pending_samples),
            "total_samples_processed": self.sample_count,
            "buffer_size": self.buffer_size,
            "fill_percentage": int(100 * ready_samples / self.buffer_size),
        }

    def __len__(self) -> int:
        """Return number of ready samples."""
        return min(self.write_index, self.buffer_size)
