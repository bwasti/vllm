# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Trainer for Online EAGLE.

Handles the actual training updates using MSE + KL divergence loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class EagleTrainer:
    """
    Handles periodic EAGLE training updates.

    Uses a combination of:
    - MSE loss for hidden state prediction
    - KL divergence for logit distribution matching
    """

    def __init__(
        self,
        online_eagle_model: nn.Module,
        learning_rate: float = 1e-5,
        mse_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        gradient_clip_norm: float = 1.0,
    ):
        """
        Args:
            online_eagle_model: The trainable OnlineEagleModel
            learning_rate: Learning rate for optimizer
            mse_loss_weight: Weight for MSE hidden state loss
            kl_loss_weight: Weight for KL divergence logit loss
            gradient_clip_norm: Max norm for gradient clipping
        """
        self.model = online_eagle_model
        self.mse_loss_weight = mse_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.gradient_clip_norm = gradient_clip_norm

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        # Track training stats
        self.train_step_count = 0
        self.total_loss_ema = 0.0
        self.mse_loss_ema = 0.0
        self.kl_loss_ema = 0.0

        logger.info(
            f"EagleTrainer initialized: lr={learning_rate}, "
            f"mse_weight={mse_loss_weight}, kl_weight={kl_loss_weight}"
        )

    def compute_loss(
        self,
        pred_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MSE + KL loss.

        Args:
            pred_hidden: Predicted hidden states from EAGLE [batch, hidden_size]
            target_hidden: Target hidden states from target model [batch, hidden_size]
            pred_logits: Predicted logits from EAGLE [batch, vocab_size]
            target_logits: Target logits from target model [batch, vocab_size]

        Returns:
            (total_loss, loss_dict)
        """
        # MSE loss for hidden states
        mse_loss = F.mse_loss(pred_hidden, target_hidden)

        # KL divergence for logits
        # Target logits -> soft labels (detached, no grad)
        target_probs = F.softmax(target_logits.detach(), dim=-1)
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)

        # KL(target || pred) = sum(target * log(target / pred))
        #                    = sum(target * (log(target) - log(pred)))
        # Using F.kl_div with log_target=False (default)
        kl_loss = F.kl_div(
            pred_log_probs, target_probs, reduction="batchmean", log_target=False
        )

        # Combined loss
        total_loss = self.mse_loss_weight * mse_loss + self.kl_loss_weight * kl_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return total_loss, loss_dict

    def train_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            input_ids: [batch_size] - Advanced token IDs
            positions: [batch_size] - Position embeddings
            hidden_states: [batch_size, hidden_size] - Target features at position t
            target_hidden: [batch_size, hidden_size] - Target features at position t+1
            target_logits: [batch_size, vocab_size] - Target logits at position t+1

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Forward pass
        pred_hidden, _ = self.model(input_ids, positions, hidden_states)

        # Compute logits from predicted hidden states
        pred_logits = self.model.compute_logits(pred_hidden)

        # Compute loss
        total_loss, loss_dict = self.compute_loss(
            pred_hidden, target_hidden, pred_logits, target_logits
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        loss_dict["grad_norm"] = grad_norm.item()

        # Optimizer step
        self.optimizer.step()

        # Update EMA stats
        alpha = 0.9  # EMA smoothing factor
        if self.train_step_count == 0:
            self.total_loss_ema = loss_dict["total_loss"]
            self.mse_loss_ema = loss_dict["mse_loss"]
            self.kl_loss_ema = loss_dict["kl_loss"]
        else:
            self.total_loss_ema = (
                alpha * self.total_loss_ema + (1 - alpha) * loss_dict["total_loss"]
            )
            self.mse_loss_ema = (
                alpha * self.mse_loss_ema + (1 - alpha) * loss_dict["mse_loss"]
            )
            self.kl_loss_ema = (
                alpha * self.kl_loss_ema + (1 - alpha) * loss_dict["kl_loss"]
            )

        self.train_step_count += 1

        return loss_dict

    def train_on_batch(self, batch: tuple) -> Dict[str, float]:
        """
        Train on a single batch.

        Args:
            batch: Tuple of (input_ids, positions, hidden_states, target_hidden, target_logits)

        Returns:
            Dictionary of loss values
        """
        input_ids, positions, hidden_states, target_hidden, target_logits = batch
        return self.train_step(
            input_ids, positions, hidden_states, target_hidden, target_logits
        )

    def train_epoch(
        self,
        buffer,
        num_steps: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Train for multiple steps on buffered data.

        Args:
            buffer: EagleTrainingBuffer with ready samples
            num_steps: Number of training steps (if None, use all samples once)
            batch_size: Batch size for training

        Returns:
            Dictionary with average loss values
        """
        if num_steps is None:
            # Train on full buffer once
            batch = buffer.get_all_samples()
            if batch is None:
                logger.warning("No samples in buffer, skipping training")
                return {}

            return self.train_on_batch(batch)

        # Train for multiple steps with mini-batches
        losses = []
        for _ in range(num_steps):
            batch = buffer.get_batch(batch_size)
            if batch is None:
                break

            loss_dict = self.train_on_batch(batch)
            losses.append(loss_dict)

        if not losses:
            return {}

        # Average losses
        avg_loss = {
            key: sum(d[key] for d in losses) / len(losses) for key in losses[0].keys()
        }

        return avg_loss

    def get_stats(self) -> Dict[str, float]:
        """
        Get training statistics.

        Returns:
            Dictionary with training stats
        """
        return {
            "train_steps": self.train_step_count,
            "total_loss_ema": self.total_loss_ema,
            "mse_loss_ema": self.mse_loss_ema,
            "kl_loss_ema": self.kl_loss_ema,
        }
