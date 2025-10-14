"""
GRPO (Group Relative Policy Optimization) Algorithm.

This module implements GRPO, an advantage-based RL algorithm for language models.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RLSample:
    """A single RL training sample."""
    prompt: str
    response: str
    reward: float
    log_probs: torch.Tensor  # shape: (seq_len,)
    values: torch.Tensor = None  # Optional baseline values


@dataclass
class RLBatch:
    """A batch of RL samples grouped together."""
    prompts: List[str]
    responses: List[str]
    rewards: torch.Tensor  # shape: (batch_size,)
    log_probs: List[torch.Tensor]  # List of tensors with varying seq_len
    ref_log_probs: List[torch.Tensor] = None  # Reference model log probs


class GRPOAlgorithm:
    """
    GRPO: Group Relative Policy Optimization.

    Computes advantages using group-relative rewards and applies policy gradient updates.
    """

    def __init__(
        self,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        normalize_advantages: bool = True,
    ):
        """
        Args:
            kl_coef: Coefficient for KL divergence penalty
            clip_range: PPO-style clipping range for policy ratio
            normalize_advantages: Whether to normalize advantages within each batch
        """
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.normalize_advantages = normalize_advantages

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        baseline: str = "mean"
    ) -> torch.Tensor:
        """
        Compute advantages using group-relative rewards.

        Args:
            rewards: Tensor of shape (batch_size,)
            baseline: "mean" or "median" for baseline computation

        Returns:
            advantages: Tensor of shape (batch_size,)
        """
        if baseline == "mean":
            baseline_value = rewards.mean()
        elif baseline == "median":
            baseline_value = rewards.median()
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        advantages = rewards - baseline_value

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_kl_penalty(
        self,
        log_probs: List[torch.Tensor],
        ref_log_probs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference model.

        Args:
            log_probs: List of log probabilities from current policy
            ref_log_probs: List of log probabilities from reference model

        Returns:
            kl_penalty: Tensor of shape (batch_size,) containing KL divergence per sample
        """
        kl_divs = []
        for lp, ref_lp in zip(log_probs, ref_log_probs):
            # KL(ref || policy) = sum(exp(ref_lp) * (ref_lp - lp))
            kl = torch.sum(torch.exp(ref_lp) * (ref_lp - lp))
            kl_divs.append(kl)

        return torch.stack(kl_divs)

    def compute_loss(
        self,
        batch: RLBatch,
        new_log_probs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss for a batch.

        Args:
            batch: RLBatch containing samples and old log probs
            new_log_probs: List of log probs from current policy

        Returns:
            loss: Scalar loss tensor
            info: Dictionary with metrics for logging
        """
        # Compute advantages
        advantages = self.compute_advantages(batch.rewards)

        # Compute policy ratio for each sample
        policy_losses = []
        ratios = []

        for i, (old_lp, new_lp) in enumerate(zip(batch.log_probs, new_log_probs)):
            # Sum log probs across sequence
            old_lp_sum = old_lp.sum()
            new_lp_sum = new_lp.sum()

            # Compute ratio
            ratio = torch.exp(new_lp_sum - old_lp_sum)
            ratios.append(ratio)

            # PPO-style clipped objective
            adv = advantages[i]
            policy_loss_unclipped = ratio * adv
            policy_loss_clipped = torch.clamp(
                ratio, 1 - self.clip_range, 1 + self.clip_range
            ) * adv
            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)

            policy_losses.append(policy_loss)

        policy_loss = torch.stack(policy_losses).mean()

        # KL penalty (if reference log probs provided)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if batch.ref_log_probs is not None:
            kl_penalty = self.compute_kl_penalty(new_log_probs, batch.ref_log_probs)
            kl_loss = self.kl_coef * kl_penalty.mean()

        total_loss = policy_loss + kl_loss

        # Collect metrics
        info = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": batch.rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": torch.stack(ratios).mean().item(),
        }

        return total_loss, info


def compute_log_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities of tokens given logits.

    Args:
        logits: Tensor of shape (seq_len, vocab_size)
        tokens: Tensor of shape (seq_len,)

    Returns:
        log_probs: Tensor of shape (seq_len,)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return selected_log_probs
