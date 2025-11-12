# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Training manager for online EAGLE training.

This module coordinates the entire online training pipeline:
- Collecting training data from speculative decoding
- Managing the trainable EAGLE model
- Triggering async training steps
- Synchronizing weights between trainable and inference models
- Tracking training metrics
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.trainable_eagle import TrainableEagleLlamaForCausalLM
from vllm.training.config import OnlineTrainingMetrics, TrainingConfig
from vllm.training.eagle_trainer import EagleTrainer, TrainingSample

if TYPE_CHECKING:
    from vllm.model_executor.models.llama_eagle import EagleLlamaForCausalLM
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

# Placeholder token ID used by rejection sampler to mark rejected tokens
PLACEHOLDER_TOKEN_ID = -1


class TrainingManager:
    """Manager for online EAGLE training during inference.

    This class coordinates training data collection from speculative decoding,
    manages the trainable EAGLE model, triggers async training, and
    synchronizes weights with the inference model.

    The training pipeline:
    1. Collect data from EAGLE propose + rejection sampling
    2. Filter for rejected tokens (high learning signal)
    3. Add samples to trainer buffer
    4. Trigger async training when conditions met
    5. Periodically sync weights to inference model
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        training_config: TrainingConfig,
        inference_drafter: "EagleLlamaForCausalLM",
    ):
        """Initialize training manager.

        Args:
            vllm_config: vLLM configuration
            training_config: Training configuration
            inference_drafter: The inference EAGLE model to sync weights to
        """
        self.vllm_config = vllm_config
        self.training_config = training_config
        self.inference_drafter = inference_drafter

        # Training state
        self.enabled = True
        self.total_requests_processed = 0
        self.total_samples_collected = 0
        self.total_rejected_tokens = 0
        self.total_accepted_tokens = 0

        # Async training state
        self.current_training_task: asyncio.Task | None = None
        self.last_weight_sync_step = 0

        # Initialize trainable model
        logger.info("Creating trainable EAGLE model for online training...")
        self.trainable_model = TrainableEagleLlamaForCausalLM(
            vllm_config=vllm_config,
            prefix="drafter",
        )

        # Copy initial weights from inference model
        logger.info("Initializing trainable model from inference model...")
        self.trainable_model.copy_weights_from_inference_model(self.inference_drafter)

        # Initialize trainer
        logger.info("Creating EAGLE trainer...")
        self.trainer = EagleTrainer(
            model=self.trainable_model,
            config=training_config,
        )

        # Metrics tracking
        self.metrics = OnlineTrainingMetrics()

        logger.info(
            "TrainingManager initialized with buffer_size=%d, "
            "train_interval_requests=%d",
            training_config.buffer_size,
            training_config.train_interval_requests,
        )

    async def collect_training_data(
        self,
        # Data from EAGLE propose
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        # Data from rejection sampling
        sampled_token_ids: torch.Tensor,
        spec_decode_metadata: "SpecDecodeMetadata",
    ) -> None:
        """Collect training data from speculative decoding outputs.

        This method extracts training samples from the EAGLE propose and
        rejection sampling outputs. We focus on rejected tokens since they
        provide high learning signal.

        Args:
            target_token_ids: Token IDs fed to EAGLE [num_tokens]
            target_positions: Position indices [num_tokens]
            target_hidden_states: Target model hidden states [num_tokens, hidden]
            draft_token_ids: Draft tokens proposed by EAGLE [batch_size, num_spec]
            next_token_ids: Ground truth next tokens [batch_size]
            sampled_token_ids: Final sampled tokens after rejection
                [batch_size, max_spec_len + 1], with -1 for rejected
            spec_decode_metadata: Metadata about speculative decoding
        """
        if not self.enabled:
            return

        try:
            batch_size = draft_token_ids.shape[0]

            # Process each request in the batch
            for req_idx in range(batch_size):
                # Get number of draft tokens for this request
                num_drafts = spec_decode_metadata.num_draft_tokens[req_idx]
                if num_drafts == 0:
                    continue

                # Get cumulative index to access flattened tensors
                start_idx = (
                    0
                    if req_idx == 0
                    else spec_decode_metadata.cu_num_draft_tokens[req_idx - 1].item()
                )

                # Get draft tokens and sampled results for this request
                req_draft_tokens = draft_token_ids[req_idx, :num_drafts]
                req_sampled_tokens = sampled_token_ids[req_idx, :num_drafts]

                # Identify rejected tokens (marked with PLACEHOLDER_TOKEN_ID)
                rejected_mask = req_sampled_tokens == PLACEHOLDER_TOKEN_ID

                num_rejected = rejected_mask.sum().item()
                num_accepted = num_drafts - num_rejected

                self.total_rejected_tokens += num_rejected
                self.total_accepted_tokens += num_accepted

                # Log acceptance rate
                acceptance_pct = 100.0 * num_accepted / num_drafts
                logger.info_once(
                    "EAGLE acceptance: accepted=%d/%d (%.1f%%), rejected=%d",
                    num_accepted,
                    num_drafts,
                    acceptance_pct,
                    num_rejected,
                )

                # Collect training samples from ALL tokens (both accepted and rejected)
                # This provides more training data and faster buffer filling
                # For accepted tokens: reinforce good predictions
                # For rejected tokens: learn from mistakes

                # For each draft token, create a training sample
                for local_idx in range(num_drafts):
                    # Global index in flattened tensors
                    global_idx = start_idx + local_idx

                    # Determine the label based on whether token was accepted
                    # If accepted: use the sampled token (what worked)
                    # If rejected: use the correct token that should have been predicted
                    if rejected_mask[local_idx]:
                        # Rejected: we want to learn the correct prediction
                        # Use the next token from target as the label
                        label_token = next_token_ids[req_idx].item()
                    else:
                        # Accepted: reinforce this prediction
                        label_token = req_draft_tokens[local_idx].item()

                    # Extract tensors for this sample
                    input_ids_slice = target_token_ids[global_idx : global_idx + 1]
                    positions_slice = target_positions[global_idx : global_idx + 1]
                    hidden_states_slice = target_hidden_states[
                        global_idx : global_idx + 1
                    ]
                    label_tensor = torch.tensor(
                        [label_token],
                        device=target_token_ids.device,
                        dtype=torch.long,
                    )

                    # Debug: Log shapes for first sample
                    if local_idx == 0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Creating TrainingSample: "
                            "input_ids shape=%s, "
                            "positions shape=%s, "
                            "hidden_states shape=%s, "
                            "labels shape=%s",
                            input_ids_slice.shape,
                            positions_slice.shape,
                            hidden_states_slice.shape,
                            label_tensor.shape,
                        )

                    sample = TrainingSample(
                        input_ids=input_ids_slice,
                        positions=positions_slice,
                        hidden_states=hidden_states_slice,
                        labels=label_tensor,
                    )

                    await self.trainer.add_training_sample(sample)
                    self.total_samples_collected += 1

                # Log once per worker
                logger.info_once(
                    "Collected %d training samples (buffer size: %d)",
                    num_drafts,
                    len(self.trainer.buffer),
                )

            # Update metrics
            self.metrics.total_samples_collected = self.total_samples_collected
            self.metrics.buffer_size = len(self.trainer.buffer)

        except Exception as e:
            logger.exception("Failed to collect training data: %s", e)
            self.metrics.training_errors += 1
            self.metrics.last_error = str(e)

    async def maybe_trigger_training(self) -> None:
        """Trigger training if conditions are met.

        Checks if we should start a training step based on:
        - Number of requests processed since last training
        - Buffer has enough samples
        - No concurrent training running (if max_concurrent_trainings=1)
        """
        if not self.enabled:
            return

        # Increment request counter
        self.total_requests_processed += 1

        # Check if we should train based on interval
        should_train_by_interval = (
            self.total_requests_processed % self.training_config.train_interval_requests
            == 0
        )

        if not should_train_by_interval:
            return

        # Check if trainer says we're ready
        if not self.trainer.should_train():
            logger.info_once(
                "Training conditions not met: buffer_size=%d, "
                "min_samples=%d, is_training=%s",
                len(self.trainer.buffer),
                self.training_config.min_samples_for_training,
                self.trainer.is_training,
            )
            return

        # Check if previous async training is still running
        if (
            self.current_training_task is not None
            and not self.current_training_task.done()
        ):
            logger.warning(
                "Previous training task still running, skipping this trigger"
            )
            return

        # Trigger training
        logger.info(
            "Triggering training: requests=%d, buffer_size=%d, acceptance_rate=%.2f%%",
            self.total_requests_processed,
            len(self.trainer.buffer),
            self.get_acceptance_rate() * 100,
        )

        try:
            # Run training steps
            n_steps = self.training_config.training_steps_per_trigger
            self.current_training_task = await self.trainer.train_async(n_steps=n_steps)

            # Check if we should sync weights
            current_step = self.trainer.training_step
            weight_sync_interval = self.training_config.checkpoint_interval_steps

            if current_step - self.last_weight_sync_step >= weight_sync_interval:
                await self.sync_weights_to_inference_model()
                self.last_weight_sync_step = current_step

        except Exception as e:
            logger.exception("Training trigger failed: %s", e)
            self.metrics.training_errors += 1
            self.metrics.last_error = str(e)

    async def sync_weights_to_inference_model(self) -> None:
        """Synchronize weights from trainable model to inference model.

        This copies the trained weights from the trainable EAGLE model to the
        inference EAGLE model used for actual speculative decoding.
        """
        try:
            logger.info(
                "Syncing weights to inference model (step %d)...",
                self.trainer.training_step,
            )
            self.trainable_model.copy_weights_to_inference_model(self.inference_drafter)
            logger.info("Weight sync complete")
        except Exception as e:
            logger.exception("Weight sync failed: %s", e)
            self.metrics.training_errors += 1
            self.metrics.last_error = str(e)

    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate.

        Returns:
            Acceptance rate as fraction (0.0 to 1.0)
        """
        total_tokens = self.total_accepted_tokens + self.total_rejected_tokens
        if total_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / total_tokens

    def get_metrics(self) -> dict:
        """Get current training metrics.

        Returns:
            Dictionary with training statistics
        """
        trainer_metrics = self.trainer.get_metrics()

        return {
            **trainer_metrics,
            "total_requests_processed": self.total_requests_processed,
            "total_samples_collected": self.total_samples_collected,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_rejected_tokens": self.total_rejected_tokens,
            "acceptance_rate": self.get_acceptance_rate(),
            "buffer_size": len(self.trainer.buffer),
            "enabled": self.enabled,
        }

    async def shutdown(self) -> None:
        """Shutdown training manager gracefully.

        Waits for pending training jobs, saves final checkpoint, and
        cleans up resources.
        """
        logger.info("Shutting down training manager...")

        # Wait for current training task to complete
        if (
            self.current_training_task is not None
            and not self.current_training_task.done()
        ):
            logger.info("Waiting for training task to complete...")
            try:
                await asyncio.wait_for(self.current_training_task, timeout=60)
            except asyncio.TimeoutError:
                logger.warning("Training task did not complete in time")

        # Sync final weights
        await self.sync_weights_to_inference_model()

        # Save final checkpoint if configured
        if self.trainer.checkpoint_dir is not None:
            logger.info("Saving final checkpoint...")
            await self.trainer.save_checkpoint(checkpoint_name="final.pt")

        logger.info("Training manager shutdown complete")
