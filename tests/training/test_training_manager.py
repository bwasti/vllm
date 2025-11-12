# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TrainingManager.

Tests the TrainingManager class that coordinates online EAGLE training.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from vllm.training.config import TrainingConfig
from vllm.training.training_manager import PLACEHOLDER_TOKEN_ID, TrainingManager


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig."""
    config = MagicMock()
    config.speculative_config = MagicMock()
    config.speculative_config.draft_model_config = MagicMock()
    config.speculative_config.draft_model_config.hf_config = MagicMock()
    config.model_config = MagicMock()
    config.model_config.get_num_layers = MagicMock(return_value=32)
    config.parallel_config = MagicMock()
    return config


@pytest.fixture
def training_config():
    """Create a minimal training config for testing."""
    return TrainingConfig(
        buffer_size=100,
        batch_size=4,
        min_samples_for_training=10,
        train_interval_requests=5,
        checkpoint_dir=None,  # Disable checkpointing for tests
    )


@pytest.fixture
def mock_inference_drafter():
    """Create a mock inference EAGLE model."""
    drafter = MagicMock()
    drafter.state_dict = MagicMock(return_value={})
    drafter.load_state_dict = MagicMock()
    return drafter


@pytest.fixture
def mock_spec_decode_metadata():
    """Create mock SpecDecodeMetadata."""
    metadata = MagicMock()
    metadata.num_draft_tokens = [3, 2]  # 2 requests with 3 and 2 draft tokens
    metadata.cu_num_draft_tokens = torch.tensor([3, 5], dtype=torch.int32)
    return metadata


class TestTrainingManagerInit:
    """Test TrainingManager initialization."""

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    def test_init_creates_components(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
    ):
        """Test that __init__ creates trainable model and trainer."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=0)

        # Create manager
        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Verify model was created
        mock_model_cls.assert_called_once()
        assert manager.trainable_model == mock_model

        # Verify model weights were copied
        mock_model.copy_weights_from_inference_model.assert_called_once_with(
            mock_inference_drafter
        )

        # Verify trainer was created
        mock_trainer_cls.assert_called_once()
        assert manager.trainer == mock_trainer

        # Verify initial state
        assert manager.enabled is True
        assert manager.total_requests_processed == 0
        assert manager.total_samples_collected == 0


class TestTrainingManagerDataCollection:
    """Test training data collection."""

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    @pytest.mark.asyncio
    async def test_collect_training_data_rejected_tokens(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
        mock_spec_decode_metadata,
    ):
        """Test that rejected tokens are collected."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=0)
        mock_trainer.add_training_sample = AsyncMock()
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Create test data
        target_token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        target_hidden_states = torch.randn(5, 128)  # 5 tokens, hidden_size=128

        # Draft tokens: [3 tokens for req0, 2 tokens for req1]
        draft_token_ids = torch.tensor([[10, 11, 12], [20, 21, -1]], dtype=torch.long)

        next_token_ids = torch.tensor([100, 200], dtype=torch.long)

        # Sampled tokens: req0 accepted 2/3, req1 accepted 0/2 (all rejected)
        sampled_token_ids = torch.tensor(
            [
                [10, 11, PLACEHOLDER_TOKEN_ID],  # req0: 2 accepted, 1 rejected
                [
                    PLACEHOLDER_TOKEN_ID,
                    PLACEHOLDER_TOKEN_ID,
                    -1,
                ],  # req1: all rejected
            ],
            dtype=torch.int32,
        )

        # Collect data
        await manager.collect_training_data(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            draft_token_ids=draft_token_ids,
            next_token_ids=next_token_ids,
            sampled_token_ids=sampled_token_ids,
            spec_decode_metadata=mock_spec_decode_metadata,
        )

        # Verify samples were collected for rejected tokens
        # req0: 1 rejected token (position 2)
        # req1: 2 rejected tokens (positions 0, 1)
        # Total: 3 samples
        assert mock_trainer.add_training_sample.call_count == 3

        # Verify acceptance/rejection stats
        assert manager.total_accepted_tokens == 2  # req0: 2 accepted
        assert manager.total_rejected_tokens == 3  # req0: 1, req1: 2

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    @pytest.mark.asyncio
    async def test_collect_training_data_no_rejected(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
        mock_spec_decode_metadata,
    ):
        """Test that no samples collected when all tokens accepted."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=0)
        mock_trainer.add_training_sample = AsyncMock()
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Create test data - all tokens accepted
        target_token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        target_hidden_states = torch.randn(5, 128)

        draft_token_ids = torch.tensor([[10, 11, 12], [20, 21, -1]], dtype=torch.long)
        next_token_ids = torch.tensor([100, 200], dtype=torch.long)

        # All tokens accepted (no PLACEHOLDER_TOKEN_ID)
        sampled_token_ids = torch.tensor(
            [[10, 11, 12], [20, 21, -1]], dtype=torch.int32
        )

        # Collect data
        await manager.collect_training_data(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            draft_token_ids=draft_token_ids,
            next_token_ids=next_token_ids,
            sampled_token_ids=sampled_token_ids,
            spec_decode_metadata=mock_spec_decode_metadata,
        )

        # No samples should be collected (all accepted)
        mock_trainer.add_training_sample.assert_not_called()
        assert manager.total_rejected_tokens == 0


class TestTrainingManagerTrainingTrigger:
    """Test training trigger logic."""

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    @pytest.mark.asyncio
    async def test_maybe_trigger_training_by_interval(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
    ):
        """Test training is triggered after train_interval_requests."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=20)  # Enough samples
        mock_trainer.should_train = MagicMock(return_value=True)
        mock_trainer.train_async = AsyncMock(return_value=None)
        mock_trainer.training_step = 0
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Trigger training after interval
        for i in range(training_config.train_interval_requests - 1):
            await manager.maybe_trigger_training()
            # Should not train yet
            mock_trainer.train_async.assert_not_called()

        # One more request should trigger training
        await manager.maybe_trigger_training()
        mock_trainer.train_async.assert_called_once()

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    @pytest.mark.asyncio
    async def test_maybe_trigger_training_not_ready(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
    ):
        """Test training not triggered when trainer not ready."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=5)  # Not enough samples
        mock_trainer.should_train = MagicMock(return_value=False)
        mock_trainer.train_async = AsyncMock()
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Call multiple times
        for _ in range(training_config.train_interval_requests * 2):
            await manager.maybe_trigger_training()

        # Should not train (not enough samples)
        mock_trainer.train_async.assert_not_called()


class TestTrainingManagerMetrics:
    """Test metrics tracking."""

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    def test_get_acceptance_rate(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
    ):
        """Test acceptance rate calculation."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=0)
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        # Initial state
        assert manager.get_acceptance_rate() == 0.0

        # Simulate some accepted/rejected tokens
        manager.total_accepted_tokens = 70
        manager.total_rejected_tokens = 30

        assert manager.get_acceptance_rate() == 0.7

    @patch("vllm.training.training_manager.TrainableEagleLlamaForCausalLM")
    @patch("vllm.training.training_manager.EagleTrainer")
    def test_get_metrics(
        self,
        mock_trainer_cls,
        mock_model_cls,
        mock_vllm_config,
        training_config,
        mock_inference_drafter,
    ):
        """Test metrics retrieval."""
        # Setup
        mock_trainer = MagicMock()
        mock_trainer.buffer = MagicMock()
        mock_trainer.buffer.__len__ = MagicMock(return_value=50)
        mock_trainer.get_metrics = MagicMock(
            return_value={"training_step": 10, "current_loss": 0.5}
        )
        mock_trainer_cls.return_value = mock_trainer

        manager = TrainingManager(
            mock_vllm_config, training_config, mock_inference_drafter
        )

        metrics = manager.get_metrics()

        # Verify metrics include both trainer and manager stats
        assert "training_step" in metrics
        assert "total_requests_processed" in metrics
        assert "acceptance_rate" in metrics
        assert "buffer_size" in metrics
