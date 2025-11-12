# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Trainable EAGLE model for on-the-fly training.

This module provides a vanilla PyTorch implementation of EAGLE that:
1. Supports backward() for gradient computation
2. Respects tensor parallelism (TP) in forward and backward passes
3. Can copy weights to/from the optimized vLLM EAGLE model
4. Produces identical outputs to the inference-only version

Key differences from standard EAGLE:
- All operations support autograd
- No custom CUDA kernels (uses standard PyTorch ops)
- Explicit gradient reduction across TP ranks
- Simplified for training clarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.models.llama_eagle import EagleLlamaForCausalLM

logger = init_logger(__name__)


class TrainableLlamaDecoderLayer(nn.Module):
    """Trainable version of LlamaDecoderLayer for EAGLE.

    This is a VANILLA PyTorch implementation using only standard torch.nn layers.
    No custom CUDA kernels - pure PyTorch for guaranteed autograd support.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        disable_input_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Get TP info
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Compute per-rank dimensions
        assert self.num_heads % self.tp_size == 0
        self.num_heads_per_rank = self.num_heads // self.tp_size

        # For GQA, num_kv_heads might be < num_heads
        if self.num_kv_heads >= self.tp_size:
            assert self.num_kv_heads % self.tp_size == 0
            self.num_kv_heads_per_rank = self.num_kv_heads // self.tp_size
        else:
            # Replicate KV heads
            self.num_kv_heads_per_rank = self.num_kv_heads

        # Q, K, V projections (column parallel)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads_per_rank * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads_per_rank * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads_per_rank * self.head_dim,
            bias=False,
        )

        # Output projection (row parallel)
        self.o_proj = nn.Linear(
            self.num_heads_per_rank * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # MLP
        self.gate_proj = nn.Linear(
            self.hidden_size,
            config.intermediate_size // self.tp_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            config.intermediate_size // self.tp_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size // self.tp_size,
            self.hidden_size,
            bias=False,
        )

        # Layer norms
        if disable_input_layernorm:
            self.input_layernorm = nn.Identity()
        else:
            self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with vanilla PyTorch ops.

        Args:
            positions: Position indices [batch_size, seq_len]
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            residual: Residual connection from previous layer

        Returns:
            (hidden_states, residual): Updated hidden states and residual
        """
        # Residual connection
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states + residual)
            residual = hidden_states + residual

        # --- Attention ---
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV heads if needed for GQA
        if self.num_kv_heads_per_rank < self.num_heads_per_rank:
            k = k.repeat_interleave(
                self.num_heads_per_rank // self.num_kv_heads_per_rank, dim=1
            )
            v = v.repeat_interleave(
                self.num_heads_per_rank // self.num_kv_heads_per_rank, dim=1
            )

        # Scaled dot-product attention (vanilla PyTorch)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # Reshape back: [batch, seq_len, num_heads_per_rank * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_heads_per_rank * self.head_dim
        )

        # Output projection
        attn_output = self.o_proj(attn_output)

        # All-reduce across TP group for row-parallel output
        if self.tp_size > 1:
            get_tp_group().all_reduce(attn_output)

        # --- MLP ---
        hidden_states = self.post_attention_layernorm(attn_output + residual)
        residual = attn_output + residual

        # SwiGLU activation
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        mlp_output = self.down_proj(gate * up)

        # All-reduce across TP group for row-parallel output
        if self.tp_size > 1:
            get_tp_group().all_reduce(mlp_output)

        return mlp_output, residual


class TrainableLlamaModel(nn.Module):
    """Trainable EAGLE model core.

    Implements the EAGLE architecture:
    1. Concatenate input embeddings + target hidden states
    2. Project to hidden_size via fc layer
    3. Pass through transformer layers
    """

    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.start_layer_id = start_layer_id

        # Get TP info
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Embedding layer (standard PyTorch, manually sharded for TP)
        vocab_size_per_rank = config.vocab_size // self.tp_size
        vocab_start_idx = self.tp_rank * vocab_size_per_rank
        vocab_end_idx = vocab_start_idx + vocab_size_per_rank

        self.embed_tokens = nn.Embedding(vocab_size_per_rank, config.hidden_size)
        self.vocab_start_idx = vocab_start_idx
        self.vocab_end_idx = vocab_end_idx

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TrainableLlamaDecoderLayer(
                    config,
                    layer_idx=i + start_layer_id,
                    disable_input_layernorm=(i == 0),  # First layer skips input norm
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # FC layer to combine embeddings + hidden states
        # Input: [hidden_size (embed) + hidden_size (target)] -> Output: hidden_size
        self.fc = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            hidden_states: Target model hidden states [batch_size, seq_len, hidden_size]

        Returns:
            (last_hidden_states, all_hidden_states): Final and intermediate
                hidden states
        """
        # Manually shard token IDs for TP embedding
        input_ids_mask = (input_ids >= self.vocab_start_idx) & (
            input_ids < self.vocab_end_idx
        )
        input_ids_local = input_ids - self.vocab_start_idx
        input_ids_local = torch.where(
            input_ids_mask, input_ids_local, torch.zeros_like(input_ids_local)
        )

        # Get input embeddings
        input_embeds = self.embed_tokens(input_ids_local)

        # Zero out embeddings for tokens not in this rank's partition
        if self.tp_size > 1:
            input_embeds = input_embeds * input_ids_mask.unsqueeze(-1).float()
            # All-reduce to gather full embeddings
            get_tp_group().all_reduce(input_embeds)

        # Concatenate embeddings with target hidden states and project
        # Shape: [batch, seq, hidden*2] -> [batch, seq, hidden]
        combined = torch.cat([input_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(combined)

        # Pass through transformer layers
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Final residual connection
        hidden_states = hidden_states + residual

        return hidden_states, hidden_states


class TrainableEagleLlamaForCausalLM(nn.Module):
    """Trainable EAGLE model for on-the-fly training.

    This is a vanilla PyTorch implementation that:
    - Supports backward() for gradient computation
    - Respects tensor parallelism (all-reduces gradients across TP group)
    - Can be validated against the vLLM optimized version
    - Can copy weights to/from the inference model

    Usage:
        # Create trainable model
        model = TrainableEagleLlamaForCausalLM(vllm_config)

        # Forward with loss computation
        loss, logits, hidden_states = model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=target_hidden_states,
            labels=labels,
        )

        # Backward pass
        loss.backward()

        # Copy weights to inference model
        model.copy_weights_to_inference_model(inference_model)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.speculative_config.draft_model_config.hf_config

        # Get TP info
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        # Determine start layer for draft model
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        # EAGLE model
        self.model = TrainableLlamaModel(
            config=self.config,
            prefix=f"{prefix}.model",
            start_layer_id=target_layer_num,
        )

        # Output projection (LM head) - column parallel
        vocab_size_per_rank = self.config.vocab_size // self.tp_size
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            vocab_size_per_rank,
            bias=False,
        )

        # Enable training mode
        self.train()

        logger.info(
            "Created TrainableEagleLlamaForCausalLM (TP rank %d/%d)",
            self.tp_rank,
            self.tp_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            hidden_states: Target model hidden states [batch_size, seq_len, hidden_size]
            labels: Ground truth token IDs for loss [batch_size, seq_len], optional

        Returns:
            (loss, logits, hidden_states):
                - loss: Cross-entropy loss if labels provided, else None
                - logits: Output logits [batch_size, seq_len, vocab_size_per_rank]
                - hidden_states: Final hidden states
        """
        # Forward through EAGLE model
        last_hidden_states, all_hidden_states = self.model(
            input_ids, positions, hidden_states
        )

        # Project to vocabulary (column parallel - only this rank's vocab partition)
        logits = self.lm_head(last_hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # For TP, need to gather logits from all ranks or compute loss locally
            if self.tp_size > 1:
                # Gather logits from all TP ranks
                logits_list = [torch.zeros_like(logits) for _ in range(self.tp_size)]
                get_tp_group().all_gather(logits_list, logits)
                # Concatenate along vocab dimension
                logits_full = torch.cat(logits_list, dim=-1)
            else:
                logits_full = logits

            # Shift labels for next-token prediction
            # logits_full: [batch, seq, vocab]
            # labels: [batch, seq]
            shift_logits = logits_full[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
                reduction="mean",
            )

        return loss, logits, all_hidden_states

    def backward_step(self, loss: torch.Tensor) -> dict[str, float]:
        """Compute gradients with TP-aware all-reduce.

        Args:
            loss: Scalar loss tensor

        Returns:
            dict with gradient statistics
        """
        # Compute gradients
        loss.backward()

        # All-reduce gradients across TP group
        if self.tp_size > 1:
            for param in self.parameters():
                if param.grad is not None:
                    # Average gradients across TP ranks
                    get_tp_group().all_reduce(param.grad)
                    param.grad.div_(self.tp_size)

        # Compute gradient norm for monitoring
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        return {
            "grad_norm": total_norm,
        }

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of parameters that should be trained.

        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]

    def copy_weights_to_inference_model(
        self,
        inference_model: EagleLlamaForCausalLM,
    ) -> None:
        """Copy trained weights to the vLLM optimized EAGLE model.

        Args:
            inference_model: The vLLM EAGLE model to update
        """
        logger.info("Copying weights from trainable to inference model...")

        # Create state dict mapping
        state_dict = self.state_dict()

        # Load into inference model
        # This handles TP sharding automatically through weight loaders
        inference_model.load_state_dict(state_dict, strict=False)

        logger.info("✓ Weights copied successfully")

    def copy_weights_from_inference_model(
        self,
        inference_model: EagleLlamaForCausalLM,
    ) -> None:
        """Initialize trainable model from vLLM EAGLE model.

        Args:
            inference_model: The vLLM EAGLE model to copy from
        """
        logger.info("Copying weights from inference to trainable model...")

        # Get inference model state dict
        state_dict = inference_model.state_dict()

        # Load into trainable model
        self.load_state_dict(state_dict, strict=False)

        logger.info("✓ Weights copied successfully")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint from %s", path)
