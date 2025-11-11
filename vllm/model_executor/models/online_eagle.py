# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online trainable EAGLE module.

This module creates a vanilla PyTorch version of EAGLE that:
1. References the same weights as the vLLM EAGLE model (no copying!)
2. Enables gradients for training
3. Produces bit-for-bit identical outputs to vLLM EAGLE (when not training)

The key challenge is that vLLM uses fused QKV projections and gate_up projections
for efficiency, but these don't support gradients. We need to "unfuse" them into
separate projections that reference the same underlying memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_tp_size():
    """Get TP size, default to 1 if not initialized (for testing)."""
    try:
        return get_tensor_model_parallel_world_size()
    except (AssertionError, AttributeError):
        # Not initialized - assume TP=1 for testing
        return 1


class VanillaAttention(nn.Module):
    """
    Vanilla PyTorch attention that references vLLM's fused QKV weights.

    vLLM uses a single fused qkv_proj for efficiency. We split this into
    separate q, k, v projections that point to slices of the same tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_weight: torch.Tensor,  # Reference to vLLM's fused weight
        o_weight: torch.Tensor,     # Reference to vLLM's o_proj weight
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm

        # Calculate sizes for Q, K, V slices
        tp_size = _get_tp_size()
        self.num_heads_per_rank = num_heads // tp_size
        self.num_kv_heads_per_rank = num_kv_heads // tp_size

        self.q_size = self.num_heads_per_rank * head_dim
        self.kv_size = self.num_kv_heads_per_rank * head_dim

        # Split the fused QKV weight into Q, K, V
        # qkv_weight shape: [q_size + 2*kv_size, hidden_size]
        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)

        # Point to slices of the fused weight (shared memory!)
        # Note: vLLM uses ModelWeightParameter which has special __torch_dispatch__ semantics
        # We use .data to access the underlying tensor, then create new Parameters from slices
        # This preserves memory sharing while allowing gradients
        self.q_proj.weight.data = qkv_weight.data[:self.q_size, :]
        self.k_proj.weight.data = qkv_weight.data[self.q_size:self.q_size + self.kv_size, :]
        self.v_proj.weight.data = qkv_weight.data[self.q_size + self.kv_size:, :]

        # O projection
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self.o_proj.weight.data = o_weight.data

        # QK norm (if used)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple self-attention for EAGLE (no fancy optimizations).
        During training, we don't need KV cache or all the vLLM optimizations.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QK norm if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # For GQA: repeat K,V heads to match Q heads
        if self.num_kv_heads_per_rank < self.num_heads_per_rank:
            n_rep = self.num_heads_per_rank // self.num_kv_heads_per_rank
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output


class VanillaMLP(nn.Module):
    """
    Vanilla PyTorch MLP that references vLLM's fused gate_up weights.

    vLLM fuses gate_proj and up_proj into a single gate_up_proj.
    We split this into separate projections that point to the same tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_up_weight: torch.Tensor,  # Reference to vLLM's fused weight
        down_weight: torch.Tensor,     # Reference to vLLM's down_proj weight
        activation: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size_per_rank = intermediate_size  # Already sharded

        # Split the fused gate_up weight
        # gate_up_weight shape: [2 * intermediate_size, hidden_size]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Point to slices of the fused weight (shared memory!)
        # Note: vLLM uses ModelWeightParameter - access underlying data
        self.gate_proj.weight.data = gate_up_weight.data[:intermediate_size, :]
        self.up_proj.weight.data = gate_up_weight.data[intermediate_size:, :]

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj.weight.data = down_weight.data

        # Activation function
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class VanillaDecoderLayer(nn.Module):
    """
    Vanilla PyTorch decoder layer that references vLLM layer weights.
    """

    def __init__(
        self,
        vllm_layer,  # The vLLM Llama4DecoderLayer to reference
        config,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Get weights from vLLM layer
        qkv_weight = vllm_layer.self_attn.qkv_proj.weight
        o_weight = vllm_layer.self_attn.o_proj.weight

        # Handle different MLP naming (mlp vs feed_forward)
        mlp_layer = getattr(vllm_layer, 'mlp', None) or getattr(vllm_layer, 'feed_forward', None)
        if mlp_layer is None:
            raise AttributeError(
                f"Layer {layer_idx} has no 'mlp' or 'feed_forward' attribute"
            )

        gate_up_weight = mlp_layer.gate_up_proj.weight
        down_weight = mlp_layer.down_proj.weight

        # Create vanilla attention and MLP that reference these weights
        self.self_attn = VanillaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_weight=qkv_weight,
            o_weight=o_weight,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            rms_norm_eps=config.rms_norm_eps,
        )

        self.mlp = VanillaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size_mlp // _get_tp_size(),
            gate_up_weight=gate_up_weight,
            down_weight=down_weight,
            activation=config.hidden_act,
        )

        # Layer norms - reference vLLM's norm weights
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Point to vLLM's norm weights using .data
        if hasattr(vllm_layer.input_layernorm, 'weight'):
            self.input_layernorm.weight.data = vllm_layer.input_layernorm.weight.data
        if hasattr(vllm_layer.post_attention_layernorm, 'weight'):
            self.post_attention_layernorm.weight.data = vllm_layer.post_attention_layernorm.weight.data

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Add residual first, then norm
            hidden_states = self.input_layernorm(hidden_states + residual)
            residual = hidden_states

        # Self attention
        hidden_states = self.self_attn(hidden_states, positions)

        # Post attention layernorm
        hidden_states = self.post_attention_layernorm(hidden_states + residual)
        residual = hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class OnlineEagleModel(nn.Module):
    """
    Trainable EAGLE model that references vLLM EAGLE weights.

    This is a vanilla PyTorch model that produces identical outputs to vLLM's
    EAGLE implementation but supports gradient computation for training.
    """

    def __init__(self, vllm_eagle_model):
        """
        Args:
            vllm_eagle_model: The vLLM EagleLlama4ForCausalLM instance to reference
        """
        super().__init__()

        self.vllm_model = vllm_eagle_model  # Keep reference
        self.config = vllm_eagle_model.config

        # Create vanilla decoder layers that reference vLLM weights
        self.layers = nn.ModuleList([
            VanillaDecoderLayer(
                vllm_layer=vllm_eagle_model.model.layers[i],
                config=self.config,
                layer_idx=i,
            )
            for i in range(len(vllm_eagle_model.model.layers))
        ])

        # Reference the embeddings and norm (keep frozen, shared with vLLM)
        self.embed_tokens = vllm_eagle_model.model.embed_tokens  # Shared, no gradients
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        if hasattr(vllm_eagle_model.model, 'norm'):
            self.norm.weight.data = vllm_eagle_model.model.norm.weight.data

        # FC layer (first layer in EAGLE)
        self.fc = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size, bias=False)
        if hasattr(vllm_eagle_model.model, 'fc'):
            self.fc.weight.data = vllm_eagle_model.model.fc.weight.data

        logger.info("OnlineEagleModel created with vanilla PyTorch structures referencing vLLM weights")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through vanilla EAGLE layers.

        Args:
            input_ids: [batch_size] - the advanced token IDs
            positions: [batch_size] or [3, batch_size] - position embeddings
            hidden_states: [batch_size, hidden_size] - target model features

        Returns:
            (hidden_states, hidden_states): Tuple of final hidden states (duplicated)
        """
        # Embed the input tokens
        inputs_embeds = self.embed_tokens(input_ids)

        # Concatenate with target hidden states
        # hidden_states shape: [batch_size, hidden_size]
        # inputs_embeds shape: [batch_size, hidden_size]
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)  # [batch_size, 2*hidden_size]

        # FC layer to project back to hidden_size
        hidden_states = self.fc(hidden_states)  # [batch_size, hidden_size]

        # Add sequence dimension if needed (EAGLE layers expect [batch, seq_len, hidden])
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Pass through decoder layers
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, positions, residual)

        # Final norm
        hidden_states = self.norm(hidden_states + residual)

        # Remove sequence dimension
        if hidden_states.size(1) == 1:
            hidden_states = hidden_states.squeeze(1)  # [batch_size, hidden_size]

        # Return duplicated for vLLM compatibility
        return hidden_states, hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits using the shared lm_head from vLLM.

        Args:
            hidden_states: [batch_size, hidden_size]

        Returns:
            logits: [batch_size, vocab_size]
        """
        return self.vllm_model.compute_logits(hidden_states)


def create_online_eagle(vllm_eagle_model) -> OnlineEagleModel:
    """
    Create a trainable EAGLE model from a vLLM EAGLE model.

    Args:
        vllm_eagle_model: vLLM's EagleLlama4ForCausalLM instance

    Returns:
        OnlineEagleModel that references the same weights
    """
    return OnlineEagleModel(vllm_eagle_model)
