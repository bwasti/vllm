#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Train an EAGLE3 draft model for Qwen3 for speculative decoding.

This script trains an EAGLE3-style draft model that uses:
- Hidden states from multiple layers of the target model (pre-concatenated)
- Transformer decoder layers
- KL divergence loss on logit distributions

The trained model can be used for speculative decoding with vLLM.

Data Format:
    Expected .pt files with structure:
    {
        'hidden_states': torch.Tensor,  # [seq_len, hidden_size * num_layers]
        'logits': torch.Tensor,          # [1, vocab_size] (if top-k not used)
        OR
        'topk_values': torch.Tensor,     # [1, k] (if top-k used)
        'topk_indices': torch.Tensor,    # [1, k] (if top-k used)
        'topk': int,                     # k value (if top-k used)
        'input_token_ids': list,
        'req_id': str,
        ...
    }

    Note: Each file is one training sample (one decoding step).
    Multiple files may share the same req_id (different steps in generation).
    Use --no-deduplicate to keep all samples (recommended).

    Top-K format: When VLLM_EAGLE3_DATA_COLLECTION_TOPK is set during data
    collection, only the top-k logits and their indices are stored to save
    disk space. The training script will reconstruct a sparse logits tensor.

Usage:
    # Single GPU
    python train_eagle3_qwen3.py \
        --data-dir ./eagle_data \
        --output-dir ./eagle3_qwen3_model \
        --base-model Qwen/Qwen3-1.7B \
        --target-layers 0,13,27 \
        --num-draft-layers 1 \
        --epochs 10 \
        --batch-size 64 \
        --lr 1e-4 \
        --no-deduplicate

    # Multi-GPU with torchrun (recommended for faster training)
    torchrun --nproc_per_node=4 train_eagle3_qwen3.py \
        --data-dir ./eagle_data \
        --output-dir ./eagle3_qwen3_model \
        --base-model Qwen/Qwen3-1.7B \
        --target-layers 0,13,27 \
        --num-draft-layers 1 \
        --epochs 10 \
        --batch-size 64 \
        --lr 5e-4 \
        --min-lr 5e-6 \
        --warmup-ratio 0.05 \
        --hidden-loss-weight 1.0 \
        --no-deduplicate \
        --num-workers 16

Requirements:
    pip install torch numpy tqdm transformers
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for Qwen3."""

    def __init__(
        self, head_dim: int, max_position: int = 131072, base: float = 1000000.0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            positions: [batch_size, seq_len] or [seq_len]
            q: [batch_size, seq_len, num_heads, head_dim]
            k: [batch_size, seq_len, num_kv_heads, head_dim]

        Returns:
            Rotated q and k tensors
        """
        # Ensure positions is 2D
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        # Compute frequencies: [batch_size, seq_len, head_dim//2]
        freqs = torch.outer(positions.flatten().float(), self.inv_freq)
        freqs = freqs.view(*positions.shape, -1)  # [batch_size, seq_len, head_dim//2]

        # Create cos and sin tensors
        cos = freqs.cos()  # [batch_size, seq_len, head_dim//2]
        sin = freqs.sin()  # [batch_size, seq_len, head_dim//2]

        # Expand to match q/k dimensions
        # q: [batch_size, seq_len, num_heads, head_dim]
        cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim//2]

        # Apply rotary embedding
        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotary_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        # Split into even and odd dimensions
        # x: [batch_size, seq_len, num_heads, head_dim]
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos

        # Interleave back
        x_rot = torch.stack([x_rot1, x_rot2], dim=-1).flatten(-2)

        return x_rot


def _index_file(args):
    """Helper function to index a single file (for parallel processing).

    Returns samples as (file_path, position_t, position_t_plus_1) tuples.
    For EAGLE3, we need consecutive pairs of hidden states.
    """
    (file_path, deduplicate) = args

    try:
        data = torch.load(file_path, map_location="cpu")

        # Check what data is available
        # Support both full logits and top-k format
        has_logits = "logits" in data or (
            "topk_values" in data and "topk_indices" in data
        )
        has_hidden_states = "hidden_states" in data

        if not has_logits or not has_hidden_states:
            return [], []

        # Get sequence length
        hidden_states = data["hidden_states"]
        seq_len = hidden_states.shape[0] if len(hidden_states.shape) > 1 else 1

        # For EAGLE3, we need consecutive pairs (t, t+1)
        # Create samples for each valid pair
        samples = []
        if seq_len >= 2:
            # Create pairs from consecutive positions
            for t in range(seq_len - 1):
                samples.append((str(file_path), t, t + 1))

        # Get request ID for deduplication
        request_id = data.get("req_id", str(file_path))
        request_ids = [request_id] * len(samples) if deduplicate else []

        return samples, request_ids
    except Exception as e:
        print(f"Error indexing {file_path}: {e}")
        return [], []


class Eagle3Dataset(Dataset):
    """Dataset for EAGLE3 training with lazy loading, parallel indexing, and file caching."""

    def __init__(
        self,
        data_files: list[Path],
        target_layers: list[int],
        vocab_size: int,
        deduplicate: bool = True,
        max_samples: int | None = None,
        num_index_workers: int = 4,
        verbose: bool = True,
        cache_size: int = 128,
    ):
        """
        Args:
            data_files: List of .pt files containing training data
            target_layers: Which layers' hidden states to use (for verification only, data is pre-concatenated)
            vocab_size: Vocabulary size from model config (used for consistent tensor shapes)
            deduplicate: Whether to deduplicate by request_id
            max_samples: Maximum number of samples to load
            num_index_workers: Number of workers for parallel indexing
            verbose: Whether to print progress
            cache_size: Number of files to cache in memory (per worker)
        """
        self.target_layers = target_layers
        self.vocab_size = vocab_size
        self.cache_size = cache_size
        self._file_cache = {}

        if verbose:
            print(f"Indexing {len(data_files)} data files...")
            print(f"Target layers: {target_layers} (data is pre-concatenated)")
            if max_samples:
                print(f"Limiting to {max_samples:,} samples")

        self.sample_index = []
        seen_request_ids = set()

        # Parallel indexing
        if num_index_workers > 0:
            index_args = [(str(f), deduplicate) for f in data_files]

            with ProcessPoolExecutor(max_workers=num_index_workers) as executor:
                futures = {
                    executor.submit(_index_file, args): args for args in index_args
                }

                iterator = as_completed(futures)
                if verbose:
                    iterator = tqdm(iterator, total=len(futures), desc="Indexing")

                for future in iterator:
                    samples, request_ids = future.result()

                    if deduplicate and request_ids:
                        for sample, req_id in zip(samples, request_ids):
                            if req_id not in seen_request_ids:
                                seen_request_ids.add(req_id)
                                self.sample_index.append(sample)
                                if (
                                    max_samples
                                    and len(self.sample_index) >= max_samples
                                ):
                                    break
                    else:
                        self.sample_index.extend(samples)

                    if max_samples and len(self.sample_index) >= max_samples:
                        self.sample_index = self.sample_index[:max_samples]
                        break
        else:
            # Sequential indexing
            file_iterator = tqdm(data_files, desc="Indexing") if verbose else data_files

            for file_path in file_iterator:
                samples, request_ids = _index_file((str(file_path), deduplicate))

                if deduplicate and request_ids:
                    for sample, req_id in zip(samples, request_ids):
                        if req_id not in seen_request_ids:
                            seen_request_ids.add(req_id)
                            self.sample_index.append(sample)
                            if max_samples and len(self.sample_index) >= max_samples:
                                break
                else:
                    self.sample_index.extend(samples)

                if max_samples and len(self.sample_index) >= max_samples:
                    break

        # Sort samples by file path to improve cache locality
        # This ensures samples from the same file are accessed together
        self.sample_index.sort(key=lambda x: x[0])

        if verbose:
            if deduplicate:
                print(
                    f"Indexed {len(self.sample_index)} training samples ({len(seen_request_ids)} unique requests)"
                )
            else:
                print(f"Indexed {len(self.sample_index)} training samples")
            print(f"File cache size: {cache_size} files per worker")

        if len(self.sample_index) == 0:
            raise ValueError("No samples indexed! Check your data files and settings.")

    def __len__(self):
        return len(self.sample_index)

    def _load_file(self, file_path: str):
        """Load file with LRU caching."""
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        # Load file
        data = torch.load(file_path, map_location="cpu")

        # Add to cache
        self._file_cache[file_path] = data

        # Evict oldest if cache is full
        if len(self._file_cache) > self.cache_size:
            # Remove first item (FIFO, simple but effective with sorted samples)
            oldest_key = next(iter(self._file_cache))
            del self._file_cache[oldest_key]

        return data

    def __getitem__(self, idx):
        # Sample is now (file_path, position_t, position_t_plus_1)
        file_path, pos_t, pos_t_plus_1 = self.sample_index[idx]

        # Load .pt file with caching
        data = self._load_file(file_path)

        # Hidden states are already concatenated from multiple layers
        # Shape: [seq_len, hidden_size * num_layers]
        # For EAGLE3: we need hidden state at position t (input) and t+1 (target)
        all_hidden_states = data["hidden_states"]

        # Clone to avoid storage sharing issues
        hidden_states_t = (
            all_hidden_states[pos_t].float().clone()
        )  # [hidden_size * num_layers]
        hidden_states_t_plus_1 = (
            all_hidden_states[pos_t_plus_1].float().clone()
        )  # [hidden_size * num_layers]

        # Load logits at position t - support both full and top-k format
        # Return log probabilities for KL divergence training
        if "logits" in data:
            # Full logits format - convert to log probabilities
            # Logits might be for the entire sequence or just last position
            logits_data = data["logits"]
            if len(logits_data.shape) > 1 and logits_data.shape[0] > 1:
                # Sequence of logits - take position t
                raw_logits = logits_data[pos_t].float()
            else:
                # Single logits tensor - assume it's for last position
                # This is a fallback, ideally we'd have logits for all positions
                raw_logits = logits_data.squeeze(0).float()

            # Ensure the logits match the expected vocab size
            if raw_logits.shape[0] != self.vocab_size:
                # Pad or truncate to match vocab size
                if raw_logits.shape[0] < self.vocab_size:
                    padding = torch.full(
                        (self.vocab_size - raw_logits.shape[0],),
                        -float("inf"),
                        dtype=torch.float32,
                    )
                    raw_logits = torch.cat([raw_logits, padding], dim=0)
                else:
                    raw_logits = raw_logits[: self.vocab_size]
            logits = F.log_softmax(raw_logits, dim=-1)  # [vocab_size]
        elif "topk_values" in data and "topk_indices" in data:
            # Top-k sparse format - reconstruct full logits tensor with uniform residual
            topk_values_data = data["topk_values"]
            topk_indices_data = data["topk_indices"]

            # Handle sequence or single position
            if len(topk_values_data.shape) > 1 and topk_values_data.shape[0] > 1:
                topk_values = topk_values_data[pos_t].float()
                topk_indices = topk_indices_data[pos_t].long()
            else:
                topk_values = topk_values_data.squeeze(0).float()
                topk_indices = topk_indices_data.squeeze(0).long()

            # Use consistent vocab size from config
            vocab_size = self.vocab_size

            # Convert top-k logits to probabilities to calculate residual
            topk_probs = F.softmax(topk_values, dim=-1)
            topk_prob_sum = topk_probs.sum().item()

            # Calculate residual probability mass for non-top-k positions
            residual_prob_mass = 1.0 - topk_prob_sum
            num_residual_positions = vocab_size - len(topk_indices)

            # Uniform probability for each non-top-k position
            uniform_residual_prob = (
                residual_prob_mass / num_residual_positions
                if num_residual_positions > 0
                else 0.0
            )

            # Create full probability distribution
            probs = torch.full(
                (vocab_size,), uniform_residual_prob, dtype=torch.float32
            )
            probs[topk_indices] = topk_probs

            # Convert back to logits (log probabilities)
            # Add small epsilon to avoid log(0)
            logits = torch.log(probs + 1e-10)
        else:
            raise ValueError(
                f"Data file {file_path} missing both 'logits' and 'topk_values/topk_indices'"
            )

        return {
            "hidden_states_t": hidden_states_t,
            "hidden_states_t_plus_1": hidden_states_t_plus_1,
            "target_logits": logits,
        }


class Qwen3Eagle3Layer(nn.Module):
    """EAGLE3-style decoder layer for Qwen3 with RoPE and QK normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
        dropout: float = 0.0,
        norm_before_residual: bool = False,
        max_position: int = 131072,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.norm_before_residual = norm_before_residual

        # Input normalization
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Hidden norm for EAGLE3 (used in layer 0)
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Multi-head attention with grouped query attention (GQA)
        # Layer 0: input is 2*hidden_size (embeds + hidden_states concatenated)
        # Other layers: input is hidden_size
        qkv_input_size = 2 * hidden_size if layer_idx == 0 else hidden_size

        # QKV projection
        # Q: num_attention_heads * head_dim = hidden_size
        # K, V: num_kv_heads * head_dim (for GQA)
        self.qkv_proj = nn.Linear(
            qkv_input_size,
            hidden_size + 2 * num_kv_heads * self.head_dim,
            bias=False,
        )

        # QK normalization (Qwen3 specific)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp_gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _norm_before_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize first, then use as residual."""
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use as residual first, then normalize."""
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(self, positions, hidden_states, embeds=None, residual=None):
        """
        Args:
            positions: Position indices [batch_size, seq_len] or [seq_len]
            hidden_states: Main hidden states [batch_size, seq_len, hidden_size]
            embeds: Input embeddings [batch_size, seq_len, hidden_size] (only used in layer 0)
            residual: Residual connection from previous layer [batch_size, seq_len, hidden_size]

        Returns:
            hidden_states: Output hidden states [batch_size, seq_len, hidden_size]
            residual: Updated residual [batch_size, seq_len, hidden_size]
        """
        if self.layer_idx == 0:
            # Layer 0: concatenate embeds with hidden_states
            embeds = self.input_layernorm(embeds)
            if self.norm_before_residual:
                hidden_states, residual = self._norm_before_residual(hidden_states)
            else:
                hidden_states, residual = self._norm_after_residual(hidden_states)
            attn_input = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Subsequent layers: process hidden_states and residuals only
            # Match vLLM's RMSNorm behavior (returns hidden_states, residual)
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            attn_input = hidden_states

        # Multi-head attention
        batch_size, seq_len, _ = attn_input.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(attn_input)

        # Split into Q, K, V
        # Q: [batch_size, seq_len, num_heads * head_dim]
        # K, V: [batch_size, seq_len, num_kv_heads * head_dim]
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape for QK normalization: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization (Qwen3 specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply Rotary Position Embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Transpose for attention: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # For GQA: repeat K and V to match number of query heads
        if self.num_kv_heads != self.num_attention_heads:
            num_repeats = self.num_attention_heads // self.num_kv_heads
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)

        # Scaled dot-product attention
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        if self.dropout:
            attn_output = self.dropout(attn_output)

        # Post-attention: add residual and normalize
        # Match vLLM's post_attention_layernorm behavior
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        gate_up = self.mlp_gate_up(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_output = self.mlp_down(F.silu(gate) * up)

        if self.dropout:
            mlp_output = self.dropout(mlp_output)

        return mlp_output, residual


class Eagle3Qwen3Model(nn.Module):
    """EAGLE3-style draft model for Qwen3."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        target_hidden_size: int,
        num_target_layers: int = 3,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        norm_before_residual: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_target_layers = 3  # Hardcode to 3 to match vLLM

        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Fusion layer to combine auxiliary hidden states
        # IMPORTANT: Hardcoded to 3 layers to match vLLM implementation
        self.fc = nn.Linear(target_hidden_size * 3, hidden_size, bias=False)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3Layer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    layer_idx=i,
                    dropout=dropout,
                    norm_before_residual=norm_before_residual,
                    max_position=131072,  # Qwen3 default
                    rope_theta=1000000.0,  # Qwen3 default
                )
                for i in range(num_hidden_layers)
            ]
        )

        # Output normalization
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # EAGLE3: Hidden state prediction head
        # Predicts next hidden states in the draft model's fused space
        # These will be fed back autoregressively without going back through fc
        self.hidden_state_projection = nn.Linear(
            hidden_size,
            hidden_size,  # Stay in draft space, not target space
            bias=False,
        )

    def forward(self, input_ids, positions, hidden_states, return_hidden_states=False):
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len] or [seq_len]
            hidden_states: Concatenated hidden states from target model [batch_size, seq_len, hidden_size * num_target_layers]
            return_hidden_states: If True, return (logits, output_hidden_states)

        Returns:
            If return_hidden_states=False: logits [batch_size, seq_len, vocab_size]
            If return_hidden_states=True: (logits, output_hidden_states)
                where output_hidden_states is [batch_size, seq_len, hidden_size * num_target_layers]
        """
        # Get input embeddings
        embeds = self.embed_tokens(input_ids)

        # Fuse target hidden states
        fused_hidden_states = self.fc(hidden_states)

        # Pass through decoder layers
        residual = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                fused_hidden_states, residual = layer(
                    positions, fused_hidden_states, embeds=embeds, residual=residual
                )
            else:
                fused_hidden_states, residual = layer(
                    positions, fused_hidden_states, residual=residual
                )

        # Final normalization
        fused_hidden_states = self.norm(fused_hidden_states + residual)

        # Project to vocabulary
        logits = self.lm_head(fused_hidden_states)

        if return_hidden_states:
            # For EAGLE3: predict what the target model's hidden states would be at next position
            output_hidden_states = self.hidden_state_projection(fused_hidden_states)
            return logits, output_hidden_states

        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    use_mixed_precision: bool = True,
    rank: int = 0,
    epoch: int = 0,
    gradient_accumulation_steps: int = 1,
    hidden_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with EAGLE3-style training."""
    model.train()
    total_loss = 0.0
    total_logit_loss = 0.0
    total_hidden_loss = 0.0
    total_samples = 0
    start_time = time.time()

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_mixed_precision and device.type == "cuda"
        else None
    )

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    else:
        pbar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # EAGLE3 training: use hidden states at t to predict logits and hidden states at t+1
        hidden_states_t = batch["hidden_states_t"].to(device)
        hidden_states_t_plus_1 = batch["hidden_states_t_plus_1"].to(device)
        target_logits = batch["target_logits"].to(device)

        batch_size = hidden_states_t.shape[0]

        # For training, we'll use a dummy input_ids of zeros
        # In real inference, this would be the actual next token
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Positions for single-token forward (always position 0)
        positions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                # Model outputs [batch_size, vocab_size] and [batch_size, hidden_size * num_layers]
                pred_logits, pred_hidden = model(
                    input_ids.unsqueeze(1),
                    positions,
                    hidden_states_t.unsqueeze(1),
                    return_hidden_states=True,
                )
                pred_logits = pred_logits.squeeze(1)
                pred_hidden = pred_hidden.squeeze(1)

                # Logit loss: KL divergence to match probability distributions
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)
                logit_loss = F.kl_div(
                    pred_log_probs,
                    target_logits,
                    reduction="batchmean",
                    log_target=True,
                )

                # Hidden state loss: MSE to match next hidden states in draft space
                # Fuse target hidden states at t+1 through fc layer to get draft space representation
                model_to_use = model.module if hasattr(model, "module") else model
                target_hidden_fused = model_to_use.fc(
                    hidden_states_t_plus_1.unsqueeze(1)
                ).squeeze(1)
                # Normalize by hidden dimension to make comparable to logit loss
                hidden_loss_raw = F.mse_loss(pred_hidden, target_hidden_fused)
                hidden_dim = pred_hidden.shape[-1]
                hidden_loss = hidden_loss_raw / hidden_dim

                # Combined loss
                loss = logit_loss + hidden_loss_weight * hidden_loss
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            pred_logits, pred_hidden = model(
                input_ids.unsqueeze(1),
                positions,
                hidden_states_t.unsqueeze(1),
                return_hidden_states=True,
            )
            pred_logits = pred_logits.squeeze(1)
            pred_hidden = pred_hidden.squeeze(1)

            # Logit loss: KL divergence to match probability distributions
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)
            logit_loss = F.kl_div(
                pred_log_probs, target_logits, reduction="batchmean", log_target=True
            )

            # Hidden state loss: MSE to match next hidden states in draft space
            # Fuse target hidden states at t+1 through fc layer to get draft space representation
            model_to_use = model.module if hasattr(model, "module") else model
            target_hidden_fused = model_to_use.fc(
                hidden_states_t_plus_1.unsqueeze(1)
            ).squeeze(1)
            # Normalize by hidden dimension to make comparable to logit loss
            hidden_loss_raw = F.mse_loss(pred_hidden, target_hidden_fused)
            hidden_dim = pred_hidden.shape[-1]
            hidden_loss = hidden_loss_raw / hidden_dim

            # Combined loss
            loss = logit_loss + hidden_loss_weight * hidden_loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        total_logit_loss += logit_loss.item() * batch_size
        total_hidden_loss += hidden_loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        if rank == 0 and hasattr(pbar, "set_postfix"):
            avg_loss = total_loss / total_samples
            avg_logit_loss = total_logit_loss / total_samples
            avg_hidden_loss = total_hidden_loss / total_samples
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "L_logit": f"{avg_logit_loss:.4f}",
                    "L_hidden": f"{avg_hidden_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "samples/s": f"{samples_per_sec:.0f}",
                }
            )

    return {
        "loss": total_loss / total_samples,
        "logit_loss": total_logit_loss / total_samples,
        "hidden_loss": total_hidden_loss / total_samples,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0,
    hidden_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate the model with EAGLE3-style losses."""
    model.eval()
    total_loss = 0.0
    total_logit_loss = 0.0
    total_hidden_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        if rank == 0:
            iterator = tqdm(dataloader, desc="Validation", unit="batch")
        else:
            iterator = dataloader

        for batch in iterator:
            hidden_states_t = batch["hidden_states_t"].to(device)
            hidden_states_t_plus_1 = batch["hidden_states_t_plus_1"].to(device)
            target_logits = batch["target_logits"].to(device)

            batch_size = hidden_states_t.shape[0]
            input_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Positions for single-token forward (always position 0)
            positions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            pred_logits, pred_hidden = model(
                input_ids.unsqueeze(1),
                positions,
                hidden_states_t.unsqueeze(1),
                return_hidden_states=True,
            )
            pred_logits = pred_logits.squeeze(1)
            pred_hidden = pred_hidden.squeeze(1)

            # Logit loss: KL divergence to match probability distributions
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)
            logit_loss = F.kl_div(
                pred_log_probs, target_logits, reduction="batchmean", log_target=True
            )

            # Hidden state loss: MSE to match next hidden states in draft space
            # Fuse target hidden states at t+1 through fc layer to get draft space representation
            model_to_use = model.module if hasattr(model, "module") else model
            target_hidden_fused = model_to_use.fc(
                hidden_states_t_plus_1.unsqueeze(1)
            ).squeeze(1)
            # Normalize by hidden dimension to make comparable to logit loss
            hidden_loss_raw = F.mse_loss(pred_hidden, target_hidden_fused)
            hidden_dim = pred_hidden.shape[-1]
            hidden_loss = hidden_loss_raw / hidden_dim

            # Combined loss
            loss = logit_loss + hidden_loss_weight * hidden_loss

            total_loss += loss.item() * batch_size
            total_logit_loss += logit_loss.item() * batch_size
            total_hidden_loss += hidden_loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            if rank == 0 and hasattr(iterator, "set_postfix"):
                avg_loss = total_loss / total_samples
                avg_logit_loss = total_logit_loss / total_samples
                avg_hidden_loss = total_hidden_loss / total_samples
                iterator.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "L_logit": f"{avg_logit_loss:.4f}",
                        "L_hidden": f"{avg_hidden_loss:.4f}",
                    }
                )

    return {
        "loss": total_loss / total_samples,
        "logit_loss": total_logit_loss / total_samples,
        "hidden_loss": total_hidden_loss / total_samples,
    }


def convert_training_weights_to_vllm(
    state_dict: dict, num_attention_heads: int, num_kv_heads: int, hidden_size: int
) -> dict:
    """
    Convert training script weight format to vLLM weight format.

    Training format has:
    - layers.{i}.qkv_proj: Combined QKV projection [hidden_size, qkv_size]
    - layers.{i}.o_proj: Output projection
    - layers.{i}.mlp_gate_up: Combined gate/up projection [hidden_size, intermediate_size * 2]
    - layers.{i}.mlp_down: Down projection

    vLLM expects:
    - layers.{i}.self_attn.q_proj: Q projection
    - layers.{i}.self_attn.k_proj: K projection
    - layers.{i}.self_attn.v_proj: V projection
    - layers.{i}.self_attn.o_proj: Output projection
    - layers.{i}.mlp.gate_proj: Gate projection
    - layers.{i}.mlp.up_proj: Up projection
    - layers.{i}.mlp.down_proj: Down projection
    """
    new_state_dict = {}
    head_dim = hidden_size // num_attention_heads

    for name, param in state_dict.items():
        if "qkv_proj.weight" in name:
            # Split QKV into separate Q, K, V weights
            layer_name = name.replace(".qkv_proj.weight", "")

            q_size = num_attention_heads * head_dim
            kv_size = num_kv_heads * head_dim

            # QKV shape: [q_size + 2*kv_size, input_size]
            q_weight, k_weight, v_weight = param.split(
                [q_size, kv_size, kv_size], dim=0
            )

            new_state_dict[f"{layer_name}.self_attn.q_proj.weight"] = q_weight
            new_state_dict[f"{layer_name}.self_attn.k_proj.weight"] = k_weight
            new_state_dict[f"{layer_name}.self_attn.v_proj.weight"] = v_weight

        elif "o_proj.weight" in name:
            # Rename o_proj to self_attn.o_proj
            layer_name = name.replace(".o_proj.weight", "")
            new_state_dict[f"{layer_name}.self_attn.o_proj.weight"] = param

        elif "mlp_gate_up.weight" in name:
            # Split gate_up into separate gate and up weights
            layer_name = name.replace(".mlp_gate_up.weight", "")
            intermediate_size = param.shape[0] // 2

            gate_weight, up_weight = param.split(
                [intermediate_size, intermediate_size], dim=0
            )

            new_state_dict[f"{layer_name}.mlp.gate_proj.weight"] = gate_weight
            new_state_dict[f"{layer_name}.mlp.up_proj.weight"] = up_weight

        elif "mlp_down.weight" in name:
            # Rename mlp_down to mlp.down_proj
            layer_name = name.replace(".mlp_down.weight", "")
            new_state_dict[f"{layer_name}.mlp.down_proj.weight"] = param

        elif "q_norm.weight" in name:
            # QK normalization layers - keep name structure
            new_state_dict[name.replace("q_norm", "self_attn.q_norm")] = param

        elif "k_norm.weight" in name:
            # QK normalization layers - keep name structure
            new_state_dict[name.replace("k_norm", "self_attn.k_norm")] = param

        else:
            # Keep other weights as-is (embed_tokens, fc, norm, lm_head, hidden_state_projection, rotary_emb)
            # Note: rotary_emb buffers (inv_freq) don't need to be saved as they're recomputed
            if "rotary_emb" not in name:
                new_state_dict[name] = param

    return new_state_dict


def save_model_for_vllm(
    model: Eagle3Qwen3Model,
    output_dir: Path,
    config: dict,
):
    """Save model in format compatible with vLLM."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get state dict and convert to vLLM format
    state_dict = model.state_dict()
    vllm_state_dict = convert_training_weights_to_vllm(
        state_dict,
        config["num_attention_heads"],
        config["num_kv_heads"],
        config["hidden_size"],
    )

    # Save converted model state dict
    torch.save(vllm_state_dict, output_dir / "model.pt")

    # Save config
    model_config = {
        "architectures": ["Eagle3Qwen3ForCausalLM"],
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_draft_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_kv_heads"],
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config["rms_norm_eps"],
        "target_hidden_size": config["target_hidden_size"],
        "num_target_layers": 3,  # Always 3 to match vLLM
        "target_layers": config["target_layers"],
        "norm_before_residual": config.get("norm_before_residual", False),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(
        f"  Converted {len(state_dict)} training weights to {len(vllm_state_dict)} vLLM weights"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train EAGLE3 for Qwen3 speculative decoding"
    )

    # Data args
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing .npz data files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save trained model"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction of data for validation"
    )

    # Model args
    parser.add_argument(
        "--base-model", type=str, required=True, help="Base Qwen3 model name or path"
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        required=True,
        help="Comma-separated list of target layer indices (e.g., '0,13,27')",
    )
    parser.add_argument(
        "--num-draft-layers", type=int, default=1, help="Number of draft model layers"
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=None,
        help="Number of attention heads (default: use base model's value). IMPORTANT: Must be divisible by target model TP size!",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (default: use base model's value)",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--norm-before-residual",
        action="store_true",
        help="Apply norm before residual (default: False)",
    )

    # Training args
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=5e-6,
        help="Minimum learning rate for cosine schedule (default: 5e-6)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio (default: 0.05 = 5%% of training)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Adam beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95, help="Adam beta2 (default: 0.95)"
    )
    parser.add_argument(
        "--hidden-loss-weight",
        type=float,
        default=1.0,
        help="Weight for hidden state prediction loss (default: 1.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Data loading workers (default: 12 for better parallelism)",
    )
    parser.add_argument(
        "--num-index-workers",
        type=int,
        default=16,
        help="Indexing workers (default: 16)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Number of data files to cache per worker (default: 128)",
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--no-deduplicate", action="store_true", help="Disable deduplication"
    )
    parser.add_argument(
        "--filter-rank",
        type=int,
        default=None,
        help="Only load data from specific rank",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum samples to load"
    )

    args = parser.parse_args()

    # Setup distributed training
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_distributed = True
        if rank == 0:
            print(f"Using DistributedDataParallel: {world_size} GPUs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0
        world_size = 1
        is_distributed = False

    if rank == 0:
        print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse target layers
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    # Validate exactly 3 target layers (required by vLLM)
    if len(target_layers) != 3:
        if rank == 0:
            print(
                f"ERROR: vLLM requires exactly 3 target layers, but got {len(target_layers)}: {target_layers}"
            )
            print(
                "Please provide exactly 3 layer indices (e.g., --target-layers 0,13,27)"
            )
        return 1

    if rank == 0:
        print(f"Target layers: {target_layers}")

    # Load base model config
    if rank == 0:
        print(f"Loading config from {args.base_model}...")
    base_config = AutoConfig.from_pretrained(args.base_model)

    # Load data files
    data_dir = Path(args.data_dir)
    if args.filter_rank is not None:
        data_files = sorted(data_dir.glob(f"*_rank{args.filter_rank}_*.pt"))
    else:
        data_files = sorted(data_dir.glob("*.pt"))

    if not data_files:
        if rank == 0:
            print(f"Error: No .pt files found in {data_dir}")
        return 1

    if rank == 0:
        print(f"Found {len(data_files)} data files")

    # Split into train/val
    num_val = max(1, int(len(data_files) * args.val_split))
    val_files = data_files[:num_val]
    train_files = data_files[num_val:]

    if rank == 0:
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Detect actual hidden size from data
    if rank == 0:
        print("\nDetecting hidden size from data...")
        sample_data = torch.load(data_files[0], map_location="cpu")
        detected_hidden_size_total = sample_data["hidden_states"].shape[-1]
        detected_hidden_size = detected_hidden_size_total // len(target_layers)
        print(
            f"  Detected hidden size: {detected_hidden_size} ({detected_hidden_size_total} / {len(target_layers)} layers)"
        )

        if detected_hidden_size != base_config.hidden_size:
            print(
                f"  WARNING: Base model config has hidden_size={base_config.hidden_size}"
            )
            print(
                f"  Using detected hidden_size={detected_hidden_size} from data instead"
            )
            target_hidden_size = detected_hidden_size
        else:
            target_hidden_size = base_config.hidden_size
    else:
        # Non-rank-0 processes: detect from first file
        sample_data = torch.load(data_files[0], map_location="cpu")
        detected_hidden_size_total = sample_data["hidden_states"].shape[-1]
        target_hidden_size = detected_hidden_size_total // len(target_layers)

    # Create datasets
    if rank == 0:
        print("\nCreating training dataset...")
    train_dataset = Eagle3Dataset(
        train_files,
        target_layers=target_layers,
        vocab_size=base_config.vocab_size,
        deduplicate=not args.no_deduplicate,
        max_samples=args.max_samples,
        num_index_workers=args.num_index_workers,
        verbose=(rank == 0),
        cache_size=args.cache_size,
    )

    if rank == 0:
        print("\nCreating validation dataset...")
    val_max_samples = int(args.max_samples * 0.1) if args.max_samples else None
    val_dataset = Eagle3Dataset(
        val_files,
        target_layers=target_layers,
        vocab_size=base_config.vocab_size,
        deduplicate=not args.no_deduplicate,
        max_samples=val_max_samples,
        num_index_workers=args.num_index_workers,
        verbose=(rank == 0),
        cache_size=args.cache_size,
    )

    # Synchronize after dataset creation
    if is_distributed:
        import torch.distributed as dist

        dist.barrier()

    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        # Prefetch batches for better GPU utilization
        prefetch_factor=2 if args.num_workers > 0 else None,
        # Use persistent workers to avoid respawning overhead
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Determine model architecture parameters
    num_attention_heads = (
        args.num_attention_heads
        if args.num_attention_heads is not None
        else base_config.num_attention_heads
    )
    num_kv_heads = (
        args.num_kv_heads
        if args.num_kv_heads is not None
        else base_config.num_key_value_heads
    )

    # Create model
    if rank == 0:
        print("\nCreating EAGLE3 model...")
        print(f"  Draft layers: {args.num_draft_layers}")
        print(f"  Hidden size: {base_config.hidden_size}")
        print(f"  Vocab size: {base_config.vocab_size}")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  KV heads: {num_kv_heads}")

    model = Eagle3Qwen3Model(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=args.num_draft_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=base_config.intermediate_size,
        target_hidden_size=target_hidden_size,
        num_target_layers=3,  # Always 3 for vLLM
        rms_norm_eps=base_config.rms_norm_eps,
        dropout=args.dropout,
        norm_before_residual=args.norm_before_residual,
    )

    model = model.to(device)

    # Wrap for DDP with optimizations
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # Gradient bucketing for better communication efficiency
            bucket_cap_mb=25,
            # Find unused parameters (shouldn't be needed but good to have)
            find_unused_parameters=False,
        )

    # Count parameters
    if rank == 0:
        model_to_count = model.module if is_distributed else model
        num_params = sum(
            p.numel() for p in model_to_count.parameters() if p.requires_grad
        )
        print(f"Model parameters: {num_params:,}")

    # Create optimizer with better defaults
    if rank == 0:
        print("\nOptimizer settings:")
        print(f"  Peak LR: {args.lr}")
        print(f"  Min LR: {args.min_lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Betas: ({args.beta1}, {args.beta2})")

    # Use fused AdamW if available (much faster on CUDA)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            fused=True if device.type == "cuda" else False,
        )
        if rank == 0:
            print("  Using fused AdamW")
    except:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        if rank == 0:
            print("  Using standard AdamW")

    # Calculate scheduler parameters
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    if rank == 0:
        print("\nLR Schedule:")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,} ({args.warmup_ratio * 100:.1f}%)")
        print(f"  Cosine decay: {args.lr} → {args.min_lr}")

    # Create cosine learning rate scheduler with warmup
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale between min_lr and peak lr
            return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_mixed_precision=not args.no_mixed_precision,
            rank=rank,
            epoch=epoch + 1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            hidden_loss_weight=args.hidden_loss_weight,
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            rank=rank,
            hidden_loss_weight=args.hidden_loss_weight,
        )

        # Synchronize metrics
        if is_distributed:
            import torch.distributed as dist

            train_loss_tensor = torch.tensor([train_metrics["loss"]], device=device)
            val_loss_tensor = torch.tensor([val_metrics["loss"]], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            train_metrics["loss"] = (train_loss_tensor / world_size).item()
            val_metrics["loss"] = (val_loss_tensor / world_size).item()

        # Log and save
        if rank == 0:
            print(
                f"\nTrain loss: {train_metrics['loss']:.6f} (logit: {train_metrics['logit_loss']:.6f}, hidden: {train_metrics['hidden_loss']:.6f})"
            )
            print(
                f"Val loss: {val_metrics['loss']:.6f} (logit: {val_metrics['logit_loss']:.6f}, hidden: {val_metrics['hidden_loss']:.6f})"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                print("New best validation loss! Saving model...")

                model_to_save = model.module if is_distributed else model
                save_model_for_vllm(
                    model_to_save,
                    output_dir / "best_model",
                    {
                        "vocab_size": base_config.vocab_size,
                        "hidden_size": base_config.hidden_size,
                        "num_draft_layers": args.num_draft_layers,
                        "num_attention_heads": num_attention_heads,
                        "num_kv_heads": num_kv_heads,
                        "intermediate_size": base_config.intermediate_size,
                        "rms_norm_eps": base_config.rms_norm_eps,
                        "target_hidden_size": target_hidden_size,
                        "target_layers": target_layers,
                        "norm_before_residual": args.norm_before_residual,
                    },
                )

    # Save final model
    if rank == 0:
        print("\nSaving final model...")
        model_to_save = model.module if is_distributed else model
        save_model_for_vllm(
            model_to_save,
            output_dir / "final_model",
            {
                "vocab_size": base_config.vocab_size,
                "hidden_size": base_config.hidden_size,
                "num_draft_layers": args.num_draft_layers,
                "num_attention_heads": num_attention_heads,
                "num_kv_heads": num_kv_heads,
                "intermediate_size": base_config.intermediate_size,
                "rms_norm_eps": base_config.rms_norm_eps,
                "target_hidden_size": target_hidden_size,
                "target_layers": target_layers,
                "norm_before_residual": args.norm_before_residual,
            },
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Model saved to: {output_dir}")

    # Cleanup
    if is_distributed:
        import torch.distributed as dist

        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    exit(main())
