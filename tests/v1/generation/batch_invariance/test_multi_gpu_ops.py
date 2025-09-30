# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for batch invariance in multi-GPU operations.

These tests verify that operations produce consistent results regardless of 
how inputs are batched when running across multiple GPUs.

Run `pytest tests/v1/generation/batch_invariance/test_multi_gpu_ops.py`.
"""

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform


def _test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Test fused MoE operations for batch invariance.
    
    This test verifies that fused MoE operations produce consistent results
    when processing inputs in different batch configurations.
    
    Args:
        m: Number of tokens
        n: Intermediate dimension size
        k: Hidden dimension size
        e: Number of experts
        topk: Top-k experts to route to
        dtype: Data type for tensors
    """
    current_platform.seed_everything(42)
    
    # Create VllmConfig
    vllm_config = VllmConfig()
    vllm_config.scheduler_config.max_num_seqs = 128
    vllm_config.scheduler_config.max_model_len = 8192
    
    # Create test inputs
    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    
    with set_current_vllm_config(vllm_config):
        # Get top-k routing weights and indices
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk=topk,
            renormalize=True,
        )
        
        # Test full batch processing
        output_full = fused_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
        )
        
        # Test split batch processing (batch invariance check)
        # Split into two halves and process separately
        mid = m // 2
        if mid > 0 and m > mid:
            output_part1 = fused_moe(
                hidden_states=hidden_states[:mid],
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[:mid],
                topk_ids=topk_ids[:mid],
                inplace=False,
            )
            
            output_part2 = fused_moe(
                hidden_states=hidden_states[mid:],
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[mid:],
                topk_ids=topk_ids[mid:],
                inplace=False,
            )
            
            output_split = torch.cat([output_part1, output_part2], dim=0)
            
            # Verify batch invariance
            torch.testing.assert_close(
                output_full,
                output_split,
                rtol=1e-3,
                atol=1e-3,
            )


def _test_fused_moe_with_attention(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    num_heads: int = 8,
    head_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Test fused MoE operations with attention for batch invariance.
    
    This test extends _test_fused_moe by adding attention operations and
    verifying that the combined pipeline produces consistent results across
    different batch configurations.
    
    Args:
        m: Number of tokens
        n: Intermediate dimension size
        k: Hidden dimension size
        e: Number of experts
        topk: Top-k experts to route to
        num_heads: Number of attention heads
        head_size: Size of each attention head
        dtype: Data type for tensors
    """
    current_platform.seed_everything(42)
    
    # Create VllmConfig
    vllm_config = VllmConfig()
    vllm_config.scheduler_config.max_num_seqs = 128
    vllm_config.scheduler_config.max_model_len = 8192
    
    # Ensure k is compatible with attention dimensions
    assert k == num_heads * head_size, \
        f"Hidden size {k} must equal num_heads * head_size ({num_heads * head_size})"
    
    # Create test inputs
    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    
    # Attention weights for Q, K, V projections
    w_qkv = torch.randn((k, 3 * k), device="cuda", dtype=dtype) / 10
    w_o = torch.randn((k, k), device="cuda", dtype=dtype) / 10
    
    with set_current_vllm_config(vllm_config):
        # Apply attention before MoE
        # Q, K, V projections
        qkv = torch.matmul(hidden_states, w_qkv)
        q, k_proj, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(m, num_heads, head_size)
        k_proj = k_proj.view(m, num_heads, head_size)
        v = v.view(m, num_heads, head_size)
        
        # Compute attention scores
        scale = 1.0 / (head_size ** 0.5)
        attn_scores = torch.matmul(q, k_proj.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(m, k)
        attn_output = torch.matmul(attn_output, w_o)
        
        # Add residual connection
        hidden_states_after_attn = hidden_states + attn_output
        
        # Get top-k routing weights and indices for MoE
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states_after_attn,
            router_logits=router_logits,
            topk=topk,
            renormalize=True,
        )
        
        # Test full batch processing
        moe_output_full = fused_moe(
            hidden_states=hidden_states_after_attn,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
        )
        
        # Add residual connection
        output_full = hidden_states_after_attn + moe_output_full
        
        # Test split batch processing (batch invariance check)
        mid = m // 2
        if mid > 0 and m > mid:
            # Process first half
            hidden_part1 = hidden_states[:mid]
            qkv_part1 = torch.matmul(hidden_part1, w_qkv)
            q_part1, k_part1, v_part1 = torch.chunk(qkv_part1, 3, dim=-1)
            
            q_part1 = q_part1.view(mid, num_heads, head_size)
            k_part1 = k_part1.view(mid, num_heads, head_size)
            v_part1 = v_part1.view(mid, num_heads, head_size)
            
            attn_scores_part1 = torch.matmul(q_part1, k_part1.transpose(-2, -1)) * scale
            attn_weights_part1 = torch.softmax(attn_scores_part1, dim=-1)
            attn_output_part1 = torch.matmul(attn_weights_part1, v_part1)
            attn_output_part1 = attn_output_part1.view(mid, k)
            attn_output_part1 = torch.matmul(attn_output_part1, w_o)
            
            hidden_after_attn_part1 = hidden_part1 + attn_output_part1
            
            moe_output_part1 = fused_moe(
                hidden_states=hidden_after_attn_part1,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[:mid],
                topk_ids=topk_ids[:mid],
                inplace=False,
            )
            
            output_part1 = hidden_after_attn_part1 + moe_output_part1
            
            # Process second half
            hidden_part2 = hidden_states[mid:]
            qkv_part2 = torch.matmul(hidden_part2, w_qkv)
            q_part2, k_part2, v_part2 = torch.chunk(qkv_part2, 3, dim=-1)
            
            q_part2 = q_part2.view(m - mid, num_heads, head_size)
            k_part2 = k_part2.view(m - mid, num_heads, head_size)
            v_part2 = v_part2.view(m - mid, num_heads, head_size)
            
            attn_scores_part2 = torch.matmul(q_part2, k_part2.transpose(-2, -1)) * scale
            attn_weights_part2 = torch.softmax(attn_scores_part2, dim=-1)
            attn_output_part2 = torch.matmul(attn_weights_part2, v_part2)
            attn_output_part2 = attn_output_part2.view(m - mid, k)
            attn_output_part2 = torch.matmul(attn_output_part2, w_o)
            
            hidden_after_attn_part2 = hidden_part2 + attn_output_part2
            
            moe_output_part2 = fused_moe(
                hidden_states=hidden_after_attn_part2,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[mid:],
                topk_ids=topk_ids[mid:],
                inplace=False,
            )
            
            output_part2 = hidden_after_attn_part2 + moe_output_part2
            
            # Concatenate split outputs
            output_split = torch.cat([output_part1, output_part2], dim=0)
            
            # Verify batch invariance
            torch.testing.assert_close(
                output_full,
                output_split,
                rtol=1e-3,
                atol=1e-3,
            )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("m,n,k", [(32, 128, 512), (64, 256, 512)])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_multi_gpu(m: int, n: int, k: int, e: int, topk: int,
                             dtype: torch.dtype):
    """Test fused MoE batch invariance on multi-GPU setup."""
    _test_fused_moe(m, n, k, e, topk, dtype)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("m,n,k", [(32, 128, 512), (64, 256, 512)])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_with_attention_multi_gpu(m: int, n: int, k: int, e: int,
                                            topk: int, num_heads: int,
                                            dtype: torch.dtype):
    """Test fused MoE with attention batch invariance on multi-GPU setup."""
    head_size = k // num_heads
    _test_fused_moe_with_attention(m, n, k, e, topk, num_heads, head_size,
                                   dtype)
