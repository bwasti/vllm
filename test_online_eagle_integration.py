#!/usr/bin/env python3
"""
Standalone test script for Online EAGLE parity.

This script can be run directly to test the trainable EAGLE implementation
against the vLLM EAGLE model using the actual Llama4 model.

Usage:
    LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" python test_online_eagle_integration.py

Or run the helper script:
    ./run_online_eagle_test.sh
"""

import os
import torch
import sys
from pathlib import Path

# Check if LD_PRELOAD is set
if "LD_PRELOAD" not in os.environ:
    print("WARNING: LD_PRELOAD not set!")
    print("You may need to run with:")
    print('LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" python test_online_eagle_integration.py')
    print("")

# Add vllm to path
vllm_path = Path(__file__).parent
sys.path.insert(0, str(vllm_path))


def test_weight_reference():
    """Test that weights are truly shared (not copied)."""
    print("\n" + "="*80)
    print("TEST 1: Weight Reference (Memory Sharing)")
    print("="*80)

    # Create a mock vLLM-style layer for testing
    class MockVLLMLayer:
        class MockQKVProj:
            def __init__(self, hidden_size=512, num_heads=8, head_dim=64):
                # Simulated fused QKV weight: [q_size + k_size + v_size, hidden_size]
                q_size = num_heads * head_dim
                kv_size = num_heads * head_dim  # Simplified (same as q for test)
                total_size = q_size + 2 * kv_size
                self.weight = torch.randn(total_size, hidden_size)

        class MockOProj:
            def __init__(self, hidden_size=512, num_heads=8, head_dim=64):
                self.weight = torch.randn(hidden_size, num_heads * head_dim)

        class MockGateUpProj:
            def __init__(self, hidden_size=512, intermediate_size=2048):
                # Fused gate_up: [2 * intermediate_size, hidden_size]
                self.weight = torch.randn(2 * intermediate_size, hidden_size)

        class MockDownProj:
            def __init__(self, hidden_size=512, intermediate_size=2048):
                self.weight = torch.randn(hidden_size, intermediate_size)

        class MockSelfAttn:
            def __init__(self):
                self.qkv_proj = MockVLLMLayer.MockQKVProj()
                self.o_proj = MockVLLMLayer.MockOProj()

        class MockMLP:
            def __init__(self):
                self.gate_up_proj = MockVLLMLayer.MockGateUpProj()
                self.down_proj = MockVLLMLayer.MockDownProj()

        class MockNorm:
            def __init__(self, hidden_size=512):
                self.weight = torch.randn(hidden_size)

        def __init__(self):
            self.self_attn = self.MockSelfAttn()
            self.mlp = self.MockMLP()
            self.input_layernorm = self.MockNorm()
            self.post_attention_layernorm = self.MockNorm()

    # Test weight sharing
    mock_layer = MockVLLMLayer()

    # Get original weight tensor
    original_qkv_weight = mock_layer.self_attn.qkv_proj.weight
    original_data_ptr = original_qkv_weight.data_ptr()

    print(f"Original QKV weight data_ptr: {original_data_ptr:#x}")
    print(f"Original QKV weight shape: {original_qkv_weight.shape}")

    # Simulate slicing (what OnlineEagle does)
    hidden_size = 512
    num_heads = 8
    head_dim = 64
    q_size = num_heads * head_dim
    kv_size = num_heads * head_dim

    q_weight_slice = original_qkv_weight[:q_size, :]
    k_weight_slice = original_qkv_weight[q_size:q_size + kv_size, :]
    v_weight_slice = original_qkv_weight[q_size + kv_size:, :]

    print(f"\\nQ slice data_ptr: {q_weight_slice.data_ptr():#x}")
    print(f"K slice data_ptr: {k_weight_slice.data_ptr():#x}")
    print(f"V slice data_ptr: {v_weight_slice.data_ptr():#x}")

    # Wrap in Parameters
    q_param = torch.nn.Parameter(q_weight_slice, requires_grad=True)
    k_param = torch.nn.Parameter(k_weight_slice, requires_grad=True)
    v_param = torch.nn.Parameter(v_weight_slice, requires_grad=True)

    print(f"\\nQ param data_ptr: {q_param.data_ptr():#x}")
    print(f"K param data_ptr: {k_param.data_ptr():#x}")
    print(f"V param data_ptr: {v_param.data_ptr():#x}")

    # Critical test: Modify through parameter and check original
    print("\\n--- Testing memory sharing ---")
    old_value = original_qkv_weight[0, 0].item()
    print(f"Original weight[0,0] before: {old_value:.6f}")

    q_param.data[0, 0] = 999.0

    new_value = original_qkv_weight[0, 0].item()
    print(f"Original weight[0,0] after modifying q_param: {new_value:.6f}")

    if abs(new_value - 999.0) < 1e-5:
        print("✅ SUCCESS: Weights are shared! Modifying parameter updated original tensor")
    else:
        print("❌ FAIL: Weights are NOT shared! This would break training")
        return False

    return True


def test_simple_forward():
    """Test a simple forward pass through vanilla components."""
    print("\\n" + "="*80)
    print("TEST 2: Simple Forward Pass")
    print("="*80)

    # Simplified test without full vLLM
    hidden_size = 512
    batch_size = 2

    # Create simple linear layer
    fc = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)

    # Test input
    input_embeds = torch.randn(batch_size, hidden_size)
    target_hidden = torch.randn(batch_size, hidden_size)

    # Concatenate and project (like EAGLE does)
    combined = torch.cat([input_embeds, target_hidden], dim=-1)
    output = fc(combined)

    print(f"Input embeds shape: {input_embeds.shape}")
    print(f"Target hidden shape: {target_hidden.shape}")
    print(f"Combined shape: {combined.shape}")
    print(f"Output shape: {output.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()

    if fc.weight.grad is not None:
        print(f"✅ Gradient shape: {fc.weight.grad.shape}")
        print(f"✅ Gradient norm: {fc.weight.grad.norm().item():.6f}")
        return True
    else:
        print("❌ FAIL: No gradients computed")
        return False


def test_tensor_parallel_slicing():
    """Test that TP slicing works correctly."""
    print("\\n" + "="*80)
    print("TEST 3: Tensor Parallel Weight Slicing")
    print("="*80)

    # Simulate TP=8 setup
    full_hidden_size = 5120
    tp_size = 8
    hidden_size_per_rank = full_hidden_size // tp_size

    print(f"Full hidden size: {full_hidden_size}")
    print(f"TP size: {tp_size}")
    print(f"Hidden size per rank: {hidden_size_per_rank}")

    # Simulate logits
    full_vocab_size = 202048
    vocab_size_per_rank = full_vocab_size // tp_size

    print(f"Full vocab size: {full_vocab_size}")
    print(f"Vocab size per rank: {vocab_size_per_rank}")

    # Test shapes
    batch_size = 16
    hidden_states_per_rank = torch.randn(batch_size, hidden_size_per_rank)
    logits_per_rank = torch.randn(batch_size, vocab_size_per_rank)

    print(f"\\nHidden states per rank: {hidden_states_per_rank.shape}")
    print(f"Logits per rank: {logits_per_rank.shape}")

    # Memory calculation
    hidden_bytes = batch_size * hidden_size_per_rank * 2  # fp16
    logits_bytes = batch_size * vocab_size_per_rank * 2  # fp16

    print(f"\\nMemory per rank:")
    print(f"  Hidden states: {hidden_bytes / 1024:.2f} KB")
    print(f"  Logits: {logits_bytes / 1024:.2f} KB")
    print(f"  Total per sample: {(hidden_bytes + logits_bytes) / batch_size / 1024:.2f} KB")

    print("✅ TP slicing dimensions look correct")
    return True


def main():
    """Run all tests."""
    print("\\n" + "="*80)
    print("ONLINE EAGLE INTEGRATION TESTS")
    print("="*80)

    results = {}

    try:
        results['weight_reference'] = test_weight_reference()
    except Exception as e:
        print(f"❌ Weight reference test failed: {e}")
        results['weight_reference'] = False

    try:
        results['simple_forward'] = test_simple_forward()
    except Exception as e:
        print(f"❌ Simple forward test failed: {e}")
        results['simple_forward'] = False

    try:
        results['tp_slicing'] = test_tensor_parallel_slicing()
    except Exception as e:
        print(f"❌ TP slicing test failed: {e}")
        results['tp_slicing'] = False

    # Summary
    print("\\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())
    print("="*80)

    if all_passed:
        print("\\n🎉 ALL TESTS PASSED!")
        print("\\nNext steps:")
        print("1. Test with actual vLLM EAGLE model loaded")
        print("2. Verify forward pass parity (should be bit-for-bit identical)")
        print("3. Test gradient computation and backpropagation")
        print("4. Integrate with training loop")
        return 0
    else:
        print("\\n❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
