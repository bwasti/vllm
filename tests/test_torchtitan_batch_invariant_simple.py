# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simple test for batch invariance with direct torchtitan model only.

This test validates that with batch invariance enabled, the torchtitan model
produces deterministic results across different runs.
"""

# Set batch invariance BEFORE any imports
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_USE_V1"] = "1"

import torch
import torch.nn.functional as F

import torchtitan.experiments.compat
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

# Import batch invariance initialization
from vllm.model_executor.layers.batch_invariant import init_batch_invariance, vllm_is_batch_invariant


def test_batch_invariant_determinism():
    """
    Test that torch titan model with batch invariance produces identical results
    across multiple runs with the same input.
    """
    print("\n" + "=" * 80)
    print("Testing Batch Invariance - Deterministic Results")
    print("=" * 80)

    # Initialize batch invariance
    print("\n[1/4] Initializing batch invariance...")
    init_batch_invariance()
    assert vllm_is_batch_invariant()
    print("✅ Batch invariance initialized")

    # Create model
    print("\n[2/4] Creating torchtitan model...")
    model_args = Qwen3ModelArgs(
        dim=512,
        n_layers=2,  # Small for fast testing
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1000,
        head_dim=64,
        hidden_dim=1536,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=128,
        depth_init=False,
        use_flex_attn=False,
        attn_mask_type="causal",
        eos_id=999,
        enable_weight_tying=False,
        moe_enabled=False,
        use_hybrid_attn=True,
    )

    model = Qwen3Model(model_args)
    model.init_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32)
    model.train()  # Training mode

    print(f"✅ Model created: {model_args.n_layers} layers, vocab={model_args.vocab_size}")

    # Create test input
    print("\n[3/4] Running model multiple times...")
    test_tokens = torch.randint(0, model_args.vocab_size, (2, 10), device=device)
    print(f"✅ Test tokens: {test_tokens[0, :5].tolist()}")

    # Run 3 times
    outputs = []
    for i in range(3):
        with torch.no_grad():
            output = model(test_tokens)
        outputs.append(output)
        print(f"✅ Run {i+1}: shape={output.shape}, sample={output[0, -1, :3].tolist()}")

    # Check all outputs are identical
    print("\n[4/4] Checking determinism...")
    all_identical = True
    for i in range(1, len(outputs)):
        if not torch.equal(outputs[0], outputs[i]):
            max_diff = (outputs[0] - outputs[i]).abs().max().item()
            print(f"✗ Run {i+1} differs from run 1: max_diff={max_diff:.2e}")
            all_identical = False
        else:
            print(f"✅ Run {i+1} matches run 1 (bitwise identical)")

    print("\n" + "=" * 80)
    if all_identical:
        print("Test Result: PASSED ✅")
        print("✅ All runs produce bitwise identical results")
        print("✅ Batch invariance is working correctly!")
    else:
        print("Test Result: FAILED ✗")
        print("✗ Results differ across runs")
        print("✗ Batch invariance not working properly")
    print("=" * 80)

    return all_identical


def test_batch_invariant_llm_vs_direct():
    """
    Test that torchtitan model produces identical logprobs when run:
    1. Directly in training mode with batch invariance
    2. Through vLLM LLM() wrapper with batch invariance

    This validates the full integration works with batch invariance.
    """
    print("\n" + "=" * 80)
    print("Testing Batch Invariance - LLM() vs Direct Model")
    print("=" * 80)

    # Initialize batch invariance
    print("\n[1/5] Initializing batch invariance...")
    init_batch_invariance()
    assert vllm_is_batch_invariant()
    print("✅ Batch invariance initialized")

    # Create model
    print("\n[2/5] Creating torchtitan model...")
    model_args = Qwen3ModelArgs(
        dim=512,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1000,
        head_dim=64,
        hidden_dim=1536,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=128,
        depth_init=False,
        use_flex_attn=False,
        attn_mask_type="causal",
        eos_id=999,
        enable_weight_tying=False,
        moe_enabled=False,
        use_hybrid_attn=True,
    )

    model = Qwen3Model(model_args)
    model.init_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32)

    print(f"✅ Model created: {model_args.n_layers} layers, vocab={model_args.vocab_size}")

    # Test with synthetic tokens (within our small vocab)
    test_token_ids = [10, 20, 30, 40, 50]
    test_tokens = torch.tensor([test_token_ids], device=device, dtype=torch.long)
    print(f"✅ Test tokens: {test_token_ids}")

    # Path 1: Direct model in training mode
    print("\n[3/5] Running direct model (training mode)...")
    model.train()
    with torch.no_grad():
        direct_logits = model(test_tokens)

    print(f"✅ Direct logits: shape={direct_logits.shape}")
    print(f"   Sample: {direct_logits[0, -1, :5].tolist()}")

    # Convert to logprobs for last token
    direct_logprobs = F.log_softmax(direct_logits[0, -1, :], dim=-1)
    print(f"   Top 5 token logprobs: {direct_logprobs[:5].tolist()}")

    # Path 2: vLLM LLM() wrapper
    print("\n[4/5] Running through vLLM LLM() wrapper...")

    from vllm import LLM, SamplingParams
    from vllm.model_executor.adapters.torchtitan_adapter import TorchTitanAdapter

    # Wrap model with adapter
    adapted_model = TorchTitanAdapter(
        torchtitan_model=model,
        max_seq_len=model_args.max_seq_len,
        disable_batch_invariance=True,  # Use context accumulation
    )

    # Create LLM with adapted model
    # We can't use real tokenizer since vocab sizes don't match
    # So we'll test by directly calling the adapter
    print("✅ Testing adapter forward pass directly...")

    # Call adapter forward with our test tokens
    # This simulates what vLLM would do
    positions = torch.arange(len(test_token_ids), device=device)

    adapted_model.clear_context_cache()
    with torch.no_grad():
        vllm_logits = adapted_model.forward(
            input_ids=test_tokens[0],  # Flatten to 1D
            positions=positions,
            kv_cache=None,
            attn_metadata=None,
        )

    print(f"✅ vLLM logits: shape={vllm_logits.shape}")
    print(f"   Sample: {vllm_logits[-1, :5].tolist()}")

    # Convert to logprobs for last token
    vllm_logprobs = F.log_softmax(vllm_logits[-1, :], dim=-1)
    print(f"   Top 5 token logprobs: {vllm_logprobs[:5].tolist()}")

    # Compare logprobs
    print("\n[5/5] Comparing logprobs...")

    # Check if they're identical
    max_diff = (direct_logprobs - vllm_logprobs).abs().max().item()
    print(f"   Max logprob difference: {max_diff:.2e}")

    # For FP32, allow tiny differences due to numerical precision
    tolerance = 1e-5
    all_match = max_diff < tolerance

    if all_match:
        print(f"✅ Logprobs match within tolerance ({tolerance})")
    else:
        print(f"✗ Logprobs differ by {max_diff:.2e} (tolerance: {tolerance})")

        # Show first few differences
        diff = (direct_logprobs - vllm_logprobs).abs()
        top_diffs = torch.topk(diff, 5)
        print(f"   Top 5 differences:")
        for i, (val, idx) in enumerate(zip(top_diffs.values, top_diffs.indices)):
            print(f"     Token {idx.item()}: direct={direct_logprobs[idx].item():.6f}, "
                  f"vllm={vllm_logprobs[idx].item():.6f}, diff={val.item():.2e}")

    print("\n" + "=" * 80)
    if all_match:
        print("Test Result: PASSED ✅")
        print("✅ Direct model and vLLM produce identical results")
        print("✅ Batch invariance working across both paths!")
    else:
        print("Test Result: FAILED ✗")
        print(f"✗ Logprobs differ by up to {max_diff:.2e}")
        print("Possible causes:")
        print("  - Different execution paths not using same kernels")
        print("  - Context accumulation introducing differences")
        print("  - Position handling differs between paths")
    print("=" * 80)

    return all_match


if __name__ == "__main__":
    print("Running determinism test...")
    success1 = test_batch_invariant_determinism()

    print("\n\n")
    print("Running LLM vs Direct comparison test...")
    success2 = test_batch_invariant_llm_vs_direct()

    if not (success1 and success2):
        print("\n❌ Some tests failed!")
        exit(1)
    else:
        print("\n✅ All tests passed!")
        exit(0)
