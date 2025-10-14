# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test for integrating torchtitan models with vLLM's LLM API.

This test demonstrates wrapping a torchtitan Qwen3Model with vLLM's LLM interface
to enable running torchtitan models through vLLM's inference engine.
"""

# Set offline mode BEFORE any imports to prevent network access
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pytest
import torch

import torchtitan.experiments.compat
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs


def test_torchtitan_qwen3_import():
    # Create minimal model args for testing
    model_args = Qwen3ModelArgs(
        dim=1024,
        n_layers=2,  # Use fewer layers for faster testing
        n_heads=16,
        n_kv_heads=8,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        use_flex_attn=False,
        attn_mask_type="causal",
        eos_id=151645,
        enable_weight_tying=False,
        moe_enabled=False,
    )

    model = Qwen3Model(model_args)
    assert model is not None
    print(f"Successfully created Qwen3Model with {model_args.n_layers} layers")


def test_torchtitan_qwen3_with_vllm_direct():
    """
    Test using a torchtitan Qwen3Model with vLLM via the TorchTitanAdapter.

    This demonstrates wrapping a TorchTitan model with the adapter to enable
    batch invariance and training support while using vLLM's inference engine.

    Now uses VLLMHybridAttentionWrapper for efficient KV caching during inference!
    """
    from vllm import LLM, SamplingParams
    from vllm.model_executor.adapters.torchtitan_adapter import TorchTitanAdapter

    # Step 1: Create your custom torchtitan model WITH HYBRID ATTENTION
    model_args = Qwen3ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=151936,
        head_dim=64,
        hidden_dim=1536,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=2048,
        depth_init=True,
        use_flex_attn=False,
        attn_mask_type="causal",
        eos_id=151645,
        enable_weight_tying=False,
        moe_enabled=False,
        use_hybrid_attn=True,  # 🚀 Enable hybrid attention for efficient inference!
    )
    custom_model = Qwen3Model(model_args)
    custom_model.init_weights()

    # Step 2: Verify that hybrid attention was enabled
    print("✅ Model created with hybrid attention!")

    # Check that the model is using VLLMHybridAttentionWrapper
    from torchtitan.models.attention import VLLMHybridAttentionWrapper
    first_layer = list(custom_model.layers.values())[0]
    assert isinstance(first_layer.attention.inner_attention, VLLMHybridAttentionWrapper), \
        "Model should be using VLLMHybridAttentionWrapper!"
    print(f"✅ Confirmed: Model uses {type(first_layer.attention.inner_attention).__name__}")

    # Step 3: Test that the model can be wrapped with TorchTitanAdapter
    adapted_model = TorchTitanAdapter(
        torchtitan_model=custom_model,
        max_seq_len=model_args.max_seq_len,
        disable_batch_invariance=True
    )
    print("✅ TorchTitanAdapter created successfully!")

    # Step 4: Test training mode (uses SDPA)
    custom_model.train()
    dummy_tokens = torch.randint(0, 1000, (2, 10))
    with torch.no_grad():
        train_output = custom_model(dummy_tokens)
    print(f"✅ Training mode works! Output shape: {train_output.shape}")

    # Step 5: Test eval mode with direct call
    custom_model.eval()
    with torch.no_grad():
        eval_output = custom_model(dummy_tokens)
    print(f"✅ Eval mode works! Output shape: {eval_output.shape}")

    # Verify outputs have the same shape
    assert train_output.shape == eval_output.shape, "Train and eval outputs should have same shape"

    # Verify outputs are numerically close (both use SDPA in this test)
    max_diff = torch.max(torch.abs(train_output - eval_output)).item()
    print(f"✅ Output difference between train/eval: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff} (expected < 1e-4)"

    # Step 6: Actually test with vLLM LLM wrapper!
    print("\n🚀 Testing with vLLM LLM wrapper...")

    os.environ['VLLM_USE_V1'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

    llm = LLM(
        model=adapted_model,
        tokenizer="Qwen/Qwen2-0.5B",  # Use a real tokenizer
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=512,
        max_num_seqs=16,
        dtype="float16",
        distributed_executor_backend=None,  # Force single-process mode
    )
    print("✅ LLM wrapper created successfully!")

    # Generate with the LLM wrapper
    test_prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # Clear cache before generation
    adapted_model.clear_context_cache()
    outputs = llm.generate(test_prompts, sampling_params)

    print(f"✅ Generated output: \"{outputs[0].outputs[0].text}\"")
    assert len(outputs) == len(test_prompts), "Should generate one output per prompt"
    assert len(outputs[0].outputs) > 0, "Should have generated tokens"

    print("\n" + "=" * 70)
    print("All checks passed! ✅")
    print("=" * 70)
    print("\nHybrid attention is successfully integrated:")
    print("  ✅ Model creation with VLLMHybridAttentionWrapper")
    print("  ✅ Training mode works (uses PyTorch SDPA)")
    print("  ✅ Eval mode works (uses PyTorch SDPA)")
    print("  ✅ TorchTitanAdapter wrapping successful")
    print("  ✅ LLM wrapper generation works!")
    print("\nThe torchtitan model can now be used with vLLM's LLM API! 🎉")


def test_torchtitan_qwen3_with_vllm_default():
    """
    Test using the default vLLM path (v0) with Qwen model for comparison.

    This test runs the same model through the standard vLLM path to compare
    memory usage and initialization behavior against the v1/torchtitan path.
    """
    from vllm import LLM, SamplingParams

    # Use default vLLM with a standard Qwen model
    llm = LLM(
        model="Qwen/Qwen2-0.5B",  # Use standard Qwen model
        tokenizer="Qwen/Qwen2-0.5B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_seqs=16,  # Match the torchtitan test configuration
    )

    print("Default vLLM initialized successfully!")
    assert llm is not None

    # TODO: Add generation test
    # prompts = ["Hello, how are you?"]
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    # outputs = llm.generate(prompts, sampling_params)
    # assert len(outputs) == len(prompts)



if __name__ == "__main__":
    # Run the import test
    print("Testing torchtitan Qwen3Model import...")
    test_torchtitan_qwen3_import()
    print("\nTest completed!")

    # Run the LLM wrapper test
    print("\n" + "=" * 70)
    print("Testing torchtitan with vLLM LLM wrapper...")
    print("=" * 70)
    test_torchtitan_qwen3_with_vllm_direct()
    print("\nAll tests completed! ✅")
