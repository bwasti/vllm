#!/usr/bin/env python3
"""
REAL DEMO: TorchTitan + vLLM Integration with LLM() class
This demonstrates full integration with vLLM's LLM.generate() API
"""

import os
# 🔑 KEY: Enable batch invariance for deterministic behavior
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_USE_V1"] = "1"  # Use V1 engine
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # 🔑 KEY: Disable multiprocessing to avoid pickling
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Offline mode to use cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import torch
import argparse

# Import torchtitan
import torchtitan.experiments.compat
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

# Import vLLM's LLM class
from vllm import LLM, SamplingParams

# Import vLLM's TorchTitanAdapter
sys.path.insert(0, 'vllm/model_executor/adapters')
from torchtitan_adapter import TorchTitanAdapter


def load_hf_weights(model: Qwen3Model, model_name: str) -> int:
    """Load HuggingFace weights into a TorchTitan model."""
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"  Loading from HuggingFace: {model_name}...")

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Load in fp32 for accurate weight transfer
        trust_remote_code=True,
    )

    hf_state_dict = hf_model.state_dict()
    tt_state_dict = model.state_dict()

    # Weight mapping from HF to TorchTitan
    weight_map = {
        'model.embed_tokens.weight': 'tok_embeddings.weight',
        'lm_head.weight': 'output.weight',
        'model.norm.weight': 'norm.weight',
    }

    # Layer mappings
    for layer_idx in range(hf_config.num_hidden_layers):
        layer_prefix_hf = f'model.layers.{layer_idx}'
        layer_prefix_tt = f'layers.{layer_idx}'

        weight_map.update({
            # Attention
            f'{layer_prefix_hf}.self_attn.q_proj.weight': f'{layer_prefix_tt}.attention.wq.weight',
            f'{layer_prefix_hf}.self_attn.k_proj.weight': f'{layer_prefix_tt}.attention.wk.weight',
            f'{layer_prefix_hf}.self_attn.v_proj.weight': f'{layer_prefix_tt}.attention.wv.weight',
            f'{layer_prefix_hf}.self_attn.o_proj.weight': f'{layer_prefix_tt}.attention.wo.weight',
            # Feed-forward
            f'{layer_prefix_hf}.mlp.gate_proj.weight': f'{layer_prefix_tt}.feed_forward.w1.weight',
            f'{layer_prefix_hf}.mlp.down_proj.weight': f'{layer_prefix_tt}.feed_forward.w2.weight',
            f'{layer_prefix_hf}.mlp.up_proj.weight': f'{layer_prefix_tt}.feed_forward.w3.weight',
            # Norms
            f'{layer_prefix_hf}.input_layernorm.weight': f'{layer_prefix_tt}.attention_norm.weight',
            f'{layer_prefix_hf}.post_attention_layernorm.weight': f'{layer_prefix_tt}.ffn_norm.weight',
        })

        # Q/K norm (if present)
        if f'{layer_prefix_hf}.self_attn.q_norm.weight' in hf_state_dict:
            weight_map.update({
                f'{layer_prefix_hf}.self_attn.q_norm.weight': f'{layer_prefix_tt}.attention.q_norm.weight',
                f'{layer_prefix_hf}.self_attn.k_norm.weight': f'{layer_prefix_tt}.attention.k_norm.weight',
            })

    # Load weights
    loaded_count = 0
    skipped_count = 0
    for hf_name, tt_name in weight_map.items():
        if hf_name in hf_state_dict and tt_name in tt_state_dict:
            if hf_state_dict[hf_name].shape == tt_state_dict[tt_name].shape:
                tt_state_dict[tt_name].copy_(hf_state_dict[hf_name])
                loaded_count += 1
            else:
                print(f"  ⚠️  Shape mismatch: {hf_name} {hf_state_dict[hf_name].shape} != {tt_name} {tt_state_dict[tt_name].shape}")
                skipped_count += 1
        else:
            if hf_name not in hf_state_dict:
                print(f"  ⚠️  Missing in HF: {hf_name}")
            if tt_name not in tt_state_dict:
                print(f"  ⚠️  Missing in TT: {tt_name}")
            skipped_count += 1

    model.load_state_dict(tt_state_dict)

    if skipped_count > 0:
        print(f"  ⚠️  Skipped {skipped_count} weights due to mismatches")

    return loaded_count


def main():
    print("\n" + "=" * 80)
    print("REAL DEMO: TorchTitan + vLLM LLM() Integration")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-hf-model', type=str, default=None,
                       help='Load weights from HuggingFace model (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                       help='Prompt text for generation')
    parser.add_argument('--max-tokens', type=int, default=20,
                       help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # Note: Batch invariance will be automatically initialized by vLLM V1 engine
    # No need to call init_batch_invariance() here - it would cause double registration
    print("\n[1/6] Batch invariance will be initialized by vLLM V1...")
    print("✅ VLLM_BATCH_INVARIANT=1 is set")

    # Flash Attention will be imported by VLLMHybridAttentionWrapper
    print("\n[2/6] Flash Attention...")
    print("✅ Flash Attention will be imported automatically by VLLMHybridAttentionWrapper")
    print("   If flash_attn library is available, it will be used")
    print("   Otherwise, PyTorch SDPA will be used as fallback")

    # Create TorchTitan model
    print("\n[3/6] Creating TorchTitan Qwen3Model...")

    if args.load_hf_model:
        # Load config from HF
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(args.load_hf_model, trust_remote_code=True)

        model_args = Qwen3ModelArgs(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            vocab_size=hf_config.vocab_size,
            head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
            hidden_dim=hf_config.intermediate_size,
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            qk_norm=True,
            max_seq_len=hf_config.max_position_embeddings,
            depth_init=False,
            use_flex_attn=False,
            attn_mask_type="causal",
            eos_id=hf_config.eos_token_id,
            enable_weight_tying=getattr(hf_config, 'tie_word_embeddings', False),
            moe_enabled=False,
            use_hybrid_attn=True,  # Use VLLMHybridAttentionWrapper for vLLM compatibility
        )
        tokenizer_path = args.load_hf_model
    else:
        # Use small test model with MATCHING vocab size to tokenizer (critical!)
        # Qwen2-0.5B tokenizer has vocab_size=151936
        model_args = Qwen3ModelArgs(
            dim=512,
            n_layers=2,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=151936,  # 🔑 MUST match tokenizer vocab size!
            head_dim=64,
            hidden_dim=1536,
            norm_eps=1e-6,
            rope_theta=1000000,
            qk_norm=True,
            max_seq_len=512,
            depth_init=False,
            use_flex_attn=False,
            attn_mask_type="causal",
            eos_id=151645,
            enable_weight_tying=False,
            moe_enabled=False,
            use_hybrid_attn=True,
        )
        # For test model, we need a tokenizer - use a locally cached one
        tokenizer_path = "Qwen/Qwen2-0.5B"  # Use Qwen2 tokenizer (cached locally)

    model = Qwen3Model(model_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load HF weights if specified (BEFORE dtype conversion!)
    if args.load_hf_model:
        print(f"\n[4/5] Loading HuggingFace weights...")
        # Initialize in fp32 for loading
        model.init_weights()
        model = model.to(device=device, dtype=torch.float32)
        loaded_count = load_hf_weights(model, args.load_hf_model)
        print(f"✅ Loaded {loaded_count} weight tensors from {args.load_hf_model}")
        # Now convert to bfloat16
        model = model.to(dtype=torch.bfloat16)
        print(f"✅ Converted to bfloat16")
        # Set to eval mode for inference
        model.eval()
        print(f"✅ Set to eval mode")
    else:
        print(f"\n[4/5] Using random weights (test mode)")
        model.init_weights()
        model = model.to(device=device, dtype=torch.bfloat16)

    print(f"✅ Model created: {model_args.n_layers} layers, vocab={model_args.vocab_size}")
    print(f"   Device: {device}")

    # Wrap with TorchTitanAdapter
    print("\n[5/5] Wrapping with TorchTitanAdapter...")
    adapter = TorchTitanAdapter(
        torchtitan_model=model,
        max_seq_len=model_args.max_seq_len,
        disable_batch_invariance=True,
        warn_on_long_sequences=False,
    )
    print("✅ Wrapped model with TorchTitanAdapter")

    # Create vLLM LLM instance
    print("\n[6/6] Creating vLLM LLM instance...")
    print(f"   Using tokenizer: {tokenizer_path}")

    try:
        llm = LLM(
            model=adapter,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=model_args.max_seq_len,
            enforce_eager=True,  # Disable CUDA graphs for simpler execution
            distributed_executor_backend=None,  # 🔑 KEY: Force single-process mode (no multiprocessing)
        )
        print("✅ vLLM LLM instance created")

        # Generate text
        print("\n" + "=" * 80)
        print("GENERATION TEST")
        print("=" * 80)

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=args.max_tokens,
        )

        print(f"\nPrompt: {args.prompt}")
        print(f"Generating {args.max_tokens} tokens...\n")

        outputs = llm.generate(
            prompts=[args.prompt],
            sampling_params=sampling_params,
        )

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")

        print("\n✅ Generation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error creating LLM or generating: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Integration with LLM() class may have limitations due to:")
        print("  - Pickling issues with TorchTitan models in multiprocessing")
        print("  - Model architecture compatibility requirements")
        print("  - See WEIGHT_LOADING_README.md for more details")
        return False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Full vLLM LLM() integration working!")
    print("\nWhat this demonstrates:")
    print("  ✅ Wrap TorchTitan model with TorchTitanAdapter")
    print("  ✅ Pass adapter to vLLM's LLM() class")
    print("  ✅ Use LLM.generate() for text generation")
    print("  ✅ Full vLLM inference pipeline")

    if args.load_hf_model:
        print("  ✅ Loaded real HuggingFace weights")

    print("\nUsage:")
    print("  # With random weights (fast test):")
    print("  python REAL_DEMO_WITH_LLM.py --prompt 'Hello, world!'")
    print("\n  # With HuggingFace weights:")
    print("  python REAL_DEMO_WITH_LLM.py --load-hf-model Qwen/Qwen3-0.6B --prompt 'Once upon a time'")

    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
