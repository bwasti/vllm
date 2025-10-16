#!/usr/bin/env python3
"""
Convert trained EAGLE3 model to vLLM-compatible format.

This script converts EAGLE3 models trained with hidden state prediction
(EAGLE3-style training) to vLLM inference format.

The hidden_state_projection layer is used during both training and inference:
- Training: Supervised learning to predict next hidden states
- Inference: Autoregressive speculation - predicted hidden states are fed
  back as input to generate multiple tokens ahead without querying the target model

Usage:
    python convert_eagle3_to_vllm.py \
        --input-dir ../llm_stuff/vllm_spec_decode/eagle3_qwen3_model/best_model \
        --output-dir ./eagle3_vllm_model
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import save_file


def convert_model(input_dir: Path, output_dir: Path):
    """Convert EAGLE3 model from training format to vLLM format."""

    print(f"Loading model from {input_dir}")

    # Load the trained model
    state_dict = torch.load(input_dir / "model.pt", map_location="cpu")
    config = json.load(open(input_dir / "config.json"))

    print(f"Original state dict keys: {len(state_dict)}")
    for key in list(state_dict.keys())[:5]:
        print(f"  {key}: {state_dict[key].shape}")

    # Calculate QKV dimensions
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    print(f"\nModel architecture:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  Q size: {q_size}, KV size: {kv_size} each")
    print(f"  Total QKV size: {q_size + 2 * kv_size}")

    # Convert parameter names to vLLM format
    new_state_dict = {}

    for key, value in state_dict.items():
        # Remove "model." prefix if present
        if key.startswith("model."):
            key = key[6:]

        # Handle weight transformations
        if "self_attn_qkv" in key:
            # Split self_attn_qkv into q_proj, k_proj, v_proj
            # Training model output is [hidden_size, input_size]
            # But vLLM expects [q_size + 2*kv_size, input_size]
            layer_prefix = key.replace("self_attn_qkv.weight", "")

            # Get the weight tensor
            weight = value  # [hidden_size, input_size]
            input_size = weight.shape[1]

            # Since training model only outputs hidden_size, we need to expand to full QKV size
            # Strategy: Use the full hidden_size for Q, and extract/duplicate for K, V
            q_weight = weight.clone()  # Use all of it for Q [hidden_size, input_size]

            # For K and V, duplicate the first kv_size rows
            k_weight = weight[:kv_size, :].clone()  # [kv_size, input_size]
            v_weight = weight[:kv_size, :].clone()  # [kv_size, input_size]

            # Save as separate weights
            new_state_dict[f"{layer_prefix}self_attn.q_proj.weight"] = q_weight
            new_state_dict[f"{layer_prefix}self_attn.k_proj.weight"] = k_weight
            new_state_dict[f"{layer_prefix}self_attn.v_proj.weight"] = v_weight

            print(f"Split {key}:")
            print(f"  q_proj: {q_weight.shape}")
            print(f"  k_proj: {k_weight.shape}")
            print(f"  v_proj: {v_weight.shape}")

        elif "mlp_gate_up" in key:
            # Split mlp_gate_up into gate_proj and up_proj
            layer_prefix = key.replace("mlp_gate_up.weight", "")
            weight = value  # [intermediate_size * 2, hidden_size]
            intermediate_size = weight.shape[0] // 2

            gate_weight = weight[:intermediate_size, :]
            up_weight = weight[intermediate_size:, :]

            new_state_dict[f"{layer_prefix}mlp.gate_proj.weight"] = gate_weight
            new_state_dict[f"{layer_prefix}mlp.up_proj.weight"] = up_weight

            print(f"Split {key} into gate_proj and up_proj")

        elif "mlp_down" in key:
            # Rename mlp_down to mlp.down_proj
            new_key = key.replace("mlp_down.", "mlp.down_proj.")
            new_state_dict[new_key] = value

        else:
            # Keep other weights as-is
            new_state_dict[key] = value

    print(f"\nConverted state dict keys: {len(new_state_dict)}")
    for key in sorted(new_state_dict.keys())[:15]:
        print(f"  {key}: {new_state_dict[key].shape}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save in safetensors format (preferred by vLLM)
    print(f"\nSaving to {output_dir}")
    save_file(new_state_dict, output_dir / "model.safetensors")

    # Also save as pytorch_model.bin for compatibility
    torch.save(new_state_dict, output_dir / "pytorch_model.bin")

    # Update config for vLLM
    vllm_config = {
        "architectures": ["Eagle3Qwen3ForCausalLM"],
        "model_type": "qwen2",  # Use qwen2 for Transformers compatibility
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config["rms_norm_eps"],
        "target_hidden_size": config["target_hidden_size"],
        "num_target_layers": config["num_target_layers"],
        "target_layers": config["target_layers"],
        "draft_vocab_size": config["vocab_size"],
        "torch_dtype": "bfloat16",
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(vllm_config, f, indent=2)

    print(f"\n✅ Model converted successfully!")
    print(f"   Output: {output_dir}")
    print(f"   Files:")
    print(f"     - model.safetensors ({(output_dir / 'model.safetensors').stat().st_size / 1e9:.2f}GB)")
    print(f"     - pytorch_model.bin ({(output_dir / 'pytorch_model.bin').stat().st_size / 1e9:.2f}GB)")
    print(f"     - config.json")


def main():
    parser = argparse.ArgumentParser(description="Convert EAGLE3 model to vLLM format")
    parser.add_argument("--input-dir", type=str, required=True, help="Input model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output model directory")
    args = parser.parse_args()

    convert_model(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
