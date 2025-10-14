# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter layer for TorchTitan models to work with vLLM's inference engine.

This adapter bridges the gap between TorchTitan's training-oriented model interface
and vLLM's inference-oriented interface. It handles the conversion of inputs/outputs
to maintain batch invariance while allowing training support.

PERFORMANCE CONSIDERATIONS:
===========================

Default Behavior (Standard SDPA):
---------------------------------
By default, TorchTitan models use PyTorch's scaled_dot_product_attention (SDPA),
which does NOT support incremental KV caching. This results in O(n²) complexity
during autoregressive generation, as attention is recomputed over all previous
tokens for each new token.

This is acceptable for:
  ✅ Research and experimentation with custom TorchTitan models
  ✅ Short sequences (< 512 tokens)
  ✅ Testing model architectures before production deployment
  ✅ Training-to-inference workflow validation

But NOT recommended for:
  ❌ Production long-context inference
  ❌ High-throughput serving with long sequences

Improved Performance (Hybrid Attention):
----------------------------------------
For efficient inference with O(n) decode complexity, use VLLMHybridAttentionWrapper
in your TorchTitan model. This provides:
  ✅ Full training support with PyTorch SDPA (gradients work)
  ✅ Efficient inference with vLLM KV caching (~8x faster)
  ✅ Automatic path selection based on train/eval mode

To enable hybrid attention in your TorchTitan model:

    from torchtitan.models.attention import VLLMHybridAttentionWrapper

    class Attention(nn.Module):
        def __init__(self, model_args):
            # Replace ScaledDotProductAttentionWrapper with:
            self.inner_attention = VLLMHybridAttentionWrapper(
                num_heads=self.n_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=self.n_kv_heads,
            )

See HYBRID_ATTENTION_GUIDE.md for complete documentation.
See KV_CACHE_ANALYSIS.md for technical deep-dive.
"""

import torch
import torch.nn as nn
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

class TorchTitanAdapter(nn.Module):
    """Adapter that wraps a TorchTitan model to work with vLLM's interface.

    TorchTitan models expect:
        forward(tokens, attention_masks=None, input_batch=None) -> logits
        - tokens: (batch_size, seq_len) token IDs
        - attention_masks: Optional attention masks
        - input_batch: Optional input batch for document masking

    vLLM expects:
        forward(input_ids, positions, kv_cache, attn_metadata, ...) -> hidden_states
        - input_ids: (num_tokens,) flattened token IDs
        - positions: (num_tokens,) token positions in each sequence
        - kv_cache: KV cache tensors (not used by TorchTitan models)
        - attn_metadata: Attention metadata (not used by TorchTitan models)

    This adapter converts between these interfaces while preserving batch invariance.
    """

    def __init__(self, torchtitan_model: nn.Module, max_seq_len: int = 4096, disable_batch_invariance: bool = False, warn_on_long_sequences: bool = True):
        """Initialize the adapter.

        Args:
            torchtitan_model: The TorchTitan model instance (e.g., Qwen3Model)
            max_seq_len: Maximum sequence length for the model
            disable_batch_invariance: If True, skip batch reconstruction (faster but not batch invariant)
            warn_on_long_sequences: If True, log warning when processing sequences > 512 tokens
        """
        logger.info("Initializing TorchTitanAdapter")
        super().__init__()
        self.model = torchtitan_model
        self.max_seq_len = max_seq_len
        self.disable_batch_invariance = disable_batch_invariance
        self.warn_on_long_sequences = warn_on_long_sequences
        self._warned_about_length = False  # Only warn once per instance

        # Context accumulation for correct RoPE positions
        # Maps request_id -> full token sequence
        self.context_cache = {}
        self._next_request_id = 0

        # TorchTitan models may need to be in eval mode for inference
        # But keep training mode for now to allow SDPA to work
        # self.model.eval()

        # Store model attributes that vLLM may need
        self.vocab_size = getattr(torchtitan_model, 'vocab_size', None)

        # Create a minimal config object to provide vLLM with reasonable defaults
        # This prevents vLLM from inferring huge default values
        if not hasattr(torchtitan_model, 'config'):
            from types import SimpleNamespace
            self.config = SimpleNamespace(
                vocab_size=self.vocab_size,
                max_position_embeddings=max_seq_len,
                hidden_size=getattr(torchtitan_model, 'dim', 512),
            )
        else:
            self.config = torchtitan_model.config

        logger.info(f"TorchTitanAdapter initialized: vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}, disable_batch_invariance={self.disable_batch_invariance}")

    def clear_context_cache(self):
        """Clear the context cache. Should be called between separate generation requests."""
        logger.info(f"Clearing context cache (had {len(self.context_cache)} entries)")
        self.context_cache = {}
        self._next_request_id = 0

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[object] = None,
        intermediate_tensors: Optional[dict] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass that converts vLLM inputs to TorchTitan format.

        Args:
            input_ids: (num_tokens,) flattened token IDs from vLLM
            positions: (num_tokens,) token positions in each sequence
            kv_cache: KV cache (ignored - TorchTitan manages its own attention)
            attn_metadata: Attention metadata (ignored)
            intermediate_tensors: Pipeline parallel tensors (ignored)
            inputs_embeds: Input embeddings (ignored, we use input_ids)
            **kwargs: Additional arguments (ignored)

        Returns:
            torch.Tensor: Output logits with shape (num_tokens, vocab_size)
        """

        if input_ids is None:
            raise ValueError("input_ids is required for TorchTitan adapter")

        # Warn about long sequences (O(n²) complexity without KV caching)
        num_tokens = input_ids.shape[0]
        if self.warn_on_long_sequences and not self._warned_about_length and num_tokens > 512:
            logger.warning(
                f"TorchTitanAdapter processing {num_tokens} tokens with O(n²) complexity. "
                "For efficient inference, use VLLMHybridAttentionWrapper in your TorchTitan model "
                "to enable KV caching (~8x faster). See HYBRID_ATTENTION_GUIDE.md for details."
            )
            self._warned_about_length = True

        # Fast path: if batch invariance is disabled, just treat everything as one sequence
        if self.disable_batch_invariance:
            # CONTEXT ACCUMULATION FIX:
            # TorchTitan models use rope_cache[0:seqlen], which assumes positions start at 0.
            # During autoregressive generation, vLLM passes actual positions (e.g., [10], [11]).
            #
            # Solution: Accumulate full context and pass complete sequences each time.
            # This ensures RoPE positions are correct. O(n²) but correct.

            logger.info(f"TorchTitanAdapter.forward: input_ids shape={input_ids.shape}, positions={positions}")
            logger.info(f"TorchTitanAdapter.forward: sample tokens={input_ids[:10]}")
            logger.info(f"TorchTitanAdapter.forward: Current context_cache keys: {list(self.context_cache.keys())}")

            # Ensure model is in eval mode for inference
            self.model.eval()

            # Extract positions to understand the request structure
            if positions is None:
                # No position info - treat as new request starting at position 0
                min_pos = 0
                max_pos = len(input_ids) - 1
                positions_list = list(range(len(input_ids)))
            else:
                if isinstance(positions, torch.Tensor):
                    positions_list = positions.cpu().tolist()
                else:
                    positions_list = list(positions)
                min_pos = min(positions_list)
                max_pos = max(positions_list)

            logger.info(f"TorchTitanAdapter.forward: positions range [{min_pos}, {max_pos}], positions_list={positions_list[:20]}")

            # PROBLEM: When vLLM processes a batch of prompts, they ALL may start at position 0
            # This means we can't use position alone to identify requests
            # For now, implement a more robust heuristic:
            # - If min_pos == 0 and len(input_ids) > 1, it's a NEW prefill request
            # - If min_pos == 0 and len(input_ids) == 1, it might be a single-token prefill OR decode step 0
            # - If min_pos > 0, it's definitely a decode step

            # Determine request ID based on position pattern
            if min_pos == 0 and len(input_ids) > 1:
                # New request prefill with multiple tokens
                request_id = self._next_request_id
                self._next_request_id += 1

                # Initialize context cache with these tokens
                self.context_cache[request_id] = input_ids.cpu().tolist()

                logger.info(f"TorchTitanAdapter: NEW REQUEST {request_id}, prefill with {len(input_ids)} tokens")
            elif min_pos == 0 and len(input_ids) == 1:
                # Single token at position 0 - could be new single-token prefill OR continuation
                # Check if we have any existing incomplete requests
                if len(self.context_cache) > 0:
                    # Assume it's a decode step for the most recent request
                    request_id = max(self.context_cache.keys())
                    logger.info(f"TorchTitanAdapter: AMBIGUOUS CASE - single token at pos 0, treating as decode for request {request_id}")
                else:
                    # No existing requests, treat as new
                    request_id = self._next_request_id
                    self._next_request_id += 1
                    self.context_cache[request_id] = input_ids.cpu().tolist()
                    logger.info(f"TorchTitanAdapter: NEW REQUEST {request_id}, single-token prefill")
            else:
                # Decode phase - find the matching request
                # For single-request case, use the most recent request
                if len(self.context_cache) == 0:
                    # Shouldn't happen, but handle gracefully
                    logger.warning("Context cache empty during decode step!")
                    request_id = 0
                    self.context_cache[request_id] = []
                else:
                    request_id = max(self.context_cache.keys())

                # Accumulate new tokens
                new_tokens = input_ids.cpu().tolist()
                self.context_cache[request_id].extend(new_tokens)

                logger.info(f"TorchTitanAdapter: DECODE STEP for request {request_id}, added {len(new_tokens)} tokens, total={len(self.context_cache[request_id])}")

            # Build full context tensor
            full_context = torch.tensor(
                [self.context_cache[request_id]],
                dtype=input_ids.dtype,
                device=input_ids.device
            )

            logger.info(f"TorchTitanAdapter: Running model with full context shape={full_context.shape}")
            logger.info(f"TorchTitanAdapter: Full context tokens: {full_context[0].tolist()}")
            logger.info(f"TorchTitanAdapter: Full context dtype: {full_context.dtype}")
            logger.info(f"TorchTitanAdapter: Model training mode: {self.model.training}")
            logger.info(f"TorchTitanAdapter: tok_embeddings dtype: {self.model.tok_embeddings.weight.dtype}")
            logger.info(f"TorchTitanAdapter: tok_embeddings[0]={self.model.tok_embeddings.weight[0, :5]}")

            # Run model with full context
            output = self.model(
                tokens=full_context,
                attention_masks=None,
                input_batch=full_context
            )

            logger.info(f"TorchTitanAdapter.forward: output shape={output.shape}")
            logger.info(f"TorchTitanAdapter.forward: output max={output.max().item():.4f}, min={output.min().item():.4f}")

            # Debug: Show top predictions for last token
            last_token_logits = output[0, -1, :]
            top5 = torch.topk(last_token_logits, 5)
            logger.info(f"TorchTitanAdapter: Top 5 predictions for last token: {top5.indices.tolist()}")
            logger.info(f"TorchTitanAdapter: Top 5 logits: {top5.values.tolist()}")

            # Output shape: (1, full_seq_len, vocab_size)
            # We need to return only the logits for the NEW tokens
            # The new tokens are at positions [min_pos:max_pos+1]

            # Extract only the logits for the positions we were asked about
            num_new_tokens = len(input_ids)
            if min_pos == 0:
                # Prefill: return all logits
                result = output[0, :num_new_tokens, :]
            else:
                # Decode: return logits for the new tokens
                # New tokens are at the END of the sequence
                result = output[0, -num_new_tokens:, :]

            logger.info(f"TorchTitanAdapter: Returning {result.shape[0]} token logits")
            logger.info(f"TorchTitanAdapter: Sample logits for last token: {result[-1, :10]}")

            return result

        # vLLM flattens the batch: input_ids is (num_tokens,) where num_tokens = sum(seq_lens)
        # We need to reconstruct the batch structure.

        logger.info(f"TorchTitanAdapter.forward: BATCH RECONSTRUCTION PATH - input_ids shape={input_ids.shape}")
        logger.info(f"TorchTitanAdapter.forward: attn_metadata type={type(attn_metadata)}, attrs={dir(attn_metadata) if attn_metadata else None}")

        # Extract batch structure from attn_metadata
        if attn_metadata is not None and hasattr(attn_metadata, 'num_reqs'):
            # We have metadata telling us the batch structure
            num_reqs = attn_metadata.num_reqs
            logger.info(f"TorchTitanAdapter.forward: num_reqs={num_reqs}")

            # Get sequence lengths from metadata
            if hasattr(attn_metadata, 'seq_lens'):
                # seq_lens tells us the current sequence length for each request
                seq_lens = attn_metadata.seq_lens
                if isinstance(seq_lens, torch.Tensor):
                    seq_lens = seq_lens.cpu().tolist()
            elif hasattr(attn_metadata, 'query_start_loc'):
                # Compute seq lens from query_start_loc
                query_start_loc = attn_metadata.query_start_loc
                if isinstance(query_start_loc, torch.Tensor):
                    query_start_loc = query_start_loc.cpu().tolist()
                seq_lens = [query_start_loc[i+1] - query_start_loc[i] for i in range(num_reqs)]
            else:
                # Fallback: assume uniform batch
                num_tokens = input_ids.shape[0]
                avg_len = num_tokens // num_reqs
                seq_lens = [avg_len] * num_reqs
        else:
            # Fallback: treat as single sequence
            num_reqs = 1
            seq_lens = [input_ids.shape[0]]

        # Reconstruct the batch
        # TorchTitan expects: (batch_size, seq_len)
        # We need to pad sequences to the same length
        max_seq_len_in_batch = max(seq_lens)

        # Create padded batch tensor
        batch_tokens = torch.zeros(
            (num_reqs, max_seq_len_in_batch),
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        # Fill in the actual tokens
        start_idx = 0
        for i, seq_len in enumerate(seq_lens):
            batch_tokens[i, :seq_len] = input_ids[start_idx:start_idx + seq_len]
            start_idx += seq_len

        # Run TorchTitan model
        # Note: TorchTitan models compute their own attention, including RoPE
        # They don't use vLLM's KV cache or attention metadata
        output = self.model(
            tokens=batch_tokens,
            attention_masks=None,  # TorchTitan will create causal masks internally
            input_batch=batch_tokens  # Pass batch for potential document masking
        )

        # Output shape: (batch_size, seq_len, vocab_size)
        # We need to flatten it back to (num_tokens, vocab_size)

        # Extract only the valid tokens (not padding)
        flat_output = []
        for i, seq_len in enumerate(seq_lens):
            flat_output.append(output[i, :seq_len, :])

        # Concatenate back to (num_tokens, vocab_size)
        flat_output = torch.cat(flat_output, dim=0)

        return flat_output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """vLLM calls this to compute logits from hidden states.

        For TorchTitan models, the forward pass already returns logits,
        so we just return them as-is.
        """
        return hidden_states

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[list] = None,
        is_multimodal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get input embeddings for tokens.

        TorchTitan models handle embeddings internally, but vLLM may call this
        for multimodal models. For now, we use the model's embedding layer.
        """
        if hasattr(self.model, 'tok_embeddings'):
            return self.model.tok_embeddings(input_ids)
        else:
            raise NotImplementedError("Model does not have tok_embeddings")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Forward to wrapped model
            return getattr(self.model, name)
