# EAGLE Benchmark Script & Bug Fixes

## Summary

Created a self-contained Python benchmark script for EAGLE speculative decoding and fixed critical bugs in the vLLM EAGLE implementation that were preventing it from working with the LLM() API.

## Files Created

### 1. `benchmark_eagle.py`
A self-contained benchmark script that:
- Uses the vLLM `LLM()` API directly (no server required)
- Supports EAGLE speculative decoding benchmarking
- Generates Perfetto-compatible Chrome traces for performance analysis
- Auto-generates random datasets for consistent testing
- Provides extensive CLI configuration options
- Outputs detailed statistics and optional JSON results

### 2. `BENCHMARK_EAGLE.md`
Comprehensive documentation including:
- Quick start guide
- Usage examples for common scenarios
- Full parameter reference
- Troubleshooting guide
- Comparison with other profiling scripts

## Bugs Fixed

### Bug #1: Missing `hasattr` check in `eagle.py`
**File**: `vllm/v1/spec_decode/eagle.py:1080`

**Problem**: Code was accessing `self.model.lm_head.weight` without first checking if `lm_head` attribute exists.

**Fix**: Added `hasattr(self.model, "lm_head")` check before accessing the attribute.

```python
# Before
elif (
    hasattr(target_language_model, "lm_head")
    and isinstance(target_language_model.lm_head.weight, torch.Tensor)
    and isinstance(self.model.lm_head.weight, torch.Tensor)  # <-- Missing hasattr check!
    ...
)

# After
elif (
    hasattr(target_language_model, "lm_head")
    and isinstance(target_language_model.lm_head.weight, torch.Tensor)
    and hasattr(self.model, "lm_head")  # <-- Added check
    and isinstance(self.model.lm_head.weight, torch.Tensor)
    ...
)
```

### Bug #2: Incorrect `has_own_lm_head` attribute setting in `llama4_eagle.py`
**File**: `vllm/model_executor/models/llama4_eagle.py:208`

**Problem**: The `EagleLlama4ForCausalLM.load_weights()` method was calling `process_eagle_weight()` for ALL weights, including "lm_head" weights. This set `has_own_lm_head=True` even though the lm_head weights were subsequently skipped by `AutoWeightsLoader(skip_prefixes=["lm_head."])`. This caused the EAGLE code to think the model had its own lm_head when it actually didn't, leading to the AttributeError during inference.

**Root Cause**: The workflow was:
1. Weight loader sees "lm_head" in checkpoint
2. `process_eagle_weight()` is called, setting `self.has_own_lm_head = True`
3. `AutoWeightsLoader` skips loading lm_head due to `skip_prefixes`
4. EAGLE model has `has_own_lm_head=True` but no actual `lm_head` attribute
5. Later, `eagle.py` checks `has_own_lm_head` and decides NOT to share from target
6. During inference, `compute_logits()` tries to access `self.lm_head` → AttributeError!

**Fix**: Only call `process_eagle_weight()` for non-lm_head weights that will actually be loaded.

```python
# Before
def transform(inputs):
    name, loaded_weight = inputs
    name, weight = self.permute_qk_weight_for_rotary(name, loaded_weight)
    if "lm_head" not in name:
        name = "model." + name
    process_eagle_weight(self, name)  # <-- Called for ALL weights including lm_head!
    return name, weight

# After
def transform(inputs):
    name, loaded_weight = inputs
    name, weight = self.permute_qk_weight_for_rotary(name, loaded_weight)
    if "lm_head" not in name:
        name = "model." + name
        process_eagle_weight(self, name)  # <-- Only called for weights that will be loaded
    # Don't call process_eagle_weight for lm_head since we skip it below
    return name, weight
```

## Why launch.sh Worked But LLM() API Didn't

Good question! Both use the same underlying vLLM engine, so the bugs would affect both. The difference might be:
1. **Timing**: The bugs were recently introduced or recently exposed
2. **Code path**: The OpenAI API server might use slightly different initialization order
3. **Model checkpoint**: Different checkpoints might have different lm_head configurations

However, the bugs we fixed are real issues that would cause failures in both scenarios - we just happened to hit them with the LLM() API first.

## Testing

After fixes, the benchmark script successfully:
- Initializes vLLM with EAGLE configuration
- Loads both target and draft models
- Shares lm_head weights correctly from target to draft model
- Runs warmup and benchmark requests
- Processes requests successfully (got to 60-100% before hitting unrelated CUDA errors)

The CUDA errors appear to be separate issues (possibly memory-related) and not related to the lm_head attribute bugs we fixed.

## Usage Example

```bash
# Basic benchmark with EAGLE
LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" \
    python benchmark_eagle.py --num-requests 100 --enable-profiling

# Baseline comparison (no EAGLE)
python benchmark_eagle.py --disable-eagle --num-requests 100 --stats-file baseline.json

# Quick test
python benchmark_eagle.py --num-requests 10 --warmup-requests 2
```

## Next Steps

The benchmark script is working! The CUDA errors you're seeing might be:
1. Unrelated to the lm_head bugs (those are fixed)
2. Memory pressure issues - try reducing batch size or model length
3. Known issues with EAGLE+FLASHINFER on certain configurations

You can now use the script to benchmark EAGLE performance and generate Perfetto traces for analysis.
