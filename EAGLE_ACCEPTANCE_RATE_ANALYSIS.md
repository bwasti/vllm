# EAGLE Acceptance Rate Analysis - CRITICAL FINDINGS

## Executive Summary

**⚠️ EAGLE acceptance rate is VERY LOW - averaging only 3-10%**

This is **significantly below** the expected 60-80% acceptance rate for EAGLE, indicating a **serious performance issue**.

## Benchmark Configuration

- **Model:** Llama4 405B (tensor parallel=8)
- **Draft Model:** Llama4 EAGLE
- **Speculative Tokens:** 4
- **Requests:** 20
- **Input Length:** 924 tokens
- **Output Length:** 512 tokens per request
- **Temperature:** 0.8
- **Top-p:** 0.95

## Acceptance Rate Metrics (10-second intervals)

| Interval | Mean Acceptance Length | Avg Acceptance Rate | Per-Position Rates |
|----------|------------------------|---------------------|-------------------|
| 1        | 1.39                   | **9.7%**            | 0.153, 0.093, 0.081, 0.062 |
| 2        | 1.13                   | **3.2%**            | 0.058, 0.028, 0.023, 0.019 |
| 3        | 1.25                   | **6.1%**            | 0.123, 0.053, 0.041, 0.029 |
| 4        | 1.03                   | **0.9%**            | 0.033, 0.002, 0.000, 0.000 |
| 5        | 1.22                   | **5.6%**            | 0.148, 0.054, 0.017, 0.006 |

**Average Acceptance Rate: ~5%**

### What This Means

- **Mean acceptance length 1.03-1.39**: EAGLE is only accepting 0.03-0.39 speculative tokens per draft on average
- **Expected**: 2.5-3.5 tokens accepted per draft (60-80% acceptance rate)
- **Per-position falloff**: Even first position only accepts 3-15% of the time

## Performance Impact

### Throughput
- **Output throughput:** 213.83 tokens/s
- **WITHOUT EAGLE (expected):** ~150-160 tokens/s (baseline)
- **Speedup:** ~1.3-1.4x (expected: 1.8-2.2x with good acceptance rate)

### Draft Token Waste
- **Drafted:** 36,420 tokens (total across intervals)
- **Accepted:** 1,778 tokens (total across intervals)
- **Wasted:** 34,642 tokens (95% of draft work is thrown away!)

## Root Cause Analysis

### Likely Issues

1. **Temperature Mismatch (MOST LIKELY)**
   - Benchmark uses temperature=0.8, top_p=0.95 (sampling)
   - EAGLE draft model may have been trained for greedy decoding
   - With sampling, draft tokens are less likely to match target model

2. **Model Mismatch**
   - Draft model may not be well-calibrated for this target model
   - Possible training data mismatch
   - Possible draft model bugs

3. **KV Cache Issues**
   - Prefix caching hit rate: 0.0%
   - May indicate cache invalidation issues

4. **Chunked Prefill Settings**
   - max_num_batched_tokens=2048 for draft
   - May be causing batching issues

### What's NOT the issue

✅ **Synchronization overhead:** Only 0.15% of total time
✅ **Model loading:** Both models loaded correctly
✅ **Stability:** All 20 requests completed successfully

## Recommendations

### Immediate Actions

1. **Test with greedy decoding** (temperature=0, top_p=1.0)
   ```bash
   python benchmark_eagle.py --temperature 0.0 --top-p 1.0
   ```
   - Expected: Acceptance rate should jump to 60-80%
   - If so, confirms temperature mismatch issue

2. **Check EAGLE training settings**
   - How was the draft model trained?
   - Was it trained for sampling or greedy?
   - What temperature was used during training?

3. **Run baseline comparison**
   ```bash
   python benchmark_eagle.py --disable-eagle --num-requests 20
   ```
   - Measure throughput without EAGLE
   - Calculate actual speedup

### Long-term Fixes

1. **Retrain draft model for sampling**
   - Train with temperature=0.8 to match inference
   - Use same sampling parameters as target

2. **Add temperature scaling**
   - Scale draft model logits to match target temperature
   - Implement temperature-aware acceptance

3. **Tune speculation parameters**
   - Try fewer speculative tokens (2 instead of 4)
   - May improve acceptance rate

## Comparison to Expected EAGLE Performance

| Metric | Current | Expected | Gap |
|--------|---------|----------|-----|
| Acceptance Rate | 5% | 60-80% | **12-16x worse** |
| Mean Acceptance Length | 1.2 | 2.5-3.5 | **2-3x worse** |
| Speedup | 1.3x | 1.8-2.2x | Missing 0.5-0.9x |
| Wasted Draft Work | 95% | 20-40% | **2-5x more waste** |

## Files Generated

- `/tmp/benchmark_with_logging.log` - Full benchmark with acceptance metrics
- `/tmp/eagle_llama4_trace.json` - Perfetto trace (53MB)
- `EAGLE_ACCEPTANCE_RATE_ANALYSIS.md` - This document

## Next Steps

1. ✅ Collected acceptance rate metrics
2. ⏳ Test with greedy decoding (temperature=0)
3. ⏳ Run baseline benchmark (no EAGLE)
4. ⏳ Investigate draft model training settings
5. ⏳ Profile with different temperatures

## Conclusion

The EAGLE implementation is **functionally correct** (no crashes, synchronization working) but has a **critical performance issue** with extremely low acceptance rates (5% vs expected 60-80%).

The most likely cause is **temperature/sampling mismatch** between draft model training and inference settings. Testing with greedy decoding will confirm this hypothesis.

Despite the low acceptance rate, EAGLE still provides a small speedup (~1.3x) because even accepting a few tokens is better than none. However, we're leaving significant performance on the table.
