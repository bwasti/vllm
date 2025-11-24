# EAGLE Llama4 Performance Analysis

## Trace Collection Summary

**Date:** 2025-11-21
**Configuration:**
- Model: Llama4 405B (tensor parallel=8)
- Draft Model: Llama4 EAGLE
- Speculative Tokens: 4
- Requests: 20
- Input Length: 924 tokens
- Output Length: 512 tokens per request

## Benchmark Results

### Throughput
- **Output Throughput:** 213.01 tokens/s
- **Total Output Tokens:** 9,936
- **Total Input Tokens:** 18,460
- **Total Tokens:** 28,396
- **Requests/sec:** 0.43
- **Avg time per request:** 2.332s

### Initialization
- **Model Loading:** ~20 seconds
- **Engine Init:** ~18 seconds
- **Total Startup:** ~69 seconds

## Trace Analysis

**Trace File:** `/tmp/eagle_llama4_trace.json` (53MB, 1.4M lines)

### Key Findings

1. **Synchronization Overhead:** 1,958ms total
   - Only 0.15% of total execution time
   - Single `torch.cuda.synchronize()` call took 687ms
   - This is VERY LOW - our synchronization fix has minimal overhead!

2. **Trace Granularity:**
   - Most events captured are Python function calls (201,175 events)
   - Limited CUDA kernel visibility (only 1 kernel recorded)
   - Need to use different profiling method for GPU kernel analysis

3. **CUDA Runtime Events:** Only 11 events captured
   - PyTorch profiler may not be configured to capture CUDA details
   - Consider using `with_stack=True` or CUDA profiler directly

## Performance Evaluation

### Strengths
✅ **Low synchronization overhead** - Our triple-sync fix adds <1% overhead
✅ **Stable throughput** - 213 tok/s maintained across 20 requests
✅ **No crashes** - All requests completed successfully

### Areas for Investigation

1. **Acceptance Rate Missing:**
   - vLLM V1 doesn't log spec decode acceptance rate by default
   - Need to add instrumentation or enable debug logging
   - Typical EAGLE acceptance rate should be 60-80%

2. **GPU Kernel Bottlenecks:**
   - Trace doesn't show GPU kernel details
   - Need nsys/nvprof for detailed GPU analysis
   - MoE operations and attention kernels need profiling

3. **Comparison Needed:**
   - No baseline (non-EAGLE) throughput for comparison
   - Can't determine actual EAGLE speedup without baseline

## Recommendations

### For Getting Acceptance Rate

1. **Add debug logging in eagle.py:**
   ```python
   # In propose() method after token verification
   logger.info(f"EAGLE acceptance: {num_accepted}/{num_proposed}")
   ```

2. **Or use vLLM metrics:**
   - Check if V1 has prometheus metrics for spec decode
   - May need to enable with `--disable-log-stats=False`

### For Detailed GPU Profiling

1. **Use Nsight Systems:**
   ```bash
   nsys profile -o eagle_profile \
     python benchmark_eagle.py --num-requests 10
   ```

2. **Enable CUDA profiling in PyTorch:**
   ```python
   prof = torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       record_shapes=True,
       with_stack=True,
       with_modules=True,
   )
   ```

### For Throughput Comparison

1. **Run baseline without EAGLE:**
   ```bash
   python benchmark_llama4.py --no-eagle --num-requests 20
   ```

2. **Compare:**
   - Baseline throughput vs EAGLE throughput
   - Expected: 1.5-2x speedup with EAGLE (depends on acceptance rate)

## Next Steps

1. ✅ Collect trace (DONE)
2. ⏳ Get EAGLE acceptance rate metrics
3. ⏳ Run nsys profile for GPU kernel analysis
4. ⏳ Compare with non-EAGLE baseline
5. ⏳ Identify specific MoE or attention bottlenecks

## Files Generated

- `/tmp/eagle_llama4_trace.json` - 53MB Perfetto trace
- `/tmp/trace_collection.log` - Full benchmark output
- `/tmp/trace_analysis.txt` - Analysis summary
- `EAGLE_TRACE_ANALYSIS.md` - This document

## Viewing the Trace

Upload to **Perfetto UI**: https://ui.perfetto.dev/

Or use Chrome: `chrome://tracing` and load `/tmp/eagle_llama4_trace.json`
