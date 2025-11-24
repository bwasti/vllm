# Async Scheduling Investigation - EAGLE Acceptance Rate

## Summary

Investigation into whether async scheduling is causing EAGLE's low acceptance rate (~5-10% vs expected 60-80%).

## Key Findings

### 1. Default Async Scheduling Value

**From `/home/bwasti/oss/vllm/vllm/config/scheduler.py:131`:**
```python
async_scheduling: bool = False
```

**The default value for `async_scheduling` is `False`!**

This means:
- LLM API without explicit config: `async_scheduling=False` (default)
- Server API: Need to check what default it uses

### 2. Test Results

| Test Configuration | Async Scheduling | Acceptance Rate | Throughput |
|-------------------|------------------|-----------------|------------|
| Default (no flag) | False (default) | 5-10% | 213 tok/s |
| `--disable-async-scheduling` | False (explicit) | 5-10% | 213 tok/s |
| `--enable-async-scheduling` | True (explicit) | 5-15% | 254 tok/s |

**Conclusion: Async scheduling status does NOT significantly affect acceptance rate!**

Both enabled and disabled async scheduling show similarly low acceptance rates (~5-15%).

### 3. User's Statement Analysis

User said: "when I run without async scheduling it works fine (the server API)"

**Possible interpretations:**
1. User might be confused about which setting corresponds to "with/without" async
2. Server API might have different defaults or configuration
3. There might be another difference between server and LLM API beyond async scheduling

### 4. What IS Affected by Async Scheduling

- **Throughput**: Slightly better with async enabled (254 vs 213 tok/s)
- **Latency**: Async scheduling should improve latency
- **Acceptance rate**: NOT significantly affected (both ~5-15%)

### 5. Async Scheduling Compatibility

From `/home/bwasti/oss/vllm/vllm/config/vllm.py:378-385`:
```python
# Currently, async scheduling only support eagle speculative decoding.
if self.speculative_config is not None:
    if self.speculative_config.method not in get_args(EagleModelTypes):
        raise ValueError(
            "Currently, async scheduling is only supported "
            "with EAGLE/MTP kind of speculative decoding"
        )
```

**Async scheduling IS supported with EAGLE** - no compatibility issues.

## Root Cause Still Unknown

The low acceptance rate persists regardless of async scheduling setting. This means:

1. ❌ **NOT async scheduling** causing the low acceptance rate
2. ❌ **NOT our synchronization fixes** (overhead is only 0.15%)
3. ⚠️ **Possibly temperature mismatch** (still most likely - need to test with temp=0)
4. ⚠️ **Possibly model calibration issue** (draft model not well-calibrated)
5. ⚠️ **Possibly another LLM vs Server API difference** we haven't identified

## Next Steps

1. **Test with greedy decoding** (temperature=0, top_p=1.0) to rule out sampling mismatch
2. **Check server API configuration** to see what's different from LLM API
3. **Investigate draft model training settings** to see if it was trained for greedy vs sampling
4. **Profile with different batch sizes** to see if batching affects acceptance rate

## Files Modified

- `benchmark_eagle.py`: Added `--enable-async-scheduling` and `--disable-async-scheduling` flags
- `benchmark_eagle.py`: Updated config printing to show async scheduling status

## Test Logs

- `/tmp/benchmark_no_async.log` - Test with `--disable-async-scheduling` (redundant, already default)
- `/tmp/benchmark_async_enabled.log` - Test with `--enable-async-scheduling`

Both show similarly low acceptance rates.
