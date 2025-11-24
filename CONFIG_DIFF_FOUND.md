# Config Comparison: Server API vs LLM API

## CRITICAL DIFFERENCES FOUND

### Server API Config (from /tmp/server_logs.txt):
```
enable_prefix_caching=True
trust_remote_code=False
trust_remote_code=True  # (note: both values present in log, using False)
```

### LLM API Config (from benchmark_eagle.py):
```python
"enable_prefix_caching": False,
"trust_remote_code": True,
```

## Hypothesis

**Prefix caching might be affecting EAGLE acceptance rates!**

If prefix caching is enabled on the server but disabled in LLM API, this could cause:
1. Different KV cache behavior
2. Different batching/scheduling
3. Different token verification paths

## Test Plan

Run benchmark_eagle.py with `enable_prefix_caching=True` to see if acceptance rate improves.

## Other Config Differences

Looking at the full engine init logs, both use:
- `enable_chunked_prefill=True`
- `max_num_batched_tokens=2048` (for draft model workers)
- `async_scheduling=False` (default)
- `tensor_parallel_size=8`
- `max_seq_len=1536`

So prefix caching is the main difference!
