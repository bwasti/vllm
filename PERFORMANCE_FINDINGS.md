# EAGLE Performance Investigation Findings

## Profiling Analysis (2025-11-26)

### Overview
Profiling revealed that the **primary bottleneck is severe load imbalance across tensor parallel ranks**, NOT individual CPU overhead operations.

---

## Issue 1: SEVERE LOAD IMBALANCE ACROSS TP RANKS (PRIMARY ISSUE)

**Problem**:
Different GPUs have wildly different utilization levels, indicating poor work distribution.

**Evidence from profiling:**

### With EAGLE enabled (`./traces`):
- **Average GPU util: 72.4%**
- **Rank 1: 45.2%** (WORST - idle 55% of time!)
- **Rank 2: 61.6%**
- **Rank 3: 64.0%**
- **Rank 0: 84.4%** (BEST)
- **Spread: 39.2% difference** between best and worst

### Without EAGLE (`./traces_no_eagle`):
- **Average GPU util: 67.1%** (WORSE than with EAGLE!)
- **Rank 0: 41.7%** (WORST - flipped!)
- **Rank 3: 42.0%**
- **Rank 1: 83.9%** (BEST - was worst with EAGLE!)
- **Spread: 42.2% difference**

**Key Insights:**

1. **EAGLE actually IMPROVES GPU utilization** (72.4% vs 67.1%)
2. **Load imbalance is NOT EAGLE-specific** - it exists in both cases
3. **Different ranks bottleneck** depending on EAGLE enabled/disabled
4. **This is a fundamental MoE + TP=8 issue**

**Root Cause:**

This is a **Llama 4 MoE model** with:
- **128 experts** (`num_local_experts: 128`)
- **1 expert per token** (`num_experts_per_tok: 1`)
- **24 MoE layers** (layers 1, 3, 5, 7, ..., 47 - every other layer)
- **TP=8** tensor parallelism

The load imbalance is likely caused by:

1. **Expert routing imbalance**: With 128 experts sharded across 8 GPUs (16 experts/GPU), token routing may not be uniform
   - Some GPUs get more active experts than others
   - Expert selection is data-dependent and dynamic

2. **MoE layer communication patterns**: Every other layer is MoE, requiring all-to-all communication
   - If expert routing is unbalanced, some ranks wait for others
   - Stragglers slow down the entire batch

3. **Shared expert overhead**: MoE layers have `shared_expert` components that may execute on specific ranks

**How Load Imbalance Normally Gets Fixed:**

1. **Expert placement optimization**:
   - Rebalance expert assignment across GPUs based on routing statistics
   - Use expert parallelism (EP) in addition to TP

2. **Load balancing losses**:
   - Router auxiliary loss to encourage balanced expert selection
   - Model already has: `"router_aux_loss_coef": 0.001`

3. **Different parallelism strategy**:
   - Try expert parallelism: `--enable-expert-parallelism`
   - Reduce TP, increase EP (e.g., TP=4, EP=2 instead of TP=8)
   - Use different sharding strategy for experts

4. **Check vLLM MoE backend options**:
   ```bash
   # Try different MoE implementations
   --moe-tp-strategy <strategy>  # Options may include: balanced, per_layer, etc.
   ```

**Status**: This is the PRIMARY performance bottleneck. The 27-33% GPU idle time is mostly due to load imbalance, not CPU overhead.

---

## Issue 2: FP8 Quantization (NOT AN ISSUE)

**Location**: Model config `/data/users/bwasti/wearable_maverick_vllm/config.json`

**Status**: FP8 quantization is INTENTIONAL and GOOD for performance.

```json
"quantization_config": {
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "dynamic": false,
                "strategy": "channel",
                "symmetric": true
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "dynamic": true,
                "strategy": "token",
                "symmetric": true
            }
        }
    }
}
```

**Conclusion**:
FP8 is being used correctly and should NOT be disabled. This is expected and optimal.

---

## Issue 3: shm_broadcast Dequeue Blocking (SECONDARY)

**Location**: `vllm/distributed/device_communicators/shm_broadcast.py:563` (dequeue operation)

**Problem**:
Profiling traces show that `shm_broadcast.py:563 dequeue` is blocking execution.

**Root Cause**:
The `dequeue` operation calls `acquire_read()` which does a **busy-wait spin loop** waiting for data:

```python
# shm_broadcast.py lines 473-506
while True:
    with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
        read_flag = metadata_buffer[self.local_reader_rank + 1]
        written_flag = metadata_buffer[0]
        if not written_flag or read_flag:
            # Block is not ready - spin wait
            self._read_spin_timer.spin()  # Does sched_yield()
            continue
        # Data ready - read it
        ...
```

**Profiling Results:**

With EAGLE:
- 19 gaps with shm_broadcast (total: 61.08 ms, ~26% of gap time)

Without EAGLE:
- Similar shm_broadcast overhead

**Code to Check:**

Two call sites for `dequeue()`:

1. **multiproc_executor.py:338** - Executor waiting for worker responses
   ```python
   # After sending RPC to workers, wait for responses
   status, result = mq.dequeue(timeout=dequeue_timeout, cancel=shutdown_event)
   ```
   - This is the executor waiting for worker to finish executing model forward pass
   - Expected to block - GPU should be busy during this time

2. **multiproc_executor.py:808** - Worker busy loop waiting for next command
   ```python
   # Worker waiting for next RPC command from executor
   method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
       cancel=cancel, indefinite=True
   )
   ```
   - This is the worker idle, waiting for work
   - Expected to block - this is when GPU SHOULD be idle

**Status**: Secondary issue. This accounts for ~26% of gap time, but the primary issue is load imbalance.

---

## Issue 4: _prepare_inputs Poor GPU Utilization (SECONDARY)

**Location**: `vllm/v1/worker/gpu_model_runner.py` (`_prepare_inputs` method)

**Profiling Results:**

With EAGLE:
- 40 gaps with prepare_input (total: 64.26 ms, ~28% of gap time)

Without EAGLE:
- Similar prepare_input overhead

**Status**: Secondary issue. Accounts for ~28% of gap time, but overshadowed by load imbalance.

---

## Summary & Recommendations

### Primary Issue: Load Imbalance (67-72% GPU util)

**Action Items:**

1. **Try Expert Parallelism**:
   ```bash
   python benchmark_eagle.py --enable-expert-parallelism
   ```

2. **Try different TP configuration**:
   ```bash
   # TP=4 instead of TP=8
   python benchmark_eagle.py --tp-size 4
   ```

3. **Check vLLM MoE optimization flags**:
   - Look for `--moe-tp-strategy` or similar options
   - Check if there's a load balancing mode

4. **Profile individual ranks**:
   - Understand which MoE experts are hot
   - Identify if specific experts are bottlenecks

### Secondary Issues: CPU Overhead (~54% of gap time)

Only worth optimizing AFTER fixing load imbalance:

- `shm_broadcast`/`dequeue`: ~26% of gap time
- `prepare_input`: ~28% of gap time
- Other: ~46% of gap time

---

## Testing Commands

### Analyze profiling traces:
```bash
# Quick summary across all GPUs
python quick_gap_summary.py --trace-dir ./traces

# Detailed gap analysis
python analyze_gpu_gaps.py --trace-dir ./traces --top-n 20

# Search for specific patterns
python analyze_gpu_gaps.py --trace-dir ./traces --search shm_broadcast
```

### Compare configurations:
```bash
# With EAGLE (current best: 72.4% util)
python benchmark_eagle.py --enable-profiling

# Without EAGLE (worse: 67.1% util)
python benchmark_eagle.py --disable-eagle --enable-profiling
```

---

## References

- Model: Llama4ForConditionalGeneration (MoE with 128 experts)
- Branch: `profiling`
- Commit: 0b336332a (sync fixes)
- Previous fixes: FLASHINFER attention backend, prefix caching enabled
