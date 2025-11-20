# EAGLE Benchmark Script

Self-contained Python script for benchmarking EAGLE speculative decoding with Perfetto trace support.

## Features

- **Self-contained**: Uses vLLM's `LLM()` API directly (no server needed)
- **EAGLE support**: Tests EAGLE speculative decoding (baseline mode also available)
- **Perfetto traces**: Generates Chrome trace format for viewing in Perfetto UI
- **Synthetic dataset**: Auto-generates random requests for consistent benchmarking
- **Configurable**: Extensive command-line options for customization

## Quick Start

```bash
# Basic benchmark with EAGLE (uses defaults from launch.sh)
python benchmark_eagle.py

# Benchmark with profiling enabled (generates trace)
python benchmark_eagle.py --enable-profiling

# Baseline comparison (no EAGLE)
python benchmark_eagle.py --disable-eagle --enable-profiling --output-trace baseline.json

# Custom configuration
python benchmark_eagle.py \
    --num-requests 200 \
    --max-tokens 256 \
    --tp-size 4 \
    --enable-profiling \
    --stats-file results.json
```

## Common Use Cases

### Compare EAGLE vs Baseline

```bash
# Run EAGLE
python benchmark_eagle.py \
    --enable-profiling \
    --output-trace eagle_trace.json \
    --stats-file eagle_stats.json

# Run baseline
python benchmark_eagle.py \
    --disable-eagle \
    --enable-profiling \
    --output-trace baseline_trace.json \
    --stats-file baseline_stats.json

# Compare the traces in Perfetto UI
# Upload both .json files to https://ui.perfetto.dev/
```

### Test Different Speculative Token Counts

```bash
# Test with 2 spec tokens
python benchmark_eagle.py --num-speculative-tokens 2 --stats-file eagle_2tok.json

# Test with 4 spec tokens (default)
python benchmark_eagle.py --num-speculative-tokens 4 --stats-file eagle_4tok.json

# Test with 8 spec tokens
python benchmark_eagle.py --num-speculative-tokens 8 --stats-file eagle_8tok.json
```

### Quick Test with Small Workload

```bash
python benchmark_eagle.py \
    --num-requests 20 \
    --max-tokens 128 \
    --input-len 512 \
    --warmup-requests 1
```

### Heavy Load Test

```bash
python benchmark_eagle.py \
    --num-requests 500 \
    --max-num-seqs 24 \
    --enable-profiling
```

## Key Arguments

### Model Configuration
- `--model`: Target model path (default: from launch.sh)
- `--draft-model`: EAGLE draft model path (default: from launch.sh)
- `--disable-eagle`: Run baseline without EAGLE

### EAGLE Settings
- `--num-speculative-tokens`: Number of spec tokens (default: 4)

### Hardware
- `--tp-size`: Tensor parallel size (default: 8)
- `--gpu-memory-utilization`: GPU memory fraction (default: 0.7)

### Workload
- `--num-requests`: Number of requests to benchmark (default: 100)
- `--input-len`: Input token length (default: 1024)
- `--max-tokens`: Max output tokens (default: 512)
- `--max-model-len`: Max sequence length (default: 1536)
- `--max-num-seqs`: Batch size (default: 12)

### Profiling
- `--enable-profiling`: Enable PyTorch profiler (required for traces)
- `--output-trace`: Output trace file path (default: eagle_benchmark_trace.json)
- `--record-shapes`: Record tensor shapes (increases trace size)
- `--stats-file`: Save JSON statistics file

### Sampling
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-p`: Top-p sampling (default: 0.95)

## Viewing Traces

### Option 1: Perfetto UI (Recommended)
1. Navigate to https://ui.perfetto.dev/
2. Click "Open trace file"
3. Select your `.json` trace file
4. Explore GPU kernels, timelines, and bottlenecks

### Option 2: Chrome Tracing
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select your `.json` file

## Output

The script outputs:
1. **Console statistics**: Throughput, latency, request/sec
2. **Chrome trace file**: For Perfetto UI (if `--enable-profiling` enabled)
3. **JSON stats file**: Machine-readable results (if `--stats-file` specified)

Example output:
```
================================================================================
BENCHMARK COMPLETE
================================================================================
Mode:                  EAGLE (spec_tokens=4)
Total time:            45.23s
Requests processed:    100
Requests/sec:          2.21

Total input tokens:    102400
Total output tokens:   48532
Total tokens:          150932
Output throughput:     1072.45 tokens/s
Avg time per request:  0.452s
Avg output per req:    485.3 tokens
================================================================================
```

## Environment Variables

The script respects these environment variables (same as launch.sh):
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: FLASHINFER)
- `LD_PRELOAD`: CUDA library preload path

## Tips

1. **Always run warmup**: The script does this automatically (default: 3 requests)
2. **Profiling overhead**: Disable profiling (`--enable-profiling` off) for pure throughput tests
3. **Memory issues**: Reduce `--max-num-seqs` or `--gpu-memory-utilization` if OOM
4. **Consistent results**: Use fixed `--input-len` and `--max-tokens` for comparisons
5. **Large traces**: Avoid `--record-shapes` unless you need detailed tensor info

## Example Workflow

```bash
# 1. Quick sanity check
python benchmark_eagle.py --num-requests 10 --warmup-requests 1

# 2. Baseline benchmark
python benchmark_eagle.py \
    --disable-eagle \
    --enable-profiling \
    --output-trace baseline.json \
    --stats-file baseline_stats.json

# 3. EAGLE benchmark
python benchmark_eagle.py \
    --enable-profiling \
    --output-trace eagle.json \
    --stats-file eagle_stats.json

# 4. Compare results
cat baseline_stats.json eagle_stats.json | jq '.results.throughput_tokens_per_sec'

# 5. Analyze traces in Perfetto
# Upload baseline.json and eagle.json to https://ui.perfetto.dev/
```

## Troubleshooting

**Error: CUDA not available**
- Ensure you're on a GPU machine
- Check `nvidia-smi`

**Error: Need X GPUs but only Y available**
- Reduce `--tp-size` to match available GPUs

**Warning: Adjusted input length**
- Input + output must fit in `--max-model-len`
- Increase `--max-model-len` or reduce `--input-len`/`--max-tokens`

**Trace file too large**
- Don't use `--record-shapes`
- Reduce `--num-requests`

## Comparison with Other Scripts

- `profile_eagle_chrome_trace.py`: Similar but less configurable
- `profile_eagle_llama4.py`: Uses vLLM internal profiler (TensorBoard format)
- `benchmark_eagle.py` (this): Most flexible, Perfetto-focused, self-contained

## Notes

- The script uses `RandomDataset` from vLLM for synthetic data generation
- EAGLE method is specified as `"eagle"` (not `"online_eagle"` which doesn't work yet)
- Same configuration as `launch.sh` for consistency
- The script sets `config_format='hf'` to avoid Mistral auto-detection issues with models that have `params.json` files

