# vLLM Profiling Scripts

This directory contains GPU profiling scripts for vLLM across 8 GPUs using real workloads.

## Scripts

### TensorBoard Format (Detailed Analysis)

#### 1. `profile_eagle_llama4.py`
Profiles EAGLE (Efficient and Accurate Generation with Language Execution) speculative decoding using DeepSeek V3 models. Outputs TensorBoard traces for detailed analysis.

#### 2. `profile_batch_invariance.py`
Profiles vLLM without EAGLE to test batch invariance behavior. Outputs TensorBoard traces.

### Chrome Trace Format (Perfetto/Chrome Tracing)

#### 3. `profile_eagle_chrome_trace.py`
Same as #1 but outputs Chrome trace JSON format that can be uploaded to Perfetto UI or viewed in chrome://tracing. **Best for sharing and quick visualization.**

#### 4. `profile_batch_invariance_chrome_trace.py`
Same as #2 but outputs Chrome trace JSON format. **Best for sharing and quick visualization.**

### Helper Scripts

#### 5. `run_batch_invariance_comparison.sh`
Convenience script that runs both baseline and batch-invariant profiling using TensorBoard format.

## Requirements

- 8 NVIDIA GPUs (can be adjusted with `PROFILE_TP_SIZE`)
- CUDA-enabled environment
- vLLM installed with all dependencies
- Sufficient disk space for profiler traces (can be large, 1-10GB+)

## Which Format to Use?

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| **Chrome Trace** | Sharing, Perfetto, quick analysis | - Single .json file<br>- Easy to share/upload<br>- Works with Perfetto UI<br>- Fast to generate<br>- Smaller file size | - Less detailed metrics<br>- No multi-run comparison in UI |
| **TensorBoard** | Deep analysis, metric tracking | - Detailed operator stats<br>- Memory profiling<br>- Multi-run comparison<br>- Per-GPU breakdown | - Larger files<br>- Requires TensorBoard server<br>- More complex to share |

**Recommendation**: Use Chrome Trace format for most profiling tasks, especially when sharing results or using Perfetto. Use TensorBoard when you need detailed operator-level analysis.

## Quick Start

### Option A: Chrome Trace Format (Recommended for Perfetto/Sharing)

**Profile EAGLE with Chrome Trace:**
```bash
# Generate Chrome trace (outputs eagle_profile.json)
python profiling_scripts/profile_eagle_chrome_trace.py

# View in Perfetto (upload the .json file)
open https://ui.perfetto.dev/

# Or view in Chrome browser
# Open chrome://tracing and load eagle_profile.json
```

**Profile Batch Invariance with Chrome Trace:**
```bash
# Baseline
VLLM_BATCH_INVARIANT=0 PROFILE_OUTPUT_FILE=baseline.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py

# With batch invariance
VLLM_BATCH_INVARIANT=1 PROFILE_OUTPUT_FILE=batch_inv.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py

# Upload both .json files to https://ui.perfetto.dev/ for comparison
```

### Option B: TensorBoard Format (For Detailed Analysis)

**Profile EAGLE with TensorBoard:**
```bash
# Basic usage with defaults (8 GPUs, ShareGPT dataset)
python profiling_scripts/profile_eagle_llama4.py

# View the profiler traces
tensorboard --logdir ./vllm_profile_eagle
```

**Profile Batch Invariance with TensorBoard:**
```bash
# Profile baseline (without batch invariance)
VLLM_BATCH_INVARIANT=0 VLLM_TORCH_PROFILER_DIR=./profile_baseline \
    python profiling_scripts/profile_batch_invariance.py

# Profile with batch invariance enabled
VLLM_BATCH_INVARIANT=1 VLLM_TORCH_PROFILER_DIR=./profile_batch_inv \
    python profiling_scripts/profile_batch_invariance.py

# Compare both profiles in TensorBoard
tensorboard --logdir_spec=baseline:./profile_baseline,batch_inv:./profile_batch_inv
```

## Configuration

Both scripts support extensive configuration through environment variables and command-line arguments.

### Common Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TORCH_PROFILER_DIR` | `./vllm_profile_*` | Directory to save profiler traces |
| `VLLM_PROFILER_DELAY_ITERS` | `2` | Number of iterations to wait before profiling starts |
| `VLLM_PROFILER_MAX_ITERS` | `10` | Maximum iterations to profile |
| `PROFILE_NUM_REQUESTS` | `100` | Number of requests to profile |
| `PROFILE_DATASET_PATH` | _(empty)_ | Path to ShareGPT dataset JSON (uses HuggingFace if not provided) |
| `PROFILE_MAX_TOKENS` | `512` | Max tokens to generate per request |
| `PROFILE_TP_SIZE` | `8` | Tensor parallel size (number of GPUs) |
| `PROFILE_GPU_MEM_UTIL` | `0.9` | GPU memory utilization |
| `PROFILE_MAX_MODEL_LEN` | `8192` | Max model length |

### Batch Invariance Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BATCH_INVARIANT` | `0` | Enable batch invariance mode (0=disabled, 1=enabled) |
| `PROFILE_BATCH_SIZE` | `256` | Max batch size (max_num_seqs) |

## Examples

### Quick Test with Fewer GPUs and Requests
```bash
# Test on single GPU with small workload
PROFILE_NUM_REQUESTS=20 PROFILE_MAX_TOKENS=128 PROFILE_TP_SIZE=1 \
    python profiling_scripts/profile_batch_invariance.py
```

### Profile EAGLE with Custom Dataset
```bash
# Use your own ShareGPT dataset
PROFILE_DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    PROFILE_NUM_REQUESTS=200 \
    python profiling_scripts/profile_eagle_llama4.py
```

### Extended Profiling Session
```bash
# Profile more iterations for deeper analysis
VLLM_PROFILER_MAX_ITERS=50 PROFILE_NUM_REQUESTS=500 \
    python profiling_scripts/profile_eagle_llama4.py
```

### Profile Different Model
```bash
# Profile with Qwen model instead of Llama
python profiling_scripts/profile_batch_invariance.py \
    --model Qwen/Qwen3-8B-Instruct \
    --tp-size 8
```

## Understanding the Profiler Output

The scripts support two output formats:

### Chrome Trace Format (JSON)

**Best for**: Sharing, uploading to Perfetto, quick visualization

**Output**: Single `.json` file per profiling run

**Viewing Options**:
1. **Perfetto UI** (Recommended):
   - Go to https://ui.perfetto.dev/
   - Click "Open trace file" and select your `.json` file
   - Interactive timeline view with GPU kernel details
   - Can compare multiple traces by loading them sequentially

2. **Chrome Tracing**:
   - Open `chrome://tracing` in Chrome/Chromium
   - Click "Load" and select your `.json` file
   - Timeline view with GPU/CPU activity

3. **Speedscope**:
   - Go to https://www.speedscope.app/
   - Upload your `.json` file
   - Flamegraph visualization

**Key Views in Perfetto**:
- Timeline view showing GPU kernels and their duration
- Thread states and scheduling
- GPU memory operations
- Python/C++ stack traces (if `--record-shapes` enabled)

### TensorBoard Format

**Best for**: Detailed analysis, comparing metrics over time

**Output Files**:
- `{profiler_dir}/` - Contains trace files for each GPU rank
- `{profiler_dir}/profiler_out_0.txt` - Summary table for rank 0 (main GPU)
- Tensorboard event files with `.pt.trace.json.gz` extension

**Viewing Traces**:

1. Start TensorBoard:
   ```bash
   tensorboard --logdir ./vllm_profile_eagle
   ```

2. Open browser to `http://localhost:6006`

3. Navigate to the "PYTORCH_PROFILER" tab

4. Analyze:
   - **GPU kernel usage** - See which kernels are running and their duration
   - **GPU memory timeline** - Track memory allocations over time
   - **Operator time breakdown** - Identify bottlenecks
   - **Kernel time** - Compare self_cuda_time_total across operations

### Key Metrics to Look For

- **Kernel execution time** - Time spent in GPU kernels
- **Memory operations** - Data transfers and allocations
- **CPU overhead** - Python/C++ execution time
- **Synchronization points** - Where GPUs wait for each other
- **Attention kernel performance** - FlashAttention vs other backends

## Datasets

Both scripts use the ShareGPT dataset for realistic conversational workloads:

- **Default**: HuggingFace dataset `Aeala/ShareGPT_Vicuna_unfiltered`
- **Custom**: Provide path via `PROFILE_DATASET_PATH`

ShareGPT contains real conversations from lmsys chat, making it ideal for:
- EAGLE profiling (needs realistic token distributions)
- Batch invariance testing (diverse prompt and output lengths)

## Profiler Configuration

### Delay Iterations (`VLLM_PROFILER_DELAY_ITERS`)
Number of iterations to run before starting profiling. This allows:
- CUDA graphs to be captured
- Caches to warm up
- First-run overhead to be excluded

**Recommendation**: Keep at 2-5 for most use cases

### Max Iterations (`VLLM_PROFILER_MAX_ITERS`)
Number of iterations to profile. More iterations provide:
- Better averaging of metrics
- Identification of patterns over time
- Larger trace files (can be 100MB+ per iteration)

**Recommendation**:
- 10-20 for quick profiling
- 50+ for detailed analysis
- Consider disk space (each iteration can be 100MB-1GB)

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce GPU memory utilization
PROFILE_GPU_MEM_UTIL=0.6 python profiling_scripts/profile_eagle_llama4.py

# Or reduce batch size
PROFILE_BATCH_SIZE=128 python profiling_scripts/profile_batch_invariance.py

# Or use fewer GPUs
PROFILE_TP_SIZE=4 python profiling_scripts/profile_eagle_llama4.py
```

### Profiler Traces Too Large
```bash
# Reduce number of profiling iterations
VLLM_PROFILER_MAX_ITERS=5 python profiling_scripts/profile_eagle_llama4.py

# Or reduce number of requests
PROFILE_NUM_REQUESTS=50 python profiling_scripts/profile_eagle_llama4.py
```

### Dataset Download Issues
```bash
# Use smaller number of requests if HuggingFace dataset is slow
PROFILE_NUM_REQUESTS=20 python profiling_scripts/profile_eagle_llama4.py

# Or provide local dataset
PROFILE_DATASET_PATH=/path/to/local/sharegpt.json \
    python profiling_scripts/profile_eagle_llama4.py
```

### Model Download Issues
```bash
# Pre-download models
huggingface-cli download deepseek-ai/DeepSeek-V3
huggingface-cli download eagle618/eagle-deepseek-v3-random

# Then run profiling
python profiling_scripts/profile_eagle_llama4.py
```

## Advanced Usage

### Comparing Multiple Configurations

Create a comparison script:
```bash
#!/bin/bash
# compare_batch_invariance.sh

# Baseline
VLLM_BATCH_INVARIANT=0 VLLM_TORCH_PROFILER_DIR=./profiles/baseline \
    python profiling_scripts/profile_batch_invariance.py

# Batch invariant
VLLM_BATCH_INVARIANT=1 VLLM_TORCH_PROFILER_DIR=./profiles/batch_inv \
    python profiling_scripts/profile_batch_invariance.py

# Compare
tensorboard --logdir_spec=baseline:./profiles/baseline,batch_inv:./profiles/batch_inv
```

### NSYS Profiling (Alternative)

For more detailed GPU profiling, you can use NVIDIA Nsight Systems:

```bash
nsys profile -o profile_eagle \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python profiling_scripts/profile_eagle_llama4.py

# View with Nsight Systems GUI
nsys-ui profile_eagle.nsys-rep
```

Note: The scripts support both PyTorch profiler (default) and CUDA profiler via environment variables.

## Environment Setup

Make sure you have the required environment:

```bash
# Install vLLM with all dependencies
pip install vllm

# Install profiling dependencies
pip install tensorboard

# Optional: For custom datasets
pip install datasets transformers

# Verify GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

## References

- [vLLM Profiling Documentation](https://docs.vllm.ai/en/latest/dev/profiling/profiling.html)
- [EAGLE Paper](https://arxiv.org/abs/2401.15077)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TensorBoard Profiler Plugin](https://github.com/pytorch/kineto/tree/main/tb_plugin)
