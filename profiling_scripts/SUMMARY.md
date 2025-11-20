# Profiling Scripts Summary

Complete profiling setup for vLLM with EAGLE (DeepSeek V3) and batch invariance testing.

## 📁 Files Created

### Main Profiling Scripts (7 files)

1. **profile_eagle_chrome_trace.py** (7.8KB)
   - Profile EAGLE + DeepSeek V3
   - Outputs: Chrome trace JSON (Perfetto/chrome://tracing compatible)
   - ✅ Upload to https://ui.perfetto.dev/

2. **profile_batch_invariance_chrome_trace.py** (8.2KB)
   - Profile batch invariance
   - Outputs: Chrome trace JSON
   - ✅ Upload to https://ui.perfetto.dev/

3. **profile_eagle_llama4.py** (9.2KB)
   - Profile EAGLE + DeepSeek V3
   - Outputs: TensorBoard format (detailed analysis)
   - ✅ View with tensorboard

4. **profile_batch_invariance.py** (11KB)
   - Profile batch invariance
   - Outputs: TensorBoard format
   - ✅ View with tensorboard

5. **run_batch_invariance_comparison.sh** (3.1KB)
   - Helper script: runs baseline + batch invariant profiling
   - Outputs: Two TensorBoard directories for comparison

6. **quick_start.sh** (4.4KB)
   - Interactive guide to run your first profile
   - Good for testing the setup

7. **README.md** (12KB)
   - Complete documentation
   - Configuration reference
   - Examples and troubleshooting

## 🚀 Quick Start

### For Perfetto (Recommended)

```bash
# Profile EAGLE with DeepSeek V3
python profiling_scripts/profile_eagle_chrome_trace.py

# Upload eagle_profile.json to https://ui.perfetto.dev/
```

### For 8 GPU Profiling

```bash
# Profile EAGLE across 8 GPUs (DeepSeek V3 requires 8 GPUs by default)
PROFILE_TP_SIZE=8 python profiling_scripts/profile_eagle_chrome_trace.py

# Profile batch invariance comparison
VLLM_BATCH_INVARIANT=0 PROFILE_OUTPUT_FILE=baseline.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py

VLLM_BATCH_INVARIANT=1 PROFILE_OUTPUT_FILE=batch_inv.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py
```

## ✨ Key Features

✅ **Chrome Trace Support** - Yes! All scripts have Chrome trace variants
✅ **Perfetto Compatible** - Upload .json files directly to ui.perfetto.dev
✅ **8-GPU Support** - Configurable tensor parallelism (default TP=8)
✅ **DeepSeek V3 + EAGLE** - Configured for large-scale MoE model profiling
✅ **Real Workloads** - Uses ShareGPT (lmsys chat) dataset automatically
✅ **Batch Invariance** - Easy baseline vs batch-invariant comparison
✅ **TensorBoard** - Alternative format for detailed analysis
✅ **Production Ready** - Warmup, error handling, progress bars

## 📊 Output Formats

| Format | File Extension | View With | Best For |
|--------|---------------|-----------|----------|
| Chrome Trace | `.json` | Perfetto UI, chrome://tracing | Sharing, quick analysis |
| TensorBoard | `.pt.trace.json.gz` | TensorBoard | Deep analysis, metrics |

## 🔧 Common Use Cases

**1. Profile EAGLE with DeepSeek V3 and share results:**
```bash
python profiling_scripts/profile_eagle_chrome_trace.py
# Upload eagle_profile.json to Perfetto
```

**2. Compare batch invariance modes:**
```bash
# Baseline
VLLM_BATCH_INVARIANT=0 PROFILE_OUTPUT_FILE=baseline.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py

# Batch invariant
VLLM_BATCH_INVARIANT=1 PROFILE_OUTPUT_FILE=batch_inv.json \
    python profiling_scripts/profile_batch_invariance_chrome_trace.py

# Load both in Perfetto for side-by-side comparison
```

**3. Deep TensorBoard analysis:**
```bash
python profiling_scripts/profile_eagle_llama4.py
tensorboard --logdir ./vllm_profile_eagle
```

**4. Quick test run:**
```bash
./profiling_scripts/quick_start.sh
# Interactive guide walks you through options
```

## 📝 Environment Variables

All scripts support extensive configuration:

- `PROFILE_TP_SIZE=8` - Number of GPUs (default: 8, required for DeepSeek V3)
- `PROFILE_NUM_REQUESTS=100` - Number of requests (default: 100)
- `PROFILE_MAX_TOKENS=512` - Max tokens per request (default: 512)
- `PROFILE_OUTPUT_FILE=trace.json` - Output file for Chrome traces
- `VLLM_BATCH_INVARIANT=1` - Enable batch invariance (0 or 1)

See README.md for complete configuration reference.

## 🎯 Answer to Original Question

**Q: Does this create a trace I can upload to Perfetto or trace?**

**A: YES!** Use these scripts:
- `profile_eagle_chrome_trace.py`
- `profile_batch_invariance_chrome_trace.py`

Both output `.json` files that work with:
- ✅ Perfetto UI (https://ui.perfetto.dev/)
- ✅ chrome://tracing
- ✅ Speedscope (https://www.speedscope.app/)

## 📚 Model Information

**Default Models:**
- Base: `deepseek-ai/DeepSeek-V3`
- EAGLE: `eagle618/eagle-deepseek-v3-random`

**Note:** DeepSeek V3 is a large MoE model that typically requires 8 GPUs with high memory. Adjust `PROFILE_TP_SIZE` based on your available resources.

## 📚 Next Steps

1. Read `README.md` for detailed documentation
2. Run `./quick_start.sh` for an interactive test
3. Try profiling with Chrome trace format first
4. Upload results to Perfetto for visualization
5. Use TensorBoard format if you need deeper analysis

## 💡 Tips

- Start with small workloads (20 requests, 128 tokens) for testing
- Chrome trace format is faster and smaller than TensorBoard
- DeepSeek V3 requires TP_SIZE=8 for production workloads
- ShareGPT dataset downloads automatically from HuggingFace
- All scripts include warmup runs to exclude initialization overhead
