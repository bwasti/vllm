# TODO: Online Trainable EAGLE Implementation

## Overview
This document outlines the plan to implement online trainable EAGLE in vLLM. The goal is to enable the EAGLE draft model to learn from inference workloads in real-time, improving speculative decoding performance on-the-fly.

**Model Directory:** `/data/users/bwasti/wearable_maverick_vllm/`
**Draft Model:** `/data/users/bwasti/wearable_maverick_vllm/draft/`

---

## Phase 0: Infrastructure & Testing Setup ✅ COMPLETE

### 0.1 Launch Scripts ✅
- [x] **Create `./launch.sh`** - Server launch script
  - [x] Parse command-line arguments for mode selection (`eagle` vs `online_eagle`)
  - [x] Set up environment variables (LD_PRELOAD, VLLM_ATTENTION_BACKEND)
  - [x] Configure vLLM server with appropriate flags:
    ```bash
    --model /data/users/bwasti/wearable_maverick_vllm/
    --speculative-config '{"model": "/data/users/bwasti/wearable_maverick_vllm/draft/", "method": "eagle", "num_speculative_tokens": 4}'
    --tensor-parallel-size <TP_SIZE>
    --quantization compressed-tensors
    --kv-cache-dtype auto
    --gpu-memory-utilization 0.85
    --max-model-len 8192
    --max-num-seqs 128
    ```
  - [x] Add optional flags for:
    - Port selection (default 8000)
    - TP size configuration
    - Training-specific parameters (training_config JSON)
    - Host configuration
  - [x] Add graceful shutdown handling (SIGINT, SIGTERM)
  - [x] Add health check loop before returning control (IPv4 curl)

- [x] **Create `./stress_test.sh`** - Client stress testing script
  - [x] Parse command-line arguments:
    - Workload type: `code`, `chat`, `debug` (same query at temp=0)
    - Number of concurrent requests / duration
    - Request rate (QPS)
    - Max tokens, temperature
  - [x] Implement workload loaders:
    - [x] **Code workload:** Load from HuggingFace dataset (bigcode/the-stack-dedup)
    - [x] **Chat workload:** Load from HuggingFace dataset (lmsys/lmsys-chat-1m)
    - [x] **Debug workload:** Repeat same prompt N times at temperature=0
  - [x] Implement request generator:
    - [x] Async request sending using `aiohttp`
    - [x] Rate limiting to achieve target QPS
    - [x] Request ID tracking
  - [x] Implement metrics collection:
    - [x] Per-request latency (TTFT, total latency)
    - [x] Throughput (tokens/sec)
    - [x] Success/error tracking
    - [x] Output length statistics
  - [x] Output real-time stats to console (every 10 requests)
  - [x] Save detailed results to JSON file
  - [x] Auto-detect model name from `/v1/models` endpoint

- [x] **Test launch scripts**
  - [x] Verify server starts correctly with standard EAGLE (TP=8)
  - [x] Run all three workload types and verify metrics collection
  - [x] Test graceful shutdown
  - [x] Verify server can handle concurrent requests

### 0.2 Baseline Metrics Collection
- [ ] **Add EAGLE metrics tracking to vLLM** (deferred to Phase 2)
  - [ ] Extend metrics with EAGLE-specific stats:
    - Speculation acceptance rate (per-token, per-request)
    - Draft model latency overhead
    - Number of accepted speculative tokens per step
  - [ ] Log metrics to StatLogger
  - [ ] Add `/metrics` endpoint to expose Prometheus metrics

- [x] **Stress test infrastructure ready**
  - [x] Can collect baseline performance for all three workloads
  - [x] Saves metrics to JSON for comparison
  - [x] Ready to document baseline acceptance rates

---

## Phase 1: Vanilla Trainable EAGLE Model ✅ COMPLETE

### 1.1 Create Trainable Model Wrapper ✅
- [x] **Create `vllm/model_executor/models/trainable_eagle.py`**
  - [x] Define `TrainableEagleLlamaForCausalLM` class
    - [x] Implemented with vanilla PyTorch (no custom CUDA ops)
    - [x] Manual TP sharding for all layers
    - [x] Store TP rank and world size
    - [x] Initialize model in training mode
    - [x] Register all parameters for gradient computation
  - [x] Implement `forward()` for training:
    - [x] Accept inputs: `input_ids`, `positions`, `hidden_states`, `labels`
    - [x] Run EAGLE forward pass through trainable layers
    - [x] Compute cross-entropy loss if labels provided
    - [x] Return `(loss, logits, hidden_states)` tuple
  - [x] Implement `backward_step()`:
    - [x] Call `loss.backward()` to compute gradients
    - [x] TP-aware gradient reduction via all_reduce:
      ```python
      if self.tp_size > 1:
          for param in self.parameters():
              if param.grad is not None:
                  get_tp_group().all_reduce(param.grad)
                  param.grad.div_(self.tp_size)
      ```
    - [x] Return gradient norms for monitoring
  - [x] Implement `get_trainable_parameters()`:
    - [x] Return list of parameters that should be trained
  - [x] Implement `copy_weights_to_inference_model(inference_model)`:
    - [x] Copy trained weights to optimized EAGLE model
    - [x] Handle TP sharding via state_dict
  - [x] Implement `copy_weights_from_inference_model(inference_model)`:
    - [x] Initialize trainable model from existing EAGLE weights
    - [x] Preserve TP sharding
  - [x] Implement checkpoint save/load:
    - [x] `save_checkpoint(path)`: Save model state and config
    - [x] `load_checkpoint(path)`: Load checkpoint

- [x] **Components implemented:**
  - [x] `TrainableLlamaDecoderLayer`: Vanilla PyTorch transformer layer
    - Standard nn.Linear layers (no vLLM custom ops)
    - F.scaled_dot_product_attention() for attention
    - Manual TP sharding for Q/K/V/O projections
    - SwiGLU MLP with TP-aware gate/up/down
    - All-reduce for row-parallel outputs
  - [x] `TrainableLlamaModel`: EAGLE model core
    - TP-aware embedding layer with manual vocab sharding
    - FC layer to concatenate embeddings + target hidden states
    - Stack of trainable transformer layers

- [ ] **Test trainable model in isolation** (deferred - need real model files)
  - [ ] Write test: `tests/training/test_trainable_eagle_correctness.py`
    - [x] Test skeletons created for integration tests
    - [x] Unit tests for helper functions (loss, grad clipping, optimizer)
    - [ ] Test forward pass produces same outputs as standard EAGLE
    - [ ] Test backward pass computes gradients
    - [ ] Test gradient shapes match parameter shapes
    - [ ] Test TP gradient reduction (if TP > 1)
    - [ ] Test weight copying to/from inference model

### 1.2 Optimizer and Training Loop ✅
- [x] **Create `vllm/training/eagle_trainer.py`**
  - [x] Define `EagleTrainer` class:
    - [x] `__init__(model, config)`:
      - [x] Initialize optimizer (AdamW with configurable LR)
      - [x] Initialize LR scheduler (linear/cosine/constant)
      - [x] Set up gradient accumulation steps
      - [x] Initialize training step counter
      - [x] Set up loss tracking (OnlineTrainingMetrics)
      - [x] Initialize TrainingBuffer for data storage
      - [x] Set up checkpoint management
    - [x] `train_step()`:
      - [x] Get batch from buffer
      - [x] Run forward pass with loss computation
      - [x] Run backward pass via model.backward_step()
      - [x] Accumulate gradients over multiple batches
      - [x] Clip gradients (max_grad_norm)
      - [x] Update weights when accumulation steps reached
      - [x] LR scheduler step
      - [x] Return loss and metrics dict
    - [x] `train_n_steps(n_steps)`: Run N training steps
    - [x] `train_async(n_steps)`: Async training support
    - [x] `get_metrics()`: Return training statistics
    - [x] `save_checkpoint(path)`: Save model checkpoint
    - [x] `load_checkpoint(path)`: Load checkpoint
    - [x] Helper methods for buffer management and old checkpoint cleanup
  - [x] Define `TrainingBuffer` class:
    - [x] Circular buffer (deque with maxlen)
    - [x] Async-safe operations with locks
    - [x] Random batch sampling
    - [x] PyTorch Dataset interface
  - [x] Define `TrainingSample` dataclass
  - [x] Define `collate_training_samples()` helper

- [x] **Create `vllm/training/config.py`**
  - [x] Define `TrainingConfig` dataclass with 30+ parameters:
    - [x] Optimizer settings: `learning_rate`, `weight_decay`, `adam_beta1/2`, `adam_epsilon`
    - [x] LR scheduler: `use_lr_scheduler`, `warmup_steps`, `lr_scheduler_type`
    - [x] Training batch: `batch_size`, `gradient_accumulation_steps`, `max_seq_len`, `max_grad_norm`
    - [x] Data collection: `buffer_size`, `min_samples_for_training`, `sample_collection_prob`
    - [x] Training schedule: `train_interval_requests`, `train_interval_samples`, `training_steps_per_trigger`
    - [x] Async training: `async_training`, `max_concurrent_trainings`
    - [x] Checkpointing: `checkpoint_interval_steps`, `checkpoint_dir`, `keep_last_n_checkpoints`
    - [x] Logging: `log_interval_steps`, `enable_tensorboard`, `tensorboard_dir`
    - [x] Validation: `validation_interval_steps`, `validation_samples`
    - [x] Resource: `training_device`, `pin_memory`, `num_workers`
    - [x] Advanced: `compile_model`, `use_mixed_precision`, `grad_checkpoint`
    - [x] Debug: `debug_mode`, `validate_gradients`
  - [x] Define `OnlineTrainingMetrics` dataclass:
    - [x] Training stats (steps, loss, lr, grad_norm)
    - [x] Performance stats (time, throughput)
    - [x] Validation stats
    - [x] Buffer stats
    - [x] Error tracking
    - [x] Helper methods: `to_dict()`, `update_from_training_step()`
  - [x] Add validation in `__post_init__`
  - [x] Add helper methods: `effective_batch_size()`, `steps_per_epoch()`, `total_training_steps()`

- [x] **Create `vllm/training/__init__.py`**
  - [x] Export TrainingConfig and OnlineTrainingMetrics

### 1.3 Validation and Correctness Testing ✅
- [x] **Create validation suite: `tests/training/test_trainable_eagle_correctness.py`**
  - [x] Test skeletons created (marked as integration tests)
  - [x] Unit tests for helper functions:
    - [x] Test loss computation correctness
    - [x] Test gradient clipping
    - [x] Test optimizer step updates parameters
    - [x] Test docstring completeness
  - [x] Integration test placeholders (require real models):
    - [x] test_trainable_vs_inference_equivalence_real()
    - [x] test_backward_pass_real()
    - [x] test_tp_training_real() (multi-GPU)
  - [x] Helper function: `create_test_vllm_config()` for test setup

- [ ] **Run validation tests** (requires real model files)
  - [ ] Unit tests pass (can run now)
  - [ ] Integration tests pass with TP=1
  - [ ] Integration tests pass with TP=2 (if multi-GPU available)
  - [ ] Integration tests pass with TP=4 (if multi-GPU available)

---

## Phase 2: High-Level Integration

### 2.1 Training Data Buffer
- [ ] **Create `vllm/training/data_buffer.py`**
  - [ ] Define `TrainingDataBuffer` class:
    - [ ] Use circular buffer for memory efficiency
    - [ ] Store tuples: `(input_ids, positions, hidden_states, labels)`
    - [ ] Implement `add(input_ids, positions, hidden_states, labels)`:
      - [ ] Add new sample to buffer
      - [ ] Evict oldest if buffer full
      - [ ] Handle batched additions efficiently
    - [ ] Implement `sample_batch(batch_size)`:
      - [ ] Sample random batch from buffer
      - [ ] Return collated tensors ready for training
      - [ ] Handle variable sequence lengths (padding)
    - [ ] Implement `__len__()`: Return current buffer size
    - [ ] Implement `clear()`: Empty buffer
    - [ ] Thread-safe implementation (use locks for concurrent access)

  - [ ] **Add filtering logic**
    - [ ] Only add samples where speculative tokens were rejected (high learning signal)
    - [ ] Add diversity sampling (avoid duplicate sequences)
    - [ ] Add quality filtering (skip very short sequences)

- [ ] **Test data buffer**
  - [ ] Write `tests/training/test_data_buffer.py`
  - [ ] Test circular buffer behavior (eviction)
  - [ ] Test sampling produces correct batch shapes
  - [ ] Test thread-safety with concurrent adds/samples

### 2.2 Integration Point: EngineCore Level
- [ ] **Modify `vllm/v1/engine/core.py`**
  - [ ] Add `training_manager: Optional[TrainingManager]` field
  - [ ] In `__init__`:
    - [ ] Check if training is enabled in config
    - [ ] Initialize `TrainingManager` if enabled
  - [ ] In `step()` method (after inference):
    ```python
    # Existing code: schedule → execute → update
    scheduler_output = self.scheduler.schedule()
    model_output = self.model_executor.execute_model(scheduler_output)
    engine_core_outputs = self.scheduler.update_from_output(...)

    # NEW: Collect training data
    if self.training_manager is not None:
        self.training_manager.collect_training_data(
            scheduler_output=scheduler_output,
            model_output=model_output,
            engine_core_outputs=engine_core_outputs
        )

        # NEW: Trigger training if conditions met
        if self.training_manager.should_train():
            self.training_manager.step()  # Async training

    return engine_core_outputs, model_executed
    ```

- [ ] **Create `vllm/training/training_manager.py`**
  - [ ] Define `TrainingManager` class:
    - [ ] `__init__(vllm_config, training_config, drafter)`:
      - [ ] Initialize data buffer
      - [ ] Create trainable model (clone from drafter)
      - [ ] Create trainer
      - [ ] Set up async training executor (ThreadPoolExecutor or separate process)
      - [ ] Initialize step counter
      - [ ] Set up stats tracking

    - [ ] `collect_training_data(scheduler_output, model_output, engine_core_outputs)`:
      - [ ] Extract relevant data from outputs:
        - [ ] Target model hidden states
        - [ ] Speculative tokens proposed by drafter
        - [ ] Accepted/rejected token masks
        - [ ] Ground truth next tokens (from final sampled outputs)
      - [ ] Filter samples:
        - [ ] Keep only rejected speculative tokens (high learning signal)
        - [ ] Skip very short sequences
      - [ ] Add to buffer:
        ```python
        for i in range(batch_size):
            if should_collect(i):
                self.buffer.add(
                    input_ids=token_ids[i],
                    positions=positions[i],
                    hidden_states=hidden_states[i],
                    labels=ground_truth_labels[i]
                )
        ```

    - [ ] `should_train() -> bool`:
      - [ ] Check if buffer has enough samples (> batch_size)
      - [ ] Check if enough steps have passed since last training
      - [ ] Check if previous async training is complete
      - [ ] Return True if ready to train

    - [ ] `step()`:
      - [ ] Sample batch from buffer
      - [ ] Submit training job to executor:
        ```python
        self.training_future = self.executor.submit(
            self._train_batch, batch
        )
        ```
      - [ ] Increment step counter

    - [ ] `_train_batch(batch)` (runs in background):
      - [ ] Run trainer.train_step(batch)
      - [ ] Collect training stats
      - [ ] Every N steps: copy weights to inference model
      - [ ] Every M steps: save checkpoint
      - [ ] Return stats

    - [ ] `get_stats()`:
      - [ ] Return training statistics (loss, buffer size, training steps, etc.)
      - [ ] Check if async training is complete and collect results

    - [ ] `shutdown()`:
      - [ ] Wait for pending training jobs
      - [ ] Save final checkpoint
      - [ ] Clean up executor

### 2.3 Worker-Level Support
- [ ] **Modify `vllm/v1/worker/gpu_worker.py`**
  - [ ] Add method `update_drafter_weights(weights)`:
    - [ ] Receive new weights from training manager
    - [ ] Update drafter model weights
    - [ ] Handle TP sharding correctly
    - [ ] Synchronize across TP group if needed

  - [ ] Add method `get_drafter_weights()`:
    - [ ] Extract current drafter weights
    - [ ] Return in format compatible with trainable model

- [ ] **Modify `vllm/v1/executor/abstract.py`**
  - [ ] Add abstract method `update_drafter_weights(weights)`
  - [ ] Add abstract method `get_drafter_weights()`

- [ ] **Implement in concrete executors**
  - [ ] `UniProcExecutor`: Direct call to worker
  - [ ] `MultiprocExecutor`: RPC to all workers
  - [ ] `RayExecutor`: Ray remote call to all workers

### 2.4 Async Training Architecture
- [ ] **Design decision: Threading vs Multiprocessing vs Remote**
  - [ ] **Option A: Threading** (simplest)
    - [ ] Use `ThreadPoolExecutor` for background training
    - [ ] Training runs on same GPU as inference (time-sliced)
    - [ ] Pros: Simple, no IPC overhead
    - [ ] Cons: May impact inference latency (GIL contention)

  - [ ] **Option B: Multiprocessing** (isolated)
    - [ ] Use separate process for training
    - [ ] Communicate via shared memory or queues
    - [ ] Pros: No inference interference
    - [ ] Cons: More complex, IPC overhead

  - [ ] **Option C: Remote** (scalable)
    - [ ] Training runs on separate machine
    - [ ] Communicate via gRPC or Ray
    - [ ] Pros: Fully decoupled, scalable
    - [ ] Cons: Network overhead, most complex

  - [ ] **Decision:** Start with Option A (threading), add Option C (remote) later

- [ ] **Implement async training with threading**
  - [ ] Use `ThreadPoolExecutor` with single worker thread
  - [ ] Use CUDA streams to overlap training with inference:
    ```python
    self.inference_stream = torch.cuda.Stream()
    self.training_stream = torch.cuda.Stream()

    # In inference
    with torch.cuda.stream(self.inference_stream):
        model_output = self.model(...)

    # In training (async)
    with torch.cuda.stream(self.training_stream):
        loss = self.trainable_model(...)
        loss.backward()
    ```
  - [ ] Test that inference latency is not significantly impacted

### 2.5 Testing Integration
- [ ] **Create `tests/integration/test_online_training.py`**
  - [ ] **Test 1: End-to-end training**
    - [ ] Start server with online training enabled
    - [ ] Send requests
    - [ ] Verify training data is collected
    - [ ] Verify training steps are executed
    - [ ] Verify drafter weights are updated

  - [ ] **Test 2: Buffer management**
    - [ ] Send many requests to fill buffer
    - [ ] Verify oldest samples are evicted
    - [ ] Verify buffer size stays within limit

  - [ ] **Test 3: Weight synchronization**
    - [ ] Train for several steps
    - [ ] Verify drafter model uses updated weights
    - [ ] Verify outputs change after training

  - [ ] **Test 4: Checkpoint saving/loading**
    - [ ] Train for N steps
    - [ ] Save checkpoint
    - [ ] Restart server
    - [ ] Load checkpoint
    - [ ] Verify training continues from correct state

  - [ ] **Test 5: Async training doesn't block inference**
    - [ ] Start training
    - [ ] Send inference requests
    - [ ] Verify requests are processed without blocking
    - [ ] Measure latency impact (should be < 5%)

---

## Phase 3: Easy Enable/Disable Flags

### 3.1 Configuration Updates
- [ ] **Modify `vllm/config/speculative.py`**
  - [ ] Add `online_training: bool = False` to `SpeculativeConfig`
  - [ ] Add `training_config: Optional[TrainingConfig] = None`
  - [ ] In `__post_init__`:
    ```python
    if self.method == "online_eagle":
        self.online_training = True
        if self.training_config is None:
            self.training_config = TrainingConfig()  # Use defaults
    ```

- [ ] **Support "online_eagle" method**
  - [ ] Modify method validation to accept "online_eagle"
  - [ ] Map "online_eagle" to standard EAGLE drafter + online training
  - [ ] Example config:
    ```bash
    --speculative-config '{
      "model": "/path/to/draft/",
      "method": "online_eagle",
      "num_speculative_tokens": 4,
      "training_config": {
        "learning_rate": 1e-4,
        "buffer_size": 1000,
        "batch_size": 32
      }
    }'
    ```

### 3.2 CLI Argument Support
- [ ] **Modify `vllm/engine/arg_utils.py`**
  - [ ] Add `--enable-online-training` flag (boolean)
  - [ ] Add `--training-config` flag (JSON string)
  - [ ] Validate that online training only works with EAGLE methods
  - [ ] Raise error if `--enable-online-training` used without speculative config

- [ ] **Update launch script to use new flags**
  - [ ] Modify `./launch.sh` to toggle between:
    ```bash
    # Standard EAGLE
    ./launch.sh --mode eagle

    # Online trainable EAGLE
    ./launch.sh --mode online_eagle
    ```

### 3.3 Documentation
- [ ] **Create `docs/online_eagle.md`**
  - [ ] Overview of online training
  - [ ] Configuration options
  - [ ] Performance considerations
  - [ ] Best practices
  - [ ] Troubleshooting guide

- [ ] **Update README**
  - [ ] Add section on online trainable EAGLE
  - [ ] Link to detailed documentation

---

## Phase 4: Stats and Metrics

### 4.1 Training Metrics
- [ ] **Extend `vllm/v1/metrics/loggers.py`**
  - [ ] Add training-specific metrics:
    - [ ] `eagle_training_loss`: Current training loss
    - [ ] `eagle_training_steps`: Total training steps
    - [ ] `eagle_training_buffer_size`: Current buffer size
    - [ ] `eagle_training_samples_collected`: Total samples collected
    - [ ] `eagle_training_grad_norm`: Gradient norm (for monitoring)
    - [ ] `eagle_training_lr`: Current learning rate
    - [ ] `eagle_training_time_ms`: Time spent training (per step)

### 4.2 Acceptance Rate Tracking
- [ ] **Modify `vllm/v1/spec_decode/eagle.py`**
  - [ ] Track acceptance rates over time:
    - [ ] Per-token acceptance rate
    - [ ] Per-request acceptance rate
    - [ ] Moving average (last 100 requests)
  - [ ] Store in `EagleProposer.stats`
  - [ ] Expose via `get_stats()` method

### 4.3 Improvement Metrics
- [ ] **Create `vllm/training/metrics_tracker.py`**
  - [ ] Define `MetricsTracker` class:
    - [ ] Track baseline acceptance rate (before training)
    - [ ] Track current acceptance rate (after training)
    - [ ] Compute improvement: `(current - baseline) / baseline * 100`
    - [ ] Track acceptance rate per workload type (if labeled)
    - [ ] Track acceptance rate over time (for visualization)

  - [ ] Add to training manager:
    ```python
    self.metrics_tracker = MetricsTracker()

    # After each inference step
    self.metrics_tracker.record_acceptance_rate(
        acceptance_rate=current_rate,
        workload_type=request.metadata.get("workload_type")
    )
    ```

### 4.4 Logging and Visualization
- [ ] **Add periodic stats logging**
  - [ ] Every N steps, log to console:
    ```
    [Online EAGLE] Step 1000 | Loss: 0.342 | Acceptance: 72.3% (+5.2%) | Buffer: 850/1000
    ```
  - [ ] Every M steps, save stats to file:
    - [ ] `stats.jsonl` (append-only log)
    - [ ] Format: `{"step": 1000, "loss": 0.342, "acceptance": 0.723, ...}`

- [ ] **Create visualization script: `scripts/visualize_training.py`**
  - [ ] Load `stats.jsonl`
  - [ ] Plot training loss over time
  - [ ] Plot acceptance rate over time
  - [ ] Plot acceptance improvement over time
  - [ ] Save plots to `plots/`

### 4.5 A/B Testing Support
- [ ] **Add ability to compare standard vs online EAGLE**
  - [ ] Modify stress test to support A/B testing:
    ```bash
    ./stress_test.sh --mode ab --workload chat --duration 300
    ```
  - [ ] Run identical workload against two servers:
    - Server A: Standard EAGLE
    - Server B: Online EAGLE
  - [ ] Collect and compare metrics:
    - Acceptance rate
    - Latency (TTFT, ITL)
    - Throughput
  - [ ] Output comparison report

---

## Phase 5: Advanced Features (Future Work)

### 5.1 Remote Training Support
- [ ] **Design remote training protocol**
  - [ ] Define gRPC/HTTP API for training service
  - [ ] Methods:
    - `add_training_sample(input_ids, hidden_states, labels)`
    - `get_updated_weights()`
    - `get_training_stats()`

- [ ] **Implement remote training server**
  - [ ] Separate service that receives training samples
  - [ ] Runs training loop independently
  - [ ] Serves updated weights on request

- [ ] **Modify training manager to use remote training**
  - [ ] Instead of local training, send samples to remote service
  - [ ] Periodically fetch updated weights
  - [ ] Handle network failures gracefully

### 5.2 Multi-Model Training
- [ ] **Support training multiple draft models**
  - [ ] For different model sizes or architectures
  - [ ] Route requests to appropriate draft model
  - [ ] Train each draft model independently

### 5.3 Reinforcement Learning from Acceptance Feedback
- [ ] **Use acceptance rate as reward signal**
  - [ ] Implement policy gradient for draft model
  - [ ] Reward: +1 for accepted tokens, -1 for rejected
  - [ ] Fine-tune draft model to maximize acceptance

### 5.4 Adaptive Learning Rate
- [ ] **Adjust LR based on acceptance rate**
  - [ ] If acceptance rate increasing: decrease LR (stabilize)
  - [ ] If acceptance rate decreasing: increase LR (explore)
  - [ ] Implement LR scheduling logic in trainer

---

## Testing Checklist

### Unit Tests
- [ ] `test_trainable_eagle.py`: Trainable model forward/backward
- [ ] `test_eagle_trainer.py`: Training loop, optimizer, checkpointing
- [ ] `test_data_buffer.py`: Buffer operations, sampling
- [ ] `test_training_manager.py`: Data collection, training triggers
- [ ] `test_metrics_tracker.py`: Metrics tracking, improvement calculation

### Integration Tests
- [ ] `test_online_training.py`: End-to-end training flow
- [ ] `test_async_training.py`: Async training doesn't block inference
- [ ] `test_weight_sync.py`: Weight synchronization between trainable and inference models
- [ ] `test_checkpoint.py`: Checkpoint save/load
- [ ] `test_multiprocess.py`: Training with multiprocess executor
- [ ] `test_tp_training.py`: Training with TP > 1

### Performance Tests
- [ ] Measure inference latency impact (should be < 5%)
- [ ] Measure training overhead (time per step)
- [ ] Measure memory overhead (buffer + trainable model)
- [ ] Measure throughput degradation (should be < 10%)

### Correctness Tests
- [ ] Verify training loss decreases over time
- [ ] Verify acceptance rate improves over time
- [ ] Verify forward pass equivalence (trainable vs inference)
- [ ] Verify weight updates propagate correctly
- [ ] Verify TP-aware gradient reduction

---

## Performance Targets

### Inference Impact
- **Latency increase:** < 5% compared to standard EAGLE
- **Throughput decrease:** < 10% compared to standard EAGLE

### Training Effectiveness
- **Acceptance rate improvement:** > 5% after 10K training steps
- **Training convergence:** Loss decreases by 50% within 5K steps

### Resource Usage
- **Memory overhead:** < 2GB for trainable model + buffer
- **Training time:** < 100ms per training step (batch_size=32)

---

## Files to Create/Modify

### New Files
```
vllm/model_executor/models/trainable_eagle.py
vllm/training/__init__.py
vllm/training/config.py
vllm/training/eagle_trainer.py
vllm/training/data_buffer.py
vllm/training/training_manager.py
vllm/training/metrics_tracker.py

tests/model_executor/test_trainable_eagle.py
tests/training/test_eagle_trainer.py
tests/training/test_data_buffer.py
tests/training/test_training_manager.py
tests/training/test_trainable_eagle_correctness.py
tests/integration/test_online_training.py

scripts/visualize_training.py
docs/online_eagle.md

launch.sh
stress_test.sh
```

### Modified Files
```
vllm/config/speculative.py
vllm/v1/engine/core.py
vllm/v1/worker/gpu_worker.py
vllm/v1/executor/abstract.py
vllm/v1/executor/uniproc_executor.py
vllm/v1/executor/multiproc_executor.py
vllm/v1/metrics/loggers.py
vllm/v1/spec_decode/eagle.py
vllm/engine/arg_utils.py
```

---

## Open Questions / Design Decisions

1. **Training frequency:** How often to train?
   - Option A: Every N inference steps (e.g., every 100 steps)
   - Option B: When buffer is full
   - Option C: Adaptive (based on acceptance rate)
   - **Recommendation:** Start with Option A, experiment with others

2. **Sample selection:** What samples to train on?
   - Option A: All samples
   - Option B: Only rejected speculative tokens (high learning signal)
   - Option C: Mix of accepted and rejected
   - **Recommendation:** Option B for efficiency

3. **Weight update strategy:** When to copy weights to inference model?
   - Option A: Every training step (high overhead)
   - Option B: Every N training steps (e.g., every 10 steps)
   - Option C: When acceptance rate improves
   - **Recommendation:** Option B

4. **TP handling:** How to handle TP during training?
   - Option A: Train only on rank 0, broadcast weights
   - Option B: Train on all ranks with data parallel
   - Option C: Train on all ranks with same data (gradient reduction)
   - **Recommendation:** Option C for correctness

5. **Catastrophic forgetting:** How to prevent forgetting?
   - Option A: Replay buffer with diverse samples
   - Option B: EWC (Elastic Weight Consolidation)
   - Option C: Small learning rate
   - **Recommendation:** Start with Option A + C

---

## Success Criteria

- [ ] **Phase 0:** Launch scripts work, baseline metrics collected
- [ ] **Phase 1:** Trainable model passes all correctness tests
- [ ] **Phase 2:** Online training integrated, acceptance rate improves
- [ ] **Phase 3:** Easy to enable/disable with single flag
- [ ] **Phase 4:** Metrics show improvement over baseline
- [ ] **Final:** A/B test shows online EAGLE outperforms standard EAGLE by > 5% acceptance rate

---

## Estimated Timeline

- **Phase 0:** 2-3 days (infrastructure + testing)
- **Phase 1:** 5-7 days (trainable model + validation)
- **Phase 2:** 7-10 days (integration + async training)
- **Phase 3:** 2-3 days (config + CLI)
- **Phase 4:** 3-4 days (metrics + visualization)
- **Testing & Refinement:** 5-7 days

**Total:** ~4-5 weeks

---

## Notes

- Start with single-GPU (TP=1) development, then extend to multi-GPU
- Use small learning rates (1e-4 or lower) to avoid instability
- Monitor training closely in early stages to catch issues
- Checkpoint frequently during development
- Test on diverse workloads to ensure generalization
