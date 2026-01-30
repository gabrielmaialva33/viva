# VIVA-QD Performance Guide

## Overview

This guide covers optimization strategies for achieving maximum throughput with VIVA-QD. Target performance: 320,000+ evaluations/second on RTX 4090.

## Hardware Requirements

### Minimum
- CPU: 8 cores, 16 threads
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- RAM: 32GB
- Storage: SSD (for BEAM cold starts)

### Recommended
- CPU: Intel i9-13900K / AMD Ryzen 9 7950X (24+ cores)
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 64GB DDR5
- Storage: NVMe SSD

### Optimal
- Same as recommended
- Multiple GPUs (multi-GPU support in development)
- 128GB RAM for very large populations

## Software Configuration

### CUDA Setup

```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### Erlang VM Tuning

Create `vm.args` file:

```
## Enable dirty schedulers
+SDcpu 16
+SDio 8

## Increase process limit
+P 1000000

## Enable HiPE (if available)
# +native

## Memory allocator tuning
+MBas aobf
+MBlmbcs 512

## Scheduler binding
+sbt db

## Enable busy waiting (reduces latency, increases CPU)
+sbwt short
```

### Rust Compilation

In `native/viva_glands/Cargo.toml`:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
target-cpu = "native"  # Via RUSTFLAGS

[profile.release.build-override]
opt-level = 3
```

Build with:

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
  cargo build --release
```

## Batch Size Optimization

### Finding Optimal Batch Size

```gleam
/// Run batch size sweep to find optimal configuration
pub fn batch_size_sweep(config: QDConfig) {
  let batch_sizes = [100, 250, 500, 1000, 2000, 4000, 4800, 6000, 8000]

  list.each(batch_sizes, fn(batch_size) {
    let start = erlang.system_time(microsecond)

    // Run 10 batches
    list.range(1, 10)
    |> list.each(fn(_) {
      batch_evaluate(generate_random_weights(batch_size, 867), config)
    })

    let elapsed = erlang.system_time(microsecond) - start
    let evals_per_sec = batch_size * 10 * 1_000_000 / elapsed

    io.println(
      "Batch " <> int.to_string(batch_size)
      <> ": " <> int.to_string(evals_per_sec) <> " evals/sec"
    )
  })
}
```

### Typical Results by GPU

| GPU | Optimal Batch | Peak Evals/sec | VRAM Usage |
|-----|---------------|----------------|------------|
| RTX 3070 | 2,000 | 145,000 | 6.5GB |
| RTX 3080 | 3,200 | 210,000 | 8.2GB |
| RTX 3090 | 4,000 | 265,000 | 14.1GB |
| RTX 4090 | 4,800 | 320,000 | 12.4GB |

### Memory vs Throughput Trade-off

```
VRAM Usage = base + (batch_size * per_genome)

where:
  base = ~430MB (CUDA context, cuFFT plans, projection matrix)
  per_genome = ~2.5KB (weights + activations + intermediates)
```

For 24GB VRAM:
```
max_batch = (24000 - 430) / 0.0025 = ~9.4M
practical_max = ~8000 (leaving headroom for cuFFT)
```

## Reducing Data Transfer Overhead

### Problem: Erlang-Rust Boundary

Each NIF call incurs:
- List -> Vec conversion (~2us per 1000 floats)
- Vec -> List conversion (~3us per 1000 floats)

### Solution 1: Batch APIs

Instead of:
```gleam
// BAD: 1000 NIF calls
list.map(genomes, fn(g) {
  glands.project(handle, g.weights)
})
```

Use:
```gleam
// GOOD: 1 NIF call
glands.batch_project(handle, list.map(genomes, fn(g) { g.weights }))
```

### Solution 2: Resource Handles

Keep GPU resources alive across calls:

```gleam
// Initialize once
let handle = glands.init(config)

// Reuse handle for all operations
list.each(generations, fn(_) {
  let results = glands.batch_forward(handle, weights, inputs)
  // ...
})
```

### Solution 3: Dirty Schedulers

NIF functions use dirty schedulers to avoid blocking BEAM:

```rust
#[rustler::nif(schedule = "DirtyCpu")]
fn glands_bind(...) -> ...
```

This allows Erlang processes to continue while GPU operations complete.

## Parallelization Strategies

### Level 1: Erlang Process Parallelism

```gleam
/// Evaluate genomes in parallel using OTP
pub fn parallel_evaluate(genomes: List(Genome), workers: Int) {
  let chunks = list.sized_chunk(genomes, list.length(genomes) / workers)

  // Spawn workers
  let tasks = list.map(chunks, fn(chunk) {
    task.async(fn() {
      list.map(chunk, evaluate_genome)
    })
  })

  // Collect results
  list.flat_map(tasks, task.await_forever)
}
```

### Level 2: Rust Rayon Parallelism

```rust
// Parallel batch evaluation in Rust
pub fn batch_evaluate(weights: &[Vec<f32>], inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    weights.par_iter()
        .zip(inputs.par_iter())
        .map(|(w, i)| forward_single(w, i))
        .collect()
}
```

### Level 3: GPU Parallelism

```rust
// Batched forward pass on GPU
fn batch_forward_gpu(
    weights_batch: &Tensor,  // [batch, weight_count]
    inputs_batch: &Tensor,   // [batch, input_dim]
) -> Tensor {
    // All batches processed in single CUDA kernel
    let layer1 = inputs_batch.matmul(&weights_batch.slice(...));
    let activated = layer1.relu();
    // ...
}
```

### Level 4: CUDA Streams (Advanced)

```rust
// Overlap H2D, compute, D2H across multiple streams
fn pipelined_evaluation(batches: &[Batch]) {
    let streams: Vec<CUstream> = (0..3).map(|_| cuda::stream_create()).collect();

    for (i, batch) in batches.iter().enumerate() {
        let stream = &streams[i % 3];

        // Previous batch's D2H overlaps with current batch's compute
        cuda::memcpy_htod_async(batch.d_input, batch.h_input, stream);
        cuda::kernel_launch_async(forward_kernel, batch, stream);
        cuda::memcpy_dtoh_async(batch.h_output, batch.d_output, stream);
    }

    cuda::device_synchronize();
}
```

## Memory Optimization

### Reducing HRR Memory

If HRR dimension is too large:

```gleam
// Use smaller HRR for initial exploration
let fast_config = HoloMapConfig(
  ..default_config(),
  hrr_dim: 4096,  // Instead of 8192
)

// Switch to full resolution for final training
let final_config = HoloMapConfig(
  ..default_config(),
  hrr_dim: 8192,
)
```

### Genome Compression

For large populations, compress inactive genomes:

```gleam
/// Compress genome by removing disabled connections
pub fn compress_genome(genome: Genome) -> CompressedGenome {
  let active_connections = list.filter(genome.connections, fn(c) { c.enabled })
  CompressedGenome(
    id: genome.id,
    node_count: list.length(genome.nodes),
    connections: active_connections,
    fitness: genome.fitness,
  )
}
```

### Archive Pruning

For very long runs, prune low-quality elites:

```gleam
/// Remove bottom 10% of elites when archive is full
pub fn prune_archive(grid: MapElitesGrid, threshold: Float) -> MapElitesGrid {
  let elites = get_elites(grid)
  let sorted = list.sort(elites, fn(a, b) { float.compare(a.fitness, b.fitness) })
  let cutoff = float.truncate(int.to_float(list.length(sorted)) *. threshold)
  let to_remove = list.take(sorted, cutoff)

  list.fold(to_remove, grid, fn(g, elite) {
    let cell = behavior_to_cell(elite.behavior, g)
    MapElitesGrid(..g, cells: dict.delete(g.cells, cell))
  })
}
```

## Profiling

### Erlang Profiling

```erlang
% In Erlang shell
fprof:trace(start).
% Run training
viva_trainer:train(50, Config).
fprof:trace(stop).
fprof:profile().
fprof:analyse([{dest, "profile.txt"}]).
```

### Rust Profiling

```bash
# CPU profiling with perf
perf record -g ./target/release/benchmark
perf report

# CUDA profiling with nsys
nsys profile --stats=true ./target/release/benchmark
nsys-ui profile.qdrep
```

### GPU Utilization Monitoring

```bash
# Real-time GPU monitoring
watch -n 0.5 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet -d 1

# With nvtop (recommended)
nvtop
```

## Common Performance Issues

### Issue 1: Low GPU Utilization

**Symptoms:** GPU util < 50%, but throughput plateau

**Causes:**
- Batch size too small
- CPU bottleneck in data preparation
- Excessive synchronization

**Solutions:**
1. Increase batch size
2. Use async data transfer
3. Pipeline CPU and GPU work

### Issue 2: Memory Thrashing

**Symptoms:** Performance degrades over time, high swap usage

**Causes:**
- Population too large for RAM
- Memory leaks in NIFs
- Inefficient Erlang term building

**Solutions:**
1. Reduce population size
2. Check for Rust memory leaks
3. Use resource handles instead of large terms

### Issue 3: BEAM Scheduler Blocking

**Symptoms:** Other Erlang processes become unresponsive

**Causes:**
- NIF running on regular scheduler
- Long-running reduction without yielding

**Solutions:**
1. Use `schedule = "DirtyCpu"` for NIFs
2. Chunk work into smaller pieces
3. Use async NIFs with callback

### Issue 4: Thermal Throttling

**Symptoms:** Performance drops after sustained load

**Causes:**
- GPU overheating
- CPU thermal limits

**Solutions:**
1. Improve case airflow
2. Reduce batch size slightly
3. Add cooling pauses between generations

## Benchmarking

### Standard Benchmark Suite

```gleam
pub fn run_benchmarks() {
  io.println("=== VIVA-QD Benchmark Suite ===")
  io.println("")

  // 1. Raw NIF throughput
  benchmark_nif_throughput()

  // 2. Forward pass throughput
  benchmark_forward_pass()

  // 3. Full evaluation throughput
  benchmark_full_evaluation()

  // 4. QD algorithm overhead
  benchmark_qd_overhead()
}

fn benchmark_nif_throughput() {
  let handle = glands.init(glands.default_config())
  let vec_a = list.repeat(0.5, 8192)
  let vec_b = list.repeat(0.3, 8192)

  let start = erlang.system_time(microsecond)
  list.range(1, 10000)
  |> list.each(fn(_) { glands.bind(handle, vec_a, vec_b) })
  let elapsed = erlang.system_time(microsecond) - start

  io.println("NIF bind: " <> int.to_string(10000 * 1_000_000 / elapsed) <> " ops/sec")
}
```

### Expected Results (RTX 4090)

```
=== VIVA-QD Benchmark Suite ===

NIF bind: 245,000 ops/sec
NIF similarity: 12,500,000 ops/sec
Forward pass (batch 4800): 880,000 forwards/sec
Full evaluation: 320,000 evals/sec
QD overhead: 2.3% of total time
Archive update: 0.12us per elite
Speciation: 45us per generation

Total training throughput: 314,000 genome-generations/sec
```

## Tuning Checklist

Before production runs:

- [ ] CUDA driver updated to latest
- [ ] Rust compiled with `--release` and native CPU flags
- [ ] Erlang VM dirty schedulers enabled
- [ ] Batch size optimized for GPU
- [ ] Memory usage monitored
- [ ] Thermal headroom verified
- [ ] Checkpoint saving configured
- [ ] Log level set appropriately (not DEBUG)
- [ ] Process limits increased (`+P` flag)

## Quick Reference

### Performance Targets by Domain

| Domain | Batch Size | Target Evals/sec | Notes |
|--------|------------|------------------|-------|
| XOR | 1000 | 1,500,000 | Simple, CPU may be faster |
| Sinuca | 4800 | 320,000 | Physics-limited |
| MuJoCo | 2000 | 150,000 | Complex physics |
| Atari | 500 | 45,000 | Frame rendering |

### GPU Memory Budget

| Component | Size |
|-----------|------|
| CUDA context | 300MB |
| cuFFT plans | 50MB |
| Projection matrix | 128MB |
| HRR buffers (3x) | 192MB |
| Batch weights | 2.5KB/genome |
| Batch activations | 1KB/genome |
| **Overhead** | **~700MB** |
| **Per batch** | **~3.5KB/genome** |
