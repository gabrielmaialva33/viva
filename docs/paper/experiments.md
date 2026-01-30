# 4. Experiments

## 4.1 Experimental Setup

### 4.1.1 Hardware Configuration

All experiments were conducted on a single workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i9-13900K (24 cores, 32 threads) |
| GPU | NVIDIA GeForce RTX 4090 (16,384 CUDA cores, 24GB VRAM) |
| RAM | 64GB DDR5-5600 |
| Storage | 2TB NVMe SSD (Samsung 990 PRO) |
| OS | Ubuntu 22.04 LTS (WSL2 on Windows 11) |

### 4.1.2 Software Configuration

| Software | Version |
|----------|---------|
| Gleam | 1.14.0 |
| Erlang/OTP | 27.0 |
| Rust | 1.75.0 |
| CUDA | 12.3 |
| cuDNN | 8.9.7 |
| burn-rs | 0.13.0 |
| Jolt Physics | 5.0.0 |

### 4.1.3 Hyperparameters

**NEAT Configuration:**
```
population_size: 50
num_inputs: 8
num_outputs: 3
weight_mutation_rate: 0.8
weight_perturb_rate: 0.9
add_node_rate: 0.03
add_connection_rate: 0.05
compatibility_threshold: 1.0
survival_threshold: 0.2
elitism: 2
```

**QD Configuration:**
```
grid_size: 10 x 10 (100 cells)
initial_novelty_weight: 0.7
final_novelty_weight: 0.3
annealing_generations: 50
shots_per_episode: 3
max_steps_per_shot: 200
```

## 4.2 Benchmark Results

### 4.2.1 Throughput Performance

We measured evaluation throughput across different batch sizes:

| Batch Size | Evals/sec | GPU Util | Memory |
|------------|-----------|----------|--------|
| 100 | 45,000 | 23% | 1.2 GB |
| 500 | 156,000 | 67% | 2.8 GB |
| 1,000 | 248,000 | 84% | 4.1 GB |
| 2,000 | 298,000 | 92% | 6.3 GB |
| 4,800 | 320,000 | 97% | 12.4 GB |
| 8,000 | 312,000 | 98% | 18.2 GB |

**Peak throughput: 320,000 evaluations/second at batch size 4,800**

The throughput plateau at 4,800 indicates optimal GPU occupancy; larger batches incur memory transfer overhead without additional parallelism benefits.

### 4.2.2 Speedup Analysis

Comparing against baselines:

| Implementation | Evals/sec | Speedup |
|----------------|-----------|---------|
| Python NEAT (CPU) | 6.4 | 1x (baseline) |
| Gleam NEAT (CPU) | 12.1 | 1.9x |
| Gleam NEAT (Rust SIMD) | 847 | 132x |
| VIVA-QD (GPU) | 320,000 | 50,000x |

The 50,000x speedup enables training runs that would take weeks on CPU to complete in minutes.

### 4.2.3 Component Latency Breakdown

Per-evaluation latency analysis:

| Component | Latency (us) | % Total |
|-----------|--------------|---------|
| Physics simulation | 1.8 | 57.7% |
| Neural forward pass | 0.6 | 19.2% |
| Fitness computation | 0.4 | 12.8% |
| Data marshalling | 0.2 | 6.4% |
| Grid update | 0.12 | 3.9% |
| **Total** | **3.12** | **100%** |

Physics simulation dominates, suggesting future optimizations should target GPU-accelerated physics.

## 4.3 Quality-Diversity Results

### 4.3.1 Training Progression

50-generation training run with QD v6 (QD-NEAT Hybrid):

| Generation | Best Fitness | Coverage | QD-Score | Novelty Weight |
|------------|--------------|----------|----------|----------------|
| 0 | 12.3 | 8.0% | 45.2 | 0.70 |
| 10 | 34.7 | 23.0% | 187.4 | 0.62 |
| 20 | 52.1 | 38.0% | 412.8 | 0.54 |
| 30 | 68.4 | 48.0% | 624.1 | 0.46 |
| 40 | 81.2 | 54.0% | 789.5 | 0.38 |
| 50 | 92.8 | 57.9% | 908.3 | 0.30 |

### 4.3.2 Final Archive Statistics

After 50 generations:

```
=== QD v6 Training Complete ===
Best fitness: 92.8
Coverage: 57.9%
QD-Score: 908.3
Cells filled: 58/100
Average elite fitness: 15.66
Median elite fitness: 12.4
Fitness std: 18.2
```

### 4.3.3 Behavior Space Visualization

The 10x10 MAP-Elites grid shows coverage across behavior dimensions:

```
Shot Angle (hit_angle)
    0.0   0.2   0.4   0.6   0.8   1.0
    +-----+-----+-----+-----+-----+
1.0 | ### | ### | ### |     | ### |
    +-----+-----+-----+-----+-----+
0.8 | ### | ### | ### | ### | ### |
    +-----+-----+-----+-----+-----+
0.6 | ### | ### |     | ### | ### |  Scatter
    +-----+-----+-----+-----+-----+  Ratio
0.4 | ### |     | ### | ### | ### |
    +-----+-----+-----+-----+-----+
0.2 | ### | ### | ### | ### |     |
    +-----+-----+-----+-----+-----+
0.0 |     | ### | ### | ### | ### |
    +-----+-----+-----+-----+-----+

### = cell occupied with elite
    = empty cell
```

### 4.3.4 Ablation Study

Comparing algorithm variants:

| Variant | Coverage | QD-Score | Best Fitness |
|---------|----------|----------|--------------|
| Pure NEAT (no QD) | 12.0% | 124.5 | 78.3 |
| MAP-Elites (random) | 42.0% | 523.7 | 45.2 |
| QD v3 (decoupled) | 51.0% | 712.4 | 84.1 |
| QD v6 (hybrid) | 57.9% | 908.3 | 92.8 |
| QD v9 (GPU-NES) | 62.0% | 1012.7 | 98.4 |

**Key findings:**
- QD improves coverage 4.8x over pure NEAT
- Decoupled selection adds +9% coverage
- Hybrid NEAT-QD achieves best balance
- GPU-accelerated NES (v9) provides additional gains

## 4.4 Behavioral Diversity Analysis

### 4.4.1 Phenotype Diversity

Analyzing discovered strategies across the archive:

| Cluster | Behavior | Count | Avg Fitness | Strategy Description |
|---------|----------|-------|-------------|---------------------|
| A | Low angle, low scatter | 12 | 23.4 | Careful positioning shots |
| B | Low angle, high scatter | 8 | 18.7 | Break shots |
| C | High angle, low scatter | 15 | 31.2 | Direct pocket shots |
| D | High angle, high scatter | 11 | 12.1 | Risky combination shots |
| E | Medium angle/scatter | 12 | 28.9 | Balanced all-rounders |

### 4.4.2 Genome Complexity Evolution

Network topology complexity over generations:

| Generation | Avg Nodes | Avg Connections | Avg Hidden |
|------------|-----------|-----------------|------------|
| 0 | 12.0 | 27.0 | 0.0 |
| 10 | 13.2 | 31.4 | 1.2 |
| 25 | 14.8 | 38.7 | 2.8 |
| 50 | 16.4 | 45.2 | 4.4 |

Networks grow modestly in complexity, consistent with NEAT's complexification principle.

## 4.5 Fault Tolerance Evaluation

### 4.5.1 Crash Recovery

We injected failures to test OTP supervision:

| Failure Type | Recovery Time | Data Loss |
|--------------|---------------|-----------|
| NIF crash | 12ms | 0 genomes |
| CUDA OOM | 45ms | Current batch |
| Process timeout | 8ms | 0 genomes |
| Node restart | 2.1s | Last checkpoint |

The supervisor tree successfully isolates failures, preventing cascade effects.

### 4.5.2 Long-Running Stability

24-hour continuous training:

```
Total evaluations: 27.6 billion
NIF crashes: 3
CUDA errors: 7
Recoveries: 10/10 (100%)
Archive integrity: Maintained
Final coverage: 89.2%
```

## 4.6 Comparison with State-of-the-Art

Comparing against published QD benchmarks (where applicable):

| System | Domain | Evals/sec | Coverage | Notes |
|--------|--------|-----------|----------|-------|
| CMA-ME | Arm Repertoire | 8,500 | 68% | CPU-only |
| PGA-MAP-Elites | QDax Suite | 125,000 | 71% | JAX/TPU |
| VIVA-QD | Sinuca | 320,000 | 57.9% | RTX 4090 |

Note: Direct comparison is limited due to different domains. Our coverage is lower because the sinuca domain has challenging sparse rewards; the throughput advantage enables more exploration.
