# 3. Methodology

## 3.1 System Architecture

VIVA-QD employs a layered architecture that separates concerns across four technology stacks:

```
+------------------------------------------------------------------+
|                         GLEAM (Application Layer)                 |
|   - Type-safe NEAT genome representation                         |
|   - QD algorithm orchestration (MAP-Elites, selection)           |
|   - Behavioral descriptor extraction                              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     ERLANG/OTP (Supervision Layer)                |
|   - Process supervision (let-it-crash philosophy)                |
|   - Parallel genome evaluation scheduling                         |
|   - Resource management and backpressure                          |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     RUST NIFs (Native Bridge)                     |
|   - Zero-copy tensor marshalling                                  |
|   - Thread-safe GPU context management                            |
|   - cuFFT wrapper for HRR operations                              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        CUDA (Compute Layer)                       |
|   - Batched neural network forward pass (burn-rs)                |
|   - Circular convolution via cuFFT                                |
|   - Custom kernels for complex multiplication                     |
+------------------------------------------------------------------+
```

### 3.1.1 Gleam Implementation

Gleam provides static typing on the Erlang BEAM, enabling:

```gleam
/// Genome representation with innovation tracking
pub type Genome {
  Genome(
    id: Int,
    nodes: List(NodeGene),
    connections: List(ConnectionGene),
    fitness: Float,
    adjusted_fitness: Float,
    species_id: Int,
  )
}

/// Connection gene with historical marking
pub type ConnectionGene {
  ConnectionGene(
    in_node: Int,
    out_node: Int,
    weight: Float,
    enabled: Bool,
    innovation: Int,  // Historical marking for crossover
  )
}
```

### 3.1.2 Rust NIF Interface

The native interface exposes GPU operations to Gleam:

```rust
#[rustler::nif(schedule = "DirtyCpu")]
fn glands_bind(
    resource: ResourceArc<GlandsResource>,
    a: Vec<f32>,
    b: Vec<f32>,
) -> Result<Vec<f32>, String> {
    // GPU-accelerated circular convolution
    if let Some(ref gpu) = resource.hrr_gpu {
        gpu.lock()?.bind(&a, &b)
    } else {
        resource.hrr_cpu.lock()?.bind(&a, &b)
    }
}
```

## 3.2 HoloMAP-Elites Algorithm

HoloMAP-Elites extends MAP-Elites with Holographic Reduced Representations for genome encoding.

### 3.2.1 Holographic Genome Encoding

Each genome is projected into an 8192-dimensional HRR space:

1. **Weight Extraction**: Extract all connection weights as a feature vector
2. **Projection**: Multiply by orthogonal projection matrix (4096 x 8192)
3. **Normalization**: L2-normalize to unit sphere

```
HRR_vector = normalize(W_proj * flatten(genome_weights))
```

### 3.2.2 Behavior Descriptor Extraction

Behavior descriptors are extracted from the first N dimensions of the HRR vector:

```gleam
pub fn hrr_to_behavior(hrr_vector: List(Float), dims: Int) -> Behavior {
  let features = list.take(hrr_vector, dims)
  novelty.behavior_from_features(features)
}
```

For the sinuca domain, we use 2D descriptors:
- **Dimension 0**: Average shot angle (normalized)
- **Dimension 1**: Ball scatter ratio (movement diversity)

### 3.2.3 Grid Update with Cell-wise Elitism

```
Algorithm 1: HoloMAP-Elites Update
Input: grid G, genome g, fitness f, behavior b
Output: updated grid G'

1. cell <- behavior_to_cell(b, G.size)
2. if cell not in G.cells:
3.     G.cells[cell] <- Elite(g, f, b)
4. else if f > G.cells[cell].fitness:
5.     G.cells[cell] <- Elite(g, f, b)
6. return G
```

## 3.3 QD-NEAT Hybrid Selection

We introduce a decoupled selection mechanism that separates fitness evaluation from parent selection pressure.

### 3.3.1 Linear Weight Annealing

The novelty weight w decays linearly over generations:

```
w(t) = w_initial - (t / T_anneal) * (w_initial - w_final)

where:
  w_initial = 0.7  (exploration-focused)
  w_final = 0.3    (exploitation-focused)
  T_anneal = 50    (annealing period)
```

### 3.3.2 Archive-Based Parent Selection

Parent selection draws from two sources with probability w:

```
Algorithm 2: Decoupled Selection
Input: archive A, population P, novelty_weight w
Output: parent genome

1. r <- uniform_random(0, 1)
2. if r < w:
3.     // Diversity selection: random cell
4.     cell <- random_choice(A.occupied_cells)
5.     return A.cells[cell].genome
6. else:
7.     // Elite selection: best fitness
8.     elites <- sort_by_fitness(A.cells, descending=True)
9.     return tournament_select(elites[:5])
```

### 3.3.3 NEAT Mutation Operators

Standard NEAT mutations are applied after parent selection:

| Mutation | Rate | Description |
|----------|------|-------------|
| Weight Perturbation | 0.9 | Gaussian noise to existing weights |
| Weight Reset | 0.05 | Replace weight with new random value |
| Add Node | 0.03 | Split existing connection with new hidden node |
| Add Connection | 0.05 | Create new connection between unconnected nodes |
| Disable Connection | 0.01 | Disable random connection |

## 3.4 GPU-Accelerated Evaluation Pipeline

### 3.4.1 Batch Neural Network Forward

The burn-rs backend processes multiple networks simultaneously:

```rust
pub fn batch_forward(
    weights_batch: Vec<Vec<f32>>,
    inputs_batch: Vec<Vec<f32>>,
    architecture: Vec<usize>,
) -> Vec<Vec<f32>> {
    // Stack into tensors [batch, features]
    let weights_tensor = stack_weights(&weights_batch, &architecture);
    let inputs_tensor = Tensor::from_data(inputs_batch);

    // Batched matmul + activation
    for layer in architecture.windows(2) {
        inputs_tensor = inputs_tensor.matmul(&weights_tensor[layer])
            .relu();  // or sigmoid for output layer
    }

    inputs_tensor.into_data().to_vec()
}
```

### 3.4.2 cuFFT HRR Operations

Circular convolution for HRR binding uses cuFFT:

```rust
fn bind_gpu(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    // 1. Copy to GPU as complex
    cuda::memcpy_htod_sync(self.d_a, &to_complex(a))?;
    cuda::memcpy_htod_sync(self.d_b, &to_complex(b))?;

    // 2. Forward FFT
    cufft::exec_c2c(self.plan, self.d_a, self.d_a, CUFFT_FORWARD)?;
    cufft::exec_c2c(self.plan, self.d_b, self.d_b, CUFFT_FORWARD)?;

    // 3. Element-wise complex multiplication (custom kernel)
    self.complex_mul_kernel(false)?;

    // 4. Inverse FFT
    cufft::exec_c2c(self.plan, self.d_result, self.d_result, CUFFT_INVERSE)?;

    // 5. Copy back and extract real part
    cuda::memcpy_dtoh_sync(&mut result, self.d_result)?;
    Ok(extract_real(&result))
}
```

### 3.4.3 Physics Simulation Batching

Jolt Physics integration enables parallel simulation:

```gleam
// Batch evaluate N genomes in parallel
let results = list.map(genomes, fn(genome) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  list.fold(range(1, shots_per_episode), #(0.0, table, episode), fn(acc, _) {
    let inputs = encode_inputs(acc.1)
    let outputs = neat.forward(genome, inputs)
    let shot = decode_outputs(outputs)

    let #(fitness, new_table, new_ep) =
      fitness.evaluate(acc.1, shot, max_steps)

    #(acc.0 +. fitness, new_table, new_ep)
  })
})
```

## 3.5 Sinuca Domain Specification

### 3.5.1 State Representation (8 inputs)

| Input | Range | Description |
|-------|-------|-------------|
| cue_x | [-1, 1] | Cue ball X position (normalized) |
| cue_z | [-1, 1] | Cue ball Z position (normalized) |
| target_x | [-1, 1] | Target ball X position |
| target_z | [-1, 1] | Target ball Z position |
| pocket_angle | [-1, 1] | Angle to best pocket (normalized by pi) |
| pocket_dist | [-1, 1] | Distance to best pocket (normalized) |
| target_value | [-1, 1] | Point value of target (1-7 mapped to [-1,1]) |
| balls_left | [-1, 1] | Remaining balls (normalized) |

### 3.5.2 Action Space (3 outputs)

| Output | Range | Description |
|--------|-------|-------------|
| angle_adj | [0, 1] | Adjustment to pocket angle (+-45 degrees) |
| power | [0, 1] | Shot power (mapped to 0.1-1.0) |
| english | [0, 1] | Cue ball spin (mapped to [-0.8, 0.8]) |

### 3.5.3 Fitness Function

```
fitness = sum(shot_rewards) where:

shot_reward =
    25.0 * balls_pocketed +
    10.0 * approach_bonus +
    5.0 * combo_bonus +
    -15.0 * if_scratch
```

The approach bonus rewards shots that move the cue ball closer to the next target, enabling multi-step planning.
