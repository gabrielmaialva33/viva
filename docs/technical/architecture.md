# VIVA-QD Technical Architecture

## Overview

VIVA-QD implements a heterogeneous neuroevolution system spanning four technology layers: Gleam (application), Erlang/OTP (supervision), Rust (native bridge), and CUDA (compute).

## System Diagram

```
+============================================================================+
|                              VIVA-QD ARCHITECTURE                          |
+============================================================================+

    +------------------------------------------------------------------+
    |                         GLEAM APPLICATION                         |
    |------------------------------------------------------------------|
    | viva/neural/neat.gleam     | NEAT genome, mutations, crossover   |
    | viva/neural/holomap.gleam  | MAP-Elites grid, behavior mapping   |
    | viva/neural/novelty.gleam  | Behavior descriptors, novelty calc  |
    | [legacy] domain-specific trainers removed    | Domain-specific trainers removed          |
    +------------------------------------------------------------------+
                                    |
                                    | Gleam FFI (@external)
                                    v
    +------------------------------------------------------------------+
    |                        ERLANG/OTP RUNTIME                         |
    |------------------------------------------------------------------|
    | viva/supervisor.gleam      | OTP supervisor tree                  |
    | gleam_otp                  | Process management, gen_server       |
    | BEAM VM                    | Scheduler, garbage collection        |
    +------------------------------------------------------------------+
                                    |
                                    | Rustler NIF
                                    v
    +------------------------------------------------------------------+
    |                         RUST NATIVE LAYER                         |
    |------------------------------------------------------------------|
    | native/viva_glands/        | HRR operations, projections          |
    |   - glands_init()          | Initialize GPU context               |
    |   - glands_bind()          | Circular convolution (cuFFT)         |
    |   - glands_project()       | LLM -> HRR projection (Candle)       |
    |   - glands_similarity()    | SIMD cosine similarity               |
    +------------------------------------------------------------------+
                                    |
                                    | CUDA Driver API
                                    v
    +------------------------------------------------------------------+
    |                          CUDA COMPUTE                             |
    |------------------------------------------------------------------|
    | cuFFT                      | FFT for circular convolution         |
    | Candle CUDA                | Tensor operations, matmul            |
    | burn-rs                    | Batched neural network forward       |
    | Custom Kernels             | Complex multiplication                |
    +------------------------------------------------------------------+
```

## Component Details

### 1. Gleam Application Layer

#### 1.1 NEAT Implementation (`viva/neural/neat.gleam`)

Core types:

```gleam
/// Node in the neural network
pub type NodeGene {
  NodeGene(id: Int, node_type: NodeType, activation: ActivationType)
}

/// Connection between nodes
pub type ConnectionGene {
  ConnectionGene(
    in_node: Int,
    out_node: Int,
    weight: Float,
    enabled: Bool,
    innovation: Int,  // Historical marking
  )
}

/// Complete genome
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

/// Population with speciation
pub type Population {
  Population(
    genomes: List(Genome),
    species: List(Species),
    generation: Int,
    innovation_counter: Int,
    node_counter: Int,
    innovation_history: Dict(#(Int, Int), Int),
  )
}
```

Key functions:

| Function | Purpose |
|----------|---------|
| `create_population(config, seed)` | Initialize minimal networks |
| `forward(genome, inputs)` | Topological forward pass |
| `mutate_weights(genome, config, seed)` | Perturb/reset weights |
| `mutate_add_node(genome, pop, seed)` | Split connection |
| `mutate_add_connection(genome, pop, seed)` | Add new synapse |
| `crossover(parent1, parent2, seed)` | Align by innovation |
| `speciate(population, config)` | Group similar genomes |
| `evolve(population, results, config, seed)` | Full generation step |

#### 1.2 HoloMAP-Elites (`viva/neural/holomap.gleam`)

Core types:

```gleam
/// Elite stored in archive cell
pub type Elite {
  Elite(
    genome_id: Int,
    behavior: Behavior,
    hrr_vector: List(Float),
    fitness: Float,
    generation_added: Int,
  )
}

/// MAP-Elites archive
pub type MapElitesGrid {
  MapElitesGrid(
    cells: Dict(#(Int, Int), Elite),
    grid_size: Int,
    behavior_dims: Int,
    min_bounds: List(Float),
    max_bounds: List(Float),
  )
}
```

Key functions:

| Function | Purpose |
|----------|---------|
| `new_grid(config)` | Create empty archive |
| `behavior_to_cell(behavior, grid)` | Map behavior to grid cell |
| `try_add_elite(grid, ...)` | Cell-wise elitism update |
| `get_elites(grid)` | Retrieve all elites |
| `coverage(grid)` | Percentage filled |
| `qd_score(grid)` | Sum of elite fitnesses |
| `adaptive_novelty_weight(gen, config)` | Sigmoid decay |
| `tournament_select(grid, size, seed)` | Selection from archive |

#### 1.3 Domain Trainers (legacy)

Legacy trainer modules were removed from this codebase. Remaining trainer behavior is now exercised through core NeuroEvolution modules (`neat`, `holomap`, `novelty`) and NIF-backed physical simulations.

No dedicated legacy domain modules are present in the active repository.

### 2. Erlang/OTP Layer

#### 2.1 Supervisor Tree

```
viva_supervisor (one_for_one)
    |
    +-- soul_pool_supervisor (simple_one_for_one)
    |       |
    |       +-- soul_worker_1
    |       +-- soul_worker_2
    |       +-- ...
    |
    +-- glands_supervisor (one_for_one)
    |       |
    |       +-- glands_manager (GPU resource)
    |
    +-- telemetry_supervisor (rest_for_one)
            |
            +-- metrics_collector
            +-- metrics_reporter
```

#### 2.2 Process Communication

```
                    +-----------------+
                    |   Trainer       |
                    | (orchestrator)  |
                    +-----------------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+---------------+  +---------------+  +---------------+
| Worker 1      |  | Worker 2      |  | Worker N      |
| (genome eval) |  | (genome eval) |  | (genome eval) |
+---------------+  +---------------+  +---------------+
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                  +------------------+
                  | Glands Manager   |
                  | (GPU resource)   |
                  +------------------+
```

### 3. Rust Native Layer

#### 3.1 NIF Interface (`native/viva_glands/src/lib.rs`)

```rust
// Resource handle wrapping GPU state
struct GlandsResource {
    llm: Option<Arc<Mutex<LlmResource>>>,
    projection_gpu: Option<Mutex<ProjectionGPU>>,
    projection_cpu: QuantizedProjection,
    hrr_gpu: Option<Mutex<HrrWorkspaceGPU>>,
    hrr_cpu: Mutex<HrrWorkspace>,
    config: GlandsConfig,
    has_gpu: bool,
}

// NIF functions
#[rustler::nif]
fn glands_init(config: NifConfig) -> Result<ResourceArc<GlandsResource>, String>

#[rustler::nif(schedule = "DirtyCpu")]
fn glands_project(resource, embedding) -> Result<Vec<f32>, String>

#[rustler::nif(schedule = "DirtyCpu")]
fn glands_bind(resource, a, b) -> Result<Vec<f32>, String>

#[rustler::nif(schedule = "DirtyCpu")]
fn glands_unbind(resource, trace, key) -> Result<Vec<f32>, String>

#[rustler::nif]
fn glands_similarity(a, b) -> Result<f32, String>

#[rustler::nif]
fn glands_batch_similarity(vectors, query) -> Result<Vec<f32>, String>

#[rustler::nif]
fn glands_superpose(vectors) -> Result<Vec<f32>, String>

#[rustler::nif]
fn glands_check() -> String

#[rustler::nif]
fn glands_benchmark(resource, iterations) -> String
```

#### 3.2 CPU Fallback (SIMD)

```rust
/// SIMD-accelerated cosine similarity (AVX2)
#[inline(always)]
fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = norm_simd(a);
    let norm_b = norm_simd(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// AVX2-friendly dot product
#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // Process 8 floats at a time
    let chunks = a.len() / 8;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let offset = i * 8;
        // Unrolled for compiler auto-vectorization
        sum += a[offset+0] * b[offset+0]
             + a[offset+1] * b[offset+1]
             + a[offset+2] * b[offset+2]
             + a[offset+3] * b[offset+3]
             + a[offset+4] * b[offset+4]
             + a[offset+5] * b[offset+5]
             + a[offset+6] * b[offset+6]
             + a[offset+7] * b[offset+7];
    }
    // Handle remainder...
    sum
}
```

### 4. CUDA Compute Layer

#### 4.1 cuFFT HRR Operations

```rust
struct HrrWorkspaceGPU {
    dim: usize,
    plan: cufft_sys::cufftHandle,
    d_a: CUdeviceptr,       // Device buffer A
    d_b: CUdeviceptr,       // Device buffer B
    d_result: CUdeviceptr,  // Result buffer
    kernel_func: CUfunction,
    _module: CUmodule,
}

impl HrrWorkspaceGPU {
    fn bind(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        // 1. H2D copy (interleaved complex)
        cuda::memcpy_htod_sync(self.d_a, &to_complex(a))?;
        cuda::memcpy_htod_sync(self.d_b, &to_complex(b))?;

        // 2. Forward FFT
        cufft::exec_c2c(self.plan, self.d_a, self.d_a, FORWARD)?;
        cufft::exec_c2c(self.plan, self.d_b, self.d_b, FORWARD)?;

        // 3. Complex multiply (custom kernel)
        self.complex_mul_kernel(false)?;

        // 4. Inverse FFT
        cufft::exec_c2c(self.plan, self.d_result, self.d_result, INVERSE)?;

        // 5. D2H copy and extract real
        cuda::memcpy_dtoh_sync(&mut result, self.d_result)?;
        Ok(extract_real_normalized(&result, self.dim))
    }
}
```

#### 4.2 Custom CUDA Kernel

```cuda
extern "C" __global__ void complex_mul(
    const float2* __restrict__ a,
    const float2* __restrict__ b,
    float2* __restrict__ out,
    int n,
    int conjugate_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[idx].x;
        float ai = a[idx].y;
        float br = b[idx].x;
        float bi = conjugate_b ? -b[idx].y : b[idx].y;
        // Complex multiplication
        out[idx].x = ar * br - ai * bi;
        out[idx].y = ar * bi + ai * br;
    }
}
```

#### 4.3 Candle Projection

```rust
struct ProjectionGPU {
    weights: Tensor,        // [hrr_dim, llm_dim] on GPU
    device: Device,
    llm_dim: usize,
    hrr_dim: usize,
}

impl ProjectionGPU {
    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>, String> {
        // Create input tensor [1, llm_dim]
        let input = Tensor::from_vec(embedding.to_vec(), (1, self.llm_dim), &self.device)?;

        // Matrix multiply: [1, llm_dim] x [llm_dim, hrr_dim]^T = [1, hrr_dim]
        let result = input.matmul(&self.weights.t()?)?;

        // L2 normalize
        let norm = result.sqr()?.sum_all()?.sqrt()?;
        let normalized = result.broadcast_div(&norm)?;

        normalized.flatten_all()?.to_vec1::<f32>()
    }
}
```

## Data Flow

### Forward Pass Data Flow

```
Gleam                    Erlang                   Rust                     CUDA
  |                        |                        |                        |
  | genome                 |                        |                        |
  +----------------------->|                        |                        |
  |                        | NIF call               |                        |
  |                        +----------------------->|                        |
  |                        |                        | weights[]              |
  |                        |                        +----------------------->|
  |                        |                        |                        |
  |                        |                        |      GPU forward       |
  |                        |                        |<-----------------------+
  |                        |                        |                        |
  |                        | outputs[]              |                        |
  |                        |<-----------------------+                        |
  |  outputs[]             |                        |                        |
  |<-----------------------+                        |                        |
  |                        |                        |                        |
```

### Memory Layout

```
BEAM Heap              Rust Stack/Heap           GPU Memory
+-------------+        +---------------+         +------------------+
| Genome      |        | Vec<f32>      |         | d_weights        |
| (Gleam)     |  --->  | weights       |  --->   | (cuBLAS format)  |
+-------------+        +---------------+         +------------------+
                              |
                       Zero-copy when possible
                       (resource_arc pattern)
```

## Performance Characteristics

### Latency Breakdown

| Operation | Time (us) | Notes |
|-----------|-----------|-------|
| Gleam -> Erlang FFI | 0.1 | Negligible |
| Erlang -> Rust NIF | 2.0 | List conversion |
| Rust -> CUDA H2D | 15.0 | For 8192 floats |
| cuFFT forward | 8.0 | 8192-point C2C |
| Complex multiply | 1.2 | Custom kernel |
| cuFFT inverse | 8.0 | 8192-point C2C |
| CUDA -> Rust D2H | 12.0 | For 8192 floats |
| Rust -> Erlang | 3.0 | Vec -> List |
| **Total HRR bind** | **~50** | Single operation |

### Throughput Optimization

1. **Batching**: Amortize transfer overhead across many operations
2. **Async Streams**: Overlap H2D, compute, and D2H
3. **Resource Pooling**: Reuse GPU handles across NIF calls
4. **Dirty Schedulers**: Prevent blocking BEAM schedulers

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| HRR buffer (per handle) | 3 x 8192 x 8 bytes = 192KB | Complex floats |
| Projection matrix | 4096 x 8192 x 4 bytes = 128MB | FP32 |
| cuFFT plan | ~1MB | Cached |
| CUDA context | ~300MB | Per-process |
| **Total per Glands** | **~430MB** | GPU VRAM |

## Error Handling

### Gleam Layer

```gleam
case glands.bind(handle, a, b) {
  Ok(result) -> process_result(result)
  Error(msg) -> {
    telemetry.log_error("glands_bind_failed", msg)
    fallback_cpu_bind(a, b)
  }
}
```

### Rust Layer

```rust
fn glands_bind(...) -> Result<Vec<f32>, String> {
    // Dimension validation
    if a.len() != resource.config.hrr_dim {
        return Err(format!("Dimension mismatch: expected {}, got {}",
            resource.config.hrr_dim, a.len()));
    }

    // GPU with CPU fallback
    if let Some(ref gpu) = resource.hrr_gpu {
        match gpu.lock() {
            Ok(g) => g.bind(&a, &b),
            Err(_) => {
                // GPU lock failed, use CPU
                resource.hrr_cpu.lock()?.bind(&a, &b)
            }
        }
    } else {
        resource.hrr_cpu.lock()?.bind(&a, &b)
    }
}
```

### OTP Supervision

```
Supervisor Strategy: one_for_one

Child crash -> Restart child only
Max restarts: 10 per 60 seconds
On max exceeded: Escalate to parent supervisor
```

## Configuration Reference

### GlandsConfig

```gleam
pub type GlandsConfig {
  GlandsConfig(
    llm_dim: Int,     // Input embedding dimension (4096)
    hrr_dim: Int,     // HRR vector dimension (8192)
    seed: Int,        // Projection matrix seed (42)
    gpu_layers: Int,  // GPU layers for LLM (99 = all)
  )
}
```

### NeatConfig

```gleam
pub type NeatConfig {
  NeatConfig(
    population_size: Int,           // 50-200
    num_inputs: Int,                // Domain-specific
    num_outputs: Int,               // Domain-specific
    weight_mutation_rate: Float,    // 0.8
    weight_perturb_rate: Float,     // 0.9
    add_node_rate: Float,           // 0.03
    add_connection_rate: Float,     // 0.05
    disable_rate: Float,            // 0.01
    compatibility_threshold: Float, // 1.0-3.0
    excess_coefficient: Float,      // 1.0
    disjoint_coefficient: Float,    // 1.0
    weight_coefficient: Float,      // 0.4
    survival_threshold: Float,      // 0.2
    elitism: Int,                   // 2
    max_stagnation: Int,            // 15
  )
}
```

### HoloMapConfig

```gleam
pub type HoloMapConfig {
  HoloMapConfig(
    grid_size: Int,               // 5-20
    behavior_dims: Int,           // 2
    hrr_dim: Int,                 // 4096-8192
    initial_novelty_weight: Float, // 0.7
    final_novelty_weight: Float,  // 0.2-0.3
    decay_midpoint: Int,          // 15-20
    batch_size: Int,              // 30-50
    tournament_size: Int,         // 3-4
  )
}

## VIVA Simulation Visualizer (legacy, removed)

This section documented a legacy Bevy-based visualizer (`bevy_simulator`) for a removed simulation client. The current repository no longer includes this client code.

### 1. WebSocket Message Protocol

Communication uses JSON-serialized messages. MessagePack is a planned optimization for reduced network overhead.

#### **Clientbound (Gleam -> Bevy)**

These messages are commands or state updates sent from the Gleam server to the Bevy client.

```json
// Example Clientbound Message Enum (conceptual)
[
  // Sets the entire state of the table. Used for initial setup and state correction.
  {
    "type": "SetTableState",
    "balls": [
      { "id": 0, "position": [0.0, 0.2, 0.5], "velocity": [0.0, 0.0, 0.0], "is_active": true },
      { "id": 1, "position": [-0.1, 0.2, -0.5], "velocity": [0.0, 0.0, 0.0], "is_active": true }
    ]
  },
  // Commands the client to allow the user to place the cue ball.
  // The provided region is a constraint for legal placement.
  {
    "type": "PlaceCueBall",
    "region": { "type": "kitchen", "extents": [2.0, 1.0] }
  },
  // Informs the client that the game has ended and who won.
  {
    "type": "GameEnd",
    "winner": "PlayerA"
  }
]
```

#### **Serverbound (Bevy -> Gleam)**

These messages are events or results sent from the Bevy client to the Gleam server.

```json
// Example Serverbound Message Enum (conceptual)
[
  // Sent by Bevy upon successful connection and initialization.
  {
    "type": "Ready"
  },
  // Sent when the player executes a shot.
  {
    "type": "ShotTaken",
    "cue_ball_id": 0,
    "force": 50.0, // A value representing shot power
    "angle_xy": 1.57, // Horizontal angle in radians
    "angle_z": 0.1, // Vertical angle for elevation
    "spin": {
      "top": 0.2, // -1.0 to 1.0
      "side": -0.5 // -1.0 to 1.0 (english)
    }
  },
  // Sent after a physics simulation has concluded (all balls stopped).
  {
    "type": "ShotResult",
    "balls": [
      { "id": 0, "position": [0.5, 0.2, 1.0], "is_active": false }, // Ball 0 was pocketed
      { "id": 1, "position": [-0.3, 0.2, -0.8], "is_active": true }
    ]
  },
  // Sent after the user places the cue ball during a ball-in-hand scenario.
  {
    "type": "CueBallPlaced",
    "position": [-0.4, 0.2, 0.6]
  }
]
```

### 2. Bevy Project Structure

The project follows a standard Bevy application layout.

**`Cargo.toml` Dependencies:**

```toml
[dependencies]
bevy = "0.13"
bevy_xpbd_3d = "0.4"
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = "0.23"
futures-util = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Folder Structure:**

```
bevy_simulator/
├── Cargo.toml
└── src/
    ├── main.rs         # App entry point, Bevy plugins
    ├── components.rs   # Custom components (e.g., Ball, Cue)
    ├── resources.rs    # Global resources (e.g., WebSocket connection)
    ├── systems/        # Game logic systems
    │   ├── setup.rs
    │   ├── network.rs
    │   ├── physics.rs
    │   └── rendering.rs
    └── plugins/        # Custom Bevy plugins for organization
```

### 3. Ball Rendering Details

For photorealistic pool balls on high-end hardware like the RTX 4090, a combination of procedural textures and advanced PBR materials is used.

**Procedural Ball Textures:**

A system runs at startup to generate the ball textures. A full font renderer is complex; a simpler method is to draw pre-rendered number sprites onto a colored circle background.

```rust
// In a Bevy system, e.g., setup_ball_materials
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

fn generate_ball_texture(
    color: Color,
    number: u8,
    number_sprite: &Image, // Pre-loaded sprite asset
    images: &mut ResMut<Assets<Image>>,
) -> Handle<Image> {
    let size = Extent3d { width: 512, height: 512, depth_or_array_layers: 1 };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &color.as_rgba_u8(),
        TextureFormat::Rgba8UnormSrgb,
        default(),
    );

    // This is a simplified concept. A real implementation would need
    // to correctly blend the number_sprite onto the background.
    // E.g., copy pixels from the sprite to the center of the `image` buffer.

    images.add(image)
}
```

**PBR Material Setup:**

The key to a realistic simulation sphere is the `StandardMaterial`'s clearcoat properties, which simulate a polished reflective coating over the base material.

```rust
// In a Bevy system, when spawning a ball entity
commands.spawn((
    PbrBundle {
        mesh: meshes.add(Sphere::new(0.057)), // Standard pool ball diameter
        material: materials.add(StandardMaterial {
            base_color_texture: Some(ball_texture_handle), // From procedural generation
            // Simulates the polished plastic/resin of the ball
            metallic: 0.1,
            perceptual_roughness: 0.08,
            // The crucial clear coat layer
            clearcoat: 1.0, // Full strength clearcoat
            clearcoat_perceptual_roughness: 0.01, // Very smooth coat
            ..default()
        }),
        ..default()
    },
    // ... other components like RigidBody, Collider, etc.
));
```

**Environment and Shadows:**

- **Reflections:** An HDRI environment map is loaded and set as the `EnvironmentMapLight` for realistic, image-based reflections on the balls.
- **Shadows:** High-quality shadows are enabled on the primary `DirectionalLight` to ground the objects.

### 4. Spin Physics in bevy_xpbd

`bevy_xpbd` allows for direct application of angular forces to simulate spin.

**Applying Spin:**

When a shot is taken, the `ShotTaken` parameters are translated into physical forces and torques applied to the cue ball's `RigidBody`.

```rust
use bevy::prelude::*;
use bevy_xpbd_3d::prelude::*;

fn apply_shot_physics(
    // Query for the cue ball entity
    mut query: Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform), With<CueBall>>,
    shot_params: Res<ShotParams>, // Resource holding shot data
) {
    for (mut lin_vel, mut ang_vel, transform) in query.iter_mut() {
        // 1. Apply linear force for the main shot power
        let forward_vector = transform.forward();
        *lin_vel = LinearVelocity(forward_vector * shot_params.power);

        // 2. Apply torque for spin (english, top/back spin)
        let transform = transform.compute_transform();
        let right_vector = transform.right();
        let up_vector = transform.up();

        // Top/back spin applies torque around the right-axis
        let topspin_torque = right_vector * shot_params.spin.top * 100.0;
        // Side spin applies torque around the up-axis
        let sidespin_torque = up_vector * shot_params.spin.side * 100.0;

        *ang_vel = AngularVelocity(topspin_torque + sidespin_torque);
    }
}
```

**Advanced Effects:**

- **Magnus Effect:** Not typically built-in. Can be implemented as a custom system applying an `ExternalForce` to the ball. The force is calculated as `F = k * cross(angular_velocity, linear_velocity)`, causing the ball to curve.
- **Friction and Damping:** `LinearDamping(0.5)` and `AngularDamping(0.8)` components are added to all balls to simulate the spin and speed decay from the table cloth. These values require careful tuning.

### 5. State Synchronization Edge Cases

A state machine in Bevy manages the game flow based on server messages.

- **Initial Table Setup:**
  1. Bevy connects to the WebSocket and sends `Serverbound::Ready`.
  2. Bevy enters a `WaitingForState` state.
  3. Gleam sends `Clientbound::SetTableState`.
  4. Bevy despawns any existing balls and spawns new ones based on the received positions, applying the `RigidBody` and `Collider` components.

- **Ball-in-Hand (Cue Ball Placement):**
  1. Gleam sends `Clientbound::PlaceCueBall`.
  2. Bevy enters a `PlacingCueBall` state. The cue ball's `RigidBody` is set to `Kinematic` and its `Collider` is temporarily disabled or moved to a non-colliding layer.
  3. A system updates the cue ball's `Transform` to follow the mouse cursor, constrained by the region sent by the server.
  4. On player click, Bevy sends `Serverbound::CueBallPlaced` with the final position.
  5. Bevy waits for a new `Clientbound::SetTableState` from Gleam to confirm the placement and restore the cue ball to a `Dynamic` `RigidBody`.

- **Foul Handling:**
  - Bevy's responsibility is limited to reporting the simulation outcome.
  - After a shot, Bevy sends `Serverbound::ShotResult`.
  - Gleam's logic determines if a foul occurred (e.g., cue ball pocketed, no rail hit).
  - If a foul occurs, Gleam's response will be a `SetTableState` (to respawn balls if necessary) followed by a `PlaceCueBall` message for the opponent.

- **Game End Conditions:**
  - Gleam tracks the game rules. When a win condition is met (e.g., 8-ball pocketed legally), it sends `Clientbound::GameEnd`.
  - Bevy receives this message, enters a `GameOver` state, and can display a UI element with the result.

### 6. Performance Metrics System

A debug overlay provides real-time performance feedback, crucial for optimization.

```rust
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

// Component to tag the FPS text UI element
#[derive(Component)]
struct FpsText;

// System to set up the UI
fn setup_performance_ui(mut commands: Commands) {
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "FPS: ",
                TextStyle { font_size: 20.0, color: Color::WHITE, ..default() },
            ),
            TextSection::from_style(TextStyle { font_size: 20.0, color: Color::GOLD, ..default() }),
            // Add more sections for other metrics
        ]).with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(5.0),
            left: Val::Px(5.0),
            ..default()
        }),
        FpsText,
    ));
}

// System to update the UI
fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                // Update the value of the second section
                text.sections[1].value = format!("{value:.2}");
            }
        }
        // Update other metrics like network latency from a resource
    }
}

// In the main App builder:
// app.add_plugins(FrameTimeDiagnosticsPlugin)
//    .add_systems(Startup, setup_performance_ui)
//    .add_systems(Update, update_fps_text);
```
- **Physics Steps:** Can be tracked by incrementing a resource counter within the fixed-update physics schedule.
- **Network Latency:** Can be measured by recording ping-pong message timings with the server.
- **Memory Usage:** More complex; requires platform-specific libraries or external tools.
```
