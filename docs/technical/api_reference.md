# VIVA-QD API Reference

## Module Index

| Module | Description |
|--------|-------------|
| `viva/neural/neat` | NEAT neuroevolution algorithm |
| `viva/neural/holomap` | MAP-Elites with HRR encoding |
| `viva/neural/novelty` | Behavior descriptors and novelty |
| `viva/neural/neat_hybrid` | NEAT with Conv/Attention modules |
| `viva/glands` | GPU-accelerated HRR operations |
| `viva/lifecycle/burn_physics` | Legacy physics batch simulation helpers |
| `viva/jolt` | Jolt Physics engine bindings |

---

## viva/neural/neat

### Types

#### `NodeType`
```gleam
pub type NodeType {
  Input
  Hidden
  Output
  Bias
}
```

#### `ActivationType`
```gleam
pub type ActivationType {
  Sigmoid
  Tanh
  ReLU
  Linear
}
```

#### `NodeGene`
```gleam
pub type NodeGene {
  NodeGene(id: Int, node_type: NodeType, activation: ActivationType)
}
```

#### `ConnectionGene`
```gleam
pub type ConnectionGene {
  ConnectionGene(
    in_node: Int,
    out_node: Int,
    weight: Float,
    enabled: Bool,
    innovation: Int,
  )
}
```

#### `Genome`
```gleam
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
```

#### `Species`
```gleam
pub type Species {
  Species(
    id: Int,
    members: List(Genome),
    representative: Genome,
    best_fitness: Float,
    stagnation: Int,
  )
}
```

#### `Population`
```gleam
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

#### `NeatConfig`
```gleam
pub type NeatConfig {
  NeatConfig(
    population_size: Int,
    num_inputs: Int,
    num_outputs: Int,
    weight_mutation_rate: Float,
    weight_perturb_rate: Float,
    add_node_rate: Float,
    add_connection_rate: Float,
    disable_rate: Float,
    compatibility_threshold: Float,
    excess_coefficient: Float,
    disjoint_coefficient: Float,
    weight_coefficient: Float,
    survival_threshold: Float,
    elitism: Int,
    max_stagnation: Int,
  )
}
```

#### `FitnessResult`
```gleam
pub type FitnessResult {
  FitnessResult(genome_id: Int, fitness: Float)
}
```

#### `PopulationStats`
```gleam
pub type PopulationStats {
  PopulationStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    num_species: Int,
    avg_nodes: Float,
    avg_connections: Float,
  )
}
```

### Functions

#### `default_config() -> NeatConfig`
Returns default NEAT configuration.

#### `xor_config() -> NeatConfig`
Returns configuration optimized for XOR problem.

#### `viva_soul_config() -> NeatConfig`
Returns configuration for VIVA soul networks (PAD -> behavior).

#### `create_population(config: NeatConfig, seed: Int) -> Population`
Creates initial population with minimal networks.

**Parameters:**
- `config`: NEAT configuration
- `seed`: Random seed for reproducibility

**Returns:** Population with `config.population_size` genomes.

#### `forward(genome: Genome, inputs: List(Float)) -> List(Float)`
Executes forward pass through the genome's neural network.

**Parameters:**
- `genome`: Network to evaluate
- `inputs`: Input values (length must match `num_inputs`)

**Returns:** Output activations (length equals `num_outputs`)

#### `mutate_weights(genome: Genome, config: NeatConfig, seed: Int) -> Genome`
Mutates connection weights.

**Parameters:**
- `genome`: Genome to mutate
- `config`: Mutation rates from config
- `seed`: Random seed

**Returns:** New genome with mutated weights.

#### `mutate_add_node(genome: Genome, population: Population, seed: Int) -> #(Genome, Population)`
Adds a new hidden node by splitting an existing connection.

**Parameters:**
- `genome`: Genome to mutate
- `population`: Population (for innovation tracking)
- `seed`: Random seed

**Returns:** Tuple of (mutated genome, updated population).

#### `mutate_add_connection(genome: Genome, population: Population, seed: Int) -> #(Genome, Population)`
Adds a new connection between unconnected nodes.

**Parameters:**
- `genome`: Genome to mutate
- `population`: Population (for innovation tracking)
- `seed`: Random seed

**Returns:** Tuple of (mutated genome, updated population).

#### `crossover(parent1: Genome, parent2: Genome, seed: Int) -> Genome`
Creates offspring from two parents using NEAT crossover.

**Note:** `parent1` should have higher or equal fitness.

**Parameters:**
- `parent1`: Fitter parent
- `parent2`: Less fit parent
- `seed`: Random seed

**Returns:** Child genome.

#### `compatibility_distance(genome1: Genome, genome2: Genome, config: NeatConfig) -> Float`
Calculates genetic distance between two genomes.

**Parameters:**
- `genome1`, `genome2`: Genomes to compare
- `config`: Coefficients for excess, disjoint, weight terms

**Returns:** Distance value (lower = more similar).

#### `speciate(population: Population, config: NeatConfig) -> Population`
Groups genomes into species based on compatibility.

**Parameters:**
- `population`: Population to speciate
- `config`: Compatibility threshold

**Returns:** Population with updated species assignments.

#### `evolve(population: Population, fitness_results: List(FitnessResult), config: NeatConfig, seed: Int) -> Population`
Evolves population to next generation.

**Parameters:**
- `population`: Current population
- `fitness_results`: Fitness for each genome
- `config`: NEAT configuration
- `seed`: Random seed

**Returns:** Next generation population.

#### `get_best(population: Population) -> Option(Genome)`
Returns the highest-fitness genome.

#### `get_stats(population: Population) -> PopulationStats`
Returns population statistics.

#### `genome_to_string(genome: Genome) -> String`
Debug string representation of genome.

---

## viva/neural/holomap

### Types

#### `Elite`
```gleam
pub type Elite {
  Elite(
    genome_id: Int,
    behavior: Behavior,
    hrr_vector: List(Float),
    fitness: Float,
    generation_added: Int,
  )
}
```

#### `MapElitesGrid`
```gleam
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

#### `HoloMapConfig`
```gleam
pub type HoloMapConfig {
  HoloMapConfig(
    grid_size: Int,
    behavior_dims: Int,
    hrr_dim: Int,
    initial_novelty_weight: Float,
    final_novelty_weight: Float,
    decay_midpoint: Int,
    batch_size: Int,
    tournament_size: Int,
  )
}
```

#### `HoloMapStats`
```gleam
pub type HoloMapStats {
  HoloMapStats(
    generation: Int,
    best_fitness: Float,
    coverage: Float,
    qd_score: Float,
    novelty_weight: Float,
  )
}
```

### Functions

#### `default_config() -> HoloMapConfig`
Returns default HoloMAP configuration (20x20 grid).

#### `fast_config() -> HoloMapConfig`
Returns fast configuration (10x10 grid).

#### `qwen3_optimized_config() -> HoloMapConfig`
Returns Qwen3-recommended configuration (5x5 grid with gradual expansion).

#### `new_grid(config: HoloMapConfig) -> MapElitesGrid`
Creates empty MAP-Elites archive.

#### `behavior_to_cell(behavior: Behavior, grid: MapElitesGrid) -> #(Int, Int)`
Maps behavior descriptor to grid cell coordinates.

#### `try_add_elite(grid: MapElitesGrid, genome_id: Int, behavior: Behavior, hrr_vector: List(Float), fitness: Float, generation: Int) -> #(MapElitesGrid, Bool)`
Attempts to add elite to grid (cell-wise elitism).

**Returns:** Tuple of (updated grid, whether added).

#### `get_elites(grid: MapElitesGrid) -> List(Elite)`
Returns all elites in the archive.

#### `coverage(grid: MapElitesGrid) -> Float`
Returns percentage of cells occupied (0-100).

#### `qd_score(grid: MapElitesGrid) -> Float`
Returns sum of all elite fitnesses.

#### `adaptive_novelty_weight(generation: Int, config: HoloMapConfig) -> Float`
Calculates novelty weight with sigmoid decay.

#### `tournament_select(grid: MapElitesGrid, tournament_size: Int, seed: Int) -> Option(Elite)`
Selects elite via tournament selection.

#### `hrr_to_behavior(hrr_vector: List(Float), dims: Int) -> Behavior`
Extracts behavior descriptor from HRR vector.

#### `hrr_to_2d_behavior(hrr_vector: List(Float)) -> Behavior`
Extracts 2D behavior (first 2 HRR dimensions).

#### `normalize_hrr(vector: List(Float)) -> List(Float)`
L2-normalizes HRR vector.

#### `normalize_grid_hrr(grid: MapElitesGrid) -> MapElitesGrid`
Normalizes all HRR vectors in grid (call every 10 generations).

#### `holographic_crossover(parent1: Elite, parent2: Elite, blend_ratio: Float) -> List(Float)`
Blends two HRR vectors with given ratio.

#### `compute_stats(grid: MapElitesGrid, generation: Int, config: HoloMapConfig) -> HoloMapStats`
Computes archive statistics.

---

## viva/glands

### Types

#### `GlandsConfig`
```gleam
pub type GlandsConfig {
  GlandsConfig(
    llm_dim: Int,
    hrr_dim: Int,
    seed: Int,
    gpu_layers: Int,
  )
}
```

#### `GlandsHandle`
```gleam
pub opaque type GlandsHandle {
  GlandsHandle(resource: Dynamic)
}
```

#### `DistillationResult`
```gleam
pub type DistillationResult {
  DistillationResult(
    text: String,
    embedding: List(Float),
    hrr_vector: List(Float),
    dimensions: Int,
  )
}
```

### Functions

#### `default_config() -> GlandsConfig`
Returns default configuration (4096 LLM dim, 8192 HRR dim).

#### `qwen_config() -> GlandsConfig`
Returns configuration for Qwen 2.5 14B (5120 dim).

#### `small_config() -> GlandsConfig`
Returns configuration for small models (2048 dim).

#### `neat_config() -> GlandsConfig`
Returns configuration for NEAT networks (64 dim).

#### `init(config: GlandsConfig) -> Result(GlandsHandle, String)`
Initializes Glands system (creates GPU context).

**Returns:** Handle for subsequent operations or error message.

#### `project(handle: GlandsHandle, embedding: List(Float)) -> Result(List(Float), String)`
Projects embedding into HRR space.

**Parameters:**
- `handle`: Glands handle
- `embedding`: Input vector (length must match `llm_dim`)

**Returns:** HRR vector (length equals `hrr_dim`).

#### `bind(handle: GlandsHandle, a: List(Float), b: List(Float)) -> Result(List(Float), String)`
Binds two HRR vectors via circular convolution (GPU-accelerated).

**Parameters:**
- `handle`: Glands handle
- `a`, `b`: HRR vectors (must have length `hrr_dim`)

**Returns:** Bound vector.

#### `unbind(handle: GlandsHandle, trace: List(Float), key: List(Float)) -> Result(List(Float), String)`
Unbinds (retrieves) from HRR trace via circular correlation.

**Parameters:**
- `handle`: Glands handle
- `trace`: Holographic trace
- `key`: Key to unbind with

**Returns:** Retrieved vector (approximate).

#### `similarity(a: List(Float), b: List(Float)) -> Result(Float, String)`
Computes cosine similarity (SIMD-accelerated).

**Returns:** Value in [-1, 1].

#### `batch_similarity(vectors: List(List(Float)), query: List(Float)) -> Result(List(Float), String)`
Computes similarities of multiple vectors against query (parallel).

#### `superpose(vectors: List(List(Float))) -> Result(List(Float), String)`
Superimposes (adds and normalizes) multiple vectors.

#### `check() -> String`
Returns system status string.

**Example:** `"GLANDS_ULTRA_OK (SIMD: AVX2, GPU: CUDA, Threads: 16)"`

#### `benchmark(handle: GlandsHandle, iterations: Int) -> String`
Runs benchmark and returns timing report.

---

## Legacy domain modules removed

The domain-specific training modules from a previous iteration were removed from the active codebase.

---

## viva/jolt

### Types

#### `Vec3`
```gleam
pub type Vec3 {
  Vec3(x: Float, y: Float, z: Float)
}
```

#### `PhysicsWorld`
Opaque type for Jolt physics world.

### Functions

#### `create_world() -> PhysicsWorld`
Creates new physics world with gravity.

#### `step(world: PhysicsWorld, dt: Float) -> PhysicsWorld`
Steps physics simulation by `dt` seconds.

#### `add_sphere(world: PhysicsWorld, position: Vec3, radius: Float, mass: Float) -> #(PhysicsWorld, Int)`
Adds sphere body, returns (world, body_id).

#### `apply_impulse(world: PhysicsWorld, body_id: Int, impulse: Vec3) -> PhysicsWorld`
Applies impulse to body.

#### `get_position(world: PhysicsWorld, body_id: Int) -> Option(Vec3)`
Gets body position.

#### `get_velocity(world: PhysicsWorld, body_id: Int) -> Option(Vec3)`
Gets body velocity.

---

## Error Handling Patterns

### Result Types

Most functions return `Result(T, String)` for error handling:

```gleam
case glands.init(glands.default_config()) {
  Ok(handle) -> {
    // Use handle
    case glands.bind(handle, vec_a, vec_b) {
      Ok(result) -> result
      Error(msg) -> {
        io.println("Bind failed: " <> msg)
        fallback_bind(vec_a, vec_b)
      }
    }
  }
  Error(msg) -> {
    io.println("Init failed: " <> msg)
    panic as "GPU initialization required"
  }
}
```

### Option Types

Position lookups return `Option`:

```gleam
case option.Some(pos) {
    io.println("Red ball at: " <> vec3_to_string(pos))
  }
  option.None -> {
    io.println("Red ball pocketed")
  }
}
```
