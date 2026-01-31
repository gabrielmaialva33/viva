//// NEAT Hybrid - Neuroevolution with Conv/Attention Modules
////
//// Extends NEAT to evolve hybrid architectures combining:
//// - Traditional dense connections (original NEAT)
//// - Convolutional modules (Conv2D)
//// - Attention modules (Multi-Head Attention)
////
//// Qwen3-235B Priority #3: Adapt NEAT for Hybrid Architectures
////
//// Philosophy: Evolution discovers optimal architecture mix.
//// Some problems need spatial processing (Conv), others need
//// global context (Attention), others just dense connections.
//// Let evolution decide.

import gleam/dict.{type Dict}
import gleam/float
import gleam/list
import gleam/option
import viva/neural/activation.{type ActivationType}
import viva/neural/math_ffi
import viva/neural/attention
import viva/neural/conv
import viva/neural/neat.{
  type Genome, type Population, Genome, Hidden, Output,
}
import viva_tensor/tensor.{type Tensor}

// =============================================================================
// TYPES - Hybrid Architecture Structures
// =============================================================================

/// Extended node types for hybrid architectures
pub type HybridNodeType {
  /// Standard NEAT node (dense)
  Dense
  /// Convolutional processing block
  ConvBlock(config: ConvConfig)
  /// Attention processing block
  AttentionBlock(config: AttentionConfig)
  /// Pooling block (max or average)
  PoolBlock(config: PoolConfig)
  /// Residual connection point
  ResidualPoint
}

/// Configuration for Conv2D module
pub type ConvConfig {
  ConvConfig(
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    activation: ActivationType,
  )
}

/// Configuration for Attention module
pub type AttentionConfig {
  AttentionConfig(d_model: Int, num_heads: Int, dropout_rate: Float)
}

/// Configuration for Pooling
pub type PoolConfig {
  PoolConfig(pool_type: PoolType, kernel_size: Int, stride: Int)
}

pub type PoolType {
  MaxPool
  AvgPool
}

/// Module gene - represents a specialized processing block
pub type ModuleGene {
  ModuleGene(
    id: Int,
    module_type: HybridNodeType,
    innovation: Int,
    enabled: Bool,
  )
}

/// Hybrid genome - extends NEAT genome with modules
pub type HybridGenome {
  HybridGenome(
    /// Base NEAT genome (nodes + connections)
    base: Genome,
    /// Specialized modules
    modules: List(ModuleGene),
    /// Module connections (which modules connect to which nodes)
    module_connections: List(ModuleConnection),
    /// Learnable parameters for modules
    module_params: Dict(Int, ModuleParams),
  )
}

/// Connection from/to a module
pub type ModuleConnection {
  ModuleConnection(
    /// Module ID
    module_id: Int,
    /// Connected node ID (from base genome)
    node_id: Int,
    /// Is this input to module or output from module
    direction: ConnectionDirection,
    /// Innovation number for crossover
    innovation: Int,
    enabled: Bool,
  )
}

pub type ConnectionDirection {
  ToModule
  FromModule
}

/// Learnable parameters for a module
pub type ModuleParams {
  ConvParams(filters: Tensor, biases: Tensor)
  AttentionParams(
    w_query: Tensor,
    w_key: Tensor,
    w_value: Tensor,
    w_out: Tensor,
  )
  NoParams
}

/// Hybrid population
pub type HybridPopulation {
  HybridPopulation(
    genomes: List(HybridGenome),
    generation: Int,
    module_innovation_counter: Int,
    connection_innovation_counter: Int,
    best_fitness: Float,
  )
}

/// Hybrid NEAT configuration
pub type HybridConfig {
  HybridConfig(
    /// Base NEAT config
    base_config: neat.NeatConfig,
    /// Probability of adding a conv module
    add_conv_rate: Float,
    /// Probability of adding an attention module
    add_attention_rate: Float,
    /// Probability of adding a pool module
    add_pool_rate: Float,
    /// Probability of mutating module parameters
    module_param_mutation_rate: Float,
    /// Max modules per genome
    max_modules: Int,
    /// Input is 2D (for conv)
    input_is_2d: Bool,
    /// Input dimensions if 2D
    input_height: Int,
    input_width: Int,
    input_channels: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Default hybrid config
pub fn default_config() -> HybridConfig {
  HybridConfig(
    base_config: neat.default_config(),
    add_conv_rate: 0.02,
    add_attention_rate: 0.02,
    add_pool_rate: 0.01,
    module_param_mutation_rate: 0.3,
    max_modules: 5,
    input_is_2d: False,
    input_height: 1,
    input_width: 1,
    input_channels: 1,
  )
}

/// Config for image processing (enables conv)
pub fn image_config(height: Int, width: Int, channels: Int) -> HybridConfig {
  HybridConfig(
    ..default_config(),
    input_is_2d: True,
    input_height: height,
    input_width: width,
    input_channels: channels,
    add_conv_rate: 0.08,
    add_pool_rate: 0.04,
    add_attention_rate: 0.02,
    max_modules: 8,
  )
}

/// Config for sequence processing (enables attention)
pub fn sequence_config(seq_len: Int, d_model: Int) -> HybridConfig {
  HybridConfig(
    ..default_config(),
    input_is_2d: False,
    input_height: seq_len,
    input_width: d_model,
    input_channels: 1,
    add_conv_rate: 0.02,
    add_attention_rate: 0.08,
    add_pool_rate: 0.01,
    max_modules: 6,
  )
}

/// Create hybrid genome from base NEAT genome
pub fn from_genome(genome: Genome) -> HybridGenome {
  HybridGenome(
    base: genome,
    modules: [],
    module_connections: [],
    module_params: dict.new(),
  )
}

/// Create hybrid population from NEAT population
pub fn from_population(pop: Population) -> HybridPopulation {
  let hybrid_genomes = list.map(pop.genomes, from_genome)
  let best =
    list.fold(hybrid_genomes, 0.0, fn(acc, g) { float.max(acc, g.base.fitness) })

  HybridPopulation(
    genomes: hybrid_genomes,
    generation: pop.generation,
    module_innovation_counter: 0,
    connection_innovation_counter: 0,
    best_fitness: best,
  )
}

// =============================================================================
// MUTATION - Add/Modify Modules
// =============================================================================

/// Mutate a hybrid genome
pub fn mutate(
  genome: HybridGenome,
  config: HybridConfig,
  rng_seed: Int,
) -> HybridGenome {
  let genome = case should_mutate(rng_seed, config.add_conv_rate) {
    True -> add_conv_module(genome, config, rng_seed)
    False -> genome
  }

  let genome = case should_mutate(rng_seed + 1, config.add_attention_rate) {
    True -> add_attention_module(genome, config, rng_seed + 1)
    False -> genome
  }

  let genome = case should_mutate(rng_seed + 2, config.add_pool_rate) {
    True -> add_pool_module(genome, config, rng_seed + 2)
    False -> genome
  }

  let genome = case
    should_mutate(rng_seed + 3, config.module_param_mutation_rate)
  {
    True -> mutate_module_params(genome, rng_seed + 3)
    False -> genome
  }

  genome
}

/// Add a convolutional module
fn add_conv_module(
  genome: HybridGenome,
  config: HybridConfig,
  rng_seed: Int,
) -> HybridGenome {
  case list.length(genome.modules) >= config.max_modules {
    True -> genome
    False -> {
      // Generate random conv config
      let kernel_size = 3 + pseudo_random(rng_seed) % 3
      // 3, 4, or 5
      let out_channels = 8 + pseudo_random(rng_seed + 1) % 24
      // 8-32

      let conv_config =
        ConvConfig(
          in_channels: config.input_channels,
          out_channels: out_channels,
          kernel_size: kernel_size,
          stride: 1,
          activation: activation.ReLU,
        )

      let module_id = list.length(genome.modules)
      let module =
        ModuleGene(
          id: module_id,
          module_type: ConvBlock(conv_config),
          innovation: module_id,
          enabled: True,
        )

      // Initialize conv parameters
      let conv_layer =
        conv.new(
          conv_config.in_channels,
          conv_config.out_channels,
          conv_config.kernel_size,
          conv_config.stride,
          conv.Same,
          conv_config.activation,
        )

      let params = ConvParams(conv_layer.filters, conv_layer.biases)

      // Connect module to input nodes
      let input_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: 0,
          // First input node
          direction: ToModule,
          innovation: list.length(genome.module_connections),
          enabled: True,
        )

      // Find a hidden or output node to connect output
      let output_node_id = find_random_target_node(genome.base, rng_seed + 2)
      let output_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: output_node_id,
          direction: FromModule,
          innovation: list.length(genome.module_connections) + 1,
          enabled: True,
        )

      HybridGenome(
        ..genome,
        modules: [module, ..genome.modules],
        module_connections: [
          input_connection,
          output_connection,
          ..genome.module_connections
        ],
        module_params: dict.insert(genome.module_params, module_id, params),
      )
    }
  }
}

/// Add an attention module
fn add_attention_module(
  genome: HybridGenome,
  config: HybridConfig,
  rng_seed: Int,
) -> HybridGenome {
  case list.length(genome.modules) >= config.max_modules {
    True -> genome
    False -> {
      // Generate random attention config
      let d_model = 16 * { 1 + pseudo_random(rng_seed) % 4 }
      // 16, 32, 48, or 64
      let num_heads = case d_model {
        16 -> 2
        32 -> 4
        48 -> 6
        _ -> 8
      }

      let attn_config =
        AttentionConfig(
          d_model: d_model,
          num_heads: num_heads,
          dropout_rate: 0.1,
        )

      let module_id = list.length(genome.modules)
      let module =
        ModuleGene(
          id: module_id,
          module_type: AttentionBlock(attn_config),
          innovation: module_id,
          enabled: True,
        )

      // Initialize attention parameters
      let mha = attention.mha_new(d_model, num_heads)
      let params =
        AttentionParams(
          w_query: mha.w_query,
          w_key: mha.w_key,
          w_value: mha.w_value,
          w_out: mha.w_out,
        )

      // Connect module
      let input_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: 0,
          direction: ToModule,
          innovation: list.length(genome.module_connections),
          enabled: True,
        )

      let output_node_id = find_random_target_node(genome.base, rng_seed + 2)
      let output_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: output_node_id,
          direction: FromModule,
          innovation: list.length(genome.module_connections) + 1,
          enabled: True,
        )

      HybridGenome(
        ..genome,
        modules: [module, ..genome.modules],
        module_connections: [
          input_connection,
          output_connection,
          ..genome.module_connections
        ],
        module_params: dict.insert(genome.module_params, module_id, params),
      )
    }
  }
}

/// Add a pooling module
fn add_pool_module(
  genome: HybridGenome,
  config: HybridConfig,
  rng_seed: Int,
) -> HybridGenome {
  case list.length(genome.modules) >= config.max_modules {
    True -> genome
    False -> {
      let pool_type = case pseudo_random(rng_seed) % 2 {
        0 -> MaxPool
        _ -> AvgPool
      }
      let kernel_size = 2 + pseudo_random(rng_seed + 1) % 2
      // 2 or 3

      let pool_config =
        PoolConfig(pool_type: pool_type, kernel_size: kernel_size, stride: 2)

      let module_id = list.length(genome.modules)
      let module =
        ModuleGene(
          id: module_id,
          module_type: PoolBlock(pool_config),
          innovation: module_id,
          enabled: True,
        )

      let input_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: 0,
          direction: ToModule,
          innovation: list.length(genome.module_connections),
          enabled: True,
        )

      let output_node_id = find_random_target_node(genome.base, rng_seed + 2)
      let output_connection =
        ModuleConnection(
          module_id: module_id,
          node_id: output_node_id,
          direction: FromModule,
          innovation: list.length(genome.module_connections) + 1,
          enabled: True,
        )

      HybridGenome(
        ..genome,
        modules: [module, ..genome.modules],
        module_connections: [
          input_connection,
          output_connection,
          ..genome.module_connections
        ],
        module_params: dict.insert(genome.module_params, module_id, NoParams),
      )
    }
  }
}

/// Mutate module parameters (weights)
fn mutate_module_params(genome: HybridGenome, rng_seed: Int) -> HybridGenome {
  let new_params =
    dict.fold(genome.module_params, dict.new(), fn(acc, id, params) {
      let mutated = case params {
        ConvParams(filters, biases) -> {
          let new_filters = perturb_tensor(filters, rng_seed + id)
          let new_biases = perturb_tensor(biases, rng_seed + id + 1)
          ConvParams(new_filters, new_biases)
        }
        AttentionParams(wq, wk, wv, wo) -> {
          let new_wq = perturb_tensor(wq, rng_seed + id)
          let new_wk = perturb_tensor(wk, rng_seed + id + 1)
          let new_wv = perturb_tensor(wv, rng_seed + id + 2)
          let new_wo = perturb_tensor(wo, rng_seed + id + 3)
          AttentionParams(new_wq, new_wk, new_wv, new_wo)
        }
        NoParams -> NoParams
      }
      dict.insert(acc, id, mutated)
    })

  HybridGenome(..genome, module_params: new_params)
}

// =============================================================================
// FORWARD PASS - Execute Hybrid Network
// =============================================================================

/// Forward pass through hybrid genome
/// Executes modules first, then standard NEAT connections
pub fn forward(
  genome: HybridGenome,
  input: Tensor,
  config: HybridConfig,
) -> List(Float) {
  // First, process through enabled modules
  let module_outputs = process_modules(genome, input, config)

  // Combine module outputs with input for NEAT forward
  let combined_input = combine_inputs(input, module_outputs, genome)

  // Run standard NEAT forward
  neat.forward(genome.base, combined_input)
}

/// Process all enabled modules
fn process_modules(
  genome: HybridGenome,
  input: Tensor,
  _config: HybridConfig,
) -> Dict(Int, Tensor) {
  genome.modules
  |> list.filter(fn(m) { m.enabled })
  |> list.fold(dict.new(), fn(acc, module) {
    let output = process_single_module(module, input, genome.module_params)
    dict.insert(acc, module.id, output)
  })
}

/// Process a single module
fn process_single_module(
  module: ModuleGene,
  input: Tensor,
  params: Dict(Int, ModuleParams),
) -> Tensor {
  case module.module_type {
    Dense -> input
    ConvBlock(config) -> {
      case dict.get(params, module.id) {
        Ok(ConvParams(filters, biases)) -> {
          // Create conv layer from stored params
          let layer =
            conv.from_weights(
              filters,
              biases,
              config.in_channels,
              config.out_channels,
              config.kernel_size,
              config.kernel_size,
              config.stride,
              config.stride,
              conv.Same,
              config.activation,
            )
          // Reshape input if needed for conv
          let conv_input = reshape_for_conv(input, config.in_channels)
          case conv.forward(layer, conv_input) {
            Ok(#(output, _cache)) -> tensor.flatten(output)
            Error(_) -> input
          }
        }
        _ -> input
      }
    }
    AttentionBlock(config) -> {
      case dict.get(params, module.id) {
        Ok(AttentionParams(wq, wk, wv, wo)) -> {
          // Create MHA from stored params
          let mha =
            attention.MultiHeadAttention(
              d_model: config.d_model,
              num_heads: config.num_heads,
              d_k: config.d_model / config.num_heads,
              w_query: wq,
              w_key: wk,
              w_value: wv,
              w_out: wo,
            )
          // Reshape input for attention
          let attn_input = reshape_for_attention(input, config.d_model)
          case
            attention.mha_forward(
              mha,
              attn_input,
              attn_input,
              attn_input,
              option.None,
            )
          {
            Ok(#(output, _cache)) -> tensor.flatten(output)
            Error(_) -> input
          }
        }
        _ -> input
      }
    }
    PoolBlock(config) -> {
      // Simple max/avg pooling
      apply_pooling(input, config)
    }
    ResidualPoint -> input
  }
}

/// Reshape 1D input for conv2d [batch, channels, height, width]
fn reshape_for_conv(input: Tensor, channels: Int) -> Tensor {
  let size = tensor.size(input)
  let spatial = size / channels
  let side = float_sqrt(int_to_float(spatial)) |> float_to_int()
  let actual_spatial = side * side

  case actual_spatial * channels == size {
    True -> {
      case tensor.reshape(input, [1, channels, side, side]) {
        Ok(reshaped) -> reshaped
        Error(_) ->
          tensor.Tensor(data: tensor.to_list(input), shape: [1, 1, 1, size])
      }
    }
    False -> {
      // Can't reshape to square, use 1D
      tensor.Tensor(data: tensor.to_list(input), shape: [1, 1, 1, size])
    }
  }
}

/// Reshape 1D input for attention [seq_len, d_model]
fn reshape_for_attention(input: Tensor, d_model: Int) -> Tensor {
  let size = tensor.size(input)
  let seq_len = size / d_model

  case seq_len * d_model == size && seq_len > 0 {
    True -> {
      case tensor.reshape(input, [seq_len, d_model]) {
        Ok(reshaped) -> reshaped
        Error(_) ->
          tensor.Tensor(data: tensor.to_list(input), shape: [1, size])
      }
    }
    False -> tensor.Tensor(data: tensor.to_list(input), shape: [1, size])
  }
}

/// Apply pooling operation
fn apply_pooling(input: Tensor, config: PoolConfig) -> Tensor {
  let data = tensor.to_list(input)
  let kernel = config.kernel_size
  let stride = config.stride

  // Simple 1D pooling
  let pooled =
    chunk_list(data, stride)
    |> list.map(fn(chunk) {
      let window = list.take(chunk, kernel)
      case config.pool_type {
        MaxPool -> list.fold(window, -1_000_000.0, float.max)
        AvgPool -> {
          let sum = list.fold(window, 0.0, fn(a, b) { a +. b })
          sum /. int_to_float(list.length(window))
        }
      }
    })

  tensor.from_list(pooled)
}

/// Combine original input with module outputs
fn combine_inputs(
  original: Tensor,
  module_outputs: Dict(Int, Tensor),
  genome: HybridGenome,
) -> List(Float) {
  // Start with original input
  let base_input = tensor.to_list(original)

  // Add module outputs that connect to input layer
  let module_contribution =
    genome.module_connections
    |> list.filter(fn(conn) { conn.direction == FromModule && conn.enabled })
    |> list.flat_map(fn(conn) {
      case dict.get(module_outputs, conn.module_id) {
        Ok(output) -> list.take(tensor.to_list(output), 4)
        // Limit contribution size
        Error(_) -> []
      }
    })

  list.append(base_input, module_contribution)
}

// =============================================================================
// CROSSOVER
// =============================================================================

/// Crossover two hybrid genomes
pub fn crossover(
  parent1: HybridGenome,
  parent2: HybridGenome,
  rng_seed: Int,
) -> HybridGenome {
  // Crossover base genomes
  let child_base = neat.crossover(parent1.base, parent2.base, rng_seed)

  // Inherit modules from fitter parent, with chance of other parent's modules
  let fitter = case parent1.base.fitness >=. parent2.base.fitness {
    True -> parent1
    False -> parent2
  }
  let other = case parent1.base.fitness >=. parent2.base.fitness {
    True -> parent2
    False -> parent1
  }

  // Inherit all modules from fitter parent
  let child_modules = fitter.modules

  // With 25% chance, add a module from other parent
  let child_modules = case
    should_mutate(rng_seed, 0.25) && !list.is_empty(other.modules)
  {
    True -> {
      let idx = pseudo_random(rng_seed) % list.length(other.modules)
      case list.drop(other.modules, idx) |> list.first() {
        Ok(module) -> [module, ..child_modules]
        Error(_) -> child_modules
      }
    }
    False -> child_modules
  }

  // Inherit module params from fitter parent
  let child_params = fitter.module_params
  let child_connections = fitter.module_connections

  HybridGenome(
    base: child_base,
    modules: list.take(child_modules, 10),
    // Limit modules
    module_connections: child_connections,
    module_params: child_params,
  )
}

// =============================================================================
// FITNESS EVALUATION
// =============================================================================

/// Evaluate fitness of hybrid genome
pub fn evaluate(
  genome: HybridGenome,
  fitness_fn: fn(HybridGenome) -> Float,
) -> HybridGenome {
  let fitness = fitness_fn(genome)
  HybridGenome(..genome, base: Genome(..genome.base, fitness: fitness))
}

/// Evaluate population
pub fn evaluate_population(
  pop: HybridPopulation,
  fitness_fn: fn(HybridGenome) -> Float,
) -> HybridPopulation {
  let evaluated = list.map(pop.genomes, fn(g) { evaluate(g, fitness_fn) })
  let best =
    list.fold(evaluated, pop.best_fitness, fn(acc, g) {
      float.max(acc, g.base.fitness)
    })

  HybridPopulation(..pop, genomes: evaluated, best_fitness: best)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn should_mutate(seed: Int, rate: Float) -> Bool {
  let random = pseudo_random_float(seed)
  random <. rate
}

fn pseudo_random(seed: Int) -> Int {
  // LCG parameters
  let a = 1_103_515_245
  let c = 12_345
  let m = 2_147_483_648
  { a * seed + c } % m
}

fn pseudo_random_float(seed: Int) -> Float {
  let r = pseudo_random(seed)
  int_to_float(r % 10_000) /. 10_000.0
}

fn find_random_target_node(genome: Genome, seed: Int) -> Int {
  let targets =
    genome.nodes
    |> list.filter(fn(n) { n.node_type == Hidden || n.node_type == Output })

  case targets {
    [] -> 0
    _ -> {
      let idx = pseudo_random(seed) % list.length(targets)
      case list.drop(targets, idx) |> list.first() {
        Ok(node) -> node.id
        Error(_) -> 0
      }
    }
  }
}

fn perturb_tensor(t: Tensor, seed: Int) -> Tensor {
  let perturbed =
    list.index_map(tensor.to_list(t), fn(x, i) {
      let perturbation = { pseudo_random_float(seed + i) -. 0.5 } *. 0.2
      x +. perturbation
    })
  tensor.Tensor(data: perturbed, shape: t.shape)
}

fn chunk_list(lst: List(a), size: Int) -> List(List(a)) {
  case lst {
    [] -> []
    _ -> {
      let chunk = list.take(lst, size)
      let rest = list.drop(lst, size)
      [chunk, ..chunk_list(rest, size)]
    }
  }
}

fn int_to_float(n: Int) -> Float {
  case n >= 0 {
    True -> positive_int_to_float(n, 0.0)
    False -> 0.0 -. positive_int_to_float(0 - n, 0.0)
  }
}

fn positive_int_to_float(n: Int, acc: Float) -> Float {
  case n {
    0 -> acc
    _ -> positive_int_to_float(n - 1, acc +. 1.0)
  }
}

// Use centralized FFI (O(1) vs O(n) Newton-Raphson)
fn float_sqrt(x: Float) -> Float {
  math_ffi.safe_sqrt(x)
}

fn float_to_int(f: Float) -> Int {
  // Simple truncation
  case f <. 0.0 {
    True -> 0 - float_to_int_positive(0.0 -. f)
    False -> float_to_int_positive(f)
  }
}

fn float_to_int_positive(f: Float) -> Int {
  float_to_int_loop(f, 0)
}

fn float_to_int_loop(f: Float, acc: Int) -> Int {
  case f <. 1.0 {
    True -> acc
    False -> float_to_int_loop(f -. 1.0, acc + 1)
  }
}
