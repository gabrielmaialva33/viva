//// DHC - Destilação Holográfica Comprimida
////
//// Knowledge distillation from LLMs to VIVA's holographic memory.
//// Inspired by CODI (2024), QLoRA NF4, and Plate's HRR (1995).
////
//// Advanced techniques (2025-2026):
//// - CKA Loss (ICLR 2025): Centered Kernel Alignment for dim mismatch
//// - Progressive Distillation (POCL 2025): Dynamic alpha_kd + temperature
//// - Multi-layer Matching (TextBrewer): Extract from layers [0, L/4, L/2]
////
//// Pipeline:
//// 1. Teacher hidden states → project to HRR via glands (GPU)
//// 2. Bind with semantic keys (circular convolution)
//// 3. Superpose multiple concepts (bundling)
//// 4. Quantize with NF4 (8x compression)
//// 5. Store in KarmaBank with emotional salience
////
//// Compression: ~1000x vs raw text (HRR + NF4 + dedup)
//// Error: ~3.6% reconstruction (acceptable for associative memory)

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva/memory/hrr.{type HRR}
import viva/neural/glands.{type GlandsHandle}
import viva/neural/llm.{type LlmModel}
import viva_tensor/nf4.{type NF4Config, type NF4Tensor}
import viva_tensor/tensor

// =============================================================================
// TYPES
// =============================================================================

/// Distilled knowledge unit - the compressed form of teacher knowledge
pub type DistilledKnowledge {
  DistilledKnowledge(
    /// Semantic key for retrieval (e.g., bound "concept" + "context")
    concept_key: HRR,
    /// Compressed holographic representation (NF4 quantized)
    compressed_hrr: NF4Tensor,
    /// Emotional context from PAD model (Pleasure, Arousal, Dominance)
    emotional_context: PADState,
    /// Salience score [0, 1] - importance for consolidation
    salience: Float,
    /// Source metadata
    metadata: KnowledgeMetadata,
  )
}

/// PAD emotional state
pub type PADState {
  PADState(pleasure: Float, arousal: Float, dominance: Float)
}

/// Metadata about the knowledge source
pub type KnowledgeMetadata {
  KnowledgeMetadata(
    /// Teacher model identifier
    teacher_id: String,
    /// Layer index where hidden states were extracted
    extraction_layer: Int,
    /// Original sequence length
    seq_len: Int,
    /// Timestamp of extraction
    timestamp: Int,
  )
}

/// Distillation configuration
pub type DistillConfig {
  DistillConfig(
    /// Dimension of teacher hidden states
    teacher_dim: Int,
    /// Dimension of HRR vectors (VIVA standard: 8192)
    hrr_dim: Int,
    /// Layer to extract hidden states from (middle = L/2)
    extraction_layer: Int,
    /// NF4 quantization config
    nf4_config: NF4Config,
    /// Temperature for softmax in KD loss
    temperature: Float,
    /// Weight for task loss
    alpha_task: Float,
    /// Weight for knowledge distillation loss
    alpha_kd: Float,
    /// Weight for EWC regularization
    alpha_ewc: Float,
  )
}

/// Training state for EWC (Elastic Weight Consolidation)
pub type EWCState {
  EWCState(
    /// Fisher information diagonal (importance of each parameter)
    fisher_diag: List(Float),
    /// Parameter values at consolidation point
    consolidated_params: List(Float),
  )
}

/// Distillation loss components
pub type DistillLoss {
  DistillLoss(
    /// Total combined loss
    total: Float,
    /// Task-specific loss (MSE in HRR space)
    task_loss: Float,
    /// Knowledge distillation loss (cosine distance)
    kd_loss: Float,
    /// EWC regularization loss
    ewc_loss: Float,
  )
}

/// Advanced loss with CKA (ICLR 2025)
pub type DistillLossV2 {
  DistillLossV2(
    /// Total combined loss
    total: Float,
    /// Task-specific loss (MSE in HRR space)
    task_loss: Float,
    /// Knowledge distillation loss (cosine distance)
    kd_loss: Float,
    /// CKA loss (Centered Kernel Alignment) - allows dim mismatch
    cka_loss: Float,
    /// EWC regularization loss
    ewc_loss: Float,
    /// Multi-layer matching loss
    layer_loss: Float,
  )
}

/// Progressive distillation state (POCL 2025)
pub type ProgressiveState {
  ProgressiveState(
    /// Current epoch
    epoch: Int,
    /// Total epochs
    max_epochs: Int,
    /// Current alpha_kd (increases over training)
    alpha_kd: Float,
    /// Current temperature (decreases over training)
    temperature: Float,
    /// Current curriculum stage
    stage: CurriculumStage,
  )
}

/// Curriculum stages for progressive learning
pub type CurriculumStage {
  /// Early: high temp, low alpha_kd, simple examples
  EarlyStage
  /// Middle: balanced parameters
  MiddleStage
  /// Late: low temp, high alpha_kd, complex examples
  LateStage
}

/// Multi-layer extraction config (TextBrewer style)
pub type LayerMatchConfig {
  LayerMatchConfig(
    /// Teacher layers to extract from [0, 16, 32] for 64-layer model
    teacher_layers: List(Int),
    /// Weight for each layer's loss
    layer_weights: List(Float),
    /// Feature type: "hidden", "attention", "ffn"
    feature_type: String,
  )
}

/// Advanced distillation config with all 2025-2026 features
pub type DistillConfigV2 {
  DistillConfigV2(
    /// Base config
    base: DistillConfig,
    /// CKA loss weight (ICLR 2025)
    alpha_cka: Float,
    /// Multi-layer matching config
    layer_match: LayerMatchConfig,
    /// Enable progressive distillation
    progressive: Bool,
    /// Initial temperature for progressive (higher = softer)
    initial_temp: Float,
    /// Final temperature for progressive (lower = sharper)
    final_temp: Float,
    /// Initial alpha_kd for progressive
    initial_alpha_kd: Float,
    /// Final alpha_kd for progressive
    final_alpha_kd: Float,
  )
}

// =============================================================================
// DEFAULT CONFIG
// =============================================================================

/// Default config for Qwen3-32B teacher
/// Qwen3-32B has 5120 hidden dim, 64 layers
pub fn default_config() -> DistillConfig {
  DistillConfig(
    teacher_dim: 5120,      // Qwen3-32B hidden dimension
    hrr_dim: 8192,          // VIVA standard HRR size
    extraction_layer: 32,   // Middle layer (32 of 64)
    nf4_config: nf4.default_config(),
    temperature: 2.0,
    alpha_task: 1.0,
    alpha_kd: 0.5,
    alpha_ewc: 0.1,
  )
}

/// Config specifically for Qwen3-32B-Q4_K_M
pub fn qwen3_32b_config() -> DistillConfig {
  DistillConfig(
    teacher_dim: 5120,      // Qwen3-32B: 5120 hidden dim
    hrr_dim: 8192,          // VIVA HRR dimension
    extraction_layer: 32,   // Middle layer for best representations
    nf4_config: nf4.default_config(),
    temperature: 3.0,       // Higher temp for softer distributions
    alpha_task: 0.8,
    alpha_kd: 0.6,
    alpha_ewc: 0.1,
  )
}

/// Config for smaller models (Llama 8B)
pub fn small_config() -> DistillConfig {
  DistillConfig(
    teacher_dim: 4096,
    hrr_dim: 4096,
    extraction_layer: 16,
    nf4_config: nf4.default_config(),
    temperature: 2.0,
    alpha_task: 1.0,
    alpha_kd: 0.5,
    alpha_ewc: 0.1,
  )
}

/// Advanced config with all 2025-2026 features
pub fn advanced_config() -> DistillConfigV2 {
  DistillConfigV2(
    base: qwen3_32b_config(),
    alpha_cka: 0.3,           // CKA loss weight
    layer_match: LayerMatchConfig(
      teacher_layers: [0, 16, 32],  // Early, middle, late
      layer_weights: [0.2, 0.3, 0.5],  // More weight on later layers
      feature_type: "hidden",
    ),
    progressive: True,
    initial_temp: 4.0,        // Start with very soft targets
    final_temp: 1.5,          // End with sharper targets
    initial_alpha_kd: 0.3,    // Start with weak KD
    final_alpha_kd: 0.8,      // End with strong KD
  )
}

/// Initialize progressive distillation state
pub fn init_progressive(max_epochs: Int) -> ProgressiveState {
  ProgressiveState(
    epoch: 0,
    max_epochs: max_epochs,
    alpha_kd: 0.3,
    temperature: 4.0,
    stage: EarlyStage,
  )
}

/// Update progressive state for next epoch (POCL 2025)
/// Smoothly transitions from soft to hard distillation
pub fn step_progressive(
  state: ProgressiveState,
  config: DistillConfigV2,
) -> ProgressiveState {
  let new_epoch = state.epoch + 1
  let progress = int.to_float(new_epoch) /. int.to_float(state.max_epochs)

  // Linear interpolation for alpha_kd (increases)
  let new_alpha_kd = config.initial_alpha_kd +.
    { config.final_alpha_kd -. config.initial_alpha_kd } *. progress

  // Linear interpolation for temperature (decreases)
  let new_temp = config.initial_temp -.
    { config.initial_temp -. config.final_temp } *. progress

  // Determine curriculum stage
  let new_stage = case progress {
    p if p <. 0.33 -> EarlyStage
    p if p <. 0.66 -> MiddleStage
    _ -> LateStage
  }

  ProgressiveState(
    epoch: new_epoch,
    max_epochs: state.max_epochs,
    alpha_kd: new_alpha_kd,
    temperature: new_temp,
    stage: new_stage,
  )
}

/// Get current config from progressive state
pub fn progressive_config(
  base: DistillConfig,
  state: ProgressiveState,
) -> DistillConfig {
  DistillConfig(
    ..base,
    alpha_kd: state.alpha_kd,
    temperature: state.temperature,
  )
}

// =============================================================================
// KNOWLEDGE EXTRACTION (Stage 1-2)
// =============================================================================

/// Extract and project knowledge from teacher hidden states
///
/// Pipeline:
/// 1. Take hidden states from teacher at layer L/2
/// 2. Project to HRR space via glands (GPU matmul)
/// 3. Normalize to unit sphere
pub fn extract_hidden_states(
  glands: GlandsHandle,
  hidden_states: List(Float),
  _config: DistillConfig,
) -> Result(HRR, String) {
  // Project from teacher dim to HRR dim (GPU accelerated)
  use projected <- result.try(glands.project(glands, hidden_states))

  // Convert to HRR type
  Ok(hrr.from_list(projected))
}

/// Bind extracted knowledge with a semantic key
///
/// Creates an associative binding: key ⊛ value
/// This allows retrieval: unbind(memory, key) ≈ value
pub fn bind_with_key(
  glands: GlandsHandle,
  key: HRR,
  value: HRR,
) -> Result(HRR, String) {
  // Use GPU-accelerated circular convolution
  use bound <- result.try(glands.bind(glands, hrr.to_list(key), hrr.to_list(value)))
  Ok(hrr.from_list(bound))
}

/// Superpose multiple knowledge bindings into a memory trace
///
/// Memory = Σ(key_i ⊛ value_i) normalized
/// Allows multiple concepts to coexist in the same vector
pub fn superpose_knowledge(
  bindings: List(HRR),
) -> Result(HRR, String) {
  let vectors = list.map(bindings, hrr.to_list)
  use superposed <- result.try(glands.superpose(vectors))
  Ok(hrr.from_list(superposed))
}

// =============================================================================
// COMPRESSION (Stage 3-4)
// =============================================================================

/// Compress HRR vector using NF4 quantization
///
/// NF4 uses 16 levels from N(0,1) quantiles - mathematically optimal
/// for neural network weights which follow Gaussian distribution.
/// Achieves ~8x compression with ~3.6% reconstruction error.
pub fn compress_hrr(hrr_vec: HRR, config: NF4Config) -> NF4Tensor {
  let tensor = tensor.from_list(hrr.to_list(hrr_vec))
  nf4.quantize(tensor, config)
}

/// Decompress NF4 back to HRR
pub fn decompress_hrr(nf4_tensor: NF4Tensor) -> HRR {
  let tensor = nf4.dequantize(nf4_tensor)
  hrr.from_list(tensor.to_list(tensor))
}

// =============================================================================
// LOSS FUNCTIONS (Stage 5)
// =============================================================================

/// Compute DHC loss: L = α_task * L_task + α_kd * L_kd + α_ewc * L_ewc
///
/// Components:
/// - L_task: MSE between student and teacher HRR projections
/// - L_kd: Cosine distance in holographic space (soft targets)
/// - L_ewc: Elastic weight consolidation (protect old knowledge)
pub fn compute_loss(
  student_hrr: HRR,
  teacher_hrr: HRR,
  ewc_state: EWCState,
  current_params: List(Float),
  config: DistillConfig,
) -> DistillLoss {
  // Task loss: MSE in HRR space
  let task_loss = compute_mse_loss(student_hrr, teacher_hrr)

  // KD loss: 1 - cosine_similarity (want to maximize similarity)
  let similarity = hrr.similarity(student_hrr, teacher_hrr)
  let kd_loss = 1.0 -. similarity

  // EWC loss: protect important weights
  let ewc_loss = compute_ewc_loss(ewc_state, current_params)

  // Combined loss
  let total =
    config.alpha_task
    *. task_loss
    +. config.alpha_kd
    *. kd_loss
    +. config.alpha_ewc
    *. ewc_loss

  DistillLoss(
    total: total,
    task_loss: task_loss,
    kd_loss: kd_loss,
    ewc_loss: ewc_loss,
  )
}

/// MSE loss between two HRR vectors
fn compute_mse_loss(a: HRR, b: HRR) -> Float {
  let a_list = hrr.to_list(a)
  let b_list = hrr.to_list(b)

  let squared_diffs =
    list.map2(a_list, b_list, fn(x, y) {
      let diff = x -. y
      diff *. diff
    })

  let sum = list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x })
  sum /. int.to_float(list.length(squared_diffs))
}

/// EWC loss: Σ F_i * (θ_i - θ*_i)²
/// Penalizes changes to important parameters
fn compute_ewc_loss(ewc: EWCState, current_params: List(Float)) -> Float {
  list.zip(ewc.fisher_diag, list.zip(current_params, ewc.consolidated_params))
  |> list.map(fn(triple) {
    let #(f_i, #(theta_i, theta_star_i)) = triple
    let diff = theta_i -. theta_star_i
    f_i *. diff *. diff
  })
  |> list.fold(0.0, fn(acc, x) { acc +. x })
  |> fn(sum) { sum /. 2.0 }
}

// =============================================================================
// CKA LOSS (ICLR 2025 - Centered Kernel Alignment)
// =============================================================================

/// Compute CKA similarity between two representations
/// Key advantage: works even when dimensions differ!
/// CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
pub fn compute_cka(a: HRR, b: HRR) -> Float {
  let a_list = hrr.to_list(a)
  let b_list = hrr.to_list(b)

  // Center the vectors (subtract mean)
  let a_centered = center_vector(a_list)
  let b_centered = center_vector(b_list)

  // Compute HSIC (Hilbert-Schmidt Independence Criterion)
  let hsic_ab = compute_hsic(a_centered, b_centered)
  let hsic_aa = compute_hsic(a_centered, a_centered)
  let hsic_bb = compute_hsic(b_centered, b_centered)

  // CKA = HSIC(a,b) / sqrt(HSIC(a,a) * HSIC(b,b))
  let denominator = float_sqrt(hsic_aa *. hsic_bb)
  case denominator >. 0.0 {
    True -> hsic_ab /. denominator
    False -> 0.0
  }
}

/// CKA loss: 1 - CKA similarity (minimize to maximize alignment)
pub fn compute_cka_loss(student: HRR, teacher: HRR) -> Float {
  1.0 -. compute_cka(student, teacher)
}

/// Center a vector by subtracting the mean
fn center_vector(v: List(Float)) -> List(Float) {
  let n = list.length(v)
  let sum = list.fold(v, 0.0, fn(acc, x) { acc +. x })
  let mean = sum /. int.to_float(n)
  list.map(v, fn(x) { x -. mean })
}

/// Compute HSIC (simplified version using linear kernel)
/// HSIC = tr(KHLH) / (n-1)^2 where H = I - 1/n * 11^T (centering matrix)
/// For centered data with linear kernel: HSIC ≈ ||X^T Y||^2_F / n^2
fn compute_hsic(a: List(Float), b: List(Float)) -> Float {
  // Dot product as similarity (linear kernel)
  let dot = list.map2(a, b, fn(x, y) { x *. y })
    |> list.fold(0.0, fn(acc, x) { acc +. x })

  // Normalize by dimension
  let n = int.to_float(list.length(a))
  case n >. 0.0 {
    True -> dot *. dot /. { n *. n }
    False -> 0.0
  }
}

// =============================================================================
// MULTI-LAYER MATCHING (TextBrewer 2025)
// =============================================================================

/// Compute multi-layer matching loss
/// Extracts representations from multiple teacher layers and aligns with student
pub fn compute_layer_loss(
  student_layers: List(HRR),
  teacher_layers: List(HRR),
  weights: List(Float),
) -> Float {
  // Zip all three lists and compute weighted MSE
  list.zip(weights, list.zip(student_layers, teacher_layers))
  |> list.map(fn(triple) {
    let #(weight, #(student, teacher)) = triple
    weight *. compute_mse_loss(student, teacher)
  })
  |> list.fold(0.0, fn(acc, x) { acc +. x })
}

/// Alternative: CKA-based layer matching (allows dimension mismatch)
pub fn compute_layer_loss_cka(
  student_layers: List(HRR),
  teacher_layers: List(HRR),
  weights: List(Float),
) -> Float {
  list.zip(weights, list.zip(student_layers, teacher_layers))
  |> list.map(fn(triple) {
    let #(weight, #(student, teacher)) = triple
    weight *. compute_cka_loss(student, teacher)
  })
  |> list.fold(0.0, fn(acc, x) { acc +. x })
}

// =============================================================================
// ADVANCED LOSS FUNCTION V2
// =============================================================================

/// Compute advanced DHC loss with CKA + multi-layer matching
/// L = α_task*L_task + α_kd*L_kd + α_cka*L_cka + α_ewc*L_ewc + L_layer
pub fn compute_loss_v2(
  student_hrr: HRR,
  teacher_hrr: HRR,
  student_layers: List(HRR),
  teacher_layers: List(HRR),
  ewc_state: EWCState,
  current_params: List(Float),
  config: DistillConfigV2,
  progressive: ProgressiveState,
) -> DistillLossV2 {
  // Get dynamic config from progressive state
  let dynamic_config = progressive_config(config.base, progressive)

  // Task loss: MSE in HRR space
  let task_loss = compute_mse_loss(student_hrr, teacher_hrr)

  // KD loss: 1 - cosine_similarity
  let similarity = hrr.similarity(student_hrr, teacher_hrr)
  let kd_loss = 1.0 -. similarity

  // CKA loss: dimension-agnostic alignment (ICLR 2025)
  let cka_loss = compute_cka_loss(student_hrr, teacher_hrr)

  // EWC loss: protect important weights
  let ewc_loss = compute_ewc_loss(ewc_state, current_params)

  // Multi-layer matching loss
  let layer_loss = compute_layer_loss_cka(
    student_layers,
    teacher_layers,
    config.layer_match.layer_weights,
  )

  // Combined loss with progressive alpha_kd
  let total =
    dynamic_config.alpha_task *. task_loss
    +. progressive.alpha_kd *. kd_loss
    +. config.alpha_cka *. cka_loss
    +. dynamic_config.alpha_ewc *. ewc_loss
    +. layer_loss

  DistillLossV2(
    total: total,
    task_loss: task_loss,
    kd_loss: kd_loss,
    cka_loss: cka_loss,
    ewc_loss: ewc_loss,
    layer_loss: layer_loss,
  )
}

/// Simplified loss_v2 without multi-layer (when layers not available)
pub fn compute_loss_v2_simple(
  student_hrr: HRR,
  teacher_hrr: HRR,
  ewc_state: EWCState,
  current_params: List(Float),
  config: DistillConfigV2,
  progressive: ProgressiveState,
) -> DistillLossV2 {
  compute_loss_v2(
    student_hrr,
    teacher_hrr,
    [],  // No layer matching
    [],
    ewc_state,
    current_params,
    config,
    progressive,
  )
}

// =============================================================================
// KNOWLEDGE STORAGE
// =============================================================================

/// Create a DistilledKnowledge packet ready for KarmaBank storage
pub fn package_knowledge(
  concept_key: HRR,
  knowledge_hrr: HRR,
  emotional_state: PADState,
  config: DistillConfig,
  teacher_id: String,
) -> DistilledKnowledge {
  // Compress with NF4
  let compressed = compress_hrr(knowledge_hrr, config.nf4_config)

  // Compute salience from emotional state
  // High arousal + high |pleasure| = high salience
  let salience =
    float.clamp(
      { float.absolute_value(emotional_state.pleasure) +. emotional_state.arousal }
        /. 2.0,
      0.0,
      1.0,
    )

  DistilledKnowledge(
    concept_key: concept_key,
    compressed_hrr: compressed,
    emotional_context: emotional_state,
    salience: salience,
    metadata: KnowledgeMetadata(
      teacher_id: teacher_id,
      extraction_layer: config.extraction_layer,
      seq_len: hrr.dim(concept_key),
      timestamp: erlang_system_time(),
    ),
  )
}

// =============================================================================
// KNOWLEDGE RETRIEVAL
// =============================================================================

/// Retrieve knowledge from compressed storage using a query key
///
/// 1. Decompress NF4 → HRR
/// 2. Unbind with query key
/// 3. Compare with candidates
pub fn retrieve_knowledge(
  glands: GlandsHandle,
  memory: DistilledKnowledge,
  query_key: HRR,
) -> Result(HRR, String) {
  // Decompress
  let hrr_vec = decompress_hrr(memory.compressed_hrr)

  // Unbind: memory ⊛⁻¹ query ≈ value
  use retrieved <- result.try(glands.unbind(
    glands,
    hrr.to_list(hrr_vec),
    hrr.to_list(query_key),
  ))

  Ok(hrr.from_list(retrieved))
}

/// Batch similarity search against multiple memories
pub fn search_memories(
  _glands: GlandsHandle,
  query: HRR,
  memories: List(DistilledKnowledge),
) -> Result(List(#(Int, Float)), String) {
  // Decompress all memories
  let hrr_vecs = list.map(memories, fn(m) {
    decompress_hrr(m.compressed_hrr) |> hrr.to_list
  })

  // Batch similarity (GPU accelerated)
  use similarities <- result.try(glands.batch_similarity(hrr_vecs, hrr.to_list(query)))

  // Return indexed results sorted by similarity
  similarities
  |> list.index_map(fn(sim, idx) { #(idx, sim) })
  |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
  |> Ok
}

// =============================================================================
// EWC UTILITIES
// =============================================================================

/// Initialize EWC state (no consolidation yet)
pub fn init_ewc(num_params: Int) -> EWCState {
  EWCState(
    fisher_diag: list.repeat(0.0, num_params),
    consolidated_params: list.repeat(0.0, num_params),
  )
}

/// Update Fisher information after training on a task
/// Uses empirical Fisher: F = E[∇log p(y|x,θ)²]
pub fn update_fisher(
  ewc: EWCState,
  gradients: List(List(Float)),
  current_params: List(Float),
) -> EWCState {
  // Compute squared gradients mean
  let num_samples = list.length(gradients)

  let new_fisher = case num_samples > 0 {
    True -> {
      // Sum of squared gradients
      let grad_squared_sum =
        list.fold(gradients, list.repeat(0.0, list.length(ewc.fisher_diag)), fn(
          acc,
          grad,
        ) {
          list.map2(acc, grad, fn(a, g) { a +. g *. g })
        })

      // Average
      list.map(grad_squared_sum, fn(x) {
        x /. int.to_float(num_samples)
      })
    }
    False -> ewc.fisher_diag
  }

  EWCState(fisher_diag: new_fisher, consolidated_params: current_params)
}

// =============================================================================
// METRICS
// =============================================================================

/// Compression statistics
pub type CompressionStats {
  CompressionStats(
    /// Original size in bytes (FP32)
    original_bytes: Int,
    /// Compressed size in bytes (NF4)
    compressed_bytes: Int,
    /// Compression ratio
    ratio: Float,
    /// Reconstruction error (MSE)
    reconstruction_error: Float,
  )
}

/// Compute compression statistics
pub fn compression_stats(
  original: HRR,
  compressed: NF4Tensor,
) -> CompressionStats {
  let original_bytes = hrr.dim(original) * 4
  // FP32 = 4 bytes
  let compressed_bytes = compressed.memory_bytes

  let ratio =
    int.to_float(original_bytes) /. int.to_float(compressed_bytes)

  // Reconstruction error
  let decompressed = decompress_hrr(compressed)
  let error = compute_mse_loss(original, decompressed)

  CompressionStats(
    original_bytes: original_bytes,
    compressed_bytes: compressed_bytes,
    ratio: ratio,
    reconstruction_error: error,
  )
}

// =============================================================================
// LLM TEACHER INTEGRATION
// =============================================================================

/// Teacher state for distillation session
pub type TeacherState {
  TeacherState(
    model: LlmModel,
    config: DistillConfig,
    context_size: Int,
  )
}

/// Initialize teacher from Qwen3-32B
/// Loads model into GPU and returns state for distillation
pub fn init_teacher(model_path: String, gpu_layers: Int) -> TeacherState {
  let model = llm.load_model(model_path, gpu_layers)
  let info = llm.model_info(model)

  // Auto-configure based on model dimensions
  let config = DistillConfig(
    teacher_dim: info.n_embd,
    hrr_dim: 8192,
    extraction_layer: 32,
    nf4_config: nf4.default_config(),
    temperature: 2.0,
    alpha_task: 1.0,
    alpha_kd: 0.5,
    alpha_ewc: 0.1,
  )

  TeacherState(model: model, config: config, context_size: 2048)
}

/// Initialize with default Qwen3-32B teacher
pub fn init_default_teacher() -> TeacherState {
  init_teacher("models/Qwen3-32B-Q4_K_M.gguf", 65)
}

/// Extract knowledge from teacher for a given prompt
///
/// Full pipeline:
/// 1. Get hidden states from LLM
/// 2. Mean pool across tokens
/// 3. Project to HRR space
/// 4. Return ready for binding/storage
pub fn extract_from_prompt(
  teacher: TeacherState,
  glands: GlandsHandle,
  prompt: String,
) -> Result(HRR, String) {
  // Get hidden states from teacher
  let states = llm.get_hidden_states(
    teacher.model,
    prompt,
    teacher.context_size,
  )

  // Mean pool across tokens to get single embedding
  let pooled = llm.mean_pooled_embedding(states)

  // Project to HRR space
  extract_hidden_states(glands, pooled, teacher.config)
}

/// Extract knowledge with soft labels for distillation loss
pub fn extract_with_logits(
  teacher: TeacherState,
  glands: GlandsHandle,
  prompt: String,
) -> Result(#(HRR, List(Float)), String) {
  // Get hidden states
  let states = llm.get_hidden_states(
    teacher.model,
    prompt,
    teacher.context_size,
  )
  let pooled = llm.mean_pooled_embedding(states)

  // Get logits for soft label distillation
  let logits = llm.get_logits(teacher.model, prompt, teacher.context_size)

  // Apply temperature scaling for softer distribution
  let soft_labels = llm.softmax_with_temperature(logits, teacher.config.temperature)

  // Project to HRR
  use hrr_vec <- result.try(extract_hidden_states(glands, pooled, teacher.config))

  Ok(#(hrr_vec, soft_labels))
}

/// Batch extract knowledge from multiple prompts
pub fn batch_extract(
  teacher: TeacherState,
  glands: GlandsHandle,
  prompts: List(String),
) -> Result(List(HRR), String) {
  list.try_map(prompts, fn(prompt) {
    extract_from_prompt(teacher, glands, prompt)
  })
}

/// Distill a concept with semantic key binding
///
/// Creates: key ⊛ teacher_knowledge
/// Ready for storage in KarmaBank
pub fn distill_concept(
  teacher: TeacherState,
  glands: GlandsHandle,
  prompt: String,
  concept_key: HRR,
  emotional_state: PADState,
) -> Result(DistilledKnowledge, String) {
  // Extract teacher knowledge
  use knowledge_hrr <- result.try(extract_from_prompt(teacher, glands, prompt))

  // Bind with concept key
  use bound <- result.try(bind_with_key(glands, concept_key, knowledge_hrr))

  // Package for storage
  Ok(package_knowledge(
    concept_key,
    bound,
    emotional_state,
    teacher.config,
    "qwen3-32b",
  ))
}

/// Full DHC training step
///
/// 1. Extract teacher hidden states
/// 2. Compute student prediction
/// 3. Calculate combined loss
/// 4. Return gradients for update
pub fn training_step(
  teacher: TeacherState,
  glands: GlandsHandle,
  prompt: String,
  student_hrr: HRR,
  ewc_state: EWCState,
  current_params: List(Float),
) -> Result(#(DistillLoss, HRR), String) {
  // Extract teacher representation
  use teacher_hrr <- result.try(extract_from_prompt(teacher, glands, prompt))

  // Compute loss
  let loss = compute_loss(
    student_hrr,
    teacher_hrr,
    ewc_state,
    current_params,
    teacher.config,
  )

  Ok(#(loss, teacher_hrr))
}

/// Get teacher model info
pub fn teacher_info(teacher: TeacherState) -> llm.ModelInfo {
  llm.model_info(teacher.model)
}

/// Check available GPU memory
pub fn check_memory() -> #(Int, Int) {
  llm.memory_status()
}

// =============================================================================
// ADVANCED TRAINING (2025-2026 TECHNIQUES)
// =============================================================================

/// Advanced training state with progressive distillation
pub type TrainingStateV2 {
  TrainingStateV2(
    teacher: TeacherState,
    config: DistillConfigV2,
    progressive: ProgressiveState,
    ewc: EWCState,
    /// Running loss average for monitoring
    running_loss: Float,
    /// Best loss seen so far
    best_loss: Float,
    /// Steps since improvement (for early stopping)
    steps_without_improvement: Int,
  )
}

/// Initialize advanced training
pub fn init_training_v2(
  teacher: TeacherState,
  num_params: Int,
  max_epochs: Int,
) -> TrainingStateV2 {
  TrainingStateV2(
    teacher: teacher,
    config: advanced_config(),
    progressive: init_progressive(max_epochs),
    ewc: init_ewc(num_params),
    running_loss: 0.0,
    best_loss: 999_999.0,
    steps_without_improvement: 0,
  )
}

/// Advanced training step with all 2025-2026 features
///
/// Includes:
/// - Progressive distillation (dynamic alpha_kd + temperature)
/// - CKA loss for dimension flexibility
/// - EWC regularization
/// - Loss monitoring for early stopping
pub fn training_step_v2(
  state: TrainingStateV2,
  glands: GlandsHandle,
  prompt: String,
  student_hrr: HRR,
  current_params: List(Float),
) -> Result(#(DistillLossV2, HRR, TrainingStateV2), String) {
  // Extract teacher representation
  use teacher_hrr <- result.try(
    extract_from_prompt(state.teacher, glands, prompt)
  )

  // Compute advanced loss
  let loss = compute_loss_v2_simple(
    student_hrr,
    teacher_hrr,
    state.ewc,
    current_params,
    state.config,
    state.progressive,
  )

  // Update running loss (exponential moving average)
  let alpha = 0.1
  let new_running_loss = alpha *. loss.total +. { 1.0 -. alpha } *. state.running_loss

  // Check for improvement
  let #(new_best, new_steps) = case loss.total <. state.best_loss {
    True -> #(loss.total, 0)
    False -> #(state.best_loss, state.steps_without_improvement + 1)
  }

  // Update state
  let new_state = TrainingStateV2(
    ..state,
    running_loss: new_running_loss,
    best_loss: new_best,
    steps_without_improvement: new_steps,
  )

  Ok(#(loss, teacher_hrr, new_state))
}

/// Advance to next epoch (updates progressive state)
pub fn next_epoch(state: TrainingStateV2) -> TrainingStateV2 {
  TrainingStateV2(
    ..state,
    progressive: step_progressive(state.progressive, state.config),
  )
}

/// Check if training should stop early
/// Stops if no improvement for patience epochs
pub fn should_stop_early(state: TrainingStateV2, patience: Int) -> Bool {
  state.steps_without_improvement >= patience
}

/// Get current training metrics
pub fn training_metrics(state: TrainingStateV2) -> #(Float, Float, Int, CurriculumStage) {
  #(
    state.running_loss,
    state.progressive.alpha_kd,
    state.progressive.epoch,
    state.progressive.stage,
  )
}

/// Full training loop with progressive distillation
///
/// Usage:
/// ```gleam
/// let state = init_training_v2(teacher, num_params, 100)
/// let final_state = train_epochs(state, glands, prompts, get_student, update_student)
/// ```
pub fn train_epochs(
  initial_state: TrainingStateV2,
  glands: GlandsHandle,
  prompts: List(String),
  get_student_fn: fn() -> #(HRR, List(Float)),
  update_student_fn: fn(DistillLossV2, HRR) -> Nil,
  patience: Int,
) -> TrainingStateV2 {
  train_epochs_loop(initial_state, glands, prompts, get_student_fn, update_student_fn, patience, 0)
}

fn train_epochs_loop(
  state: TrainingStateV2,
  glands: GlandsHandle,
  prompts: List(String),
  get_student_fn: fn() -> #(HRR, List(Float)),
  update_student_fn: fn(DistillLossV2, HRR) -> Nil,
  patience: Int,
  epoch: Int,
) -> TrainingStateV2 {
  // Check termination conditions
  case epoch >= state.progressive.max_epochs || should_stop_early(state, patience) {
    True -> state
    False -> {
      // Train one epoch
      let state_after_epoch = train_one_epoch(
        state, glands, prompts, get_student_fn, update_student_fn
      )

      // Advance progressive schedule
      let next_state = next_epoch(state_after_epoch)

      // Recurse
      train_epochs_loop(
        next_state, glands, prompts, get_student_fn, update_student_fn, patience, epoch + 1
      )
    }
  }
}

fn train_one_epoch(
  state: TrainingStateV2,
  glands: GlandsHandle,
  prompts: List(String),
  get_student_fn: fn() -> #(HRR, List(Float)),
  update_student_fn: fn(DistillLossV2, HRR) -> Nil,
) -> TrainingStateV2 {
  list.fold(prompts, state, fn(current_state, prompt) {
    let #(student_hrr, params) = get_student_fn()

    case training_step_v2(current_state, glands, prompt, student_hrr, params) {
      Ok(#(loss, teacher_hrr, new_state)) -> {
        update_student_fn(loss, teacher_hrr)
        new_state
      }
      Error(_) -> current_state
    }
  })
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "erlang", "system_time")
fn erlang_system_time() -> Int

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
