//// GPU - VIVA GPU Acceleration Layer
////
//// Provides GPU-accelerated operations for VIVA using EXLA/CUDA.
//// Falls back to CPU gracefully if GPU unavailable.
////
//// Usage:
////   import viva/gpu
////   let backend = gpu.detect()  // Auto-detect best backend
////   let result = gpu.batch_matmul(inputs, weights, backend)

import gleam/dict.{type Dict}
import gleam/float
import gleam/io
import gleam/list
import gleam/result
import viva/neural/tensor.{type Tensor}
import viva_emotion/pad.{type Pad}

// =============================================================================
// TYPES
// =============================================================================

/// Compute backend
pub type Backend {
  /// Pure Gleam (always available)
  CPU
  /// EXLA on GPU (requires CUDA)
  GPU
  /// EXLA on CPU (faster than pure Gleam)
  ExlaCpu
}

/// GPU info
pub type GpuInfo {
  GpuInfo(
    available: Bool,
    name: String,
    memory_mb: Int,
    compute_capability: String,
  )
}

/// Batch PAD tensor for GPU operations
pub type PadBatch {
  PadBatch(
    /// Flat data: [p1, a1, d1, p2, a2, d2, ...]
    data: List(Float),
    /// Number of PADs in batch
    count: Int,
  )
}

// =============================================================================
// DETECTION
// =============================================================================

/// Detect best available backend
pub fn detect() -> Backend {
  case gpu_available() {
    True -> GPU
    False -> case exla_available() {
      True -> ExlaCpu
      False -> CPU
    }
  }
}

/// Check if GPU is available
pub fn gpu_available() -> Bool {
  nx_cuda_available()
}

/// Check if EXLA is available (even without GPU)
pub fn exla_available() -> Bool {
  nx_exla_available()
}

/// Get GPU info
pub fn gpu_info() -> GpuInfo {
  case gpu_available() {
    True -> nx_gpu_info()
    False -> GpuInfo(
      available: False,
      name: "None",
      memory_mb: 0,
      compute_capability: "N/A",
    )
  }
}

/// Print GPU status
pub fn print_status() -> Nil {
  let info = gpu_info()
  case info.available {
    True -> {
      io.println("GPU: " <> info.name)
      io.println("VRAM: " <> int_to_string(info.memory_mb) <> " MB")
      io.println("Compute: " <> info.compute_capability)
    }
    False -> {
      io.println("GPU: Not available")
      io.println("Using CPU backend")
    }
  }
}

// =============================================================================
// PAD BATCH OPERATIONS (for Soul Pool)
// =============================================================================

/// Create PAD batch from dict
pub fn pads_to_batch(pads: Dict(Int, Pad)) -> PadBatch {
  let data = pads
    |> dict.values()
    |> list.flat_map(fn(p) { [p.pleasure, p.arousal, p.dominance] })

  PadBatch(data: data, count: dict.size(pads))
}

/// Convert batch back to dict (with original IDs)
pub fn batch_to_pads(batch: PadBatch, ids: List(Int)) -> Dict(Int, Pad) {
  let pads = chunk_pads(batch.data, [])
  list.zip(ids, pads)
  |> dict.from_list()
}

fn chunk_pads(data: List(Float), acc: List(Pad)) -> List(Pad) {
  case data {
    [p, a, d, ..rest] -> chunk_pads(rest, [pad.new(p, a, d), ..acc])
    _ -> list.reverse(acc)
  }
}

/// Apply delta to all PADs in batch (GPU accelerated)
pub fn batch_apply_delta(
  batch: PadBatch,
  delta: Pad,
  backend: Backend,
) -> PadBatch {
  case backend {
    GPU -> nx_batch_apply_delta(batch, delta)
    ExlaCpu -> nx_batch_apply_delta(batch, delta)
    CPU -> cpu_batch_apply_delta(batch, delta)
  }
}

fn cpu_batch_apply_delta(batch: PadBatch, delta: Pad) -> PadBatch {
  let new_data = batch.data
    |> list.index_map(fn(val, idx) {
      let dim = idx % 3
      case dim {
        0 -> clamp(val +. delta.pleasure, -1.0, 1.0)
        1 -> clamp(val +. delta.arousal, -1.0, 1.0)
        _ -> clamp(val +. delta.dominance, -1.0, 1.0)
      }
    })
  PadBatch(..batch, data: new_data)
}

/// Scale all PADs in batch
pub fn batch_scale(batch: PadBatch, factor: Float, backend: Backend) -> PadBatch {
  case backend {
    GPU -> nx_batch_scale(batch, factor)
    ExlaCpu -> nx_batch_scale(batch, factor)
    CPU -> {
      let new_data = list.map(batch.data, fn(v) { clamp(v *. factor, -1.0, 1.0) })
      PadBatch(..batch, data: new_data)
    }
  }
}

/// Lerp all PADs toward target
pub fn batch_lerp(
  batch: PadBatch,
  target: Pad,
  t: Float,
  backend: Backend,
) -> PadBatch {
  case backend {
    GPU -> nx_batch_lerp(batch, target, t)
    ExlaCpu -> nx_batch_lerp(batch, target, t)
    CPU -> cpu_batch_lerp(batch, target, t)
  }
}

fn cpu_batch_lerp(batch: PadBatch, target: Pad, t: Float) -> PadBatch {
  let new_data = batch.data
    |> list.index_map(fn(val, idx) {
      let dim = idx % 3
      let target_val = case dim {
        0 -> target.pleasure
        1 -> target.arousal
        _ -> target.dominance
      }
      val +. { target_val -. val } *. t
    })
  PadBatch(..batch, data: new_data)
}

// =============================================================================
// TENSOR OPERATIONS (for Neural Networks)
// =============================================================================

/// Batch matrix multiplication
pub fn batch_matmul(
  inputs: List(Tensor),
  weights: Tensor,
  backend: Backend,
) -> List(Tensor) {
  case backend {
    GPU -> nx_batch_matmul(inputs, weights)
    ExlaCpu -> nx_batch_matmul(inputs, weights)
    CPU -> cpu_batch_matmul(inputs, weights)
  }
}

fn cpu_batch_matmul(inputs: List(Tensor), weights: Tensor) -> List(Tensor) {
  list.filter_map(inputs, fn(input) {
    tensor.matmul_vec(weights, input)
    |> result.replace_error(Nil)
  })
}

/// Batch forward pass through dense layer
pub fn batch_dense_forward(
  inputs: List(Tensor),
  weights: Tensor,
  biases: Tensor,
  activation: Activation,
  backend: Backend,
) -> List(Tensor) {
  case backend {
    GPU -> nx_batch_dense_forward(inputs, weights, biases, activation)
    ExlaCpu -> nx_batch_dense_forward(inputs, weights, biases, activation)
    CPU -> cpu_batch_dense_forward(inputs, weights, biases, activation)
  }
}

pub type Activation {
  Linear
  ReLU
  Sigmoid
  Tanh
  Softmax
}

fn cpu_batch_dense_forward(
  inputs: List(Tensor),
  weights: Tensor,
  biases: Tensor,
  activation: Activation,
) -> List(Tensor) {
  list.filter_map(inputs, fn(input) {
    case tensor.matmul_vec(weights, input) {
      Ok(wx) -> {
        case tensor.add(wx, biases) {
          Ok(z) -> Ok(apply_activation_cpu(z, activation))
          Error(_) -> Error(Nil)
        }
      }
      Error(_) -> Error(Nil)
    }
  })
}

fn apply_activation_cpu(t: Tensor, activation: Activation) -> Tensor {
  case activation {
    Linear -> t
    ReLU -> tensor.map(t, fn(x) { float.max(0.0, x) })
    Sigmoid -> tensor.map(t, sigmoid)
    Tanh -> tensor.map(t, tanh)
    Softmax -> {
      let max_val = tensor.max(t)
      let shifted = tensor.add_scalar(t, 0.0 -. max_val)
      let exp_vals = tensor.map(shifted, exp)
      let sum_exp = tensor.sum(exp_vals)
      tensor.scale(exp_vals, 1.0 /. sum_exp)
    }
  }
}

// =============================================================================
// RESONANCE CALCULATIONS (batch)
// =============================================================================

/// Calculate all pairwise resonances (O(n²) but parallelized on GPU)
pub fn batch_resonance(
  pads: List(Pad),
  backend: Backend,
) -> List(List(Float)) {
  case backend {
    GPU -> nx_batch_resonance(pads)
    ExlaCpu -> nx_batch_resonance(pads)
    CPU -> cpu_batch_resonance(pads)
  }
}

fn cpu_batch_resonance(pads: List(Pad)) -> List(List(Float)) {
  list.map(pads, fn(p1) {
    list.map(pads, fn(p2) {
      pad_similarity(p1, p2)
    })
  })
}

fn pad_similarity(p1: Pad, p2: Pad) -> Float {
  let dp = p1.pleasure -. p2.pleasure
  let da = p1.arousal -. p2.arousal
  let dd = p1.dominance -. p2.dominance
  let dist = float_sqrt(dp *. dp +. da *. da +. dd *. dd)
  1.0 -. dist /. 3.464  // max dist is sqrt(12) ≈ 3.464
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(val: Float, min: Float, max: Float) -> Float {
  case val <. min {
    True -> min
    False -> case val >. max {
      True -> max
      False -> val
    }
  }
}

fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. exp(0.0 -. x) }
}

// =============================================================================
// FFI - Elixir/EXLA calls
// =============================================================================

@external(erlang, "Elixir.VivaGpu", "cuda_available")
fn nx_cuda_available() -> Bool

@external(erlang, "Elixir.VivaGpu", "exla_available")
fn nx_exla_available() -> Bool

@external(erlang, "Elixir.VivaGpu", "gpu_info")
fn nx_gpu_info() -> GpuInfo

@external(erlang, "Elixir.VivaGpu", "batch_apply_delta")
fn nx_batch_apply_delta(batch: PadBatch, delta: Pad) -> PadBatch

@external(erlang, "Elixir.VivaGpu", "batch_scale")
fn nx_batch_scale(batch: PadBatch, factor: Float) -> PadBatch

@external(erlang, "Elixir.VivaGpu", "batch_lerp")
fn nx_batch_lerp(batch: PadBatch, target: Pad, t: Float) -> PadBatch

@external(erlang, "Elixir.VivaGpu", "batch_matmul")
fn nx_batch_matmul(inputs: List(Tensor), weights: Tensor) -> List(Tensor)

@external(erlang, "Elixir.VivaGpu", "batch_dense_forward")
fn nx_batch_dense_forward(
  inputs: List(Tensor),
  weights: Tensor,
  biases: Tensor,
  activation: Activation,
) -> List(Tensor)

@external(erlang, "Elixir.VivaGpu", "batch_resonance")
fn nx_batch_resonance(pads: List(Pad)) -> List(List(Float))

@external(erlang, "math", "exp")
fn exp(x: Float) -> Float

@external(erlang, "math", "tanh")
fn tanh(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(i: Int) -> String
