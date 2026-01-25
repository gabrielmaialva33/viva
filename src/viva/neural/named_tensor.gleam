//// NamedTensor - Tensors with named axes (inspired by Nx/xarray)
////
//// Named axes provide semantic meaning to dimensions, enabling:
//// - Self-documenting code: `tensor |> sum(axis: Batch)` vs `sum(axis: 0)`
//// - Safe broadcasting: only compatible names can be broadcast
//// - Flexible operations: reference axes by name, not position
////
//// This module re-exports from:
//// - axis.gleam: Axis types and constructors
//// - named_tensor_core.gleam: Core type and basic operations
//// - named_ops.gleam: Reduction, element-wise, matrix operations
//// - einsum.gleam: Einstein summation-like API

import viva/neural/axis
import viva/neural/einsum
import viva/neural/named_ops.{SliceFirst, SliceIndex, SliceLast, SliceRange}
import viva/neural/named_tensor_core
import viva/neural/tensor.{type Tensor}

// =============================================================================
// RE-EXPORTS: TYPES
// =============================================================================

pub type Axis =
  axis.Axis

pub type AxisSpec =
  axis.AxisSpec

pub type NamedTensor =
  named_tensor_core.NamedTensor

pub type NamedTensorError =
  named_tensor_core.NamedTensorError

pub type SliceSpec =
  named_ops.SliceSpec

pub type EinsumSpec =
  einsum.EinsumSpec

// =============================================================================
// AXIS CONSTRUCTORS
// =============================================================================

/// Create axis spec with shorthand
pub fn axis(name: Axis, size: Int) -> AxisSpec {
  axis.axis(name, size)
}

/// Batch dimension
pub fn batch(size: Int) -> AxisSpec {
  axis.batch(size)
}

/// Sequence dimension
pub fn seq(size: Int) -> AxisSpec {
  axis.seq(size)
}

/// Feature dimension
pub fn feature(size: Int) -> AxisSpec {
  axis.feature(size)
}

/// Height dimension
pub fn height(size: Int) -> AxisSpec {
  axis.height(size)
}

/// Width dimension
pub fn width(size: Int) -> AxisSpec {
  axis.width(size)
}

/// Input dimension
pub fn input(size: Int) -> AxisSpec {
  axis.input(size)
}

/// Output dimension
pub fn output(size: Int) -> AxisSpec {
  axis.output(size)
}

/// Head dimension
pub fn head(size: Int) -> AxisSpec {
  axis.head(size)
}

/// Embed dimension
pub fn embed(size: Int) -> AxisSpec {
  axis.embed(size)
}

/// Custom named dimension
pub fn named(name: String, size: Int) -> AxisSpec {
  axis.named(name, size)
}

/// Axis to string
pub fn axis_to_string(a: Axis) -> String {
  axis.axis_to_string(a)
}

// =============================================================================
// CORE CONSTRUCTORS
// =============================================================================

/// Create named tensor from data and axis specs
pub fn new(
  data: Tensor,
  axes: List(AxisSpec),
) -> Result(NamedTensor, NamedTensorError) {
  named_tensor_core.new(data, axes)
}

/// Create from tensor with inferred anonymous axes
pub fn from_tensor(t: Tensor) -> NamedTensor {
  named_tensor_core.from_tensor(t)
}

/// Create named tensor of zeros
pub fn zeros(axes: List(AxisSpec)) -> NamedTensor {
  named_tensor_core.zeros(axes)
}

/// Create named tensor of ones
pub fn ones(axes: List(AxisSpec)) -> NamedTensor {
  named_tensor_core.ones(axes)
}

/// Create named tensor with random values
pub fn random(axes: List(AxisSpec)) -> NamedTensor {
  named_tensor_core.random(axes)
}

/// Create named tensor with normal distribution
pub fn randn(axes: List(AxisSpec), mean: Float, std: Float) -> NamedTensor {
  named_tensor_core.randn(axes, mean, std)
}

// =============================================================================
// AXIS LOOKUP
// =============================================================================

/// Find axis index by name
pub fn find_axis(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  named_tensor_core.find_axis(t, name)
}

/// Get axis size by name
pub fn axis_size(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  named_tensor_core.axis_size(t, name)
}

/// Check if tensor has axis
pub fn has_axis(t: NamedTensor, name: Axis) -> Bool {
  named_tensor_core.has_axis(t, name)
}

/// Get all axis names
pub fn axis_names(t: NamedTensor) -> List(Axis) {
  named_tensor_core.axis_names(t)
}

/// Get shape as list
pub fn shape(t: NamedTensor) -> List(Int) {
  named_tensor_core.shape(t)
}

/// Get rank (number of dimensions)
pub fn rank(t: NamedTensor) -> Int {
  named_tensor_core.rank(t)
}

/// Total number of elements
pub fn size(t: NamedTensor) -> Int {
  named_tensor_core.size(t)
}

// =============================================================================
// AXIS OPERATIONS
// =============================================================================

/// Rename an axis
pub fn rename_axis(
  t: NamedTensor,
  from: Axis,
  to: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_tensor_core.rename_axis(t, from, to)
}

/// Transpose/reorder axes by names
pub fn transpose(
  t: NamedTensor,
  new_order: List(Axis),
) -> Result(NamedTensor, NamedTensorError) {
  named_tensor_core.transpose(t, new_order)
}

/// Add a new axis of size 1
pub fn unsqueeze(t: NamedTensor, name: Axis, position: Int) -> NamedTensor {
  named_tensor_core.unsqueeze(t, name, position)
}

/// Remove axis of size 1 by name
pub fn squeeze(
  t: NamedTensor,
  name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_tensor_core.squeeze(t, name)
}

// =============================================================================
// CONVERSION
// =============================================================================

/// Convert to plain tensor (drop names)
pub fn to_tensor(t: NamedTensor) -> Tensor {
  named_tensor_core.to_tensor(t)
}

/// Pretty print tensor info
pub fn describe(t: NamedTensor) -> String {
  named_tensor_core.describe(t)
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

/// Sum along named axis
pub fn sum(t: NamedTensor, along: Axis) -> Result(NamedTensor, NamedTensorError) {
  named_ops.sum(t, along)
}

/// Mean along named axis
pub fn mean(
  t: NamedTensor,
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.mean(t, along)
}

/// Max value along named axis
pub fn max_along(
  t: NamedTensor,
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.max_along(t, along)
}

/// Argmax along named axis
pub fn argmax_along(
  t: NamedTensor,
  along: Axis,
) -> Result(List(Int), NamedTensorError) {
  named_ops.argmax_along(t, along)
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

/// Element-wise addition
pub fn add(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.add(a, b)
}

/// Element-wise subtraction
pub fn sub(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.sub(a, b)
}

/// Element-wise multiplication
pub fn mul(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.mul(a, b)
}

/// Scale by constant
pub fn scale(t: NamedTensor, s: Float) -> NamedTensor {
  named_ops.scale(t, s)
}

/// Apply function to each element
pub fn map(t: NamedTensor, f: fn(Float) -> Float) -> NamedTensor {
  named_ops.map(t, f)
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/// Matrix multiplication with explicit contraction axis
pub fn matmul(
  a: NamedTensor,
  b: NamedTensor,
  contract_a: Axis,
  contract_b: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.matmul(a, b, contract_a, contract_b)
}

/// Dot product over named axis
pub fn dot(
  a: NamedTensor,
  b: NamedTensor,
  along: Axis,
) -> Result(Float, NamedTensorError) {
  named_ops.dot(a, b, along)
}

// =============================================================================
// SLICING
// =============================================================================

/// Apply slice to tensor
pub fn slice(
  t: NamedTensor,
  spec: SliceSpec,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.slice(t, spec)
}

// =============================================================================
// STACKING & CONCATENATION
// =============================================================================

/// Stack tensors along new axis
pub fn stack(
  tensors: List(NamedTensor),
  new_axis: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.stack(tensors, new_axis)
}

/// Concatenate tensors along existing axis
pub fn concat(
  tensors: List(NamedTensor),
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  named_ops.concat(tensors, along)
}

// =============================================================================
// EINSUM
// =============================================================================

/// Einsum-like contraction
pub fn einsum(
  tensors: List(NamedTensor),
  spec: EinsumSpec,
) -> Result(NamedTensor, NamedTensorError) {
  einsum.einsum(tensors, spec)
}

// =============================================================================
// EINSUM CONSTRUCTORS
// =============================================================================

/// Create Contract einsum spec
pub fn contract(axes: List(Axis)) -> EinsumSpec {
  einsum.Contract(axes)
}

/// Create Keep einsum spec
pub fn keep(axes: List(Axis)) -> EinsumSpec {
  einsum.Keep(axes)
}

/// Create BatchAxes einsum spec
pub fn batch_axes_spec(axes: List(Axis)) -> EinsumSpec {
  einsum.BatchAxes(axes)
}

// =============================================================================
// SLICE CONSTRUCTORS
// =============================================================================

/// Create SliceRange spec
pub fn slice_range(axis_name: Axis, start: Int, length: Int) -> SliceSpec {
  SliceRange(axis: axis_name, start: start, length: length)
}

/// Create SliceIndex spec
pub fn slice_index(axis_name: Axis, index: Int) -> SliceSpec {
  SliceIndex(axis: axis_name, index: index)
}

/// Create SliceFirst spec
pub fn slice_first(axis_name: Axis, n: Int) -> SliceSpec {
  SliceFirst(axis: axis_name, n: n)
}

/// Create SliceLast spec
pub fn slice_last(axis_name: Axis, n: Int) -> SliceSpec {
  SliceLast(axis: axis_name, n: n)
}
