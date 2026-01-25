//// Neural Module Tests
////
//// Tests for tensor, network, layer, and train modules

import gleam/float
import gleam/result
import gleeunit/should
import viva/neural/activation
import viva/neural/tensor

// =============================================================================
// TENSOR CONSTRUCTORS
// =============================================================================

pub fn tensor_zeros_test() {
  let t = tensor.zeros([2, 3])
  tensor.size(t) |> should.equal(6)
  t.shape |> should.equal([2, 3])
  tensor.sum(t) |> should.equal(0.0)
}

pub fn tensor_ones_test() {
  let t = tensor.ones([3, 3])
  tensor.size(t) |> should.equal(9)
  tensor.sum(t) |> should.equal(9.0)
}

pub fn tensor_fill_test() {
  let t = tensor.fill([2, 2], 5.0)
  tensor.sum(t) |> should.equal(20.0)
}

pub fn tensor_from_list_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  tensor.size(t) |> should.equal(3)
  tensor.rank(t) |> should.equal(1)
  t.shape |> should.equal([3])
}

pub fn tensor_from_list2d_test() {
  let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let result = tensor.from_list2d(rows)

  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  t.shape |> should.equal([3, 2])
  tensor.size(t) |> should.equal(6)
}

pub fn tensor_from_list2d_invalid_test() {
  let rows = [[1.0, 2.0], [3.0]]  // Different lengths
  let result = tensor.from_list2d(rows)
  result |> should.be_error
}

pub fn tensor_matrix_test() {
  let result = tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  t.shape |> should.equal([2, 3])
}

pub fn tensor_matrix_size_mismatch_test() {
  let result = tensor.matrix(2, 3, [1.0, 2.0, 3.0])  // Only 3 elements
  result |> should.be_error
}

// =============================================================================
// TENSOR PROPERTIES
// =============================================================================

pub fn tensor_rank_test() {
  tensor.rank(tensor.zeros([5])) |> should.equal(1)
  tensor.rank(tensor.zeros([3, 4])) |> should.equal(2)
  tensor.rank(tensor.zeros([2, 3, 4])) |> should.equal(3)
}

pub fn tensor_dim_test() {
  let t = tensor.zeros([2, 3, 4])
  tensor.dim(t, 0) |> should.equal(Ok(2))
  tensor.dim(t, 1) |> should.equal(Ok(3))
  tensor.dim(t, 2) |> should.equal(Ok(4))
  tensor.dim(t, 3) |> should.be_error
}

pub fn tensor_rows_cols_test() {
  let t = tensor.zeros([3, 5])
  tensor.rows(t) |> should.equal(3)
  tensor.cols(t) |> should.equal(5)
}

// =============================================================================
// TENSOR ELEMENT ACCESS
// =============================================================================

pub fn tensor_get_test() {
  let t = tensor.from_list([10.0, 20.0, 30.0])
  tensor.get(t, 0) |> should.equal(Ok(10.0))
  tensor.get(t, 1) |> should.equal(Ok(20.0))
  tensor.get(t, 2) |> should.equal(Ok(30.0))
  tensor.get(t, 3) |> should.be_error
}

pub fn tensor_get2d_test() {
  let t =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))

  tensor.get2d(t, 0, 0) |> should.equal(Ok(1.0))
  tensor.get2d(t, 0, 2) |> should.equal(Ok(3.0))
  tensor.get2d(t, 1, 0) |> should.equal(Ok(4.0))
  tensor.get2d(t, 1, 2) |> should.equal(Ok(6.0))
}

pub fn tensor_get_row_test() {
  let t =
    tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))

  let row0 = tensor.get_row(t, 0) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(row0) |> should.equal([1.0, 2.0])

  let row2 = tensor.get_row(t, 2) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(row2) |> should.equal([5.0, 6.0])
}

pub fn tensor_get_col_test() {
  let t =
    tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))

  let col0 = tensor.get_col(t, 0) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(col0) |> should.equal([1.0, 3.0, 5.0])

  let col1 = tensor.get_col(t, 1) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(col1) |> should.equal([2.0, 4.0, 6.0])
}

// =============================================================================
// TENSOR ELEMENT-WISE OPERATIONS
// =============================================================================

pub fn tensor_map_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let doubled = tensor.map(t, fn(x) { x *. 2.0 })
  tensor.to_list(doubled) |> should.equal([2.0, 4.0, 6.0])
}

pub fn tensor_add_test() {
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([10.0, 20.0, 30.0])
  let result = tensor.add(a, b)

  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(t) |> should.equal([11.0, 22.0, 33.0])
}

pub fn tensor_add_shape_mismatch_test() {
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([1.0, 2.0])
  tensor.add(a, b) |> should.be_error
}

pub fn tensor_sub_test() {
  let a = tensor.from_list([10.0, 20.0, 30.0])
  let b = tensor.from_list([1.0, 2.0, 3.0])
  let result = tensor.sub(a, b) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(result) |> should.equal([9.0, 18.0, 27.0])
}

pub fn tensor_mul_test() {
  let a = tensor.from_list([2.0, 3.0, 4.0])
  let b = tensor.from_list([10.0, 10.0, 10.0])
  let result = tensor.mul(a, b) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(result) |> should.equal([20.0, 30.0, 40.0])
}

pub fn tensor_div_test() {
  let a = tensor.from_list([10.0, 20.0, 30.0])
  let b = tensor.from_list([2.0, 4.0, 5.0])
  let result = tensor.div(a, b) |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(result) |> should.equal([5.0, 5.0, 6.0])
}

pub fn tensor_scale_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let scaled = tensor.scale(t, 10.0)
  tensor.to_list(scaled) |> should.equal([10.0, 20.0, 30.0])
}

pub fn tensor_add_scalar_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let result = tensor.add_scalar(t, 100.0)
  tensor.to_list(result) |> should.equal([101.0, 102.0, 103.0])
}

pub fn tensor_negate_test() {
  let t = tensor.from_list([1.0, -2.0, 3.0])
  let negated = tensor.negate(t)
  tensor.to_list(negated) |> should.equal([-1.0, 2.0, -3.0])
}

// =============================================================================
// TENSOR REDUCTIONS
// =============================================================================

pub fn tensor_sum_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  tensor.sum(t) |> should.equal(15.0)
}

pub fn tensor_product_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  tensor.product(t) |> should.equal(24.0)
}

pub fn tensor_mean_test() {
  let t = tensor.from_list([2.0, 4.0, 6.0, 8.0])
  tensor.mean(t) |> should.equal(5.0)
}

pub fn tensor_max_test() {
  let t = tensor.from_list([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
  tensor.max(t) |> should.equal(9.0)
}

pub fn tensor_min_test() {
  let t = tensor.from_list([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
  tensor.min(t) |> should.equal(1.0)
}

pub fn tensor_argmax_test() {
  let t = tensor.from_list([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
  tensor.argmax(t) |> should.equal(5)  // Index of 9.0
}

pub fn tensor_argmin_test() {
  let t = tensor.from_list([3.0, 1.0, 4.0, 0.5, 5.0, 9.0])
  tensor.argmin(t) |> should.equal(3)  // Index of 0.5
}

// =============================================================================
// TENSOR MATRIX OPERATIONS
// =============================================================================

pub fn tensor_dot_test() {
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([4.0, 5.0, 6.0])
  tensor.dot(a, b) |> should.equal(Ok(32.0))  // 1*4 + 2*5 + 3*6
}

pub fn tensor_matmul_vec_test() {
  // [2x3] @ [3] -> [2]
  let mat =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))
  let vec = tensor.from_list([1.0, 1.0, 1.0])

  let result = tensor.matmul_vec(mat, vec)
  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(t) |> should.equal([6.0, 15.0])  // [1+2+3, 4+5+6]
}

pub fn tensor_matmul_test() {
  // [2x3] @ [3x2] -> [2x2]
  let a =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))
  let b =
    tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))

  let result = tensor.matmul(a, b)
  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  t.shape |> should.equal([2, 2])
  // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
  // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
  tensor.to_list(t) |> should.equal([22.0, 28.0, 49.0, 64.0])
}

pub fn tensor_transpose_test() {
  // [2x3] -> [3x2]
  let t =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))

  let transposed = tensor.transpose(t) |> result.unwrap(tensor.zeros([0]))
  transposed.shape |> should.equal([3, 2])
  // Original: [[1,2,3], [4,5,6]]
  // Transposed: [[1,4], [2,5], [3,6]]
  tensor.to_list(transposed) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

pub fn tensor_outer_test() {
  let a = tensor.from_list([1.0, 2.0])
  let b = tensor.from_list([3.0, 4.0, 5.0])

  let result = tensor.outer(a, b)
  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  t.shape |> should.equal([2, 3])
  // [1*3, 1*4, 1*5, 2*3, 2*4, 2*5]
  tensor.to_list(t) |> should.equal([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
}

// =============================================================================
// TENSOR UTILITY
// =============================================================================

pub fn tensor_reshape_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let reshaped = tensor.reshape(t, [2, 3])

  reshaped |> should.be_ok
  let r = reshaped |> result.unwrap(tensor.zeros([0]))
  r.shape |> should.equal([2, 3])
}

pub fn tensor_reshape_invalid_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  tensor.reshape(t, [2, 2]) |> should.be_error  // 3 != 4
}

pub fn tensor_flatten_test() {
  let t =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))
  let flat = tensor.flatten(t)
  flat.shape |> should.equal([6])
}

pub fn tensor_concat_test() {
  let a = tensor.from_list([1.0, 2.0])
  let b = tensor.from_list([3.0, 4.0, 5.0])
  let result = tensor.concat([a, b])
  tensor.to_list(result) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0])
}

pub fn tensor_norm_test() {
  let t = tensor.from_list([3.0, 4.0])
  tensor.norm(t) |> should.equal(5.0)  // sqrt(9 + 16)
}

pub fn tensor_normalize_test() {
  let t = tensor.from_list([3.0, 4.0])
  let normalized = tensor.normalize(t)
  let len = tensor.norm(normalized)
  should.be_true(float.loosely_equals(len, 1.0, 0.0001))
}

pub fn tensor_clamp_test() {
  let t = tensor.from_list([-2.0, 0.5, 3.0])
  let clamped = tensor.clamp(t, 0.0, 1.0)
  tensor.to_list(clamped) |> should.equal([0.0, 0.5, 1.0])
}

// =============================================================================
// TENSOR STATISTICS
// =============================================================================

pub fn tensor_variance_test() {
  let t = tensor.from_list([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  let var = tensor.variance(t)
  // Mean = 5, variance = ((2-5)² + ... + (9-5)²) / 8 = 4
  should.be_true(float.loosely_equals(var, 4.0, 0.0001))
}

pub fn tensor_std_test() {
  let t = tensor.from_list([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  let std = tensor.std(t)
  should.be_true(float.loosely_equals(std, 2.0, 0.0001))
}

// =============================================================================
// TENSOR BROADCASTING
// =============================================================================

pub fn tensor_can_broadcast_test() {
  tensor.can_broadcast([3, 4], [3, 4]) |> should.be_true
  tensor.can_broadcast([3, 4], [4]) |> should.be_true
  tensor.can_broadcast([3, 4], [1, 4]) |> should.be_true
  tensor.can_broadcast([3, 4], [3, 1]) |> should.be_true
  tensor.can_broadcast([3, 4], [2, 4]) |> should.be_false
}

pub fn tensor_broadcast_shape_test() {
  tensor.broadcast_shape([3, 4], [4]) |> should.equal(Ok([3, 4]))
  tensor.broadcast_shape([3, 1], [1, 4]) |> should.equal(Ok([3, 4]))
  tensor.broadcast_shape([1], [5]) |> should.equal(Ok([5]))
}

pub fn tensor_add_broadcast_test() {
  let a =
    tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    |> result.unwrap(tensor.zeros([0]))
  let b = tensor.from_list([10.0, 20.0, 30.0])  // [3]

  let result = tensor.add_broadcast(a, b)
  result |> should.be_ok
  let t = result |> result.unwrap(tensor.zeros([0]))
  t.shape |> should.equal([2, 3])
  // [1+10, 2+20, 3+30, 4+10, 5+20, 6+30]
  tensor.to_list(t) |> should.equal([11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
}

// =============================================================================
// TENSOR SHAPE MANIPULATION
// =============================================================================

pub fn tensor_squeeze_test() {
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3, 1])
  let squeezed = tensor.squeeze(t)
  squeezed.shape |> should.equal([3])
}

pub fn tensor_unsqueeze_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let expanded = tensor.unsqueeze(t, 0)
  expanded.shape |> should.equal([1, 3])
}

pub fn tensor_expand_dims_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let expanded = tensor.expand_dims(t, 1)
  expanded.shape |> should.equal([3, 1])
}

// =============================================================================
// TENSOR SLICING
// =============================================================================

pub fn tensor_slice_1d_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let sliced = tensor.slice(t, [1], [3])

  sliced |> should.be_ok
  let s = sliced |> result.unwrap(tensor.zeros([0]))
  tensor.to_list(s) |> should.equal([2.0, 3.0, 4.0])
}

pub fn tensor_take_first_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let first3 = tensor.take_first(t, 3)
  tensor.to_list(first3) |> should.equal([1.0, 2.0, 3.0])
}

pub fn tensor_take_last_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let last2 = tensor.take_last(t, 2)
  tensor.to_list(last2) |> should.equal([4.0, 5.0])
}

// =============================================================================
// TENSOR RANDOM
// =============================================================================

pub fn tensor_random_uniform_test() {
  let t = tensor.random_uniform([100])
  tensor.size(t) |> should.equal(100)

  // All values should be in [0, 1)
  let all_in_range =
    tensor.to_list(t)
    |> list.all(fn(x) { x >=. 0.0 && x <. 1.0 })
  should.be_true(all_in_range)
}

pub fn tensor_xavier_init_test() {
  let t = tensor.xavier_init(100, 50)
  t.shape |> should.equal([100, 50])

  // Xavier limit = sqrt(6 / 150) ≈ 0.2
  let limit = 0.3  // Slightly larger for safety
  let all_in_range =
    tensor.to_list(t)
    |> list.all(fn(x) { x >=. -1.0 *. limit && x <=. limit })
  should.be_true(all_in_range)
}

// =============================================================================
// ACTIVATION FUNCTIONS
// =============================================================================

pub fn activation_sigmoid_test() {
  let result = activation.sigmoid(0.0)

  // sigmoid(0) = 0.5
  should.be_true(float.loosely_equals(result.value, 0.5, 0.0001))

  // sigmoid approaches 1 for large positive
  let high = activation.sigmoid(10.0)
  should.be_true(high.value >. 0.99)

  // sigmoid approaches 0 for large negative
  let low = activation.sigmoid(-10.0)
  should.be_true(low.value <. 0.01)
}

pub fn activation_tanh_test() {
  let result = activation.tanh(0.0)

  // tanh(0) = 0
  should.be_true(float.loosely_equals(result.value, 0.0, 0.0001))

  // tanh approaches 1 for large positive
  let high = activation.tanh(10.0)
  should.be_true(high.value >. 0.99)

  // tanh approaches -1 for large negative
  let low = activation.tanh(-10.0)
  should.be_true(low.value <. -0.99)
}

pub fn activation_relu_test() {
  let pos = activation.relu(5.0)
  pos.value |> should.equal(5.0)

  let neg = activation.relu(-5.0)
  neg.value |> should.equal(0.0)

  let zero = activation.relu(0.0)
  zero.value |> should.equal(0.0)
}

pub fn activation_leaky_relu_test() {
  let pos = activation.leaky_relu(5.0, 0.01)
  pos.value |> should.equal(5.0)

  let neg = activation.leaky_relu(-5.0, 0.01)
  should.be_true(float.loosely_equals(neg.value, -0.05, 0.0001))
}

pub fn activation_linear_test() {
  let result = activation.linear(42.0)
  result.value |> should.equal(42.0)

  let neg = activation.linear(-100.0)
  neg.value |> should.equal(-100.0)
}

pub fn activation_from_name_test() {
  activation.from_name("sigmoid") |> should.equal(activation.Sigmoid)
  activation.from_name("relu") |> should.equal(activation.ReLU)
  activation.from_name("tanh") |> should.equal(activation.Tanh)
  // Unknown defaults to ReLU
  activation.from_name("unknown") |> should.equal(activation.ReLU)
}

// =============================================================================
// IMPORT
// =============================================================================

import gleam/list
