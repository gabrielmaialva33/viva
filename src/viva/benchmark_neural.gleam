// VIVA Neural Benchmarks - gleamy_bench
// Statistical validation for thesis defense

import gleam/io
import gleam/list
import gleam/result
import gleam/string
import gleamy/bench
import viva/neural/activation
import viva/neural/layer
import viva/neural/network
import viva/neural/tensor

// =============================================================================
// TENSOR BENCHMARKS
// =============================================================================

fn benchmark_tensor_ops() {
  io.println("\n=== TENSOR OPERATIONS ===")

  let small = tensor.zeros([64])
  let medium = tensor.zeros([256])
  let large = tensor.zeros([1024])

  bench.run(
    [
      bench.Input("64 elements", small),
      bench.Input("256 elements", medium),
      bench.Input("1024 elements", large),
    ],
    [
      bench.Function("add", fn(t) {
        tensor.add(t, t) |> result.unwrap(t)
      }),
      bench.Function("mul", fn(t) {
        tensor.mul(t, t) |> result.unwrap(t)
      }),
      bench.Function("scale", fn(t) { tensor.scale(t, 2.0) }),
    ],
    [bench.Duration(1000), bench.Warmup(50)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

fn benchmark_tensor_reduce() {
  io.println("\n=== TENSOR REDUCTIONS ===")

  let t1024 = tensor.random_uniform([1024])

  // Note: tensor.sum/mean/max/min return Float, not Tensor
  // Using Float result wrapped in a dummy operation for benchmarking
  bench.run(
    [bench.Input("1024 elements", t1024)],
    [
      bench.Function("sum", fn(t) {
        let _ = tensor.sum(t)
        t
      }),
      bench.Function("mean", fn(t) {
        let _ = tensor.mean(t)
        t
      }),
      bench.Function("max", fn(t) {
        let _ = tensor.max(t)
        t
      }),
      bench.Function("min", fn(t) {
        let _ = tensor.min(t)
        t
      }),
      bench.Function("variance", fn(t) {
        let _ = tensor.variance(t)
        t
      }),
    ],
    [bench.Duration(1000), bench.Warmup(50)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

fn benchmark_matmul() {
  io.println("\n=== MATRIX MULTIPLICATION ===")

  // Create matrices using tensor.matrix(rows, cols, data)
  let mat_8x8 =
    tensor.matrix(8, 8, list.repeat(0.5, 64)) |> result.unwrap(tensor.zeros([8, 8]))
  let mat_16x16 =
    tensor.matrix(16, 16, list.repeat(0.5, 256))
    |> result.unwrap(tensor.zeros([16, 16]))
  let mat_32x32 =
    tensor.matrix(32, 32, list.repeat(0.5, 1024))
    |> result.unwrap(tensor.zeros([32, 32]))

  bench.run(
    [
      bench.Input("8x8", mat_8x8),
      bench.Input("16x16", mat_16x16),
      bench.Input("32x32", mat_32x32),
    ],
    [
      bench.Function("matmul", fn(t) {
        tensor.matmul(t, t) |> result.unwrap(t)
      }),
      bench.Function("transpose", fn(t) {
        tensor.transpose(t) |> result.unwrap(t)
      }),
    ],
    [bench.Duration(2000), bench.Warmup(100)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

// =============================================================================
// ACTIVATION BENCHMARKS
// =============================================================================

fn benchmark_activations() {
  io.println("\n=== ACTIVATION FUNCTIONS ===")

  let t256 = tensor.random_uniform([256])

  bench.run(
    [bench.Input("256 elements", t256)],
    [
      // activation.forward applies activation to tensor
      bench.Function("relu", fn(t) { activation.forward(t, activation.ReLU) }),
      bench.Function("sigmoid", fn(t) {
        activation.forward(t, activation.Sigmoid)
      }),
      bench.Function("tanh", fn(t) { activation.forward(t, activation.Tanh) }),
      bench.Function("softmax", fn(t) { activation.softmax(t) }),
      bench.Function("leaky_relu", fn(t) {
        activation.forward(t, activation.LeakyReLU(0.01))
      }),
    ],
    [bench.Duration(1000), bench.Warmup(50)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

// =============================================================================
// LAYER BENCHMARKS
// =============================================================================

fn benchmark_dense_layer() {
  io.println("\n=== DENSE LAYER FORWARD ===")

  let input_64 = tensor.random_uniform([64])
  let input_256 = tensor.random_uniform([256])

  // layer.dense takes 3 args: input_size, output_size, activation
  let layer_64_32 = layer.dense(64, 32, activation.ReLU)
  let layer_256_64 = layer.dense(256, 64, activation.ReLU)

  bench.run(
    [
      bench.Input("64->32", #(input_64, layer_64_32)),
      bench.Input("256->64", #(input_256, layer_256_64)),
    ],
    [
      bench.Function("forward", fn(pair) {
        let #(input, l) = pair
        // layer.forward returns Result(#(Tensor, Cache), Error)
        case layer.forward(l, input) {
          Ok(#(output, _)) -> output
          Error(_) -> input
        }
      }),
    ],
    [bench.Duration(2000), bench.Warmup(100)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

// =============================================================================
// NETWORK BENCHMARKS
// =============================================================================

fn benchmark_network() {
  io.println("\n=== NETWORK FORWARD PASS ===")

  let input = tensor.random_uniform([64])

  // network.builder creates a builder, add_dense adds layers, build finalizes
  let net_shallow =
    network.builder(64)
    |> network.add_dense(32, activation.ReLU)
    |> network.add_dense(10, activation.Linear)
    |> network.build()
    |> result.unwrap(network.Network(layers: [], input_size: 64, output_size: 10))

  let net_deep =
    network.builder(64)
    |> network.add_dense(48, activation.ReLU)
    |> network.add_dense(32, activation.ReLU)
    |> network.add_dense(16, activation.ReLU)
    |> network.add_dense(10, activation.Linear)
    |> network.build()
    |> result.unwrap(network.Network(layers: [], input_size: 64, output_size: 10))

  bench.run(
    [
      bench.Input("2 layers (64->32->10)", #(input, net_shallow)),
      bench.Input("4 layers (64->48->32->16->10)", #(input, net_deep)),
    ],
    [
      bench.Function("forward", fn(pair) {
        let #(inp, net) = pair
        // network.forward returns Result(#(Tensor, Cache), Error)
        case network.forward(net, inp) {
          Ok(#(output, _)) -> output
          Error(_) -> inp
        }
      }),
    ],
    [bench.Duration(2000), bench.Warmup(100)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

// =============================================================================
// TENSOR ADVANCED OPS
// =============================================================================

fn benchmark_tensor_advanced() {
  io.println("\n=== TENSOR ADVANCED OPS ===")

  let t1024 = tensor.random_uniform([1024])

  bench.run(
    [bench.Input("1024 elements", t1024)],
    [
      bench.Function("normalize", fn(t) { tensor.normalize(t) }),
      bench.Function("clamp", fn(t) { tensor.clamp(t, 0.0, 1.0) }),
      bench.Function("negate", fn(t) { tensor.negate(t) }),
    ],
    [bench.Duration(1000), bench.Warmup(50)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

fn benchmark_tensor_search() {
  io.println("\n=== TENSOR SEARCH OPS ===")

  let t1024 = tensor.random_uniform([1024])

  bench.run(
    [bench.Input("1024 elements", t1024)],
    [
      bench.Function("argmax", fn(t) {
        let _ = tensor.argmax(t)
        t
      }),
      bench.Function("argmin", fn(t) {
        let _ = tensor.argmin(t)
        t
      }),
    ],
    [bench.Duration(1000), bench.Warmup(50)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.P(99)])
  |> io.println()
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println(string.repeat("=", 70))
  io.println("  VIVA NEURAL BENCHMARKS - Thesis Defense Metrics")
  io.println(string.repeat("=", 70))

  benchmark_tensor_ops()
  benchmark_tensor_reduce()
  benchmark_matmul()
  benchmark_activations()
  benchmark_dense_layer()
  benchmark_network()
  benchmark_tensor_advanced()
  benchmark_tensor_search()

  io.println("\n" <> string.repeat("=", 70))
  io.println("  BENCHMARK COMPLETE - Use for thesis statistics")
  io.println(string.repeat("=", 70))
}
