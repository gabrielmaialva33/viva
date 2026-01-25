//// CUDA Neural Network Benchmark
////
//// Compara Pure Gleam vs Nx/CUDA para operações neurais
//// Uso: gleam run -m viva/neural/benchmark_cuda

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva/neural/activation
import viva/neural/layer
import viva/neural/network
import viva/neural/network_accelerated as accel
import viva/neural/nx_backend.{CUDA, Nx, Pure}
import viva/neural/tensor

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("")
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║         CUDA NEURAL BENCHMARK - RTX 4090 24GB                ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  Pure Gleam vs Nx/EXLA (CUDA)                                ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Check CUDA availability
  io.println("Checking CUDA...")
  let cuda_ok = nx_backend.cuda_available()
  let nx_ok = accel.nx_available()

  io.println("  Nx available:   " <> bool_str(nx_ok))
  io.println("  CUDA available: " <> bool_str(cuda_ok))
  io.println("  Backend:        " <> backend_str(accel.best_backend()))
  io.println("")

  // Initialize if available
  case nx_ok {
    True -> {
      let _ = nx_backend.init()
      io.println("  Initialized Nx backend")
    }
    False -> io.println("  Using Pure Gleam only")
  }
  io.println("")

  // Run benchmarks
  io.println(string.repeat("═", 60))
  io.println("[1/5] TENSOR OPERATIONS")
  io.println(string.repeat("─", 60))
  bench_tensor_ops()

  io.println("")
  io.println(string.repeat("═", 60))
  io.println("[2/5] MATRIX MULTIPLICATION")
  io.println(string.repeat("─", 60))
  bench_matmul()

  io.println("")
  io.println(string.repeat("═", 60))
  io.println("[3/5] NEURAL NETWORK FORWARD")
  io.println(string.repeat("─", 60))
  bench_forward()

  io.println("")
  io.println(string.repeat("═", 60))
  io.println("[4/5] BATCH PROCESSING (where GPU shines)")
  io.println(string.repeat("─", 60))
  bench_batch()

  io.println("")
  io.println(string.repeat("═", 60))
  io.println("[5/5] VIVA BRAIN (16->64->32->8)")
  io.println(string.repeat("─", 60))
  bench_viva_brain()

  io.println("")
  io.println(string.repeat("═", 60))
  io.println("BENCHMARK COMPLETE")
  io.println(string.repeat("═", 60))
}

// =============================================================================
// TENSOR OPS BENCHMARK
// =============================================================================

fn bench_tensor_ops() {
  let size = 256
  let a = tensor.random_uniform([size])
  let b = tensor.random_uniform([size])

  io.println("  Vector size: " <> int.to_string(size))
  io.println("")

  // Addition
  let #(pure_add_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.add(a, b, Pure)
      Nil
    })
  })

  let #(nx_add_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.add(a, b, Nx)
      Nil
    })
  })

  io.println("  ADD (1000 ops):")
  io.println("    Pure:  " <> fmt_us(pure_add_us) <> " (" <> fmt_ops(1000.0 /. { pure_add_us /. 1000.0 }) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_add_us) <> " (" <> fmt_ops(1000.0 /. { nx_add_us /. 1000.0 }) <> ")")
  print_speedup(pure_add_us, nx_add_us)

  // Multiplication
  let #(pure_mul_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.mul(a, b, Pure)
      Nil
    })
  })

  let #(nx_mul_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.mul(a, b, Nx)
      Nil
    })
  })

  io.println("")
  io.println("  MUL (1000 ops):")
  io.println("    Pure:  " <> fmt_us(pure_mul_us) <> " (" <> fmt_ops(1000.0 /. { pure_mul_us /. 1000.0 }) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_mul_us) <> " (" <> fmt_ops(1000.0 /. { nx_mul_us /. 1000.0 }) <> ")")
  print_speedup(pure_mul_us, nx_mul_us)

  // Softmax
  let #(pure_sm_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.softmax(a, Pure)
      Nil
    })
  })

  let #(nx_sm_us, _) = time_us(fn() {
    list.range(1, 1000)
    |> list.each(fn(_) {
      let _ = nx_backend.softmax(a, Nx)
      Nil
    })
  })

  io.println("")
  io.println("  SOFTMAX (1000 ops):")
  io.println("    Pure:  " <> fmt_us(pure_sm_us) <> " (" <> fmt_ops(1000.0 /. { pure_sm_us /. 1000.0 }) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_sm_us) <> " (" <> fmt_ops(1000.0 /. { nx_sm_us /. 1000.0 }) <> ")")
  print_speedup(pure_sm_us, nx_sm_us)
}

// =============================================================================
// MATMUL BENCHMARK
// =============================================================================

fn bench_matmul() {
  // Small matrix
  bench_matmul_size(64, 64, 100)
  // Medium matrix
  bench_matmul_size(256, 256, 50)
  // Large matrix (where GPU really shines)
  bench_matmul_size(512, 512, 20)
}

fn bench_matmul_size(rows: Int, cols: Int, iterations: Int) {
  let a = tensor.random_uniform([rows, cols])
  let b = tensor.random_uniform([cols, rows])

  io.println("")
  io.println("  MATMUL [" <> int.to_string(rows) <> "x" <> int.to_string(cols) <> "] x [" <> int.to_string(cols) <> "x" <> int.to_string(rows) <> "] (" <> int.to_string(iterations) <> " ops):")

  let #(pure_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = nx_backend.matmul(a, b, Pure)
      Nil
    })
  })

  let #(nx_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = nx_backend.matmul(a, b, Nx)
      Nil
    })
  })

  let pure_ops = int.to_float(iterations) /. { pure_us /. 1_000_000.0 }
  let nx_ops = int.to_float(iterations) /. { nx_us /. 1_000_000.0 }

  io.println("    Pure:  " <> fmt_us(pure_us) <> " (" <> fmt_ops(pure_ops) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_us) <> " (" <> fmt_ops(nx_ops) <> ")")
  print_speedup(pure_us, nx_us)
}

// =============================================================================
// FORWARD PASS BENCHMARK
// =============================================================================

fn bench_forward() {
  // Create network: 16 -> 64 -> 32 -> 8
  let assert Ok(net) = network.from_layers([
    layer.dense(16, 64, activation.ReLU),
    layer.dense(64, 32, activation.ReLU),
    layer.dense(32, 8, activation.Softmax),
  ])

  let input = tensor.random_uniform([16])
  let iterations = 500

  io.println("  Network: 16 -> 64 -> 32 -> 8")
  io.println("  Iterations: " <> int.to_string(iterations))
  io.println("")

  let #(pure_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = accel.predict_with_backend(net, input, Pure)
      Nil
    })
  })

  let #(nx_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = accel.predict_with_backend(net, input, Nx)
      Nil
    })
  })

  let pure_ops = int.to_float(iterations) /. { pure_us /. 1_000_000.0 }
  let nx_ops = int.to_float(iterations) /. { nx_us /. 1_000_000.0 }

  io.println("  FORWARD PASS:")
  io.println("    Pure:  " <> fmt_us(pure_us) <> " (" <> fmt_ops(pure_ops) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_us) <> " (" <> fmt_ops(nx_ops) <> ")")
  print_speedup(pure_us, nx_us)
}

// =============================================================================
// BATCH PROCESSING BENCHMARK
// =============================================================================

fn bench_batch() {
  // Create network
  let assert Ok(net) = network.from_layers([
    layer.dense(16, 64, activation.ReLU),
    layer.dense(64, 32, activation.ReLU),
    layer.dense(32, 8, activation.Softmax),
  ])

  // Batch sizes to test
  bench_batch_size(net, 64)
  bench_batch_size(net, 256)
  bench_batch_size(net, 1024)
}

fn bench_batch_size(net: network.Network, batch_size: Int) {
  let inputs = list.range(1, batch_size)
    |> list.map(fn(_) { tensor.random_uniform([16]) })

  io.println("")
  io.println("  BATCH SIZE: " <> int.to_string(batch_size))

  let #(pure_us, _) = time_us(fn() {
    let _ = accel.predict_batch_with_backend(net, inputs, Pure)
    Nil
  })

  let #(nx_us, _) = time_us(fn() {
    let _ = accel.predict_batch_with_backend(net, inputs, Nx)
    Nil
  })

  let pure_throughput = int.to_float(batch_size) /. { pure_us /. 1_000_000.0 }
  let nx_throughput = int.to_float(batch_size) /. { nx_us /. 1_000_000.0 }

  io.println("    Pure:  " <> fmt_us(pure_us) <> " (" <> fmt_throughput(pure_throughput) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_us) <> " (" <> fmt_throughput(nx_throughput) <> ")")
  print_speedup(pure_us, nx_us)
}

// =============================================================================
// VIVA BRAIN BENCHMARK
// =============================================================================

fn bench_viva_brain() {
  // VIVA brain: perceives 16 inputs, decides 8 outputs
  let assert Ok(brain) = accel.viva_brain(16, 8)

  let iterations = 1000
  let input = tensor.random_uniform([16])

  io.println("  VIVA Brain: 16 -> 64 -> 32 -> 8")
  io.println("  Iterations: " <> int.to_string(iterations))
  io.println("")

  let #(pure_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = accel.predict_with_backend(brain, input, Pure)
      Nil
    })
  })

  let #(nx_us, _) = time_us(fn() {
    list.range(1, iterations)
    |> list.each(fn(_) {
      let _ = accel.predict_with_backend(brain, input, Nx)
      Nil
    })
  })

  let pure_ticks = int.to_float(iterations) /. { pure_us /. 1_000_000.0 }
  let nx_ticks = int.to_float(iterations) /. { nx_us /. 1_000_000.0 }

  io.println("  VIVA DECIDE (soul-ticks/sec):")
  io.println("    Pure:  " <> fmt_us(pure_us) <> " (" <> fmt_ticks(pure_ticks) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_us) <> " (" <> fmt_ticks(nx_ticks) <> ")")
  print_speedup(pure_us, nx_us)

  // Batch decision (multiple souls thinking at once)
  io.println("")
  io.println("  BATCH DECISION (100 souls):")

  let soul_inputs = list.range(1, 100)
    |> list.map(fn(_) { tensor.random_uniform([16]) })

  let #(pure_batch_us, _) = time_us(fn() {
    let _ = accel.predict_batch_with_backend(brain, soul_inputs, Pure)
    Nil
  })

  let #(nx_batch_us, _) = time_us(fn() {
    let _ = accel.predict_batch_with_backend(brain, soul_inputs, Nx)
    Nil
  })

  let pure_batch_ticks = 100.0 /. { pure_batch_us /. 1_000_000.0 }
  let nx_batch_ticks = 100.0 /. { nx_batch_us /. 1_000_000.0 }

  io.println("    Pure:  " <> fmt_us(pure_batch_us) <> " (" <> fmt_ticks(pure_batch_ticks) <> ")")
  io.println("    Nx:    " <> fmt_us(nx_batch_us) <> " (" <> fmt_ticks(nx_batch_ticks) <> ")")
  print_speedup(pure_batch_us, nx_batch_us)
}

// =============================================================================
// UTILITIES
// =============================================================================

fn time_us(f: fn() -> a) -> #(Float, a) {
  let start = erlang_monotonic_time_micro()
  let result = f()
  let end = erlang_monotonic_time_micro()
  #(int.to_float(end - start), result)
}

@external(erlang, "erlang", "monotonic_time")
fn erlang_monotonic_time_native() -> Int

fn erlang_monotonic_time_micro() -> Int {
  // Convert native to microseconds
  let native = erlang_monotonic_time_native()
  native / 1000  // Approximate, depends on system
}

fn bool_str(b: Bool) -> String {
  case b {
    True -> "YES"
    False -> "NO"
  }
}

fn backend_str(b: nx_backend.Backend) -> String {
  case b {
    Pure -> "Pure Gleam"
    Nx -> "Nx/EXLA"
    CUDA(i) -> "CUDA:" <> int.to_string(i)
  }
}

fn fmt_us(us: Float) -> String {
  case us >. 1_000_000.0 {
    True -> fstr(us /. 1_000_000.0) <> "s"
    False -> case us >. 1000.0 {
      True -> fstr(us /. 1000.0) <> "ms"
      False -> fstr(us) <> "us"
    }
  }
  |> string.pad_start(10, " ")
}

fn fmt_ops(ops: Float) -> String {
  case ops >. 1_000_000.0 {
    True -> fstr(ops /. 1_000_000.0) <> "M ops/s"
    False -> case ops >. 1000.0 {
      True -> fstr(ops /. 1000.0) <> "K ops/s"
      False -> fstr(ops) <> " ops/s"
    }
  }
}

fn fmt_throughput(t: Float) -> String {
  case t >. 1_000_000.0 {
    True -> fstr(t /. 1_000_000.0) <> "M samples/s"
    False -> case t >. 1000.0 {
      True -> fstr(t /. 1000.0) <> "K samples/s"
      False -> fstr(t) <> " samples/s"
    }
  }
}

fn fmt_ticks(t: Float) -> String {
  case t >. 1_000_000.0 {
    True -> fstr(t /. 1_000_000.0) <> "M ticks/s"
    False -> case t >. 1000.0 {
      True -> fstr(t /. 1000.0) <> "K ticks/s"
      False -> fstr(t) <> " ticks/s"
    }
  }
}

fn print_speedup(pure_us: Float, nx_us: Float) {
  case nx_us <. pure_us {
    True -> {
      let speedup = pure_us /. nx_us
      io.println("    Speedup: " <> fstr(speedup) <> "x (Nx faster)")
    }
    False -> {
      let slowdown = nx_us /. pure_us
      io.println("    Slowdown: " <> fstr(slowdown) <> "x (Pure faster - overhead)")
    }
  }
}

fn fstr(f: Float) -> String {
  let abs_f = case f <. 0.0 {
    True -> 0.0 -. f
    False -> f
  }
  let scaled = float.round(abs_f *. 100.0)
  let int_part = scaled / 100
  let dec_part = scaled % 100
  int.to_string(int_part) <> "." <> string.pad_start(int.to_string(dec_part), 2, "0")
}
