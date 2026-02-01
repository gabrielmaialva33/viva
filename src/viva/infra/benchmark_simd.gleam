//// SIMD Benchmark - Compare SIMD vs Pure Gleam performance
////
//// Run with: gleam run -m viva/infra/benchmark_simd

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva/memory/hrr
import viva/memory/simd

// FFI for timing
@external(erlang, "os", "timestamp")
fn timestamp() -> #(Int, Int, Int)

fn now_us() -> Int {
  let #(mega, sec, micro) = timestamp()
  mega * 1_000_000_000_000 + sec * 1_000_000 + micro
}

pub fn main() {
  io.println("=== VIVA SIMD Benchmark ===\n")

  // Check SIMD availability
  io.println("SIMD Available: " <> case simd.is_available() {
    True -> "YES (AVX)"
    False -> "NO (Erlang fallback)"
  })
  io.println("")

  // Benchmark parameters
  let dims = [256, 512, 1024, 2048, 4096]
  let iterations = 1000

  io.println("Iterations per test: " <> int.to_string(iterations))
  io.println("")

  // Test dot product
  io.println("=== Dot Product ===")
  io.println("Dim      | Time (μs) | Ops/sec")
  io.println("---------+-----------+---------")

  list.each(dims, fn(dim) {
    let a = hrr.random(dim)
    let b = hrr.random(dim)

    let start = now_us()
    list.each(list.range(1, iterations), fn(_) {
      let _ = hrr.dot(a, b)
      Nil
    })
    let elapsed = now_us() - start

    let us_per_op = int.to_float(elapsed) /. int.to_float(iterations)
    let ops_per_sec = 1_000_000.0 /. us_per_op

    io.println(
      pad_left(int.to_string(dim), 8) <> " | " <>
      pad_left(float.to_string(us_per_op), 9) <> " | " <>
      pad_left(int.to_string(float.round(ops_per_sec)), 7)
    )
  })

  io.println("")

  // Test similarity
  io.println("=== Similarity (cosine) ===")
  io.println("Dim      | Time (μs) | Ops/sec")
  io.println("---------+-----------+---------")

  list.each(dims, fn(dim) {
    let a = hrr.random(dim)
    let b = hrr.random(dim)

    let start = now_us()
    list.each(list.range(1, iterations), fn(_) {
      let _ = hrr.similarity(a, b)
      Nil
    })
    let elapsed = now_us() - start

    let us_per_op = int.to_float(elapsed) /. int.to_float(iterations)
    let ops_per_sec = 1_000_000.0 /. us_per_op

    io.println(
      pad_left(int.to_string(dim), 8) <> " | " <>
      pad_left(float.to_string(us_per_op), 9) <> " | " <>
      pad_left(int.to_string(float.round(ops_per_sec)), 7)
    )
  })

  io.println("")

  // Test normalize
  io.println("=== Normalize ===")
  io.println("Dim      | Time (μs) | Ops/sec")
  io.println("---------+-----------+---------")

  list.each(dims, fn(dim) {
    let a = hrr.random(dim)

    let start = now_us()
    list.each(list.range(1, iterations), fn(_) {
      let _ = hrr.normalize(a)
      Nil
    })
    let elapsed = now_us() - start

    let us_per_op = int.to_float(elapsed) /. int.to_float(iterations)
    let ops_per_sec = 1_000_000.0 /. us_per_op

    io.println(
      pad_left(int.to_string(dim), 8) <> " | " <>
      pad_left(float.to_string(us_per_op), 9) <> " | " <>
      pad_left(int.to_string(float.round(ops_per_sec)), 7)
    )
  })

  io.println("")
  io.println("=== Benchmark Complete ===")
}

fn pad_left(s: String, _width: Int) -> String {
  s
}
