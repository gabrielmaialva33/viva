//// VIVA Burn Test - Quick benchmark of batch neural forward

import gleam/io
import gleam/int
import gleam/float
import gleam/list
import viva/lifecycle/burn

pub fn main() {
  io.println("=== VIVA Burn GPU Test ===")
  io.println("")

  // Check status
  io.println("Status: " <> burn.check())
  io.println("")

  // Test batch forward
  io.println("Testing batch forward...")

  let architecture = [8, 32, 16, 3]
  let weight_count = burn.weight_count(architecture)
  io.println("Architecture: [8, 32, 16, 3]")
  io.println("Weight count: " <> int.to_string(weight_count))

  // Create test data
  let pop_size = 100
  let weights_list = generate_weights(pop_size, weight_count, 42)
  let inputs_list = generate_inputs(pop_size, 8, 123)

  io.println("Population: " <> int.to_string(pop_size))
  io.println("")

  // Run batch forward
  case burn.batch_forward(weights_list, inputs_list, architecture) {
    Ok(results) -> {
      io.println("Batch forward SUCCESS!")
      io.println("Results count: " <> int.to_string(list.length(results)))

      // Show first result
      case results {
        [first, ..] -> {
          io.println("First output: " <> format_floats(first))
        }
        [] -> io.println("Empty results")
      }
    }
    Error(e) -> {
      io.println("Batch forward FAILED: " <> e)
    }
  }

  io.println("")

  // Run benchmark
  io.println("Running benchmark...")
  let bench_result = burn.benchmark(1000, 8, 32, 3, 100)
  io.println(bench_result)
}

fn generate_weights(count: Int, size: Int, seed: Int) -> List(List(Float)) {
  generate_list(count, fn(i) {
    generate_floats(size, seed + i * 1000)
  })
}

fn generate_inputs(count: Int, size: Int, seed: Int) -> List(List(Float)) {
  generate_list(count, fn(i) {
    generate_floats(size, seed + i * 100)
  })
}

fn generate_list(n: Int, f: fn(Int) -> a) -> List(a) {
  do_generate_list(n, 0, f, [])
}

fn do_generate_list(n: Int, i: Int, f: fn(Int) -> a, acc: List(a)) -> List(a) {
  case i >= n {
    True -> list.reverse(acc)
    False -> do_generate_list(n, i + 1, f, [f(i), ..acc])
  }
}

fn generate_floats(n: Int, seed: Int) -> List(Float) {
  do_generate_floats(n, seed, [])
}

fn do_generate_floats(n: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case n <= 0 {
    True -> list.reverse(acc)
    False -> {
      let next = { seed * 1103515245 + 12345 } % 2147483648
      let value = int.to_float(next % 1000 - 500) /. 1000.0
      do_generate_floats(n - 1, next, [value, ..acc])
    }
  }
}

fn format_floats(lst: List(Float)) -> String {
  "[" <> format_floats_inner(lst) <> "]"
}

fn format_floats_inner(lst: List(Float)) -> String {
  case lst {
    [] -> ""
    [x] -> float_to_string(x)
    [x, ..rest] -> float_to_string(x) <> ", " <> format_floats_inner(rest)
  }
}

fn float_to_string(x: Float) -> String {
  let scaled = float.truncate(x *. 1000.0)
  let whole = scaled / 1000
  let frac = int.absolute_value(scaled % 1000)
  let sign = case x <. 0.0 {
    True -> "-"
    False -> ""
  }
  sign <> int.to_string(int.absolute_value(whole)) <> "." <> pad_zeros(frac, 3)
}

fn pad_zeros(n: Int, width: Int) -> String {
  let s = int.to_string(n)
  let len = string_length(s)
  case len >= width {
    True -> s
    False -> pad_zeros_left(s, width - len)
  }
}

fn pad_zeros_left(s: String, n: Int) -> String {
  case n <= 0 {
    True -> s
    False -> pad_zeros_left("0" <> s, n - 1)
  }
}

@external(erlang, "string", "length")
fn string_length(s: String) -> Int
