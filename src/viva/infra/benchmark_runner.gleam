//// VIVA Benchmark Runner
////
//// Entry point for running standard benchmarks from CLI.
////
//// Usage:
////   gleam run -m viva/infra/benchmark_runner
////   gleam run -m viva/infra/benchmark_runner -- quick
////   gleam run -m viva/infra/benchmark_runner -- full

import gleam/list
import viva/infra/benchmark_standard
import viva/infra/environments/environment.{type BenchmarkMetrics}
import viva_telemetry/log

pub fn main() {
  let args = get_args()

  case list.first(args) {
    Ok("quick") -> {
      log.info("Running QUICK benchmark...", [])
      let _ = benchmark_standard.run_quick()
      Nil
    }
    Ok("full") -> {
      log.info("Running FULL benchmark...", [])
      let _ = benchmark_standard.run_all()
      Nil
    }
    Ok("cartpole") -> {
      log.info("Running CartPole benchmark only...", [])
      let config = benchmark_standard.fast_config()
      let metrics = benchmark_standard.benchmark_cartpole(config)
      print_single_metrics(metrics)
    }
    Ok("pendulum") -> {
      log.info("Running Pendulum benchmark only...", [])
      let config = benchmark_standard.fast_config()
      let metrics = benchmark_standard.benchmark_pendulum(config)
      print_single_metrics(metrics)
    }
    _ -> {
      log.info("VIVA Benchmark Runner", [])
      log.info("=====================", [])
      log.info("", [])
      log.info("Usage:", [])
      log.info(
        "  gleam run -m viva/infra/benchmark_runner -- quick     # Fast benchmarks",
        [],
      )
      log.info(
        "  gleam run -m viva/infra/benchmark_runner -- full      # Full benchmarks",
        [],
      )
      log.info(
        "  gleam run -m viva/infra/benchmark_runner -- cartpole  # CartPole only",
        [],
      )
      log.info(
        "  gleam run -m viva/infra/benchmark_runner -- pendulum  # Pendulum only",
        [],
      )
      log.info("", [])
      log.info("No mode provided, running quick benchmark by default...", [])
      log.info("Running quick benchmark by default...", [])
      let _ = benchmark_standard.run_quick()
      Nil
    }
  }
}

fn print_single_metrics(m: BenchmarkMetrics) -> Nil {
  log.info("", [])
  log.info("Results:", [])
  log.info("  Environment:  " <> m.env_name, [])
  log.info("  Evals/sec:    " <> format_number(m.evals_per_sec), [])
  log.info("  Steps/sec:    " <> format_number(m.steps_per_sec), [])
  log.info("  Wall time:    " <> float_str(m.wall_time) <> " sec", [])
  log.info("  Total evals:  " <> int_str(m.num_evals), [])
  log.info("  Final return: " <> float_str(m.final_return), [])
}

fn format_number(n: Float) -> String {
  case n >=. 1_000_000.0 {
    True -> float_str(n /. 1_000_000.0) <> "M"
    False ->
      case n >=. 1000.0 {
        True -> float_str(n /. 1000.0) <> "K"
        False -> float_str(n)
      }
  }
}

fn float_str(f: Float) -> String {
  let scaled = float_round(f *. 100.0)
  let int_part = scaled / 100
  let dec_part = int_abs(scaled % 100)
  int_str(int_part) <> "." <> pad_left(int_str(dec_part), 2)
}

fn int_str(i: Int) -> String {
  case i < 0 {
    True -> "-" <> do_int_str(0 - i, "")
    False -> do_int_str(i, "")
  }
}

fn do_int_str(i: Int, acc: String) -> String {
  case i < 10 {
    True -> digit_char(i) <> acc
    False -> do_int_str(i / 10, digit_char(i % 10) <> acc)
  }
}

fn digit_char(d: Int) -> String {
  case d {
    0 -> "0"
    1 -> "1"
    2 -> "2"
    3 -> "3"
    4 -> "4"
    5 -> "5"
    6 -> "6"
    7 -> "7"
    8 -> "8"
    _ -> "9"
  }
}

fn pad_left(s: String, width: Int) -> String {
  let len = string_length(s)
  case len >= width {
    True -> s
    False -> "0" <> pad_left(s, width - 1)
  }
}

fn int_abs(i: Int) -> Int {
  case i < 0 {
    True -> 0 - i
    False -> i
  }
}

@external(erlang, "erlang", "round")
fn float_round(f: Float) -> Int

@external(erlang, "init", "get_plain_arguments")
fn get_args() -> List(String)

@external(erlang, "string", "length")
fn string_length(s: String) -> Int
