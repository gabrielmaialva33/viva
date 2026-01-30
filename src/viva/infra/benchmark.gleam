//// VIVA Benchmarking Suite
////
//// Mede performance dos módulos do VIVA

import gleam/dict
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import gleamy/bench.{
  type BenchResults, Duration, Function, IPS, Input, Max, Mean, Min, Quiet, SD,
  Warmup,
}
import viva/lifecycle/bardo
import viva/infra/gpu
import viva/memory/memory
import viva/soul/reflexivity
import viva/soul/resonance
import viva/soul/soul
import viva/soul/soul_pool
import viva_emotion/pad
import viva_glyph/glyph

// =============================================================================
// BENCHMARK SUITE
// =============================================================================

/// Run all benchmarks
pub fn run_all() -> Nil {
  print_banner()

  io.println("\n[1/8] GLYPH SIMILARITY")
  io.println(string.repeat("─", 60))
  bench_glyph_similarity() |> print_results()

  io.println("\n[2/8] PAD OPERATIONS")
  io.println(string.repeat("─", 60))
  bench_pad() |> print_results()

  io.println("\n[3/8] SOUL ACTOR (individual)")
  io.println(string.repeat("─", 60))
  bench_soul() |> print_results()

  io.println("\n[4/8] SOUL POOL (batched) ⚡")
  io.println(string.repeat("─", 60))
  bench_soul_pool() |> print_results()

  io.println("\n[5/8] BARDO CYCLE")
  io.println(string.repeat("─", 60))
  bench_bardo() |> print_results()

  io.println("\n[6/8] RESONANCE")
  io.println(string.repeat("─", 60))
  bench_resonance() |> print_results()

  io.println("\n[7/8] REFLEXIVITY")
  io.println(string.repeat("─", 60))
  bench_reflexivity() |> print_results()

  io.println("\n[8/8] GPU STATUS")
  io.println(string.repeat("─", 60))
  bench_gpu()

  io.println("\n" <> string.repeat("═", 60))
  io.println("BENCHMARK COMPLETE")
  io.println(string.repeat("═", 60))
}

/// Run quick benchmarks
pub fn run_quick() -> Nil {
  print_banner()
  io.println("(Quick mode)\n")

  io.println("[GLYPH]")
  bench_glyph_quick() |> print_results()

  io.println("\n[PAD]")
  bench_pad_quick() |> print_results()

  io.println("\n[SOUL ACTOR]")
  bench_soul_quick() |> print_results()

  io.println("\n[SOUL POOL] ⚡")
  bench_soul_pool_quick() |> print_results()

  io.println("\nDone.")
}

fn bench_soul_pool_quick() -> BenchResults {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 100)

  let results =
    bench.run(
      [Input("pool_100", pool)],
      [
        Function("tick_all", fn(p) {
          soul_pool.tick_all(p, 0.1)
          soul_pool.count(p)
        }),
      ],
      [Warmup(200), Duration(500), Quiet],
    )

  soul_pool.kill_all(pool)
  results
}

// =============================================================================
// BENCHMARKS - Each returns same type per benchmark
// =============================================================================

fn bench_glyph_similarity() -> BenchResults {
  let small = glyph.new([100, 150, 200, 50])
  let large = glyph.new([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

  let inputs = [Input("small_4tok", small), Input("large_10tok", large)]

  // All return Float
  let functions = [
    Function("similarity", fn(g) { glyph.similarity(g, small) }),
    Function("weighted_sim", fn(g) { glyph.weighted_similarity(g, small) }),
  ]

  bench.run(inputs, functions, [Warmup(500), Duration(2000), Quiet])
}

fn bench_glyph_quick() -> BenchResults {
  let g = glyph.new([100, 150, 200, 50])
  bench.run(
    [Input("glyph", g)],
    [Function("similarity", fn(g) { glyph.similarity(g, g) })],
    [Warmup(200), Duration(500), Quiet],
  )
}

fn bench_pad() -> BenchResults {
  let neutral = pad.neutral()
  let excited = pad.new(0.8, 0.9, 0.5)

  let inputs = [Input("neutral", neutral), Input("excited", excited)]

  // All return Pad
  let functions = [
    Function("add", fn(p) { pad.add(p, excited) }),
    Function("scale", fn(p) { pad.scale(p, 0.5) }),
    Function("lerp", fn(p) { pad.lerp(p, excited, 0.3) }),
  ]

  bench.run(inputs, functions, [Warmup(500), Duration(2000), Quiet])
}

fn bench_pad_quick() -> BenchResults {
  let p = pad.neutral()
  bench.run(
    [Input("pad", p)],
    [Function("lerp", fn(p) { pad.lerp(p, p, 0.5) })],
    [Warmup(200), Duration(500), Quiet],
  )
}

fn bench_soul() -> BenchResults {
  let assert Ok(s) = soul.start(1)

  // All return Pad (use get_pad for consistent type)
  let results =
    bench.run(
      [Input("soul", s)],
      [
        Function("get_pad", fn(s) { soul.get_pad(s) }),
        Function("tick+get", fn(s) {
          soul.tick(s, 0.1)
          soul.get_pad(s)
        }),
      ],
      [Warmup(500), Duration(2000), Quiet],
    )

  soul.kill(s)
  results
}

fn bench_soul_quick() -> BenchResults {
  let assert Ok(s) = soul.start(99)
  let results =
    bench.run(
      [Input("soul", s)],
      [
        Function("get_pad", fn(s) { soul.get_pad(s) }),
        Function("tick+get", fn(s) {
          soul.tick(s, 0.1)
          soul.get_pad(s)
        }),
      ],
      [Warmup(200), Duration(500), Quiet],
    )
  soul.kill(s)
  results
}

fn bench_soul_pool() -> BenchResults {
  let assert Ok(pool) = soul_pool.start()

  // Spawn 100 souls
  let _ids = soul_pool.spawn_many(pool, 100)

  // Benchmark batch operations
  let results =
    bench.run(
      [Input("pool_100", pool)],
      [
        // Single message ticks ALL 100 souls
        Function("tick_100", fn(p) {
          soul_pool.tick_all(p, 0.1)
          soul_pool.count(p)
        }),
        // Single message gets ALL 100 PADs
        Function("get_100_pads", fn(p) { dict.size(soul_pool.get_all_pads(p)) }),
      ],
      [Warmup(500), Duration(2000), Quiet],
    )

  soul_pool.kill_all(pool)
  results
}

/// Comparison benchmark: equivalent work
pub fn bench_comparison() -> Nil {
  io.println("\n╔════════════════════════════════════════════════════════════╗")
  io.println("║           SOUL vs SOUL POOL COMPARISON                     ║")
  io.println("╚════════════════════════════════════════════════════════════╝\n")

  // Test with different scales
  bench_scale(10)
  bench_scale(50)
  bench_scale(100)

  io.println("\n" <> string.repeat("═", 60))
  io.println("Pool wins at scale: fewer messages = less overhead")
}

fn bench_scale(n: Int) -> Nil {
  io.println("\n── " <> int.to_string(n) <> " SOULS ──")

  // Soul Actor approach: n individual actors
  let souls =
    list.range(1, n)
    |> list.filter_map(fn(i) {
      case soul.start(i) {
        Ok(s) -> Ok(s)
        Error(_) -> Error(Nil)
      }
    })

  // Fair comparison: both use async tick + sync read
  let actor_results =
    bench.run(
      [Input("actors", souls)],
      [
        Function("tick+read", fn(souls_list) {
          // Tick all (async)
          list.each(souls_list, fn(s) { soul.tick(s, 0.1) })
          // Read all (sync) - this is the fair comparison
          list.each(souls_list, fn(s) { soul.get_pad(s) })
          list.length(souls_list)
        }),
      ],
      [Warmup(100), Duration(500), Quiet],
    )

  list.each(souls, fn(s) { soul.kill(s) })

  // Soul Pool approach: 1 actor with n souls
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, n)

  let pool_results =
    bench.run(
      [Input("pool", pool)],
      [
        Function("tick+read", fn(p) {
          // Tick all (async) - 1 message for n souls
          soul_pool.tick_all(p, 0.1)
          // Read all (sync) - 1 message for n pads
          dict.size(soul_pool.get_all_pads(p))
        }),
      ],
      [Warmup(100), Duration(500), Quiet],
    )

  soul_pool.kill_all(pool)

  // Extract IPS for comparison
  let actor_ips = get_first_ips(actor_results)
  let pool_ips = get_first_ips(pool_results)
  let speedup = pool_ips /. actor_ips

  io.println(
    "  Actors: "
    <> fmt(actor_ips)
    <> " ops/sec (n="
    <> int.to_string(n)
    <> " msgs/tick)",
  )
  io.println("  Pool:   " <> fmt(pool_ips) <> " ops/sec (2 msgs/tick) ⚡")
  io.println("  Speedup: " <> fstr(speedup) <> "x")
}

fn get_first_ips(results: BenchResults) -> Float {
  case list.first(results.sets) {
    Ok(set) -> {
      let sum = float.sum(set.reps)
      let count = int.to_float(list.length(set.reps))
      case sum >. 0.0 {
        True -> 1000.0 *. count /. sum
        False -> 0.0
      }
    }
    Error(_) -> 0.0
  }
}

fn bench_bardo() -> BenchResults {
  let final_glyph = glyph.new([200, 100, 50, 150])
  let bank = memory.new(1)

  // Return type: #(LiberationOutcome, List(BardoPhase))
  bench.run(
    [Input("bardo", #(final_glyph, bank))],
    [
      Function("full_cycle", fn(state) {
        let #(g, b) = state
        bardo.run_bardo_cycle(g, b)
      }),
    ],
    [Warmup(500), Duration(2000), Quiet],
  )
}

fn bench_resonance() -> BenchResults {
  let v1 =
    resonance.VivaState(
      id: 1,
      pad: pad.new(0.5, 0.3, 0.2),
      glyph: glyph.new([100, 150, 200, 50]),
      alive: True,
      tick: 100,
    )

  let v2 =
    resonance.VivaState(
      id: 2,
      pad: pad.new(-0.3, 0.7, 0.1),
      glyph: glyph.new([120, 140, 180, 60]),
      alive: True,
      tick: 100,
    )

  // All return Float
  bench.run(
    [Input("vivas", #(v1, v2))],
    [
      Function("calculate", fn(state) {
        let #(a, b) = state
        resonance.calculate_resonance(a, b)
      }),
    ],
    [Warmup(500), Duration(2000), Quiet],
  )
}

fn bench_reflexivity() -> BenchResults {
  let sm = reflexivity.new()
  let p = pad.new(0.5, 0.3, -0.2)
  let g = glyph.new([100, 150, 200, 50])

  // Return Introspection
  bench.run(
    [Input("self", #(sm, p, g))],
    [
      Function("introspect", fn(state) {
        let #(sm, p, g) = state
        reflexivity.introspect(sm, p, g, 100)
      }),
    ],
    [Warmup(500), Duration(2000), Quiet],
  )
}

// =============================================================================
// OUTPUT
// =============================================================================

fn print_banner() -> Nil {
  io.println("")
  io.println("╔════════════════════════════════════════════════════════════╗")
  io.println("║           VIVA BENCHMARK SUITE                             ║")
  io.println("║           Measuring Consciousness Performance              ║")
  io.println("╚════════════════════════════════════════════════════════════╝")
  io.println("")
}

fn print_results(results: BenchResults) -> Nil {
  let table =
    bench.table(results, [IPS, Mean, Min, Max, SD])
    |> string.split("\n")
    |> list.map(fn(line) { "  " <> line })
    |> string.join("\n")
  io.println(table)
}

// =============================================================================
// METRICS
// =============================================================================

pub type BenchMetrics {
  BenchMetrics(
    glyph_ips: Float,
    pad_ips: Float,
    soul_ips: Float,
    bardo_ips: Float,
    resonance_ips: Float,
    reflexivity_ips: Float,
  )
}

fn avg_ips(results: BenchResults) -> Float {
  let total =
    list.fold(results.sets, 0.0, fn(acc, set) {
      let sum = float.sum(set.reps)
      let count = int.to_float(list.length(set.reps))
      case sum >. 0.0 {
        True -> acc +. { 1000.0 *. count /. sum }
        False -> acc
      }
    })
  let len = list.length(results.sets)
  case len > 0 {
    True -> total /. int.to_float(len)
    False -> 0.0
  }
}

pub fn collect_metrics() -> BenchMetrics {
  let opts = [Warmup(200), Duration(500), Quiet]

  let g = glyph.new([100, 150, 200, 50])
  let glyph_r =
    bench.run(
      [Input("g", g)],
      [Function("sim", fn(g) { glyph.similarity(g, g) })],
      opts,
    )

  let p = pad.neutral()
  let pad_r =
    bench.run(
      [Input("p", p)],
      [Function("lerp", fn(p) { pad.lerp(p, p, 0.5) })],
      opts,
    )

  let assert Ok(s) = soul.start(999)
  let soul_r =
    bench.run(
      [Input("s", s)],
      [
        Function("tick", fn(s) {
          soul.tick(s, 0.1)
          soul.get_pad(s)
        }),
      ],
      opts,
    )
  soul.kill(s)

  let bank = memory.new(1)
  let bardo_r =
    bench.run(
      [Input("b", #(g, bank))],
      [
        Function("cycle", fn(state) {
          let #(glyph, bank) = state
          bardo.run_bardo_cycle(glyph, bank)
        }),
      ],
      opts,
    )

  let v = resonance.VivaState(id: 1, pad: p, glyph: g, alive: True, tick: 0)
  let res_r =
    bench.run(
      [Input("r", v)],
      [Function("calc", fn(v) { resonance.calculate_resonance(v, v) })],
      opts,
    )

  let sm = reflexivity.new()
  let ref_r =
    bench.run(
      [Input("rf", #(sm, p, g))],
      [
        Function("intro", fn(state) {
          let #(sm, p, g) = state
          reflexivity.introspect(sm, p, g, 100)
        }),
      ],
      opts,
    )

  BenchMetrics(
    glyph_ips: avg_ips(glyph_r),
    pad_ips: avg_ips(pad_r),
    soul_ips: avg_ips(soul_r),
    bardo_ips: avg_ips(bardo_r),
    resonance_ips: avg_ips(res_r),
    reflexivity_ips: avg_ips(ref_r),
  )
}

pub fn print_metrics(m: BenchMetrics) -> Nil {
  io.println("\n┌────────────────────────────────────────┐")
  io.println("│       VIVA PERFORMANCE METRICS         │")
  io.println("├────────────────────────────────────────┤")
  io.println("│ Glyph:       " <> fmt(m.glyph_ips) <> " ops/sec │")
  io.println("│ PAD:         " <> fmt(m.pad_ips) <> " ops/sec │")
  io.println("│ Soul tick:   " <> fmt(m.soul_ips) <> " ops/sec │")
  io.println("│ Bardo:       " <> fmt(m.bardo_ips) <> " ops/sec │")
  io.println("│ Resonance:   " <> fmt(m.resonance_ips) <> " ops/sec │")
  io.println("│ Reflexivity: " <> fmt(m.reflexivity_ips) <> " ops/sec │")
  io.println("└────────────────────────────────────────┘")
}

fn fmt(ips: Float) -> String {
  let str = case ips >. 1_000_000.0 {
    True -> fstr(ips /. 1_000_000.0) <> "M"
    False ->
      case ips >. 1000.0 {
        True -> fstr(ips /. 1000.0) <> "K"
        False -> fstr(ips)
      }
  }
  string.pad_start(str, 12, " ")
}

fn fstr(f: Float) -> String {
  let abs_f = case f <. 0.0 {
    True -> 0.0 -. f
    False -> f
  }
  let scaled = float.round(abs_f *. 100.0)
  let int_part = scaled / 100
  let dec_part = scaled % 100
  int.to_string(int_part)
  <> "."
  <> string.pad_start(int.to_string(dec_part), 2, "0")
}

// =============================================================================
// GPU BENCHMARK
// =============================================================================

fn bench_gpu() -> Nil {
  let backend = gpu.detect()

  case backend {
    gpu.GPU -> {
      io.println("  Backend: EXLA/CUDA (RTX 4090)")
      gpu.print_status()
      bench_gpu_ops()
    }
    gpu.ExlaCpu -> {
      io.println("  Backend: EXLA/CPU")
      io.println("  (GPU not detected, using CPU EXLA)")
    }
    gpu.CPU -> {
      io.println("  Backend: Pure Gleam (CPU)")
      io.println("  (EXLA not available)")
    }
  }
}

fn bench_gpu_ops() -> Nil {
  // Benchmark batch PAD operations
  let pads =
    list.range(1, 100)
    |> list.map(fn(i) {
      let f = int.to_float(i) /. 100.0
      pad.new(f, 0.0 -. f, f *. 0.5)
    })

  let ids = list.range(1, 100)
  let pad_dict = list.zip(ids, pads) |> dict.from_list()
  let batch = gpu.pads_to_batch(pad_dict)
  let delta = pad.new(0.1, 0.1, 0.1)

  // Benchmark CPU vs GPU
  let cpu_results =
    bench.run(
      [Input("cpu_100", #(batch, delta))],
      [
        Function("apply_delta", fn(input) {
          let #(b, d) = input
          gpu.batch_apply_delta(b, d, gpu.CPU)
        }),
      ],
      [Warmup(100), Duration(500), Quiet],
    )

  let gpu_results =
    bench.run(
      [Input("gpu_100", #(batch, delta))],
      [
        Function("apply_delta", fn(input) {
          let #(b, d) = input
          gpu.batch_apply_delta(b, d, gpu.GPU)
        }),
      ],
      [Warmup(100), Duration(500), Quiet],
    )

  let cpu_ips = get_first_ips(cpu_results)
  let gpu_ips = get_first_ips(gpu_results)

  io.println("\n  Batch PAD (100 souls):")
  io.println("    CPU: " <> fmt(cpu_ips) <> " ops/sec")
  io.println("    GPU: " <> fmt(gpu_ips) <> " ops/sec")

  case gpu_ips >. cpu_ips {
    True -> io.println("    Speedup: " <> fstr(gpu_ips /. cpu_ips) <> "x ⚡")
    False -> io.println("    (CPU faster for small batch)")
  }
}
