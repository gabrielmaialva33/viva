import gleam/erlang/process
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import gleamy/bench.{
  type BenchResults, Duration, Function, IPS, Input, Max, Mean, Min, Quiet, SD,
  Warmup,
}
import viva/neural_swarm

pub fn main() {
  io.println("\n╔════════════════════════════════════════════════════════════╗")
  io.println("║           NEURAL SWARM GPU BENCHMARK                       ║")
  io.println("║           Jolt/Vampire Survivors Architecture              ║")
  io.println("╚════════════════════════════════════════════════════════════╝\n")

  // Scenario 1: Small (100)
  bench_scenario(100)

  // Scenario 2: Medium (1,000)
  bench_scenario(1000)

  // Scenario 3: Large (5,000)
  bench_scenario(5000)

  // Scenario 4: Massive (10,000)
  bench_scenario(10_000)
}

fn bench_scenario(count: Int) {
  io.println("\n── " <> int.to_string(count) <> " BOIDS ──")

  // Initialize Actor (One-time cost, not measured)
  let assert Ok(swarm) = neural_swarm.start(count)

  // Warmup to compile shaders (JIT)
  neural_swarm.tick(swarm, 0.016)

  let results =
    bench.run(
      [Input("neural_swarm", swarm)],
      [
        Function("tick(16ms)", fn(s) {
          neural_swarm.tick(s, 0.016)
          // Ensure sync? tick is async (cast).
          // To measure throughput we usually want to sync or flood.
          // Getting count forces a sync roundtrip.
          neural_swarm.get_count(s)
        }),
      ],
      [Warmup(100), Duration(2000), Quiet],
    )

  print_results(results)

  // Cleanup
  let assert Ok(pid) = process.subject_owner(swarm)
  process.kill(pid)
}

fn print_results(results: BenchResults) -> Nil {
  let table =
    bench.table(results, [IPS, Mean, Min, Max, SD])
    |> string.split("\n")
    |> list.map(fn(line) { "  " <> line })
    |> string.join("\n")
  io.println(table)
}
