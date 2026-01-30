//// VIVA GPU vs CPU Benchmark
////
//// Compara performance de GPU (CUDA) vs CPU (Rayon)

import gleam/io
import viva/lifecycle/burn
import viva/lifecycle/burn_physics

pub fn main() {
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║           VIVA GPU vs CPU Benchmark - RTX 4090               ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Backend info
  io.println("Backend: " <> burn.check())
  io.println("")

  // Batch Physics Benchmark
  io.println("┌──────────────────────────────────────────────────────────────┐")
  io.println("│ Batch Physics Simulation                                     │")
  io.println("├──────────────────────────────────────────────────────────────┤")

  io.println("│ Config: 1000 tables, 200 steps, 5 iterations                 │")
  let physics_result = burn_physics.benchmark(1000, 200, 5)
  io.println("│ " <> physics_result)
  io.println("└──────────────────────────────────────────────────────────────┘")
  io.println("")

  // NES Operations Benchmark
  io.println("┌──────────────────────────────────────────────────────────────┐")
  io.println("│ NES Operations (Neural Evolution)                            │")
  io.println("├──────────────────────────────────────────────────────────────┤")

  io.println("│ Config: 867 weights, 16 perturbations, 100 iterations        │")
  let nes_result = burn.benchmark_nes(867, 16, 100)
  io.println("│ " <> nes_result)
  io.println("└──────────────────────────────────────────────────────────────┘")
  io.println("")

  // Neural Forward Benchmark
  io.println("┌──────────────────────────────────────────────────────────────┐")
  io.println("│ Neural Network Batch Forward                                 │")
  io.println("├──────────────────────────────────────────────────────────────┤")

  io.println("│ Config: 1000 networks, [8,32,16,3] arch, 100 iterations      │")
  let forward_result = burn.benchmark(1000, 8, 32, 3, 100)
  io.println("│ " <> forward_result)
  io.println("└──────────────────────────────────────────────────────────────┘")
  io.println("")

  // Large batch test
  io.println("┌──────────────────────────────────────────────────────────────┐")
  io.println("│ Large Batch Physics (5000 tables)                            │")
  io.println("├──────────────────────────────────────────────────────────────┤")

  io.println("│ Config: 5000 tables, 200 steps, 3 iterations                 │")
  let large_result = burn_physics.benchmark(5000, 200, 3)
  io.println("│ " <> large_result)
  io.println("└──────────────────────────────────────────────────────────────┘")
  io.println("")

  io.println("═══════════════════════════════════════════════════════════════")
  io.println(" Benchmark complete!")
  io.println("═══════════════════════════════════════════════════════════════")
}
