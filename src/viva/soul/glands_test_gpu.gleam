//// Quick test for viva_glands GPU NIF

import gleam/io
import viva/soul/glands

pub fn main() {
  io.println("=== VIVA Glands GPU Test ===\n")

  // Check native status
  io.println("Status: " <> glands.check())

  // Try to initialize
  io.println("\nInitializing with default config...")
  let config = glands.default_config()
  io.println("  llm_dim: 4096")
  io.println("  hrr_dim: 8192")
  io.println("  gpu_layers: 99")

  case glands.init(config) {
    Ok(handle) -> {
      io.println("  ✅ Glands initialized!")

      // Run benchmark
      io.println("\n" <> glands.benchmark(handle, 100))
    }
    Error(msg) -> {
      io.println("  ❌ Init failed: " <> msg)
    }
  }
}
