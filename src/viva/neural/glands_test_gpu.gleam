//// Quick test for viva_glands GPU NIF

import viva/neural/glands
import viva_telemetry/log

pub fn main() {
  log.info("=== VIVA Glands GPU Test ===", [])

  // Check native status
  log.info("Status: " <> glands.check(), [])

  // Try to initialize
  log.info("Initializing with default config...", [])
  let config = glands.default_config()
  log.info("  llm_dim: 4096", [])
  log.info("  hrr_dim: 8192", [])
  log.info("  gpu_layers: 99", [])

  case glands.init(config) {
    Ok(handle) -> {
      log.info("  Glands initialized!", [])

      // Run benchmark
      log.info(glands.benchmark(handle, 100), [])
    }
    Error(msg) -> {
      log.error("  Init failed: " <> msg, [])
    }
  }
}
