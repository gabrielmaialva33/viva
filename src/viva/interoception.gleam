//// Interoception - Sensing the hardware
////
//// Reads system metrics (/proc on Linux) and converts to
//// Free Energy to feed emotional state.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import simplifile
import viva_math
import viva_math/free_energy.{type Feeling}
import viva_math/vector.{type Vec3}

// =============================================================================
// TYPES
// =============================================================================

/// System metrics
pub type SystemMetrics {
  SystemMetrics(
    /// Load average (0.0 - N.0, generally < 1.0 is good)
    cpu_load: Float,
    /// Memory used (0.0 - 1.0)
    memory_used: Float,
    /// Temperature in Celsius
    temperature: Float,
  )
}

/// Interoceptive state
pub type InteroceptiveState {
  InteroceptiveState(
    /// Predicted metrics (baseline)
    predicted: SystemMetrics,
    /// Observed metrics
    observed: SystemMetrics,
    /// Total Free Energy
    free_energy: Float,
    /// Feeling classification
    feeling: Feeling,
  )
}

// =============================================================================
// SENSING
// =============================================================================

/// Read system metrics (Linux)
pub fn sense() -> Result(SystemMetrics, Nil) {
  use loadavg <- result.try(read_loadavg())
  use meminfo <- result.try(read_meminfo())
  let temperature = read_temperature() |> result.unwrap(50.0)

  Ok(SystemMetrics(
    cpu_load: loadavg,
    memory_used: meminfo,
    temperature: temperature,
  ))
}

/// Create baseline metrics (idle system)
pub fn baseline() -> SystemMetrics {
  SystemMetrics(cpu_load: 0.5, memory_used: 0.5, temperature: 50.0)
}

// =============================================================================
// FREE ENERGY COMPUTATION
// =============================================================================

/// Compute Free Energy between predicted and observed
pub fn compute_free_energy(
  predicted: SystemMetrics,
  observed: SystemMetrics,
) -> InteroceptiveState {
  let pred_vec = metrics_to_vec3(predicted)
  let obs_vec = metrics_to_vec3(observed)

  let fe_state = viva_math.free_energy(pred_vec, obs_vec)

  InteroceptiveState(
    predicted: predicted,
    observed: observed,
    free_energy: fe_state.free_energy,
    feeling: fe_state.feeling,
  )
}

/// Convert InteroceptiveState to PAD delta
pub fn to_pad_delta(state: InteroceptiveState) -> Vec3 {
  // Map feeling to PAD delta
  case state.feeling {
    free_energy.Homeostatic -> viva_math.pad(0.05, -0.05, 0.05)
    free_energy.Surprised -> viva_math.pad(-0.1, 0.2, -0.1)
    free_energy.Alarmed -> viva_math.pad(-0.2, 0.4, -0.2)
    free_energy.Overwhelmed -> viva_math.pad(-0.3, 0.5, -0.3)
  }
}

// =============================================================================
// INTERNAL
// =============================================================================

fn metrics_to_vec3(m: SystemMetrics) -> Vec3 {
  // Map metrics to PAD space:
  // - high cpu_load -> high arousal
  // - high memory -> low dominance (loss of control)
  // - high temperature -> low pleasure (discomfort)
  let pleasure = 1.0 -. float.min(m.temperature /. 80.0, 1.0)
  let arousal = float.min(m.cpu_load, 2.0) /. 2.0
  let dominance = 1.0 -. m.memory_used

  viva_math.pad(pleasure, arousal, dominance)
}

fn read_loadavg() -> Result(Float, Nil) {
  case simplifile.read("/proc/loadavg") {
    Ok(content) -> {
      content
      |> string.trim()
      |> string.split(" ")
      |> list.first()
      |> result.try(float.parse)
    }
    Error(_) -> Error(Nil)
  }
}

fn read_meminfo() -> Result(Float, Nil) {
  case simplifile.read("/proc/meminfo") {
    Ok(content) -> {
      let lines = string.split(content, "\n")
      use total <- result.try(find_meminfo_value(lines, "MemTotal:"))
      use available <- result.try(find_meminfo_value(lines, "MemAvailable:"))

      case total {
        0 -> Error(Nil)
        _ -> {
          let used_ratio =
            int.to_float(total - available) /. int.to_float(total)
          Ok(used_ratio)
        }
      }
    }
    Error(_) -> Error(Nil)
  }
}

fn find_meminfo_value(lines: List(String), prefix: String) -> Result(Int, Nil) {
  lines
  |> list.find(fn(line) { string.starts_with(line, prefix) })
  |> result.try(fn(line) {
    line
    |> string.replace(prefix, "")
    |> string.trim()
    |> string.split(" ")
    |> list.first()
    |> result.try(int.parse)
  })
}

fn read_temperature() -> Result(Float, Nil) {
  // Try to read thermal zone 0 (CPU)
  case simplifile.read("/sys/class/thermal/thermal_zone0/temp") {
    Ok(content) -> {
      content
      |> string.trim()
      |> int.parse()
      |> result.map(fn(millidegrees) {
        int.to_float(millidegrees) /. 1000.0
      })
    }
    Error(_) -> Error(Nil)
  }
}
