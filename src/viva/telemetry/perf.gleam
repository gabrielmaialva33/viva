//// Performance Metrics
////
//// Measures tick performance, GC activity, actor latency, and system overhead.
//// Essential for optimizing VIVA's real-time consciousness simulation.

import gleam/json.{type Json}
import gleam/list

// =============================================================================
// TYPES
// =============================================================================

/// Performance snapshot
pub type PerfMetrics {
  PerfMetrics(
    tick_time_us: Int,
    gc_runs: Int,
    memory_words: Int,
    reductions: Int,
    scheduler_usage: Float,
    energy_loss_rate: Float,
    timestamp: Int,
  )
}

/// Timing measurement result
pub type Measurement(a) {
  Measurement(result: a, elapsed_us: Int)
}

/// Rolling performance history
pub type PerfHistory {
  PerfHistory(samples: List(PerfMetrics), max_samples: Int)
}

// =============================================================================
// TIMING FUNCTIONS
// =============================================================================

/// Measure execution time of a function in microseconds
pub fn measure(f: fn() -> a) -> Measurement(a) {
  let start = monotonic_time_us()
  let result = f()
  let elapsed = monotonic_time_us() - start
  Measurement(result: result, elapsed_us: elapsed)
}

/// Get current monotonic time in microseconds
pub fn monotonic_time_us() -> Int {
  erlang_monotonic_time_micro()
}

// =============================================================================
// PROCESS METRICS (via Erlang process_info)
// =============================================================================

/// Get reductions (work units) for current process
pub fn get_reductions() -> Int {
  erlang_get_reductions()
}

/// Get heap size in words for current process
pub fn get_heap_words() -> Int {
  erlang_get_heap_size()
}

/// Get GC count for current process
pub fn get_gc_count() -> Int {
  erlang_get_gc_count()
}

// =============================================================================
// SYSTEM-WIDE METRICS
// =============================================================================

/// Get scheduler utilization (0.0 to 1.0)
pub fn get_scheduler_usage() -> Float {
  // This would require scheduler_wall_time statistics
  // For now, return placeholder
  0.0
}

/// Calculate energy loss rate from two metric snapshots
pub fn calculate_energy_loss(
  prev_energy: Float,
  curr_energy: Float,
  elapsed_us: Int,
) -> Float {
  case elapsed_us > 0 {
    True -> {
      let delta = curr_energy -. prev_energy
      let seconds = int_to_float(elapsed_us) /. 1_000_000.0
      delta /. seconds
    }
    False -> 0.0
  }
}

// =============================================================================
// COLLECTOR
// =============================================================================

/// Collect current performance metrics
pub fn collect(tick_time_us: Int, energy_loss_rate: Float) -> PerfMetrics {
  PerfMetrics(
    tick_time_us: tick_time_us,
    gc_runs: get_gc_count(),
    memory_words: get_heap_words(),
    reductions: get_reductions(),
    scheduler_usage: get_scheduler_usage(),
    energy_loss_rate: energy_loss_rate,
    timestamp: system_time_seconds(),
  )
}

/// Create empty metrics (for initialization)
pub fn empty() -> PerfMetrics {
  PerfMetrics(
    tick_time_us: 0,
    gc_runs: 0,
    memory_words: 0,
    reductions: 0,
    scheduler_usage: 0.0,
    energy_loss_rate: 0.0,
    timestamp: system_time_seconds(),
  )
}

// =============================================================================
// HISTORY MANAGEMENT
// =============================================================================

/// Create new performance history
pub fn new_history(max_samples: Int) -> PerfHistory {
  PerfHistory(samples: [], max_samples: max_samples)
}

/// Add sample to history (FIFO)
pub fn add_sample(history: PerfHistory, metrics: PerfMetrics) -> PerfHistory {
  let new_samples = [metrics, ..history.samples]
  let trimmed = case list.length(new_samples) > history.max_samples {
    True -> list.take(new_samples, history.max_samples)
    False -> new_samples
  }
  PerfHistory(..history, samples: trimmed)
}

/// Get average tick time from history
pub fn avg_tick_time(history: PerfHistory) -> Float {
  case history.samples {
    [] -> 0.0
    samples -> {
      let sum =
        list.fold(samples, 0, fn(acc, m) { acc + m.tick_time_us })
      int_to_float(sum) /. int_to_float(list.length(samples))
    }
  }
}

/// Get max tick time from history
pub fn max_tick_time(history: PerfHistory) -> Int {
  list.fold(history.samples, 0, fn(acc, m) {
    case m.tick_time_us > acc {
      True -> m.tick_time_us
      False -> acc
    }
  })
}

// =============================================================================
// JSON SERIALIZATION
// =============================================================================

/// Metrics to JSON
pub fn to_json(m: PerfMetrics) -> Json {
  json.object([
    #("tick_time_us", json.int(m.tick_time_us)),
    #("gc_runs", json.int(m.gc_runs)),
    #("memory_words", json.int(m.memory_words)),
    #("memory_mb", json.float(int_to_float(m.memory_words * 8) /. 1_048_576.0)),
    #("reductions", json.int(m.reductions)),
    #("scheduler_usage", json.float(m.scheduler_usage)),
    #("energy_loss_rate", json.float(m.energy_loss_rate)),
    #("timestamp", json.int(m.timestamp)),
  ])
}

/// Metrics to JSON string
pub fn to_string(m: PerfMetrics) -> String {
  to_json(m) |> json.to_string
}

/// History to JSON (recent samples)
pub fn history_to_json(history: PerfHistory, n: Int) -> Json {
  let recent = list.take(history.samples, n) |> list.reverse
  json.object([
    #("samples", json.array(recent, to_json)),
    #("avg_tick_us", json.float(avg_tick_time(history))),
    #("max_tick_us", json.int(max_tick_time(history))),
    #("sample_count", json.int(list.length(history.samples))),
  ])
}

// =============================================================================
// ERLANG FFI
// =============================================================================

@external(erlang, "viva_perf_ffi", "monotonic_time_micro")
fn erlang_monotonic_time_micro() -> Int

@external(erlang, "viva_perf_ffi", "get_reductions")
fn erlang_get_reductions() -> Int

@external(erlang, "viva_perf_ffi", "get_heap_size")
fn erlang_get_heap_size() -> Int

@external(erlang, "viva_perf_ffi", "get_gc_count")
fn erlang_get_gc_count() -> Int

@external(erlang, "viva_system_ffi", "system_time_seconds")
fn system_time_seconds() -> Int

@external(erlang, "erlang", "float")
fn int_to_float(i: Int) -> Float
