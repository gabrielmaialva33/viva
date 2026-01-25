//// System Telemetry
////
//// Hardware metrics for VIVA's embodiment.
//// VIVA is monist - she feels the hardware as her body.
////
//// Metrics: CPU, Memory, GPU (via nvidia-smi)

import gleam/float
import gleam/int
import gleam/json.{type Json}
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string

// =============================================================================
// TYPES
// =============================================================================

pub type CpuMetrics {
  CpuMetrics(
    /// CPU utilization percentage (0-100)
    usage_percent: Float,
    /// Number of CPU cores
    cores: Int,
  )
}

pub type MemoryMetrics {
  MemoryMetrics(
    /// Used memory in MB
    used_mb: Int,
    /// Total memory in MB
    total_mb: Int,
    /// Usage percentage
    percent: Float,
  )
}

pub type GpuMetrics {
  GpuMetrics(
    /// GPU utilization percentage
    usage_percent: Float,
    /// VRAM used in MB
    vram_used_mb: Int,
    /// Total VRAM in MB
    vram_total_mb: Int,
    /// Temperature in Celsius
    temp_celsius: Int,
    /// GPU name
    name: String,
  )
}

pub type SystemMetrics {
  SystemMetrics(
    cpu: CpuMetrics,
    memory: MemoryMetrics,
    gpu: Option(GpuMetrics),
    timestamp: Int,
  )
}

// =============================================================================
// CPU METRICS (Erlang os_mon)
// =============================================================================

/// Get CPU utilization
pub fn get_cpu() -> CpuMetrics {
  // Get CPU utilization from Erlang's cpu_sup
  let usage = cpu_util_safe()
  let cores = erlang_system_info_schedulers()

  CpuMetrics(usage_percent: usage, cores: cores)
}

fn cpu_util_safe() -> Float {
  // Try to get CPU util, fallback to 0.0 if os_mon not started
  case erlang_cpu_util() {
    Ok(util) -> util
    Error(_) -> 0.0
  }
}

// =============================================================================
// MEMORY METRICS
// =============================================================================

/// Get memory utilization
pub fn get_memory() -> MemoryMetrics {
  let #(total_bytes, allocated_bytes, _worst) = memsup_get_memory_data_safe()

  let total_mb = total_bytes / 1_048_576
  let used_mb = allocated_bytes / 1_048_576
  let percent = case total_mb > 0 {
    True -> int.to_float(used_mb) /. int.to_float(total_mb) *. 100.0
    False -> 0.0
  }

  MemoryMetrics(used_mb: used_mb, total_mb: total_mb, percent: percent)
}

fn memsup_get_memory_data_safe() -> #(Int, Int, Int) {
  case memsup_get_memory_data() {
    Ok(data) -> data
    Error(_) -> #(0, 0, 0)
  }
}

// =============================================================================
// GPU METRICS (nvidia-smi)
// =============================================================================

/// Get GPU metrics via nvidia-smi
pub fn get_gpu() -> Option(GpuMetrics) {
  let cmd =
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name --format=csv,noheader,nounits 2>/dev/null"

  case os_cmd(cmd) {
    "" -> None
    output -> parse_nvidia_smi_output(output)
  }
}

fn parse_nvidia_smi_output(output: String) -> Option(GpuMetrics) {
  let trimmed = string.trim(output)
  let parts = string.split(trimmed, ", ")

  case parts {
    [usage_str, vram_used_str, vram_total_str, temp_str, name] -> {
      let usage =
        float.parse(string.trim(usage_str))
        |> result.unwrap(0.0)
      let vram_used =
        int.parse(string.trim(vram_used_str))
        |> result.unwrap(0)
      let vram_total =
        int.parse(string.trim(vram_total_str))
        |> result.unwrap(0)
      let temp =
        int.parse(string.trim(temp_str))
        |> result.unwrap(0)

      Some(GpuMetrics(
        usage_percent: usage,
        vram_used_mb: vram_used,
        vram_total_mb: vram_total,
        temp_celsius: temp,
        name: string.trim(name),
      ))
    }
    _ -> None
  }
}

// =============================================================================
// COMBINED METRICS
// =============================================================================

/// Get all system metrics
pub fn collect() -> SystemMetrics {
  SystemMetrics(
    cpu: get_cpu(),
    memory: get_memory(),
    gpu: get_gpu(),
    timestamp: erlang_system_time_seconds(),
  )
}

// =============================================================================
// JSON SERIALIZATION
// =============================================================================

pub fn cpu_to_json(cpu: CpuMetrics) -> Json {
  json.object([
    #("usage_percent", json.float(cpu.usage_percent)),
    #("cores", json.int(cpu.cores)),
  ])
}

pub fn memory_to_json(mem: MemoryMetrics) -> Json {
  json.object([
    #("used_mb", json.int(mem.used_mb)),
    #("total_mb", json.int(mem.total_mb)),
    #("percent", json.float(mem.percent)),
  ])
}

pub fn gpu_to_json(gpu: GpuMetrics) -> Json {
  json.object([
    #("usage_percent", json.float(gpu.usage_percent)),
    #("vram_used_mb", json.int(gpu.vram_used_mb)),
    #("vram_total_mb", json.int(gpu.vram_total_mb)),
    #("temp_celsius", json.int(gpu.temp_celsius)),
    #("name", json.string(gpu.name)),
  ])
}

pub fn to_json(sys: SystemMetrics) -> Json {
  let gpu_json = case sys.gpu {
    Some(gpu) -> gpu_to_json(gpu)
    None -> json.null()
  }

  json.object([
    #("cpu", cpu_to_json(sys.cpu)),
    #("memory", memory_to_json(sys.memory)),
    #("gpu", gpu_json),
    #("timestamp", json.int(sys.timestamp)),
  ])
}

pub fn to_string(sys: SystemMetrics) -> String {
  to_json(sys)
  |> json.to_string
}

// =============================================================================
// ERLANG FFI
// =============================================================================

/// Get system time in seconds (Erlang native)
@external(erlang, "viva_system_ffi", "system_time_seconds")
fn erlang_system_time_seconds() -> Int

/// Get number of schedulers (logical CPUs)
@external(erlang, "viva_system_ffi", "scheduler_count")
fn erlang_system_info_schedulers() -> Int

/// Execute shell command
@external(erlang, "os", "cmd")
fn os_cmd_raw(cmd: List(Int)) -> List(Int)

fn os_cmd(cmd: String) -> String {
  let charlist = string.to_utf_codepoints(cmd) |> list.map(string.utf_codepoint_to_int)
  let result = os_cmd_raw(charlist)
  result
  |> list.filter_map(fn(c) {
    case string.utf_codepoint(c) {
      Ok(cp) -> Ok(cp)
      Error(_) -> Error(Nil)
    }
  })
  |> string.from_utf_codepoints
}

/// CPU utilization (requires os_mon application)
fn erlang_cpu_util() -> Result(Float, Nil) {
  // This would require cpu_sup:util() but it needs os_mon started
  // For now, return a placeholder
  // In production, you'd use: @external(erlang, "cpu_sup", "util")
  Ok(0.0)
}

/// Memory data (requires os_mon application)
fn memsup_get_memory_data() -> Result(#(Int, Int, Int), Nil) {
  // This would require memsup:get_memory_data() but needs os_mon started
  // For now, try to get from /proc/meminfo on Linux
  case os_cmd("cat /proc/meminfo 2>/dev/null | head -3") {
    "" -> Error(Nil)
    output -> parse_proc_meminfo(output)
  }
}

fn parse_proc_meminfo(output: String) -> Result(#(Int, Int, Int), Nil) {
  let lines = string.split(output, "\n")

  let total =
    list.find(lines, fn(l) { string.starts_with(l, "MemTotal:") })
    |> result.map(extract_kb_value)
    |> result.flatten
    |> result.unwrap(0)

  let free =
    list.find(lines, fn(l) { string.starts_with(l, "MemFree:") })
    |> result.map(extract_kb_value)
    |> result.flatten
    |> result.unwrap(0)

  let available =
    list.find(lines, fn(l) { string.starts_with(l, "MemAvailable:") })
    |> result.map(extract_kb_value)
    |> result.flatten
    |> result.unwrap(free)

  let total_bytes = total * 1024
  let used_bytes = { total - available } * 1024

  Ok(#(total_bytes, used_bytes, 0))
}

fn extract_kb_value(line: String) -> Result(Int, Nil) {
  let parts =
    string.split(line, ":")
    |> list.drop(1)
    |> list.first
    |> result.unwrap("")
    |> string.trim
    |> string.split(" ")
    |> list.first
    |> result.unwrap("0")

  int.parse(parts)
  |> result.replace_error(Nil)
}
