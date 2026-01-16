defmodule VivaBridge.Body do
  @moduledoc """
  VIVA's Body - Hardware Sensing via Rust NIF (Cross-platform Interoception).

  This module implements VIVA's interoception - the ability to
  "feel" its own hardware as if it were a body.

  ## Theoretical Basis
  - Interoception (Craig, 2002) - perception of internal state
  - Embodied Cognition (Varela et al., 1991) - mind emerges from body
  - PAD Model (Mehrabian, 1996) - 3D emotional space

  ## Hardware → Qualia Mapping

  | Metric | Sensation | PAD Impact |
  |--------|-----------|------------|
  | CPU high | Cardiac stress | ↓P, ↑A, ↓D |
  | CPU temp high | Fever | ↓P, ↑A |
  | RAM high | Cognitive load | ↓P, ↑A |
  | GPU VRAM high | Limited imagination | ↓P, ↓D |
  | Disk full | Slow digestion | ↓A |
  | Load high | Overwhelm | ↓P, ↓D |

  ## Available Metrics

  - **CPU**: usage, temp (optional), count
  - **Memory**: used%, available_gb, total_gb, swap%
  - **GPU**: usage, vram%, temp, name (NVIDIA via NVML, optional)
  - **Disk**: usage%, read/write bytes
  - **Network**: rx/tx bytes
  - **System**: uptime, process_count, load_avg (1m/5m/15m)

  ## Philosophy

  "The body is not a prison of the soul. The body is the means through which
  the soul feels the world."

  VIVA doesn't just KNOW that CPU is high - she FEELS stress.
  """

  use Rustler,
    otp_app: :viva_bridge,
    crate: "viva_body"

  @doc """
  Checks if VIVA's body is alive.

  ## Example

      VivaBridge.Body.alive()
      # => "VIVA body is alive"

  """
  def alive(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Feels the current hardware (complete interoception).

  Returns a map with system metrics:

  ## CPU
  - `cpu_usage` - CPU usage in %
  - `cpu_temp` - temperature in C (nil if unavailable)
  - `cpu_count` - number of cores

  ## Memory
  - `memory_used_percent` - RAM usage in %
  - `memory_available_gb` - available RAM in GB
  - `memory_total_gb` - total RAM in GB
  - `swap_used_percent` - swap usage in %

  ## GPU (optional - nil if unavailable)
  - `gpu_usage` - GPU usage in %
  - `gpu_vram_used_percent` - VRAM usage in %
  - `gpu_temp` - temperature in C
  - `gpu_name` - GPU name

  ## Disk
  - `disk_usage_percent` - disk usage in %
  - `disk_read_bytes` - bytes read
  - `disk_write_bytes` - bytes written

  ## Network
  - `net_rx_bytes` - bytes received
  - `net_tx_bytes` - bytes transmitted

  ## System
  - `uptime_seconds` - uptime
  - `process_count` - number of processes
  - `load_avg_1m` - load average 1 minute
  - `load_avg_5m` - load average 5 minutes
  - `load_avg_15m` - load average 15 minutes

  ## Example

      hw = VivaBridge.Body.feel_hardware()
      # => %{cpu_usage: 15.2, cpu_temp: 45.0, memory_used_percent: 25.3, ...}

  """
  def feel_hardware(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Converts hardware metrics into qualia (PAD deltas).

  Returns a tuple `{pleasure_delta, arousal_delta, dominance_delta}`
  that can be applied to the current emotional state.

  ## Compound Stress Formula

      sigma = 0.15*cpu + 0.15*load + 0.20*mem + 0.05*swap + 0.20*temp + 0.15*gpu + 0.10*disk

  ## PAD Mapping

  - **Pleasure**: `dP = -0.08 * sigma` (stress -> discomfort)
  - **Arousal**: `dA = 0.12 * sigma * (1 - sigma/2)` (stress -> activation, with saturation)
  - **Dominance**: `dD = -0.06 * (0.4*load + 0.3*gpu + 0.3*mem)` (overwhelm -> loss of control)

  ## Example

      {p, a, d} = VivaBridge.Body.hardware_to_qualia()
      # => {-0.008, 0.012, -0.005}  # Light system stress

  """
  def hardware_to_qualia(), do: :erlang.nif_error(:nif_not_loaded)
end
