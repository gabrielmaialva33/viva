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

  @skip_nif System.get_env("VIVA_SKIP_NIF") == "true"

  unless @skip_nif do
    use Rustler,
      otp_app: :viva_bridge,
      crate: "viva_body"
  end

  @doc """
  Checks if VIVA's body is alive.

  ## Example

      VivaBridge.Body.alive()
      # => "VIVA body is alive"

  """
  if @skip_nif do
    def alive(), do: "VIVA body is alive (stub)"
    def get_cycles(), do: 0
  else
    def alive(), do: :erlang.nif_error(:nif_not_loaded)
    def get_cycles(), do: :erlang.nif_error(:nif_not_loaded)
  end

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

  ## Low-Level (New)
  - `cpu_freq_mhz` - Real-time CPU frequency in MHz (via /sys)
  - `l3_cache_kb` - L3 Cache size in KB (via CPUID)
  - `context_switches` - OS context switches (measure of noise)
  - `interrupts` - Hardware interrupts

  ## Example

      hw = VivaBridge.Body.feel_hardware()
      # => %{cpu_usage: 15.2, cpu_temp: 45.0, memory_used_percent: 25.3, ...}

  """
  if @skip_nif do
    def feel_hardware() do
      %{
        cpu_usage: 10.0,
        cpu_temp: 40.0,
        cpu_count: 4,
        memory_used_percent: 30.0,
        memory_available_gb: 16.0,
        memory_total_gb: 32.0,
        swap_used_percent: 0.0,
        gpu_usage: nil,
        gpu_vram_used_percent: nil,
        gpu_temp: nil,
        gpu_name: nil,
        disk_usage_percent: 20.0,
        disk_read_bytes: 0,
        disk_write_bytes: 0,
        net_rx_bytes: 0,
        net_tx_bytes: 0,
        uptime_seconds: 3600,
        process_count: 100,
        load_avg_1m: 0.5,
        load_avg_5m: 0.5,
        load_avg_15m: 0.5
      }
    end
  else
    def feel_hardware(), do: :erlang.nif_error(:nif_not_loaded)
  end

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
  if @skip_nif do
    def hardware_to_qualia(), do: {0.0, 0.0, 0.0}
  else
    def hardware_to_qualia(), do: :erlang.nif_error(:nif_not_loaded)
  end

  # ============================================================================
  # Dynamics NIFs - Stochastic Emotional State Evolution
  # ============================================================================

  @doc """
  Single Ornstein-Uhlenbeck step for PAD state.

  The O-U process models mean-reversion: emotional states naturally drift
  back towards equilibrium (neutral) over time, with random perturbations.

  ## Parameters

  - `p, a, d` - Current PAD state (each in [-1, 1])
  - `dt` - Time step (typically 0.5 for 500ms ticks)
  - `noise_p, noise_a, noise_d` - Gaussian noise samples (N(0,1))

  ## Returns

  Tuple `{p', a', d'}` - New PAD state (bounded to [-1, 1])

  ## Example

      # Neutral state with small perturbation
      {p, a, d} = VivaBridge.Body.dynamics_ou_step(0.0, 0.0, 0.0, 0.5, 0.1, -0.2, 0.05)
      # => {0.0075, -0.025, 0.005}

  ## Default O-U Parameters (Kuppens et al., 2010)

  - Pleasure: θ=0.3, σ=0.15 (slow reversion, moderate volatility)
  - Arousal: θ=0.5, σ=0.25 (fast reversion, high volatility)
  - Dominance: θ=0.2, σ=0.10 (slowest reversion, low volatility)
  """
  if @skip_nif do
    def dynamics_ou_step(_p, _a, _d, _dt, _noise_p, _noise_a, _noise_d), do: {0.0, 0.0, 0.0}
  else
    def dynamics_ou_step(_p, _a, _d, _dt, _noise_p, _noise_a, _noise_d),
      do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Find equilibrium points of the cusp catastrophe.

  The cusp catastrophe models sudden transitions (bifurcations) in mood.
  Given control parameters (c, y), returns the equilibria where dV/dx = 0.

  ## Parameters

  - `c` - "Splitting factor" (controls bifurcation strength)
  - `y` - "Normal factor" (asymmetry/bias)

  ## Returns

  List of equilibrium x values (1 or 3 depending on bifurcation region)

  ## Example

      # Outside bifurcation region - single equilibrium
      VivaBridge.Body.dynamics_cusp_equilibria(0.5, 0.0)
      # => [0.0]

      # Inside bifurcation region - three equilibria
      VivaBridge.Body.dynamics_cusp_equilibria(2.0, 0.0)
      # => [-1.414, 0.0, 1.414]

  """
  if @skip_nif do
    def dynamics_cusp_equilibria(_c, _y), do: [0.0]
  else
    def dynamics_cusp_equilibria(_c, _y), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Check if (c, y) is in the bifurcation region.

  The bifurcation region is where the cusp has 3 equilibria, allowing
  sudden mood transitions. Condition: 27y² < 4c³

  ## Example

      VivaBridge.Body.dynamics_cusp_is_bifurcation(2.0, 0.0)
      # => true

      VivaBridge.Body.dynamics_cusp_is_bifurcation(0.5, 0.0)
      # => false

  """
  if @skip_nif do
    def dynamics_cusp_is_bifurcation(_c, _y), do: false
  else
    def dynamics_cusp_is_bifurcation(_c, _y), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Cusp-modulated mood step.

  Applies cusp catastrophe dynamics to pleasure (mood) based on arousal level.
  Higher arousal increases bifurcation potential, enabling sudden mood swings.

  ## Parameters

  - `mood` - Current pleasure/mood value
  - `arousal` - Current arousal (controls bifurcation strength)
  - `external_bias` - External influence (maps to cusp asymmetry)
  - `dt` - Time step

  ## Returns

  New mood value after cusp dynamics

  ## Example

      # High arousal can trigger mood flip
      VivaBridge.Body.dynamics_cusp_mood_step(-0.8, 0.9, 0.1, 0.5)
      # => 0.7  (sudden flip from negative to positive mood)

  """
  if @skip_nif do
    def dynamics_cusp_mood_step(mood, _arousal, _external_bias, _dt), do: mood
  else
    def dynamics_cusp_mood_step(_mood, _arousal, _external_bias, _dt),
      do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Full DynAffect step: O-U dynamics + optional Cusp overlay.

  This is the main function for evolving VIVA's emotional state over time.
  Combines:
  1. Ornstein-Uhlenbeck mean-reversion (baseline dynamics)
  2. Cusp catastrophe (optional sudden transitions)

  ## Parameters

  - `p, a, d` - Current PAD state
  - `dt` - Time step
  - `noise_p, noise_a, noise_d` - Gaussian noise samples
  - `cusp_enabled` - Whether to apply cusp overlay
  - `cusp_sensitivity` - How strongly cusp affects mood (0.0-1.0)
  - `external_bias` - External emotional influence

  ## Returns

  Tuple `{p', a', d'}` - New PAD state

  ## Example

      # Full dynamics with cusp enabled
      {p, a, d} = VivaBridge.Body.dynamics_step(
        0.5, 0.3, 0.2,      # Current PAD
        0.5,                 # dt = 500ms
        0.1, -0.2, 0.05,    # Noise
        true,                # Cusp enabled
        0.5,                 # Medium sensitivity
        0.0                  # No external bias
      )

  """
  if @skip_nif do
    def dynamics_step(p, a, d, _dt, _np, _na, _nd, _cusp, _sens, _bias), do: {p, a, d}
  else
    def dynamics_step(_p, _a, _d, _dt, _np, _na, _nd, _cusp, _sens, _bias),
      do: :erlang.nif_error(:nif_not_loaded)
  end

  # ============================================================================
  # Body Engine NIFs - Unified Interoception
  # ============================================================================

  @doc """
  Creates a new body engine with default configuration.

  Returns a reference to the Rust BodyEngine resource.

  ## Example

      engine = VivaBridge.Body.body_engine_new()

  """
  if @skip_nif do
    def body_engine_new(), do: :stub_engine
  else
    def body_engine_new(), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Creates a new body engine with custom configuration.

  ## Parameters

  - `dt` - Time step per tick in seconds (default: 0.5)
  - `cusp_enabled` - Enable cusp catastrophe overlay (default: true)
  - `cusp_sensitivity` - Cusp effect strength 0-1 (default: 0.5)
  - `seed` - RNG seed, 0 = use system time (default: 0)

  ## Example

      engine = VivaBridge.Body.body_engine_new_with_config(0.5, true, 0.5, 0)

  """
  if @skip_nif do
    def body_engine_new_with_config(_dt, _cusp, _sens, _seed), do: :stub_engine
  else
    def body_engine_new_with_config(_dt, _cusp, _sens, _seed),
      do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Executes one body tick - the main integration function.

  This performs the complete interoception cycle:
  1. Senses hardware (CPU, memory, GPU, etc.)
  2. Computes stress and qualia (hardware → PAD deltas)
  3. Generates stochastic noise
  4. Evolves PAD state via O-U dynamics + Cusp
  5. Returns complete BodyState

  ## Returns

  Map with:
  - `:pleasure`, `:arousal`, `:dominance` - PAD state [-1, 1]
  - `:stress_level` - Composite stress [0, 1]
  - `:in_bifurcation` - Whether in cusp bifurcation region
  - `:tick` - Monotonic tick counter
  - `:timestamp_ms` - Unix timestamp
  - `:hardware` - Complete hardware metrics map

  ## Example

      engine = VivaBridge.Body.body_engine_new()
      state = VivaBridge.Body.body_engine_tick(engine)
      # => %{pleasure: 0.05, arousal: -0.1, dominance: 0.2, stress_level: 0.15, ...}

  """
  if @skip_nif do
    def body_engine_tick(_engine) do
      %{
        pleasure: 0.0,
        arousal: 0.0,
        dominance: 0.0,
        stress_level: 0.0,
        in_bifurcation: false,
        tick: 0,
        timestamp_ms: System.system_time(:millisecond),
        hardware: feel_hardware()
      }
    end
  else
    def body_engine_tick(_engine), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Gets current PAD state without ticking.

  Returns `{pleasure, arousal, dominance}` tuple.
  """
  if @skip_nif do
    def body_engine_get_pad(_engine), do: {0.0, 0.0, 0.0}
  else
    def body_engine_get_pad(_engine), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Sets PAD state directly.

  Values are clamped to [-1, 1].
  """
  if @skip_nif do
    def body_engine_set_pad(_engine, _p, _a, _d), do: :ok
  else
    def body_engine_set_pad(_engine, _p, _a, _d), do: :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Applies an external emotional stimulus.

  Allows external events to influence PAD state additively.
  Values are clamped after application.

  ## Example

      # Positive interaction
      VivaBridge.Body.body_engine_apply_stimulus(engine, 0.3, 0.1, 0.1)

  """
  if @skip_nif do
    def body_engine_apply_stimulus(_engine, _p, _a, _d), do: :ok
  else
    def body_engine_apply_stimulus(_engine, _p, _a, _d), do: :erlang.nif_error(:nif_not_loaded)
  end
end
