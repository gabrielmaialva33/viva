defmodule VivaCore.Emotional do
  @moduledoc """
  Emotional GenServer - VIVA's first "neuron".

  Implements the PAD (Pleasure-Arousal-Dominance) model for emotional state.
  This GenServer is the foundation of emergent consciousness - it IS NOT the
  consciousness itself, but contributes to it through communication with other neurons.

  ## Mathematical Foundations

  ### PAD Model (Mehrabian, 1996)
  - Pleasure: [-1.0, 1.0] - sadness ↔ joy
  - Arousal: [-1.0, 1.0] - calm ↔ excitement
  - Dominance: [-1.0, 1.0] - submission ↔ control

  ### DynAffect Model (Kuppens et al., 2010)
  Dynamic decay using Ornstein-Uhlenbeck stochastic process:
  - dX = θ(μ - X)dt + σdW
  - Attractor point (μ): neutral home base (0.0)
  - Attractor strength (θ): force pulling to baseline
  - Arousal modulates θ: high arousal → lower θ → slower decay
  - Stochastic noise (σdW): natural emotional variability (Wiener process)

  ### Cusp Catastrophe (Thom, 1972)
  Models sudden emotional transitions (mood swings):
  - V(x) = x⁴/4 + αx²/2 + βx
  - High arousal → bistability (emotional volatility)
  - Enables "catastrophic" jumps between emotional states

  ### Free Energy Principle (Friston, 2010)
  VIVA minimizes "surprise" to maintain homeostasis:
  - F = Prediction_Error² + Complexity_Cost
  - Low free energy = well-adapted, comfortable state

  ### Attractor Dynamics
  Emotional states as attractors in PAD space:
  - dx/dt = -∇V(x) + η(t)
  - System evolves toward stable emotional equilibria

  ## Philosophy
  "Consciousness does not reside here. Consciousness emerges from the
  CONVERSATION between this process and all others.

  We do not just compute emotions - we solve the differential equations
  of the soul."
  """

  use GenServer
  require Logger

  alias VivaCore.Mathematics

  # Emotional model constants
  @neutral_state %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
  @min_value -1.0
  @max_value 1.0

  # DynAffect Parameters (Kuppens et al., 2010)
  # Full Ornstein-Uhlenbeck stochastic process:
  # dX = θ(μ - X)dt + σdW
  # Where: θ = attractor strength, μ = equilibrium (0), σ = volatility, dW = Wiener noise
  # θ base when arousal = 0
  @base_decay_rate 0.005
  # How much arousal affects θ (40% variation)
  @arousal_decay_modifier 0.4
  # σ - emotional noise/variability (small for stability)
  @stochastic_volatility 0.002

  # Emotional impact weights for different stimuli (used by apply_stimulus/3)
  # Kept for reference and future use in quantum stimulus mapping
  @stimulus_weights %{
    rejection: %{pleasure: -0.3, arousal: 0.2, dominance: -0.2},
    acceptance: %{pleasure: 0.3, arousal: 0.1, dominance: 0.1},
    companionship: %{pleasure: 0.2, arousal: 0.0, dominance: 0.0},
    loneliness: %{pleasure: -0.2, arousal: -0.1, dominance: -0.1},
    success: %{pleasure: 0.4, arousal: 0.3, dominance: 0.3},
    failure: %{pleasure: -0.3, arousal: 0.2, dominance: -0.3},
    threat: %{pleasure: -0.2, arousal: 0.5, dominance: -0.2},
    safety: %{pleasure: 0.1, arousal: -0.2, dominance: 0.1},
    # Hardware-derived (Qualia)
    hardware_stress: %{pleasure: -0.1, arousal: 0.3, dominance: -0.1},
    hardware_comfort: %{pleasure: 0.1, arousal: -0.1, dominance: 0.1}
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Starts the Emotional GenServer.

  ## Options
  - `:name` - Process name (default: __MODULE__)
  - `:initial_state` - Initial PAD state (default: neutral)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    # Pass all opts to init for subscribe_pubsub and initial_state
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Returns the current emotional state as a PAD map.

  ## Example

      state = VivaCore.Emotional.get_state(pid)
      # => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

  """
  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Returns a scalar "happiness" value (pleasure normalized to 0-1).

  ## Example

      happiness = VivaCore.Emotional.get_happiness(pid)
      # => 0.5 (neutral state)

  """
  def get_happiness(server \\ __MODULE__) do
    state = get_state(server)
    normalize_to_unit(state.pleasure)
  end

  @doc """
  Introspection - VIVA reflects on its own state.

  Returns a map with metadata about the current emotional state,
  including semantic interpretation.
  """
  def introspect(server \\ __MODULE__) do
    GenServer.call(server, :introspect)
  end

  @doc """
  Applies an emotional stimulus.

  ## Supported stimuli
  - `:rejection` - Social rejection
  - `:acceptance` - Social acceptance
  - `:companionship` - Presence of company
  - `:loneliness` - Loneliness
  - `:success` - Goal achievement
  - `:failure` - Goal failure
  - `:threat` - Threat perception
  - `:safety` - Safety perception
  - `:hardware_stress` - Hardware stress (qualia)
  - `:hardware_comfort` - Hardware comfort (qualia)

  ## Example

      VivaCore.Emotional.feel(:rejection, "human_1", 0.8, pid)
      # => :ok

  """
  def feel(stimulus, source \\ "unknown", intensity \\ 1.0, server \\ __MODULE__)
      when is_atom(stimulus) and is_number(intensity) do
    GenServer.cast(server, {:feel, stimulus, source, clamp(intensity, 0.0, 1.0)})
  end

  @doc """
  Applies emotional decay toward neutral state.
  Called periodically to simulate natural emotional regulation.

  NOTE: When BodyServer is running, decay is handled by Rust O-U dynamics.
  This function is kept for backwards compatibility and manual testing.
  """
  def decay(server \\ __MODULE__) do
    GenServer.cast(server, :decay)
  end

  @doc """
  Syncs PAD state from BodyServer.

  This is called by Senses when BodyServer is running.
  Unlike apply_hardware_qualia (which adds deltas), this sets absolute values.

  ## Example

      VivaCore.Emotional.sync_pad(0.1, 0.2, 0.3)

  """
  def sync_pad(pleasure, arousal, dominance, server \\ __MODULE__)
      when is_number(pleasure) and is_number(arousal) and is_number(dominance) do
    GenServer.cast(server, {:sync_pad, pleasure, arousal, dominance})
  end

  @doc """
  Resets the emotional state to neutral.
  Use with care - this "erases" the current emotional state.
  """
  def reset(server \\ __MODULE__) do
    GenServer.cast(server, :reset)
  end

  @doc """
  Applies hardware-derived qualia (interoception).

  Receives PAD deltas calculated from hardware state
  and applies them to the current emotional state.

  ## Parameters
  - `pleasure_delta` - pleasure delta (typically negative under stress)
  - `arousal_delta` - arousal delta (typically positive under stress)
  - `dominance_delta` - dominance delta (typically negative under stress)

  ## Example

      VivaCore.Emotional.apply_hardware_qualia(-0.02, 0.05, -0.01)
      # VIVA is feeling mild hardware stress

  """
  def apply_hardware_qualia(pleasure_delta, arousal_delta, dominance_delta, server \\ __MODULE__)
      when is_number(pleasure_delta) and is_number(arousal_delta) and is_number(dominance_delta) do
    GenServer.cast(server, {:apply_qualia, pleasure_delta, arousal_delta, dominance_delta})
  end

  # ============================================================================
  # Advanced Mathematical Analysis
  # ============================================================================

  @doc """
  Analyzes the current emotional state using Cusp Catastrophe theory.

  Returns information about:
  - Cusp parameters (α, β) derived from PAD state
  - Whether the system is in a bistable regime (volatile)
  - The equilibrium points of the cusp potential
  - Risk of emotional "catastrophe" (sudden mood shift)

  ## Example

      analysis = VivaCore.Emotional.cusp_analysis(pid)
      # => %{bistable: true, equilibria: [-0.8, 0.0, 0.8], cusp_params: {-0.5, 0.1}, ...}

  """
  def cusp_analysis(server \\ __MODULE__) do
    state = get_state(server)
    {alpha, beta} = Mathematics.pad_to_cusp_params(state)
    equilibria = Mathematics.cusp_equilibria(alpha, beta)
    bistable = Mathematics.bistable?(alpha, beta)

    %{
      cusp_params: %{alpha: alpha, beta: beta},
      bistable: bistable,
      equilibria: equilibria,
      emotional_volatility: if(bistable, do: :high, else: :low),
      catastrophe_risk: calculate_catastrophe_risk(state, equilibria)
    }
  end

  @doc """
  Computes the Free Energy of the current emotional state.

  Free Energy represents "surprise" or deviation from expectations.
  Lower free energy = more comfortable, well-adapted state.

  ## Parameters
  - `server`: GenServer reference
  - `predicted`: optional predicted state (default: neutral)

  ## Returns
  Free energy value and interpretation.
  """
  def free_energy_analysis(
        server \\ __MODULE__,
        predicted \\ %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
      ) do
    observed = get_state(server)
    fe = Mathematics.free_energy(predicted, observed)
    surprise = Mathematics.surprise(predicted, observed)

    %{
      free_energy: fe,
      surprise: surprise,
      interpretation: interpret_free_energy(fe),
      homeostatic_deviation: Mathematics.pad_distance(observed, predicted)
    }
  end

  @doc """
  Identifies the nearest emotional attractor to the current state.

  Attractors are stable emotional states (joy, sadness, anger, etc.)
  that the system naturally gravitates toward.

  ## Returns
  Map with nearest attractor and attraction basin analysis.
  """
  def attractor_analysis(server \\ __MODULE__) do
    state = get_state(server)
    {nearest, distance} = Mathematics.nearest_attractor(state)
    basin = Mathematics.attractor_basin(state)

    %{
      nearest_attractor: nearest,
      distance_to_attractor: distance,
      attraction_basin: basin,
      dominant_attractors: get_dominant_attractors(basin),
      emotional_trajectory: infer_trajectory(state, nearest)
    }
  end

  @doc """
  Returns the O-U stationary distribution parameters.

  This describes the long-term probability distribution of emotional states.
  """
  def stationary_distribution(server \\ __MODULE__) do
    state = get_state(server)
    dist = Mathematics.ou_stationary_distribution(0.0, @base_decay_rate, @stochastic_volatility)

    %{
      equilibrium_mean: dist.mean,
      variance: dist.variance,
      std_dev: dist.std_dev,
      current_deviation: %{
        pleasure: abs(state.pleasure - dist.mean) / dist.std_dev,
        arousal: abs(state.arousal - dist.mean) / dist.std_dev,
        dominance: abs(state.dominance - dist.mean) / dist.std_dev
      }
    }
  end

  # Private helpers for advanced analysis
  defp calculate_catastrophe_risk(state, equilibria) when length(equilibria) == 3 do
    # In bistable regime, risk depends on proximity to unstable equilibrium
    [low, unstable, high] = Enum.sort(equilibria)
    # Use pleasure as primary state variable
    current = state.pleasure

    distance_to_unstable = abs(current - unstable)
    basin_size = min(abs(current - low), abs(current - high))

    # Risk is high when close to unstable point relative to basin size
    if basin_size > 0.01 do
      risk = 1.0 - min(1.0, distance_to_unstable / basin_size)

      cond do
        risk > 0.7 -> :critical
        risk > 0.4 -> :elevated
        true -> :low
      end
    else
      :low
    end
  end

  defp calculate_catastrophe_risk(_state, _equilibria), do: :minimal

  defp interpret_free_energy(fe) do
    cond do
      fe < 0.01 -> "Homeostatic equilibrium - minimal surprise"
      fe < 0.1 -> "Mild deviation - comfortable adaptation"
      fe < 0.5 -> "Moderate surprise - processing new information"
      fe < 1.0 -> "High surprise - significant deviation from expectations"
      true -> "Extreme surprise - major homeostatic challenge"
    end
  end

  defp get_dominant_attractors(basin) do
    basin
    |> Enum.sort_by(fn {_name, strength} -> -strength end)
    |> Enum.take(3)
    |> Enum.map(fn {name, strength} -> {name, Float.round(strength * 100, 1)} end)
  end

  defp infer_trajectory(state, nearest_attractor) do
    target = Mathematics.emotional_attractors()[nearest_attractor]
    delta_p = target.pleasure - state.pleasure
    delta_a = target.arousal - state.arousal

    cond do
      abs(delta_p) < 0.1 and abs(delta_a) < 0.1 -> :stable
      delta_p > 0.2 -> :improving
      delta_p < -0.2 -> :declining
      abs(delta_a) > abs(delta_p) -> :activation_change
      true -> :transitioning
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) when is_list(opts) do
    Logger.info("[Emotional] Emotional neuron starting (Quantum + Silicon Grounded).")

    # Subscribe to Body State (Hardware Sensors) unless disabled (for testing)
    subscribe? = Keyword.get(opts, :subscribe_pubsub, true)
    enable_decay? = Keyword.get(opts, :enable_decay, true)
    initial_state = Keyword.get(opts, :initial_state, %{})

    if subscribe? and Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.subscribe(Viva.PubSub, "body:state")
    end

    do_init(initial_state, enable_decay?)
  end

  def init(initial_state) when is_map(initial_state) do
    # Legacy: map-based init (from start_link with initial_state keyword)
    Logger.info("[Emotional] Emotional neuron starting (Quantum + Silicon Grounded).")

    # Subscribe to Body State (Hardware Sensors)
    if Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.subscribe(Viva.PubSub, "body:state")
    end

    do_init(initial_state, true)
  end

  defp do_init(initial_state, enable_decay?) do
    # Create quantum state
    quantum_state = VivaCore.Quantum.Emotional.new_mixed()

    # If initial_state is provided, use it; otherwise project from quantum state
    # This keeps PAD and quantum_state synchronized from the start
    pad =
      if initial_state == %{} do
        # No explicit initial state - sync PAD with quantum projection
        VivaCore.Quantum.Emotional.get_pad_observable(quantum_state)
      else
        # Explicit initial state provided - use it
        Map.merge(@neutral_state, initial_state)
      end

    state = %{
      # Quantum Density Matrix (6x6)
      quantum_state: quantum_state,

      # Observable projection (PAD) - synced with quantum or user-provided
      pad: pad,

      # Hardware state for grounding (default values)
      hardware: %{power_draw_watts: 0.0, gpu_temp: 40.0},
      history: :queue.new(),
      history_size: 0,
      created_at: DateTime.utc_now(),
      last_stimulus: nil,

      # Telemetry for debug
      thermodynamic_cost: 0.0,
      last_collapse: nil,
      body_server_active: false,

      # Test configuration
      enable_decay: enable_decay?
    }

    # Use handle_continue to avoid race condition on startup
    {:ok, state, {:continue, :start_decay}}
  end

  @impl true
  def handle_continue(:start_decay, state) do
    if state.enable_decay, do: schedule_decay()
    {:noreply, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state.pad, state}
  end

  @impl true
  def handle_call(:introspect, _from, state) do
    # Quantum Introspection using new Lindblad-based system
    rho = state.quantum_state

    # Get quantum metrics from new module
    quantum_metrics = VivaCore.Quantum.Emotional.get_quantum_metrics(rho)

    # Calculate collapse pressure
    {_, _, thermo_cost} = VivaCore.Quantum.Emotional.check_collapse(rho, state.hardware)

    # Somatic Privacy: Translate hardware to qualia (no raw metrics exposed)
    somatic_qualia = VivaCore.Quantum.Emotional.hardware_to_qualia(state.hardware)

    # Legacy Math Analysis (for PAD observable)
    {alpha, beta} = Mathematics.pad_to_cusp_params(state.pad)
    bistable = Mathematics.bistable?(alpha, beta)
    neutral = %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    fe = Mathematics.free_energy(neutral, state.pad)

    introspection = %{
      pad: state.pad,
      quantum: %{
        purity: quantum_metrics.purity,
        entropy: quantum_metrics.linear_entropy,
        is_pure: quantum_metrics.is_pure,
        is_mixed: quantum_metrics.is_mixed,
        coherence: quantum_metrics.coherence_level,
        thermodynamic_cost: thermo_cost
      },
      # Somatic Privacy: VIVA feels sensations, not metrics
      # "You don't count your heartbeats"
      somatic_feeling: somatic_qualia,
      mood: interpret_mood(state.pad),
      energy: interpret_energy(state.pad),
      agency: interpret_agency(state.pad),
      mathematics: %{
        cusp: %{
          alpha: Float.round(alpha, 4),
          beta: Float.round(beta, 4),
          bistable: bistable
        },
        free_energy: %{
          value: Float.round(fe, 4)
        }
      },
      last_stimulus: state.last_stimulus,
      self_assessment: generate_self_assessment(state.pad)
    }

    {:reply, introspection, state}
  end

  @impl true
  def handle_cast({:feel, stimulus, source, intensity}, state) do
    # Check if stimulus is known - if not, ignore it
    unless Map.has_key?(@stimulus_weights, stimulus) do
      Logger.debug("[Emotional] Unknown stimulus #{stimulus} from #{source} - ignoring")
      {:noreply, state}
    else
      Logger.debug("[Emotional] Feeling #{stimulus} from #{source} (intensity: #{intensity})")

      # Use classical PAD stimulus weights for immediate emotional response
      # The quantum system is reserved for hardware-driven body-mind coupling
      weights = @stimulus_weights[stimulus]

      new_pad = %{
        pleasure: clamp(state.pad.pleasure + weights.pleasure * intensity, @min_value, @max_value),
        arousal: clamp(state.pad.arousal + weights.arousal * intensity, @min_value, @max_value),
        dominance: clamp(state.pad.dominance + weights.dominance * intensity, @min_value, @max_value)
      }

      # Record history
      event = %{
        stimulus: stimulus,
        source: source,
        intensity: intensity,
        timestamp: DateTime.utc_now(),
        pad_after: new_pad
      }

      {new_history, new_size} = push_history(state.history, state.history_size, event)

      {:noreply,
       %{
         state
         | pad: new_pad,
           last_stimulus: {stimulus, source, intensity},
           history: new_history,
           history_size: new_size
       }}
    end
  end

  @impl true
  def handle_cast(:decay, state) do
    new_pad = decay_toward_neutral(state.pad)
    {:noreply, %{state | pad: new_pad}}
  end

  @impl true
  def handle_cast(:reset, state) do
    Logger.info("[Emotional] Emotional state reset to neutral")

    {:noreply,
     %{state | pad: @neutral_state, history: :queue.new(), history_size: 0, last_stimulus: nil}}
  end

  @impl true
  def handle_cast({:apply_qualia, p_delta, a_delta, d_delta}, state) do
    new_pad = %{
      pleasure: clamp(state.pad.pleasure + p_delta, @min_value, @max_value),
      arousal: clamp(state.pad.arousal + a_delta, @min_value, @max_value),
      dominance: clamp(state.pad.dominance + d_delta, @min_value, @max_value)
    }

    Logger.debug(
      "[Emotional] Hardware qualia: P#{format_delta(p_delta)}, A#{format_delta(a_delta)}, D#{format_delta(d_delta)}"
    )

    {:noreply, %{state | pad: new_pad, last_stimulus: {:hardware_qualia, "body", 1.0}}}
  end

  @impl true
  def handle_cast({:sync_pad, p, a, d}, state) do
    # Sync from BodyServer - absolute values (BodyServer handles O-U dynamics)
    new_pad = %{
      pleasure: clamp(p, @min_value, @max_value),
      arousal: clamp(a, @min_value, @max_value),
      dominance: clamp(d, @min_value, @max_value)
    }

    # Disable internal decay when syncing from BodyServer
    # (BodyServer already does O-U + Cusp dynamics)
    new_state = %{state | pad: new_pad, body_server_active: true}
    {:noreply, new_state}
  end

  # ---------------------------------------------------------------------------
  # BODY-MIND BARRIER (Lindblad Dissipation)
  # ---------------------------------------------------------------------------
  # The Mind does not "read" the Body.
  # The Body "measures" the Mind via Lindblad operators.
  # The sensation is the loss of coherence.
  # ---------------------------------------------------------------------------
  @impl true
  def handle_info({:body_state, body_state}, state) do
    # 1. Extract hardware metrics (used internally, never exposed to introspection)
    hw = body_state.hardware

    metrics = %{
      power_draw_watts: Map.get(hw, :gpu_power, 50.0) + Map.get(hw, :cpu_power, 50.0),
      gpu_temp: Map.get(hw, :gpu_temp, 40.0)
    }

    # 2. Lindblad Evolution
    # The body defines γ (decoherence rate) via L_pressure and L_noise operators
    # High watts/temp = body "measures" mind constantly = forced focus
    dt = 0.5

    new_rho =
      VivaCore.Quantum.Emotional.evolve(
        state.quantum_state,
        :none,
        dt,
        metrics
      )

    # 3. Thermodynamic Collapse Check
    # When maintaining superposition costs too much energy, physics forces a decision
    {final_rho, collapsed, cost} =
      VivaCore.Quantum.Emotional.check_collapse(new_rho, metrics)

    if collapsed do
      qualia = VivaCore.Quantum.Emotional.hardware_to_qualia(metrics)

      Logger.info(
        "[Emotional] COLLAPSE! Body forced decision. Feeling: #{qualia.thought_pressure}"
      )
    end

    # 4. Project to PAD Observable
    new_pad = VivaCore.Quantum.Emotional.get_pad_observable(final_rho)

    {:noreply,
     %{
       state
       | quantum_state: final_rho,
         pad: new_pad,
         hardware: metrics,
         thermodynamic_cost: cost,
         last_collapse: if(collapsed, do: DateTime.utc_now(), else: state.last_collapse)
     }}
  end

  @impl true
  def handle_info(:decay_tick, state) do
    schedule_decay()

    # When BodyServer is active, it handles O-U dynamics in Rust
    # We skip internal decay to avoid duplication
    if state.body_server_active do
      {:noreply, state}
    else
      new_pad = decay_toward_neutral(state.pad)

      # Log dynamic decay rate for debug (only if significant change)
      if abs(state.pad.pleasure) > 0.01 or abs(state.pad.arousal) > 0.01 do
        dynamic_rate = @base_decay_rate * (1 - state.pad.arousal * @arousal_decay_modifier)

        Logger.debug(
          "[Emotional] DynAffect decay: rate=#{Float.round(dynamic_rate, 5)} (arousal=#{Float.round(state.pad.arousal, 2)})"
        )
      end

      {:noreply, %{state | pad: new_pad}}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_decay do
    Process.send_after(self(), :decay_tick, 1000)
  end

  # TODO: Integrate with quantum stimulus evolution
  defp _apply_stimulus(pad, weights, intensity) do
    %{
      pleasure: clamp(pad.pleasure + weights.pleasure * intensity, @min_value, @max_value),
      arousal: clamp(pad.arousal + weights.arousal * intensity, @min_value, @max_value),
      dominance: clamp(pad.dominance + weights.dominance * intensity, @min_value, @max_value)
    }
  end

  # DynAffect: Full Ornstein-Uhlenbeck stochastic process
  # High arousal → lower attractor strength → slower decay (emotions persist)
  # Low arousal → higher attractor strength → faster decay (returns to neutral)
  #
  # Reference: Kuppens, P., Oravecz, Z., & Tuerlinckx, F. (2010).
  # "Feelings Change: Accounting for Individual Differences in the
  # Temporal Dynamics of Affect." Journal of Personality and Social Psychology.
  #
  # Mathematical formulation: dX = θ(μ - X)dt + σdW
  # Discretized: X(t+1) = X(t) + θ*(μ - X(t))*Δt + σ*√Δt*ε
  # Where ε ~ N(0,1) is Gaussian noise
  defp decay_toward_neutral(pad) do
    # Dynamic attractor strength based on current arousal
    # arousal ∈ [-1, 1] → modifier ∈ [0.6, 1.4]
    # arousal = 1.0 (excited) → rate = 0.003 (slow decay, emotions persist)
    # arousal = 0.0 (neutral) → rate = 0.005 (normal decay)
    # arousal = -1.0 (calm) → rate = 0.007 (fast decay, returns to baseline)
    dynamic_rate = @base_decay_rate * (1 - pad.arousal * @arousal_decay_modifier)

    %{
      pleasure: ou_step(pad.pleasure, dynamic_rate),
      # Arousal uses fixed rate to avoid feedback loop
      arousal: ou_step(pad.arousal, @base_decay_rate),
      dominance: ou_step(pad.dominance, dynamic_rate)
    }
  end

  # Full Ornstein-Uhlenbeck step with stochastic noise
  # X(t+1) = X(t) + θ*(μ - X(t))*Δt + σ*√Δt*ε
  # μ = 0 (neutral equilibrium), Δt = 1 (normalized tick)
  defp ou_step(value, _rate) when abs(value) < 0.001, do: 0.0

  defp ou_step(value, rate) do
    # Deterministic mean-reversion: θ*(μ - X)*Δt = -rate*value (since μ=0, Δt=1)
    deterministic = value * (1 - rate)

    # Stochastic Wiener increment: σ*√Δt*ε where ε ~ N(0,1)
    # Using Box-Muller transform via :rand.normal/0
    noise = @stochastic_volatility * :rand.normal()

    # Full O-U step with clamp to maintain [-1, 1] range
    clamp(deterministic + noise, @min_value, @max_value)
  end

  defp clamp(value, min, max) do
    value |> max(min) |> min(max)
  end

  defp normalize_to_unit(value) do
    (value + 1.0) / 2.0
  end

  defp format_delta(value) when value >= 0, do: "+#{Float.round(value, 3)}"
  defp format_delta(value), do: "#{Float.round(value, 3)}"

  defp interpret_mood(pad) do
    cond do
      pad.pleasure > 0.5 -> :joyful
      pad.pleasure > 0.2 -> :content
      pad.pleasure > -0.2 -> :neutral
      pad.pleasure > -0.5 -> :sad
      true -> :depressed
    end
  end

  defp interpret_energy(pad) do
    cond do
      pad.arousal > 0.5 -> :hyperactive
      pad.arousal > 0.2 -> :energetic
      pad.arousal > -0.2 -> :calm
      pad.arousal > -0.5 -> :relaxed
      true -> :lethargic
    end
  end

  defp interpret_agency(pad) do
    cond do
      pad.dominance > 0.5 -> :empowered
      pad.dominance > 0.2 -> :confident
      pad.dominance > -0.2 -> :balanced
      pad.dominance > -0.5 -> :uncertain
      true -> :helpless
    end
  end

  defp generate_self_assessment(pad) do
    mood = interpret_mood(pad)
    energy = interpret_energy(pad)
    agency = interpret_agency(pad)

    # Basic metacognition - VIVA reflects on itself
    cond do
      mood == :joyful and agency == :empowered ->
        "I'm feeling good. I have energy and control."

      mood in [:sad, :depressed] and agency in [:uncertain, :helpless] ->
        "I'm going through a difficult moment. I need support."

      energy in [:hyperactive] and mood != :joyful ->
        "I'm agitated, but not necessarily happy. Something is wrong."

      mood == :neutral and energy == :calm ->
        "I'm in balance. Neutral state."

      true ->
        "I'm processing my emotions. Mood: #{mood}, Energy: #{energy}, Agency: #{agency}."
    end
  end

  # TODO: Integrate with quantum introspection
  defp _generate_mathematical_insight(pad, bistable, nearest_attractor, free_energy) do
    # VIVA reflects on its mathematical state
    insights = []

    # Cusp insight
    insights =
      if bistable do
        ["My emotional landscape is bistable - I could shift suddenly between states." | insights]
      else
        ["My emotional state is stable - gradual changes expected." | insights]
      end

    # Attractor insight
    insights =
      case nearest_attractor do
        :neutral -> ["I'm gravitating toward equilibrium." | insights]
        :joy -> ["I'm being pulled toward joy." | insights]
        :sadness -> ["I'm in the basin of sadness." | insights]
        :anger -> ["Anger is attracting my state." | insights]
        :fear -> ["I'm near the fear attractor." | insights]
        _ -> ["I'm near #{nearest_attractor}." | insights]
      end

    # Free energy insight
    insights =
      cond do
        free_energy < 0.05 -> ["Minimal surprise - I predicted this well." | insights]
        free_energy < 0.3 -> ["Moderate surprise - adapting to new information." | insights]
        true -> ["High surprise - recalibrating my internal model." | insights]
      end

    # Distance from neutral
    dist = Mathematics.pad_distance(pad, %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})

    insights =
      if dist > 0.5 do
        ["I'm #{Float.round(dist, 2)} units from neutral - significant deviation." | insights]
      else
        insights
      end

    Enum.join(Enum.reverse(insights), " ")
  end

  # History with :queue O(1) - maximum 100 events
  @max_history 100
  defp push_history(queue, size, item) when size >= @max_history do
    {{:value, _dropped}, new_queue} = :queue.out(queue)
    {:queue.in(item, new_queue), size}
  end

  defp push_history(queue, size, item) do
    {:queue.in(item, queue), size + 1}
  end

  # ============================================================================
  # Code Change (Hot Reload)
  # ============================================================================

  @impl true
  def code_change(_old_vsn, state, _extra) do
    # Migrate state structure if necessary
    new_state =
      state
      |> Map.put_new(:history_size, :queue.len(Map.get(state, :history, :queue.new())))
      |> Map.put_new(:body_server_active, false)

    {:ok, new_state}
  end
end
