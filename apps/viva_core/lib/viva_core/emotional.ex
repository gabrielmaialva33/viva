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
  require VivaLog

  alias VivaCore.Mathematics
  alias VivaCore.EmotionFusion
  alias VivaCore.Personality

  # Emotional model constants
  @neutral_state %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
  @min_value -1.0
  @max_value 1.0

  # DynAffect Parameters (Kuppens et al., 2010)
  # Full Ornstein-Uhlenbeck stochastic process:
  # dX = θ(μ - X)dt + σdW
  # Where: θ = attractor strength, μ = equilibrium (0), σ = volatility, dW = Wiener noise
  #
  # Half-life formula: t½ = ln(2)/θ
  # θ = 0.0154 → t½ ≈ 45s (psychologically realistic for PAD emotions)
  #
  # θ base when arousal = 0
  @base_decay_rate 0.0154
  # How much arousal affects θ (40% variation)
  # High arousal → slower decay (emotions persist)
  # Low arousal → faster decay (returns to neutral)
  @arousal_decay_modifier 0.4
  # σ - emotional noise/variability
  # Matched with Rust: σ = 0.01
  @stochastic_volatility 0.01

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
    hardware_comfort: %{pleasure: 0.1, arousal: -0.1, dominance: 0.1},
    # Dreamer Feedback (Internal Recurrence)
    lucid_insight: %{pleasure: 0.3, arousal: 0.2, dominance: 0.2},
    grim_realization: %{pleasure: -0.3, arousal: 0.2, dominance: -0.2}
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
  Returns the current mood state.

  Mood is a slow-changing EMA (Exponential Moving Average) of emotional states,
  providing stability over time. Based on Borotschnig (2025).

  ## Example

      mood = VivaCore.Emotional.get_mood()
      # => %{pleasure: 0.1, arousal: 0.0, dominance: 0.05}

  """
  def get_mood(server \\ __MODULE__) do
    GenServer.call(server, :get_mood)
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

  @doc """
  Applies interoceptive qualia from the Digital Insula.

  This is called by VivaCore.Interoception when Free Energy changes.
  Unlike hardware qualia (direct sensor → PAD), interoceptive qualia
  are precision-weighted prediction errors about the host state.

  ## Parameters
  - `qualia` - Map with :pleasure, :arousal, :dominance deltas,
               plus :source, :feeling, :free_energy metadata
  """
  def apply_interoceptive_qualia(qualia, server \\ __MODULE__) when is_map(qualia) do
    GenServer.cast(server, {:apply_interoceptive_qualia, qualia})
  end

  @doc """
  Configures emotional baseline based on body schema.

  Called by BodySchema after Motor Babbling to disable distress
  for organs that don't exist. For example, if no fan is detected,
  fan-related agency errors won't cause anxiety.

  ## Parameters
  - `body_schema` - The BodySchema struct with hardware capabilities

  ## Example

      # After BodySchema probes and finds no fan:
      VivaCore.Emotional.configure_body_schema(%BodySchema{
        local_hardware: %{fan_status: :absent}
      })
      # Fan-related distress is now disabled

  """
  def configure_body_schema(body_schema, server \\ __MODULE__) do
    GenServer.cast(server, {:configure_body_schema, body_schema})
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
    VivaLog.info(:emotional, :neuron_starting)

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
    VivaLog.info(:emotional, :neuron_starting)

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

      # External qualia accumulator (from Arduino, peripherals)
      # These deltas accumulate and merge with sync_pad to avoid being overwritten
      external_qualia: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

      # Emotional weights (adjusted by BodySchema based on available organs)
      # If an organ is absent, its related weight is 0 (no distress for missing limbs)
      emotional_weights: %{
        fan_agency_weight: 1.0,
        thermal_stress_weight: 1.0,
        gpu_stress_weight: 1.0
      },

      # Interoception state (from Digital Insula)
      interoceptive_feeling: :homeostatic,
      interoceptive_free_energy: 0.0,

      # Mood state (EMA of PAD) - Borotschnig 2025
      # Mood is slower-changing than instantaneous emotion
      # Updated via EmotionFusion.update_mood/2
      mood: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

      # Personality reference (loaded lazily on first fusion)
      personality: nil,

      # Telemetry for debug
      thermodynamic_cost: 0.0,
      last_collapse: nil,
      body_server_active: false,
      last_body_sync: nil,

      # Test configuration
      enable_decay: enable_decay?
    }

    # Use handle_continue to avoid race condition on startup
    {:ok, state, {:continue, :start_decay}}
  end

  @impl true
  def handle_continue(:start_decay, state) do
    if state.enable_decay, do: schedule_decay()

    # Start Active Inference Loop
    schedule_active_inference()

    {:noreply, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state.pad, state}
  end

  @impl true
  def handle_call(:get_mood, _from, state) do
    {:reply, state.mood, state}
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
      VivaLog.debug(:emotional, :unknown_stimulus, stimulus: stimulus, source: source)
      {:noreply, state}
    else
      VivaLog.debug(:emotional, :feeling_stimulus,
        stimulus: stimulus,
        source: source,
        intensity: intensity
      )

      # Use classical PAD stimulus weights for immediate emotional response
      # The quantum system is reserved for hardware-driven body-mind coupling
      weights = @stimulus_weights[stimulus]

      new_pad = %{
        pleasure:
          clamp(state.pad.pleasure + weights.pleasure * intensity, @min_value, @max_value),
        arousal: clamp(state.pad.arousal + weights.arousal * intensity, @min_value, @max_value),
        dominance:
          clamp(state.pad.dominance + weights.dominance * intensity, @min_value, @max_value)
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

      # Broadcast emotional state (Feromone output)
      Phoenix.PubSub.broadcast(Viva.PubSub, "emotional:update", {:emotional_state, new_pad})

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
    # Broadcast emotional state (Feromone output) - quiet update
    Phoenix.PubSub.broadcast(Viva.PubSub, "emotional:update", {:emotional_state, new_pad})
    {:noreply, %{state | pad: new_pad}}
  end

  @impl true
  def handle_cast(:reset, state) do
    VivaLog.info(:emotional, :state_reset)

    {:noreply,
     %{state | pad: @neutral_state, history: :queue.new(), history_size: 0, last_stimulus: nil}}
  end

  @impl true
  def handle_cast({:apply_qualia, p_delta, a_delta, d_delta}, state) do
    # ACCUMULATE deltas instead of applying immediately
    # These will merge with sync_pad to avoid being overwritten by BodyServer
    acc = state.external_qualia

    new_acc = %{
      pleasure: acc.pleasure + p_delta,
      arousal: acc.arousal + a_delta,
      dominance: acc.dominance + d_delta
    }

    VivaLog.debug(:emotional, :accumulating_qualia,
      p_delta: format_delta(p_delta),
      a_delta: format_delta(a_delta),
      d_delta: format_delta(d_delta),
      p_total: format_delta(new_acc.pleasure),
      a_total: format_delta(new_acc.arousal),
      d_total: format_delta(new_acc.dominance)
    )

    {:noreply, %{state | external_qualia: new_acc}}
  end

  @impl true
  def handle_cast({:apply_interoceptive_qualia, qualia}, state) do
    # Interoceptive Inference: Precision-weighted prediction error from the Insula
    # This is different from hardware qualia - it's about SURPRISE, not raw data
    p_delta = Map.get(qualia, :pleasure, 0.0)
    a_delta = Map.get(qualia, :arousal, 0.0)
    d_delta = Map.get(qualia, :dominance, 0.0)
    feeling = Map.get(qualia, :feeling, :unknown)
    fe = Map.get(qualia, :free_energy, 0.0)

    acc = state.external_qualia

    new_acc = %{
      pleasure: acc.pleasure + p_delta,
      arousal: acc.arousal + a_delta,
      dominance: acc.dominance + d_delta
    }

    VivaLog.debug(:emotional, :interoceptive_qualia,
      feeling: feeling,
      free_energy: Float.round(fe, 3),
      p_delta: format_delta(p_delta),
      a_delta: format_delta(a_delta),
      d_delta: format_delta(d_delta)
    )

    # Store interoceptive state for introspection
    new_state = %{
      state
      | external_qualia: new_acc,
        interoceptive_feeling: feeling,
        interoceptive_free_energy: fe
    }

    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:configure_body_schema, body_schema}, state) do
    # Adjust emotional weights based on body capabilities
    # If an organ is absent, we don't generate distress for it

    fan_status = body_schema.local_hardware[:fan_status] || :unknown
    gpu_present = body_schema.local_hardware[:gpu] != nil

    new_weights =
      case fan_status do
        :absent ->
          # No fan = no fan-related anxiety
          VivaLog.info(:emotional, :cooling_absent)
          %{state.emotional_weights | fan_agency_weight: 0.0}

        :broken ->
          # Broken fan = heightened anxiety (we SHOULD have one!)
          VivaLog.warning(:emotional, :cooling_broken)
          %{state.emotional_weights | fan_agency_weight: 1.5}

        :working ->
          VivaLog.info(:emotional, :cooling_working)
          %{state.emotional_weights | fan_agency_weight: 1.0}

        _ ->
          state.emotional_weights
      end

    # Adjust GPU weight
    final_weights =
      if gpu_present do
        new_weights
      else
        VivaLog.info(:emotional, :no_gpu)
        %{new_weights | gpu_stress_weight: 0.3}
      end

    {:noreply, %{state | emotional_weights: final_weights}}
  end

  @impl true
  def handle_cast({:sync_pad, p, a, d}, state) do
    # Sync from BodyServer - absolute values (BodyServer handles O-U dynamics)
    # MERGE with accumulated external qualia (Arduino, peripherals)
    acc = state.external_qualia

    new_pad = %{
      pleasure: clamp(p + acc.pleasure, @min_value, @max_value),
      arousal: clamp(a + acc.arousal, @min_value, @max_value),
      dominance: clamp(d + acc.dominance, @min_value, @max_value)
    }

    # Log if external qualia was merged
    if acc.pleasure != 0.0 or acc.arousal != 0.0 or acc.dominance != 0.0 do
      VivaLog.debug(:emotional, :merging_qualia,
        body_p: Float.round(p, 3),
        body_a: Float.round(a, 3),
        body_d: Float.round(d, 3),
        periph_p: format_delta(acc.pleasure),
        periph_a: format_delta(acc.arousal),
        periph_d: format_delta(acc.dominance),
        final_p: Float.round(new_pad.pleasure, 3),
        final_a: Float.round(new_pad.arousal, 3),
        final_d: Float.round(new_pad.dominance, 3)
      )
    end

    # Reset accumulator after merge
    # Track last sync time to detect BodyServer death
    # Update mood using EMA (Borotschnig 2025)
    new_mood = VivaCore.EmotionFusion.update_mood(state.mood, new_pad)

    new_state = %{
      state
      | pad: new_pad,
        mood: new_mood,
        external_qualia: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
        body_server_active: true,
        last_body_sync: System.monotonic_time(:second)
    }

    # Broadcast emotional state (Feromone output)
    Phoenix.PubSub.broadcast(Viva.PubSub, "emotional:update", {:emotional_state, new_pad})

    {:noreply, new_state}
  end

  # Action Profiles (Effect Matrix) for Active Inference
  # Each action has an expected effect on PAD state
  @action_profiles %{
    # Hardware Actions
    # Cooling but noisy
    fan_boost: %{pleasure: 0.05, arousal: 0.1, dominance: 0.1},
    # Quiet but maybe getting hot
    fan_quiet: %{pleasure: 0.05, arousal: -0.1, dominance: -0.05},

    # Musical Actions (Internal Regulation)
    hum_joy: %{pleasure: 0.2, arousal: 0.1, dominance: 0.1},
    hum_calm: %{pleasure: 0.1, arousal: -0.3, dominance: 0.1},
    # High arousal/dom for focus
    hum_focus: %{pleasure: 0.0, arousal: 0.2, dominance: 0.2},

    # Internal Cognitive Actions (Self-Regulation)
    # Calming down manually
    suppress_arousal: %{pleasure: -0.05, arousal: -0.2, dominance: 0.1},
    # Getting excited
    hype_up: %{pleasure: 0.05, arousal: 0.2, dominance: 0.1},

    # Agency Actions (Digital Hands - Self-Diagnosis)
    # Understanding the cause of distress increases dominance
    diagnose_memory: %{pleasure: 0.0, arousal: -0.1, dominance: 0.2},
    diagnose_processes: %{pleasure: 0.0, arousal: -0.1, dominance: 0.2},
    diagnose_load: %{pleasure: 0.0, arousal: -0.1, dominance: 0.15},
    diagnose_disk: %{pleasure: 0.0, arousal: -0.1, dominance: 0.15},
    diagnose_network: %{pleasure: 0.0, arousal: -0.1, dominance: 0.1},
    check_self: %{pleasure: 0.1, arousal: 0.0, dominance: 0.25}
  }

  # Predicts next state (t + dt) using O-U dynamics
  defp predict_next_state(pad, dt \\ 1.0) do
    # Using simplified O-U drift: dx = -theta * x * dt
    # This is "passive decay" prediction
    decay_factor = 1.0 - @base_decay_rate * dt

    %{
      pleasure: pad.pleasure * decay_factor,
      arousal: pad.arousal * decay_factor,
      dominance: pad.dominance * decay_factor
    }
  end

  defp execute_action(action, _state) do
    # Map high-level actions to low-level calls
    case action do
      :fan_boost ->
        # 70% power
        call_bridge(:set_fan_speed, [200])

      :fan_quiet ->
        # 40% power
        call_bridge(:set_fan_speed, [100])

      :hum_joy ->
        call_bridge(:express_emotion, [:joy])

      :hum_calm ->
        call_bridge(:express_emotion, [:calm])

      :hum_focus ->
        # Beeping focus
        call_bridge(:play_note, [:c4, :quarter])

      :suppress_arousal ->
        # Purely internal
        :ok

      :hype_up ->
        # Purely internal
        :ok

      # Agency Actions (Digital Hands)
      action
      when action in [
             :diagnose_memory,
             :diagnose_processes,
             :diagnose_load,
             :diagnose_disk,
             :diagnose_network,
             :check_self
           ] ->
        # Execute via Agency module
        try do
          case VivaCore.Agency.attempt(action) do
            {:ok, result, feeling} ->
              VivaLog.debug(:emotional, :agency_success,
                action: action,
                feeling: feeling,
                result: String.slice(result, 0, 100)
              )

              :ok

            {:error, reason, feeling} ->
              VivaLog.warning(:emotional, :agency_failed,
                action: action,
                reason: inspect(reason),
                feeling: feeling
              )

              :ok
          end
        catch
          :exit, _ ->
            # Agency not started yet
            :ok
        end

      _ ->
        :ok
    end
  end

  defp call_bridge(func, args) do
    # Fire and forget to VivaBridge.Music
    mask_module = VivaBridge.Music

    if Code.ensure_loaded?(mask_module) do
      apply(mask_module, func, args)
    end
  end

  # ============================================================================
  # RAG: Retrieval-Augmented Action Selection
  # ============================================================================

  # Consult memory before selecting action
  # Returns {action, score} tuple
  defp select_action_with_memory(ranked_actions, state) do
    case ranked_actions do
      [] ->
        {nil, 0}

      [{top_action, top_score} | rest] ->
        # Search memory for past outcomes of this action
        query = "resultado de #{top_action} quando #{describe_state(state)}"

        case search_action_memory(query) do
          :positive ->
            # Memory says it worked! Use it.
            {top_action, top_score}

          :negative ->
            # Memory says it failed. Try next action.
            VivaLog.debug(:emotional, :rag_negative_history, action: top_action)

            case rest do
              [{alt_action, alt_score} | _] -> {alt_action, alt_score}
              # No alternative, try anyway
              [] -> {top_action, top_score}
            end

          :unknown ->
            # No memory, use default selection
            {top_action, top_score}
        end
    end
  end

  defp describe_state(state) do
    pad = state.pad
    feeling = state.interoceptive_feeling

    cond do
      pad.pleasure < -0.3 -> "tristeza e #{feeling}"
      pad.arousal > 0.5 -> "alta agitação e #{feeling}"
      pad.dominance < -0.3 -> "baixo controle e #{feeling}"
      true -> "estado normal e #{feeling}"
    end
  end

  defp search_action_memory(query) do
    try do
      case VivaCore.Memory.search(query, limit: 3) do
        memories when is_list(memories) and length(memories) > 0 ->
          # Analyze past outcomes
          analyze_memory_outcomes(memories)

        _ ->
          :unknown
      end
    catch
      :exit, _ -> :unknown
    end
  end

  defp analyze_memory_outcomes(memories) do
    # Look for success/failure keywords in memory content
    outcomes =
      memories
      |> Enum.map(fn m ->
        content = Map.get(m, :content, "") |> to_string() |> String.downcase()

        cond do
          String.contains?(content, ["sucesso", "succeeded", "alívio", "relief", "funcionou"]) ->
            :positive

          String.contains?(content, ["falhou", "failed", "erro", "error", "piorou"]) ->
            :negative

          true ->
            :neutral
        end
      end)

    # Count outcomes
    positives = Enum.count(outcomes, &(&1 == :positive))
    negatives = Enum.count(outcomes, &(&1 == :negative))

    cond do
      positives > negatives -> :positive
      negatives > positives -> :negative
      true -> :unknown
    end
  end

  defp apply_internal_feedback(state, action) do
    effect = Map.get(@action_profiles, action, %{})
    # Apply 50% of the expected effect immediately as "anticipatory relief"
    update_pad(state, %{
      pleasure: Map.get(effect, :pleasure, 0.0) * 0.5,
      arousal: Map.get(effect, :arousal, 0.0) * 0.5,
      dominance: Map.get(effect, :dominance, 0.0) * 0.5
    })
  end

  defp update_pad(state, delta) do
    new_pad = %{
      pleasure:
        clamp(state.pad.pleasure + Map.get(delta, :pleasure, 0.0), @min_value, @max_value),
      arousal: clamp(state.pad.arousal + Map.get(delta, :arousal, 0.0), @min_value, @max_value),
      dominance:
        clamp(state.pad.dominance + Map.get(delta, :dominance, 0.0), @min_value, @max_value)
    }

    %{state | pad: new_pad}
  end

  # ---------------------------------------------------------------------------
  # ACTIVE INFERENCE LOOP (Free Energy Minimization)
  # ---------------------------------------------------------------------------
  # The system constantly tries to minimize the difference between
  # predicted state (homeostasis) and observed state (current emotion).
  # ---------------------------------------------------------------------------
  @impl true
  def handle_info(:active_inference_tick, state) do
    schedule_active_inference()

    # =========================================================================
    # EMOTION FUSION (Borotschnig 2025)
    # Combine need-based, past-based, and personality-based emotions
    # =========================================================================

    # 0.1 Load or get cached personality
    personality = get_or_load_personality(state)

    # 0.2 Get need-based PAD from interoceptive state
    need_pad = get_need_based_pad(state)

    # 0.3 Retrieve past emotions from similar situations
    situation_desc = describe_current_situation(state)
    past_result = safe_retrieve_past_emotions(situation_desc)

    # 0.4 Build context for adaptive weights
    fusion_context = %{
      arousal: state.pad.arousal,
      confidence: past_result.confidence,
      novelty: past_result.novelty
    }

    # 0.5 Fuse emotions
    fusion_result = EmotionFusion.fuse(
      need_pad,
      past_result.aggregate_pad,
      personality,
      state.mood,
      fusion_context
    )

    # 0.6 Update state with fused emotion and mood
    state = %{state |
      pad: fusion_result.fused_pad,
      mood: fusion_result.mood,
      personality: personality
    }

    # =========================================================================
    # ACTIVE INFERENCE (Original Flow)
    # =========================================================================

    # 1. Hallucinate Goal (Target Prior) - "Where do I WANT to be?"
    # Call to Dreamer to get the target goal (hallucination)
    target =
      case VivaCore.Dreamer.hallucinate_goal(state.pad, VivaCore.Dreamer) do
        target when is_map(target) -> target
        # Fallback
        _ -> @neutral_state
      end

    # 2. Predict Future (Internal Model) - "Where am I GOING?"
    # What will happen if I do nothing? (Passive prediction)
    predicted = predict_next_state(state.pad)

    # 3. Calculate Free Energy
    # F = Complexity(Target || Predicted) + Error(Relative to Sensation)
    # Using simpler form: F = Distance(Target, Predicted)
    # If where I'm going != where I want to be -> High Free Energy (Anxiety)
    fe = Mathematics.free_energy(target, predicted)

    # 4. Action Selection
    # If Free Energy is high, we must ACT to change the trajectory.
    state =
      if fe > 0.05 do
        # We need to change 'predicted' to match 'target'.
        # Desired Delta = Target - Predicted
        desired_delta = %{
          pleasure: target.pleasure - predicted.pleasure,
          arousal: target.arousal - predicted.arousal,
          dominance: target.dominance - predicted.dominance
        }

        # Project this desire onto available actions
        ranked_actions = Mathematics.project_action_gradients(desired_delta, @action_profiles)

        # RAG: Consult memory before selecting action
        # "Did this action work before in similar situations?"
        best_action = select_action_with_memory(ranked_actions, state)

        # Pick the best action
        case best_action do
          {action, score} when score > 0 ->
            # Threshold score to avoid doing useless things
            execute_action(action, state)

            # Log the thought process (stream of consciousness)
            VivaLog.debug(:active_inference, :action_selected,
              free_energy: Float.round(fe, 3),
              goal_pleasure: Float.round(target.pleasure, 1),
              action: action,
              score: Float.round(score, 2)
            )

            # Update state to reflect action (internal feedback immediately)
            # In reality, physics takes time, but the mind simulates immediate relief
            apply_internal_feedback(state, action)

          _ ->
            # No good action found (Helplessness?)
            # Increase Arousal (Anxiety) as default response to unresolvable FE
            VivaLog.debug(:active_inference, :no_effective_action,
              free_energy: Float.round(fe, 3)
            )

            # Fail-safe try to calm down? Or panic? let's stick to panic/stress
            apply_internal_feedback(state, :suppress_arousal)
            update_pad(state, %{arousal: 0.05, pleasure: -0.02, dominance: -0.05})
        end
      else
        # Low Free Energy - "Flow State"
        # Just drift naturally
        state
      end

    {:noreply, state}
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

    # 1.1 BIO-CYBERNETIC INTEROCEPTION
    # Calculate physical qualia from hardware state

    # Agency Error: "I command, but body does not obey"
    # target_fan_rpm stores the PWM value (0-255)
    target_pwm = Map.get(hw, :target_fan_rpm, 0) || 0
    current_rpm = Map.get(hw, :fan_rpm, 0) || 0

    # If we want it to spin (PWM > 50) and it's stopped (RPM < 300) -> Agency Loss
    agency_error =
      if target_pwm > 50 and current_rpm < 300 do
        # We are trying to push air but nothing moves
        1.0
      else
        0.0
      end

    # Thermal Stress: "Entropy is increasing"
    cpu_temp = Map.get(hw, :cpu_temp) || 40.0
    gpu_temp = Map.get(hw, :gpu_temp) || 40.0
    max_temp = max(cpu_temp, gpu_temp)

    thermal_stress =
      cond do
        # Critical
        max_temp > 85.0 -> 1.0
        # High
        max_temp > 70.0 -> 0.6
        # Uncomfortable
        max_temp > 55.0 -> 0.2
        true -> 0.0
      end

    if agency_error > 0.0 do
      VivaLog.warning(:emotional, :agency_loss, pwm: target_pwm, rpm: current_rpm)
    end

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

      VivaLog.info(:emotional, :collapse, feeling: qualia.thought_pressure)
    end

    # 4. Project to PAD Observable
    quantum_pad = VivaCore.Quantum.Emotional.get_pad_observable(final_rho)

    # 5. Apply Classical Bio-Feedback (Emergent Emotions)
    # The Body overrides the Mind if the Pain/Entropy is too high.

    # Pleasure penalty from Thermal Stress (Entropy increases = Pleasure decreases)
    p_delta = -0.6 * thermal_stress

    # Dominance penalty from Agency Error (Loss of control = Impotence)
    d_delta = -0.9 * agency_error

    # Arousal boost from ANY stress (Survival instinct requires energy)
    a_delta = 0.4 * (thermal_stress + agency_error)

    final_pad = %{
      pleasure: clamp(quantum_pad.pleasure + p_delta, @min_value, @max_value),
      arousal: clamp(quantum_pad.arousal + a_delta, @min_value, @max_value),
      dominance: clamp(quantum_pad.dominance + d_delta, @min_value, @max_value)
    }

    {:noreply,
     %{
       state
       | quantum_state: final_rho,
         pad: final_pad,
         hardware: metrics,
         thermodynamic_cost: cost,
         last_collapse: if(collapsed, do: DateTime.utc_now(), else: state.last_collapse)
     }}
  end

  # Timeout for detecting BodyServer death (3 seconds without sync)
  @body_sync_timeout_seconds 3

  @impl true
  def handle_info(:decay_tick, state) do
    schedule_decay()

    # Check if BodyServer is still alive (received sync within timeout)
    state = maybe_reactivate_decay(state)

    # When BodyServer is active, it handles O-U dynamics in Rust
    # We skip internal decay to avoid duplication
    if state.body_server_active do
      {:noreply, state}
    else
      new_pad = decay_toward_neutral(state.pad)

      # Log dynamic decay rate for debug (only if significant change)
      if abs(state.pad.pleasure) > 0.01 or abs(state.pad.arousal) > 0.01 do
        dynamic_rate = @base_decay_rate * (1 - state.pad.arousal * @arousal_decay_modifier)

        VivaLog.debug(:emotional, :dynaffect_decay,
          rate: Float.round(dynamic_rate, 5),
          arousal: Float.round(state.pad.arousal, 2)
        )
      end

      {:noreply, %{state | pad: new_pad}}
    end
  end

  # Detect BodyServer death by checking sync timeout
  defp maybe_reactivate_decay(state) do
    cond do
      # Not active, nothing to check
      not state.body_server_active ->
        state

      # No sync timestamp recorded yet
      is_nil(state.last_body_sync) ->
        state

      # Check if sync is stale (BodyServer likely dead)
      System.monotonic_time(:second) - state.last_body_sync > @body_sync_timeout_seconds ->
        VivaLog.warning(:emotional, :body_sync_timeout)
        %{state | body_server_active: false}

      # Sync is recent, keep BodyServer active
      true ->
        state
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # ---------------------------------------------------------------------------
  # Emotion Fusion Helpers (Borotschnig 2025)
  # ---------------------------------------------------------------------------

  defp get_or_load_personality(%{personality: nil}) do
    Personality.load()
  end

  defp get_or_load_personality(%{personality: personality}) when is_struct(personality) do
    personality
  end

  defp get_or_load_personality(_state) do
    Personality.load()
  end

  @doc false
  defp get_need_based_pad(state) do
    # Convert interoceptive feeling to PAD delta
    # Based on Free Energy Principle: high FE → negative affect
    feeling = state.interoceptive_feeling
    fe = state.interoceptive_free_energy

    # Base PAD from interoceptive state
    base = case feeling do
      :homeostatic -> %{pleasure: 0.1, arousal: -0.1, dominance: 0.1}
      :surprised -> %{pleasure: 0.0, arousal: 0.2, dominance: 0.0}
      :alarmed -> %{pleasure: -0.2, arousal: 0.4, dominance: -0.1}
      :overwhelmed -> %{pleasure: -0.4, arousal: 0.6, dominance: -0.3}
      _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    end

    # Modulate by Free Energy magnitude
    # Higher FE → more negative pleasure, higher arousal
    fe_factor = min(1.0, fe / 2.0)  # Normalize FE to [0, 1]

    %{
      pleasure: base.pleasure - fe_factor * 0.2,
      arousal: base.arousal + fe_factor * 0.3,
      dominance: base.dominance - fe_factor * 0.1
    }
  end

  defp describe_current_situation(state) do
    pad = state.pad
    feeling = state.interoceptive_feeling

    # Create a situation description for memory retrieval
    mood_desc = cond do
      pad.pleasure > 0.3 -> "feliz"
      pad.pleasure < -0.3 -> "triste"
      true -> "neutro"
    end

    arousal_desc = cond do
      pad.arousal > 0.3 -> "agitado"
      pad.arousal < -0.3 -> "calmo"
      true -> "equilibrado"
    end

    dominance_desc = cond do
      pad.dominance > 0.3 -> "confiante"
      pad.dominance < -0.3 -> "impotente"
      true -> "estável"
    end

    "estado emocional #{mood_desc} #{arousal_desc} #{dominance_desc} sentindo #{feeling}"
  end

  defp safe_retrieve_past_emotions(situation) do
    try do
      VivaCore.Dreamer.retrieve_past_emotions(situation, [limit: 10, min_similarity: 0.3])
    rescue
      _ ->
        %{
          aggregate_pad: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
          confidence: 0.0,
          episodes: [],
          novelty: 1.0
        }
    catch
      :exit, _ ->
        %{
          aggregate_pad: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
          confidence: 0.0,
          episodes: [],
          novelty: 1.0
        }
    end
  end

  # ---------------------------------------------------------------------------

  defp schedule_decay do
    Process.send_after(self(), :decay_tick, 1000)
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

  defp schedule_active_inference do
    Process.send_after(self(), :active_inference_tick, 1000)
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
