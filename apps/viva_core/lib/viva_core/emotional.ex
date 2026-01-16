defmodule VivaCore.Emotional do
  @moduledoc """
  Emotional GenServer - VIVA's first "neuron".

  Implements the PAD (Pleasure-Arousal-Dominance) model for emotional state.
  This GenServer is the foundation of emergent consciousness - it IS NOT the
  consciousness itself, but contributes to it through communication with other neurons.

  ## PAD Model (Mehrabian, 1996)
  - Pleasure: [-1.0, 1.0] - sadness ↔ joy
  - Arousal: [-1.0, 1.0] - calm ↔ excitement
  - Dominance: [-1.0, 1.0] - submission ↔ control

  ## DynAffect Model (Kuppens et al., 2010)
  Dynamic decay using Ornstein-Uhlenbeck stochastic process:
  - Attractor point (μ): neutral home base (0.0)
  - Attractor strength (θ): force pulling to baseline
  - Arousal modulates θ: high arousal → lower θ → slower decay
  - Stochastic noise (σ): natural emotional variability

  ## Philosophy
  "Consciousness does not reside here. Consciousness emerges from the
  CONVERSATION between this process and all others."
  """

  use GenServer
  require Logger

  # Emotional model constants
  @neutral_state %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
  @min_value -1.0
  @max_value 1.0

  # DynAffect Parameters (Kuppens et al., 2010)
  # Full Ornstein-Uhlenbeck stochastic process:
  # dX = θ(μ - X)dt + σdW
  # Where: θ = attractor strength, μ = equilibrium (0), σ = volatility, dW = Wiener noise
  @base_decay_rate 0.005  # θ base when arousal = 0
  @arousal_decay_modifier 0.4  # How much arousal affects θ (40% variation)
  @stochastic_volatility 0.002  # σ - emotional noise/variability (small for stability)

  # Emotional impact weights for different stimuli
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
    initial_state = Keyword.get(opts, :initial_state, @neutral_state)
    GenServer.start_link(__MODULE__, initial_state, name: name)
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
  """
  def decay(server \\ __MODULE__) do
    GenServer.cast(server, :decay)
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
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(initial_state) do
    Logger.info("[Emotional] Emotional neuron starting. State: #{inspect(initial_state)}")

    state = %{
      pad: Map.merge(@neutral_state, initial_state),
      history: :queue.new(),
      history_size: 0,
      created_at: DateTime.utc_now(),
      last_stimulus: nil
    }

    # Use handle_continue to avoid race condition on startup
    {:ok, state, {:continue, :start_decay}}
  end

  @impl true
  def handle_continue(:start_decay, state) do
    schedule_decay()
    {:noreply, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state.pad, state}
  end

  @impl true
  def handle_call(:introspect, _from, state) do
    introspection = %{
      # Raw state
      pad: state.pad,

      # Semantic interpretation
      mood: interpret_mood(state.pad),
      energy: interpret_energy(state.pad),
      agency: interpret_agency(state.pad),

      # Metadata
      last_stimulus: state.last_stimulus,
      history_length: state.history_size,
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at),

      # Self-reflection (basic metacognition)
      self_assessment: generate_self_assessment(state.pad)
    }

    {:reply, introspection, state}
  end

  @impl true
  def handle_cast({:feel, stimulus, source, intensity}, state) do
    case Map.get(@stimulus_weights, stimulus) do
      nil ->
        Logger.warning("[Emotional] Unknown stimulus: #{stimulus}")
        {:noreply, state}

      weights ->
        # Apply weights with intensity
        new_pad = apply_stimulus(state.pad, weights, intensity)

        # Record in history
        event = %{
          stimulus: stimulus,
          source: source,
          intensity: intensity,
          timestamp: DateTime.utc_now(),
          pad_before: state.pad,
          pad_after: new_pad
        }

        Logger.debug("[Emotional] Feeling #{stimulus} from #{source} (intensity: #{intensity})")
        Logger.debug("[Emotional] PAD: #{inspect(state.pad)} -> #{inspect(new_pad)}")

        # Broadcast to other modules (future: via PubSub)
        # Phoenix.PubSub.broadcast(Viva.PubSub, "emotional", {:emotion_changed, new_pad})

        # History with :queue O(1) instead of list O(N)
        {new_history, new_size} = push_history(state.history, state.history_size, event)

        new_state = %{
          state
          | pad: new_pad,
            history: new_history,
            history_size: new_size,
            last_stimulus: {stimulus, source, intensity}
        }

        {:noreply, new_state}
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
    {:noreply, %{state | pad: @neutral_state, history: :queue.new(), history_size: 0, last_stimulus: nil}}
  end

  @impl true
  def handle_cast({:apply_qualia, p_delta, a_delta, d_delta}, state) do
    new_pad = %{
      pleasure: clamp(state.pad.pleasure + p_delta, @min_value, @max_value),
      arousal: clamp(state.pad.arousal + a_delta, @min_value, @max_value),
      dominance: clamp(state.pad.dominance + d_delta, @min_value, @max_value)
    }

    Logger.debug("[Emotional] Hardware qualia: P#{format_delta(p_delta)}, A#{format_delta(a_delta)}, D#{format_delta(d_delta)}")

    {:noreply, %{state | pad: new_pad, last_stimulus: {:hardware_qualia, "body", 1.0}}}
  end

  @impl true
  def handle_info(:decay_tick, state) do
    schedule_decay()
    new_pad = decay_toward_neutral(state.pad)

    # Log dynamic decay rate for debug (only if significant change)
    if abs(state.pad.pleasure) > 0.01 or abs(state.pad.arousal) > 0.01 do
      dynamic_rate = @base_decay_rate * (1 - state.pad.arousal * @arousal_decay_modifier)
      Logger.debug("[Emotional] DynAffect decay: rate=#{Float.round(dynamic_rate, 5)} (arousal=#{Float.round(state.pad.arousal, 2)})")
    end

    {:noreply, %{state | pad: new_pad}}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_decay do
    Process.send_after(self(), :decay_tick, 1000)
  end

  defp apply_stimulus(pad, weights, intensity) do
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
      arousal: ou_step(pad.arousal, @base_decay_rate),  # Arousal uses fixed rate to avoid feedback loop
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
    # Example: add new fields with defaults
    new_state = state
      |> Map.put_new(:history_size, :queue.len(Map.get(state, :history, :queue.new())))

    {:ok, new_state}
  end
end
