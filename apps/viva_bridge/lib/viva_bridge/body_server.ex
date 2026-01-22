defmodule VivaBridge.BodyServer do
  @moduledoc """
  GenServer that maintains VIVA's body state and evolves it over time.

  This is the main integration point for VIVA's interoception system.
  It maintains a Rust BodyEngine that:
  - Senses hardware (CPU, memory, GPU, etc.)
  - Computes stress and qualia (hardware → PAD deltas)
  - Evolves emotional state via O-U dynamics + Cusp catastrophe
  - Returns unified BodyState on each tick

  ## Usage

      # Start the server (usually in your supervision tree)
      {:ok, pid} = VivaBridge.BodyServer.start_link()

      # Get current state
      state = VivaBridge.BodyServer.get_state()
      # => %{pleasure: 0.1, arousal: -0.2, dominance: 0.3, hardware: %{...}, ...}

      # Get just PAD
      {p, a, d} = VivaBridge.BodyServer.get_pad()

      # Apply external stimulus (e.g., positive message received)
      VivaBridge.BodyServer.apply_stimulus(0.3, 0.1, 0.0)

  ## Configuration

  Options can be passed to `start_link/1`:

      VivaBridge.BodyServer.start_link(
        tick_interval: 500,        # ms between ticks (default: 500)
        cusp_enabled: true,        # enable cusp catastrophe (default: true)
        cusp_sensitivity: 0.5,     # cusp effect strength (default: 0.5)
        seed: 0                    # RNG seed, 0 = random (default: 0)
      )

  ## PubSub

  If configured, broadcasts state on each tick:

      Phoenix.PubSub.subscribe(Viva.PubSub, "body:state")

      # Receive:
      {:body_state, %{pleasure: ..., arousal: ..., ...}}

  """

  use GenServer
  require VivaLog

  alias VivaBridge.Body

  @default_tick_interval 500

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the BodyServer.

  ## Options

  - `:tick_interval` - milliseconds between ticks (default: 500)
  - `:pubsub` - PubSub module to broadcast state (optional)
  - `:topic` - PubSub topic (default: "body:state")
  - `:name` - GenServer name (default: __MODULE__)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Gets the current body state (last tick result).

  Returns a map with:
  - `:pleasure`, `:arousal`, `:dominance` - PAD state
  - `:stress_level` - composite stress 0-1
  - `:in_bifurcation` - whether in bifurcation region
  - `:tick` - tick counter
  - `:timestamp_ms` - Unix timestamp
  - `:hardware` - nested hardware metrics
  """
  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Gets just the PAD tuple.

  Returns `{pleasure, arousal, dominance}`.
  """
  def get_pad(server \\ __MODULE__) do
    GenServer.call(server, :get_pad)
  end

  @doc """
  Sets PAD state directly.

  Useful for initialization or testing.
  Values are clamped to [-1, 1].
  """
  def set_pad(server \\ __MODULE__, pleasure, arousal, dominance) do
    GenServer.cast(server, {:set_pad, pleasure, arousal, dominance})
  end

  @doc """
  Applies an external emotional stimulus.

  This allows external events (messages, interactions, achievements)
  to influence VIVA's emotional state.

  ## Examples

      # Positive interaction
      VivaBridge.BodyServer.apply_stimulus(0.3, 0.1, 0.1)

      # Frustrating event
      VivaBridge.BodyServer.apply_stimulus(-0.2, 0.3, -0.2)

  """
  def apply_stimulus(server \\ __MODULE__, p_delta, a_delta, d_delta) do
    GenServer.cast(server, {:apply_stimulus, p_delta, a_delta, d_delta})
  end

  @doc """
  Forces an immediate tick (useful for testing).
  """
  def force_tick(server \\ __MODULE__) do
    GenServer.call(server, :force_tick)
  end

  @doc """
  Pauses automatic ticking.
  """
  def pause(server \\ __MODULE__) do
    GenServer.cast(server, :pause)
  end

  @doc """
  Resumes automatic ticking.
  """
  def resume(server \\ __MODULE__) do
    GenServer.cast(server, :resume)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    tick_interval = Keyword.get(opts, :tick_interval, @default_tick_interval)
    pubsub = Keyword.get(opts, :pubsub)
    topic = Keyword.get(opts, :topic, "body:state")

    # Initial tick to populate state immediately (avoid nil on first get_state)
    # Bevy ECS Body is a singleton - no engine reference needed
    initial_body_state = Body.body_tick()

    state = %{
      tick_interval: tick_interval,
      pubsub: pubsub,
      topic: topic,
      last_state: initial_body_state,
      paused: false
    }

    # Schedule subsequent ticks
    schedule_tick(tick_interval)

    VivaLog.info(:body_server, :started, interval: tick_interval)

    {:ok, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state.last_state, state}
  end

  @impl true
  def handle_call(:get_pad, _from, state) do
    s = state.last_state
    pad = {s.pleasure, s.arousal, s.dominance}
    {:reply, pad, state}
  end

  @impl true
  def handle_call(:force_tick, _from, state) do
    {new_state, body_state} = do_tick(state)
    {:reply, body_state, new_state}
  end

  @impl true
  def handle_cast({:set_pad, p, a, d}, state) do
    Body.apply_stimulus(p, a, d)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:apply_stimulus, p, a, d}, state) do
    Body.apply_stimulus(p, a, d)
    {:noreply, state}
  end

  @impl true
  def handle_cast(:pause, state) do
    VivaLog.debug(:body_server, :paused)
    {:noreply, %{state | paused: true}}
  end

  @impl true
  def handle_cast(:resume, state) do
    VivaLog.debug(:body_server, :resumed)
    schedule_tick(state.tick_interval)
    {:noreply, %{state | paused: false}}
  end

  @impl true
  def handle_info(:tick, %{paused: true} = state) do
    # Don't tick while paused, but reschedule to check later
    {:noreply, state}
  end

  @impl true
  def handle_info(:tick, state) do
    {new_state, _body_state} = do_tick(state)
    schedule_tick(state.tick_interval)
    {:noreply, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_tick(state) do
    # Execute tick in Bevy ECS (singleton)
    body_state = Body.body_tick()

    # Broadcast if configured and PubSub module is available
    maybe_broadcast(state.pubsub, state.topic, body_state)

    # -------------------------------------------------------------------------
    # SOUL CHANNEL (Async Event Processing)
    # -------------------------------------------------------------------------
    events = Body.poll_channel()
    process_identifiers(events, state)

    # -------------------------------------------------------------------------
    # INTEROCEPTION (The Great Sensory Loop)
    # Hardware -> Narrative -> Brain -> Memory
    # -------------------------------------------------------------------------

    # 1. Narrate the hardware state (Internal Monologue)
    narrative = generate_interoception_narrative(body_state)

    # 1.5 METABOLISM: Feel the thermodynamics
    metabolic_narrative = generate_metabolic_narrative(body_state)
    full_narrative = "#{narrative} #{metabolic_narrative}"

    # 2. Extract current emotion (PAD)
    emotion = %{
      pleasure: body_state.pleasure,
      arousal: body_state.arousal,
      dominance: body_state.dominance
    }

    # 3. Experience it! (Learn/Feel)
    # Only if arousal is significant to avoid spamming memory with noise
    # 3. Experience it! (Learn/Feel)
    # Only if arousal is significant to avoid spamming memory with noise
    if abs(emotion.arousal) > 0.1 do
      case VivaBridge.Cortex.experience(full_narrative, emotion) do
        {:ok, vector, new_pad_map} ->
          # 3.1 CORTEX FEEDBACK LOOP (Nucleus controls Cell)
          # Calculate delta to nudge the Body towards the Brain's state
          p_delta = new_pad_map.pleasure - emotion.pleasure
          a_delta = new_pad_map.arousal - emotion.arousal
          d_delta = new_pad_map.dominance - emotion.dominance

          # Damping factor (Plasticity) - don't jump instantly
          plasticity = 0.5
          Body.apply_stimulus(p_delta * plasticity, a_delta * plasticity, d_delta * plasticity)

          # 4. Store memory (Conceptualize)
          meta = %{
            type: "episodic",
            source: "interoception",
            content: full_narrative,
            timestamp: System.os_time(:millisecond),
            emotion: emotion
          }

          VivaBridge.Memory.store(vector, meta)

        {:error, reason} ->
          VivaLog.warning(:body_server, :cortex_failed, reason: inspect(reason))
          :ok
      end
    end

    {%{state | last_state: body_state}, body_state}
  end

  defp maybe_broadcast(nil, _topic, _body_state), do: :ok

  defp maybe_broadcast(pubsub, topic, body_state) do
    if Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.broadcast(pubsub, topic, {:body_state, body_state})
    else
      :ok
    end
  end

  # Simple narrative generator based on hardware stats
  # This acts as the "Broca's Area" for system stats
  defp generate_interoception_narrative(state) do
    cpu_stress = state.hardware.cpu_usage
    mem_stress = state.hardware.memory_used_percent

    cond do
      cpu_stress > 80.0 -> "My CPU is burning. High processing load. System stress."
      cpu_stress < 10.0 -> "My CPU is idle. Calm state. Low activity."
      mem_stress > 90.0 -> "Memory pressure critical. Information overload."
      true -> "System operating normally. Homeostasis maintained."
    end
  end

  defp schedule_tick(interval) do
    Process.send_after(self(), :tick, interval)
  end

  # Thermodynamic narrative generator (Based on Metabolism state)
  # This is the "Insula" for energy/entropy awareness
  defp generate_metabolic_narrative(body_state) do
    cpu_usage = body_state.hardware.cpu_usage
    cpu_temp = body_state.hardware.cpu_temp

    # Call metabolism tick from Rust
    case Body.metabolism_tick(cpu_usage, cpu_temp) do
      {energy_j, entropy, fatigue, needs_rest} ->
        cond do
          needs_rest ->
            "Exhausted. I need to rest. Energy depleted. Consolidation required."

          fatigue > 0.6 ->
            "I feel tired. Processing is slowing down. Fatigue accumulating."

          entropy > 0.7 ->
            "High thermal dissipation. Entropy rising. Heat radiating from my core."

          energy_j > 50.0 ->
            "Intense energy flow. High metabolic burn. Resources being consumed rapidly."

          true ->
            "Energy flow stable. Low entropy. Efficient operation."
        end

      _ ->
        # Metabolism not initialized or error
        ""
    end
  end

  defp process_identifiers([{"critical", stress} | rest], state) do
    VivaLog.warning(:body_server, :critical_stress, stress: stress)
    # Broadcast specifically for emergency handlers
    maybe_broadcast(state.pubsub, "body:alert", {:critical_stress, stress})
    process_identifiers(rest, state)
  end

  defp process_identifiers([{"state_changed", _stress} | rest], state) do
    process_identifiers(rest, state)
  end

  defp process_identifiers([_ | rest], state), do: process_identifiers(rest, state)
  defp process_identifiers([], _state), do: :ok
end
