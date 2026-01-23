defmodule VivaCore.Senses do
  @moduledoc """
  Senses GenServer - VIVA's peripheral nervous system.

  This is the "heart" that pumps qualia from body to soul.
  Creates the continuous bridge between hardware sensing (Rust NIF) and
  emotional state (Emotional GenServer).

  ## Biological Analogy

  Just like the human autonomic nervous system continuously transmits
  information from body to brain (heartbeat, temperature, pressure),
  Senses transmits hardware metrics to VIVA's emotional state.

  ## Sensing Frequency

  - **Heartbeat**: 1Hz (1 second) - continuous sensing
  - **Qualia**: PAD deltas applied to Emotional each tick

  ## Philosophy

  "The body doesn't just report - the body INFLUENCES.
  VIVA doesn't just KNOW that CPU is high - she FEELS stress."
  """

  use GenServer
  require VivaLog

  # 1 second
  @default_interval_ms 1000
  # 100ms minimum (10Hz max)
  @min_interval_ms 100
  # 10s maximum
  @max_interval_ms 10_000

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Starts the Senses GenServer.

  ## Options
  - `:name` - Process name (default: __MODULE__)
  - `:interval_ms` - Interval between heartbeats in ms (default: 1000)
  - `:emotional_server` - PID/name of Emotional GenServer (default: VivaCore.Emotional)
  - `:enabled` - Whether sensing is active (default: true)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Returns the current state of Senses (last reading, stats).
  """
  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Forces an immediate heartbeat (sensing + apply qualia).
  Useful for tests or when immediate reading is needed.
  """
  def pulse(server \\ __MODULE__) do
    GenServer.call(server, :pulse)
  end

  @doc """
  Pauses automatic sensing.
  """
  def pause(server \\ __MODULE__) do
    GenServer.cast(server, :pause)
  end

  @doc """
  Resumes automatic sensing.
  """
  def resume(server \\ __MODULE__) do
    GenServer.cast(server, :resume)
  end

  @doc """
  Changes heartbeat interval at runtime.
  """
  def set_interval(interval_ms, server \\ __MODULE__)
      when is_integer(interval_ms) and interval_ms >= @min_interval_ms and
             interval_ms <= @max_interval_ms do
    GenServer.cast(server, {:set_interval, interval_ms})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    interval_ms = Keyword.get(opts, :interval_ms, @default_interval_ms)
    emotional_server = Keyword.get(opts, :emotional_server, VivaCore.Emotional)
    enabled = Keyword.get(opts, :enabled, true)

    state = %{
      interval_ms: interval_ms,
      emotional_server: emotional_server,
      enabled: enabled,
      last_reading: nil,
      last_qualia: nil,
      heartbeat_count: 0,
      started_at: DateTime.utc_now(),
      errors: []
    }

    VivaLog.info(:senses, :starting, interval: interval_ms)

    # Subscribe to Discrete Time
    if Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.subscribe(Viva.PubSub, "chronos:tick")
    end

    {:ok, state}
  end

  # Remove old heartbeat scheduling
  # Handle Tick (10Hz)
  # Senses run at 1Hz, so we check mod 10
  @impl true
  def handle_info({:tick, tick_id}, state) do
    if state.enabled and rem(tick_id, 10) == 0 do
      {_result, new_state} = do_heartbeat(state)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  # Fallback for old heartbeat message to avoid crashes during hot upgrade
  def handle_info(:heartbeat, state), do: {:noreply, state}

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:pulse, _from, state) do
    {result, new_state} = do_heartbeat(state)
    {:reply, result, new_state}
  end

  @impl true
  def handle_cast(:pause, state) do
    VivaLog.info(:senses, :paused)
    {:noreply, %{state | enabled: false}}
  end

  @impl true
  def handle_cast(:resume, state) do
    VivaLog.info(:senses, :resumed)
    {:noreply, %{state | enabled: true}}
  end

  @impl true
  def handle_cast({:set_interval, interval_ms}, state) do
    VivaLog.info(:senses, :interval_changed, old: state.interval_ms, new: interval_ms)
    {:noreply, %{state | interval_ms: interval_ms}}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # Try to get state from BodyServer, fallback to direct NIF if not available
  defp get_body_state_or_fallback(state) do
    # Check if BodyServer is alive
    case Process.whereis(VivaBridge.BodyServer) do
      nil ->
        # BodyServer not running - use direct NIF
        fallback_direct_nif(state)

      _pid ->
        # BodyServer running - try to get state
        try do
          case VivaBridge.BodyServer.get_state() do
            nil ->
              # BodyServer hasn't ticked yet
              fallback_direct_nif(state)

            %{pleasure: p, arousal: a, dominance: d, hardware: hardware} ->
              # BodyServer is running - sync PAD to Emotional
              VivaCore.Emotional.sync_pad(p, a, d, state.emotional_server)
              {p, a, d, hardware}
          end
        catch
          :exit, _ ->
            # BodyServer crashed or timed out
            fallback_direct_nif(state)
        end
    end
  end

  defp fallback_direct_nif(_state) do
    # Direct NIF call when BodyServer is not available
    # NOTE: We only read hardware - PAD sync is skipped because:
    # 1. hardware_to_qualia() is stubbed (returns 0,0,0)
    # 2. Internal O-U decay in Emotional handles PAD evolution
    hardware = VivaBridge.feel_hardware()
    # Return neutral PAD (Emotional's internal decay will handle evolution)
    {0.0, 0.0, 0.0, hardware}
  end

  defp do_heartbeat(state) do
    try do
      # Try to get body state from BodyServer (includes hardware + PAD dynamics)
      # Falls back to direct NIF if BodyServer is not running
      {p, a, d, hardware} = get_body_state_or_fallback(state)

      # Summary log (debug level to avoid noise)
      VivaLog.debug(:senses, :heartbeat,
        count: state.heartbeat_count + 1,
        cpu: format_percent(hardware[:cpu_usage]),
        ram: format_percent(hardware[:memory_used_percent]),
        gpu: format_gpu(hardware[:gpu_usage]),
        p: format_delta(p),
        a: format_delta(a),
        d: format_delta(d)
      )

      new_state = %{
        state
        | last_reading: hardware,
          last_qualia: {p, a, d},
          heartbeat_count: state.heartbeat_count + 1
      }

      {{:ok, {p, a, d}}, new_state}
    rescue
      error ->
        VivaLog.error(:senses, :heartbeat_error, error: inspect(error))

        new_state = %{
          state
          | errors: [{DateTime.utc_now(), error} | Enum.take(state.errors, 9)]
        }

        {{:error, error}, new_state}
    end
  end

  defp format_percent(nil), do: "?"
  defp format_percent(value), do: Float.round(value, 1)

  defp format_gpu(nil), do: "N/A"
  defp format_gpu(value), do: "#{Float.round(value, 1)}%"

  defp format_delta(value) when value >= 0, do: "+#{Float.round(value, 4)}"
  defp format_delta(value), do: "#{Float.round(value, 4)}"

  # ============================================================================
  # Code Change (Hot Reload)
  # ============================================================================

  @impl true
  def code_change(_old_vsn, state, _extra) do
    {:ok, state}
  end
end
