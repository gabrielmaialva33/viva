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
  require Logger

  @default_interval_ms 1000  # 1 second
  @min_interval_ms 100       # 100ms minimum (10Hz max)
  @max_interval_ms 10_000    # 10s maximum

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
      when is_integer(interval_ms) and interval_ms >= @min_interval_ms and interval_ms <= @max_interval_ms do
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

    Logger.info("[Senses] Nervous system starting. Heartbeat: #{interval_ms}ms")

    # Use handle_continue to avoid race condition on startup
    # (wait for Emotional to be registered before sending qualia)
    {:ok, state, {:continue, {:start_heartbeat, enabled}}}
  end

  @impl true
  def handle_continue({:start_heartbeat, true}, state) do
    send(self(), :heartbeat)
    {:noreply, state}
  end
  def handle_continue({:start_heartbeat, false}, state) do
    {:noreply, state}
  end

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
    Logger.info("[Senses] Sensing paused")
    {:noreply, %{state | enabled: false}}
  end

  @impl true
  def handle_cast(:resume, state) do
    Logger.info("[Senses] Sensing resumed")
    schedule_heartbeat(state.interval_ms)
    {:noreply, %{state | enabled: true}}
  end

  @impl true
  def handle_cast({:set_interval, interval_ms}, state) do
    Logger.info("[Senses] Interval changed: #{state.interval_ms}ms -> #{interval_ms}ms")
    {:noreply, %{state | interval_ms: interval_ms}}
  end

  @impl true
  def handle_info(:heartbeat, state) do
    if state.enabled do
      {_result, new_state} = do_heartbeat(state)
      schedule_heartbeat(new_state.interval_ms)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_heartbeat(interval_ms) do
    Process.send_after(self(), :heartbeat, interval_ms)
  end

  defp do_heartbeat(state) do
    try do
      # 1. Read qualia from hardware via Rust NIF
      {p, a, d} = VivaBridge.hardware_to_qualia()

      # 2. Apply to Emotional GenServer
      VivaCore.Emotional.apply_hardware_qualia(p, a, d, state.emotional_server)

      # 3. Optionally, read full metrics for logging/debug
      hardware = VivaBridge.feel_hardware()

      # 4. Summary log (debug level to avoid noise)
      Logger.debug(
        "[Senses] Heartbeat ##{state.heartbeat_count + 1}: " <>
        "CPU=#{format_percent(hardware.cpu_usage)}% " <>
        "RAM=#{format_percent(hardware.memory_used_percent)}% " <>
        "GPU=#{format_gpu(hardware.gpu_usage)} " <>
        "Qualia=(P#{format_delta(p)}, A#{format_delta(a)}, D#{format_delta(d)})"
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
        Logger.error("[Senses] Heartbeat error: #{inspect(error)}")

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
