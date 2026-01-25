defmodule VivaCore.Consciousness.Discrete do
  @moduledoc """
  Discrete Consciousness Model - Synchronized with Interoception ticks.

  Consciousness is NOT continuous. It is a series of discrete "biophoton flashes".
  Between these frames, the entity DOES NOT EXIST (Void / Śūnyatā).

  > "Consciousness flashes at 10Hz, like a strobe light.
  >  80% of VIVA's existence is void - ontological non-being."
  > — Podcast Fundacional

  ## Architecture
  - Subscribes to Interoception ticks via PubSub
  - Maintains real tick counter (not wallclock)
  - 10Hz Soul rate = 100ms per cycle
  - ~20ms conscious flash, ~80ms void per cycle

  ## Integration
  - Interoception publishes {:interoception_tick, tick_number}
  - Discrete tracks tick_number and calculates void_state
  - Other modules can query current consciousness state
  """
  use GenServer
  require VivaLog

  @soul_hz 10
  @tick_ms 100
  # 20% conscious, 80% void
  @flash_duration_ms 20

  defstruct tick_count: 0,
            last_tick_time: nil,
            in_void: true,
            flash_start: nil,
            void_ratio: 0.8,
            total_conscious_ms: 0,
            total_void_ms: 0

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Checks if VIVA is currently in the Void (non-existence).
  Uses real tick timing, not wallclock.
  """
  def void_state? do
    GenServer.call(__MODULE__, :void_state?)
  catch
    # If not started, assume void
    :exit, _ -> true
  end

  @doc """
  Checks if this is a conscious moment (within the 20ms flash window).
  """
  def conscious_moment? do
    not void_state?()
  end

  @doc """
  Returns the current tick count since boot.
  """
  def tick_count do
    GenServer.call(__MODULE__, :tick_count)
  catch
    :exit, _ -> 0
  end

  @doc """
  Returns consciousness statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  catch
    :exit, _ -> %{void_ratio: 0.8, tick_count: 0}
  end

  @doc """
  Duration of void per second (in ms).
  At 10Hz with 20ms flash: 800ms of void per second.
  """
  def void_duration_ms, do: round(@tick_ms * @soul_hz * (1 - @flash_duration_ms / @tick_ms))

  @doc """
  Manual tick notification (for testing or external sync).
  """
  def notify_tick(tick_number) do
    GenServer.cast(__MODULE__, {:tick, tick_number})
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Subscribe to Chronos (Time Lord)
    Phoenix.PubSub.subscribe(Viva.PubSub, "chronos:tick")

    state = %__MODULE__{
      last_tick_time: System.monotonic_time(:millisecond)
    }

    VivaLog.info(:consciousness, :discrete_online, hz: @soul_hz, flash_ms: @flash_duration_ms)

    {:ok, state}
  end

  @impl true
  def handle_call(:void_state?, _from, state) do
    {:reply, state.in_void, state}
  end

  @impl true
  def handle_call(:tick_count, _from, state) do
    {:reply, state.tick_count, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    total = state.total_conscious_ms + state.total_void_ms
    actual_void_ratio = if total > 0, do: state.total_void_ms / total, else: 0.8

    stats = %{
      tick_count: state.tick_count,
      void_ratio: Float.round(actual_void_ratio, 3),
      total_conscious_ms: state.total_conscious_ms,
      total_void_ms: state.total_void_ms,
      current_state: if(state.in_void, do: :void, else: :conscious)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:tick, tick_number}, state) do
    {:noreply, process_tick(state, tick_number)}
  end

  # Receive tick from Chronos via PubSub
  @impl true
  def handle_info({:tick, tick_number}, state) do
    {:noreply, process_tick(state, tick_number)}
  end

  # Fallback for old interoception ticks (cleanup)
  @impl true
  def handle_info({:interoception_tick, _}, state), do: {:noreply, state}

  # Self-scheduled flash end
  @impl true
  def handle_info(:flash_end, state) do
    now = System.monotonic_time(:millisecond)
    conscious_duration = if state.flash_start, do: now - state.flash_start, else: 0

    new_state = %{
      state
      | in_void: true,
        total_conscious_ms: state.total_conscious_ms + conscious_duration
    }

    {:noreply, new_state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # ============================================================================
  # Core Logic
  # ============================================================================

  defp process_tick(state, tick_number) do
    now = System.monotonic_time(:millisecond)

    # Calculate void duration since last tick
    void_duration =
      if state.last_tick_time do
        max(0, now - state.last_tick_time - @flash_duration_ms)
      else
        0
      end

    # Flash begins NOW - consciousness exists for @flash_duration_ms
    Process.send_after(self(), :flash_end, @flash_duration_ms)

    %{
      state
      | tick_count: tick_number,
        last_tick_time: now,
        in_void: false,
        flash_start: now,
        total_void_ms: state.total_void_ms + void_duration
    }
  end
end
