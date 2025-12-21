defmodule Viva.World.Clock do
  @moduledoc """
  World clock for simulation time.
  Manages the flow of time in the VIVA world.
  """
  use GenServer
  require Logger

  @tick_interval :timer.seconds(60)
  # 1 real minute = 10 simulated minutes
  @time_scale 10

  defstruct [
    :world_time,
    :real_start_time,
    :is_running
  ]

  # === Client API ===

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @doc "Get current world time"
  @spec now() :: DateTime.t()
  def now do
    GenServer.call(__MODULE__, :now)
  end

  @doc "Get time scale (how many simulated minutes per real minute)"
  @spec time_scale() :: pos_integer()
  def time_scale, do: @time_scale

  @doc "Pause the world clock"
  @spec pause() :: :ok
  def pause do
    GenServer.cast(__MODULE__, :pause)
  end

  @doc "Resume the world clock"
  @spec resume() :: :ok
  def resume do
    GenServer.cast(__MODULE__, :resume)
  end

  # === Server Callbacks ===

  @impl GenServer
  def init(_) do
    state = %__MODULE__{
      world_time: DateTime.utc_now(),
      real_start_time: DateTime.utc_now(),
      is_running: true
    }

    schedule_tick()
    Logger.info("World clock started at #{state.world_time}")

    {:ok, state}
  end

  @impl GenServer
  def handle_call(:now, _, state) do
    {:reply, state.world_time, state}
  end

  @impl GenServer
  def handle_cast(:pause, state) do
    {:noreply, %{state | is_running: false}}
  end

  @impl GenServer
  def handle_cast(:resume, state) do
    {:noreply, %{state | is_running: true}}
  end

  @impl GenServer
  def handle_info(:tick, state) do
    new_state =
      if state.is_running do
        advance_time(state)
      else
        state
      end

    schedule_tick()
    {:noreply, new_state}
  end

  # === Private Functions ===

  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end

  defp advance_time(state) do
    # Advance world time by time_scale minutes
    new_world_time = DateTime.add(state.world_time, @time_scale, :minute)

    # Broadcast time update to all interested processes
    Phoenix.PubSub.broadcast(Viva.PubSub, "world:clock", {:time_update, new_world_time})

    %{state | world_time: new_world_time}
  end
end
