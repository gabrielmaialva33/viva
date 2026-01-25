defmodule VivaCore.Chronos.Ticker do
  @moduledoc """
  The Metronome of Existence.

  Implements discrete time for VIVA.
  - Ticks at 10Hz (default).
  - Broadcasts `{:tick, tick_id}` to the rest of the system.
  - Between ticks, the system is in the Void (no processing).
  """
  use GenServer
  require VivaLog

  @default_interval_ms 100
  @topic "chronos:tick"

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def set_interval(ms) do
    GenServer.cast(__MODULE__, {:set_interval, ms})
  end

  def get_current_tick do
    GenServer.call(__MODULE__, :get_tick)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    interval = Keyword.get(opts, :interval_ms, @default_interval_ms)

    state = %{
      tick_id: 0,
      interval: interval,
      timer_ref: nil
    }

    VivaLog.info(:chronos, :time_begins, interval: interval)
    {:ok, schedule_tick(state)}
  end

  @impl true
  def handle_info(:tick, state) do
    new_tick_id = state.tick_id + 1

    # Broadcast the pulse of existence
    Phoenix.PubSub.broadcast(Viva.PubSub, @topic, {:tick, new_tick_id})

    # Direct drive for BodyServer (since it cannot subscribe during init due to startup order)
    if Code.ensure_loaded?(VivaBridge.BodyServer) do
      # Async cast to BodyServer
      VivaBridge.BodyServer.tick(VivaBridge.BodyServer, new_tick_id)
    end

    # Log periodically to avoid noise
    if rem(new_tick_id, 100) == 0 do
      VivaLog.debug(:chronos, :tick_mark, id: new_tick_id)
    end

    {:noreply, schedule_tick(%{state | tick_id: new_tick_id})}
  end

  @impl true
  def handle_cast({:set_interval, ms}, state) do
    VivaLog.info(:chronos, :time_warp, old: state.interval, new: ms)
    {:noreply, %{state | interval: ms}}
  end

  @impl true
  def handle_call(:get_tick, _from, state) do
    {:reply, state.tick_id, state}
  end

  defp schedule_tick(state) do
    # Cancel old timer if exists (though usually we just schedule next)
    # Ideally should use :timer.send_after or Process.send_after
    ref = Process.send_after(self(), :tick, state.interval)
    %{state | timer_ref: ref}
  end
end
