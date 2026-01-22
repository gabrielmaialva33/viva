defmodule VivaCore.Consciousness.Workspace do
  @moduledoc """
  The Global Workspace (Theater of Consciousness).

  Implementation of the "Thoughtseeds" paper (2024).
  Manages a collection of active mental objects (Seeds) that compete for
  system-wide broadcast (Focus).

  Concepts:
  - **Seed**: A bundle of info + salience (importance).
  - **Competition**: Seeds decay or grow based on resonance.
  - **Broadcast**: The winner is sent to all subsystems (Motor, Speech, etc).
  """

  use GenServer
  require Logger

  # ============================================================================
  # STRUCT & TYPES
  # ============================================================================

  defmodule Seed do
    defstruct [
      :id,
      # Map or String
      :content,
      # :liquid, :ultra, :memory, :sensory
      :source,
      # 0.0 - 1.0 (Importance)
      :salience,
      # Associated emotion
      :emotion,
      :created_at
    ]
  end

  # ============================================================================
  # CLIENT API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Plant a new thoughtseed in the workspace.
  """
  def sow(content, source, salience, emotion \\ nil) do
    GenServer.cast(__MODULE__, {:sow, content, source, salience, emotion})
  end

  @doc """
  Get the current "Conscious Focus" (Winning Seed).
  """
  def current_focus do
    GenServer.call(__MODULE__, :get_focus)
  end

  # ============================================================================
  # GENSERVER
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Consciousness] Global Workspace Online. Theory: Thoughtseeds (2024).")

    # Tick every 100ms for consciousness updates (10Hz alpha wave)
    :timer.send_interval(100, :conscious_cycle)

    {:ok,
     %{
       seeds: [],
       focus: nil,
       history: []
     }}
  end

  @impl true
  def handle_cast({:sow, content, source, salience, emotion}, state) do
    # Generate simple unique ID
    id = "#{System.os_time(:millisecond)}-#{System.unique_integer([:positive])}"

    seed = %Seed{
      id: id,
      content: content,
      source: source,
      salience: min(salience, 1.0),
      emotion: emotion,
      created_at: System.os_time(:millisecond)
    }

    # Logger.debug("[Workspace] New Seed from #{source}: #{inspect(content)} (Sal: #{salience})")

    {:noreply, %{state | seeds: [seed | state.seeds]}}
  end

  @impl true
  def handle_info(:conscious_cycle, state) do
    state
    |> decay_seeds()
    |> compete()
    |> broadcast_focus()
  end

  @impl true
  def handle_call(:get_focus, _from, state) do
    {:reply, state.focus, state}
  end

  # ============================================================================
  # LOGIC
  # ============================================================================

  defp decay_seeds(state) do
    # Seeds decay fast if not reinforced (Short-term memory)
    decay_rate = 0.05

    active_seeds =
      state.seeds
      |> Enum.map(fn s -> %{s | salience: s.salience - decay_rate} end)
      # Cull weak thoughts
      |> Enum.filter(fn s -> s.salience > 0.1 end)

    %{state | seeds: active_seeds}
  end

  defp compete(state) do
    if Enum.empty?(state.seeds) do
      %{state | focus: nil}
    else
      # Winner takes all (Softmax-like selection could be better, but WTA is cleaner for now)
      winner = Enum.max_by(state.seeds, & &1.salience)

      # Persistence: If same winner, boost slightly (Focus momentum)
      # But if dominance is low, switch.

      %{state | focus: winner}
    end
  end

  defp broadcast_focus(state) do
    if state.focus && state.focus != state.history |> List.first() do
      # New thought entered consciousness!
      Logger.info("💡 [CONSCIOUS] #{state.focus.source}: #{inspect(state.focus.content)}")

      # Broadcast to System (Motor, Speech, etc)
      Phoenix.PubSub.broadcast(Viva.PubSub, "consciousness:stream", {:focus, state.focus})

      %{state | history: [state.focus | state.history] |> Enum.take(10)}
    else
      state
    end
  end
end
