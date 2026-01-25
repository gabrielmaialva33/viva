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
  require VivaLog

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

  If emotion context is provided, queries CogGNN for attention-based
  salience boost. The GNN attention reflects how relevant this thought
  is given the current emotional state.

  ## Parameters
  - `content`: The thought content (string or map)
  - `source`: Origin of the thought (:liquid, :ultra, :memory, :sensory)
  - `salience`: Base importance (0.0 - 1.0)
  - `emotion`: Optional PAD state map for GNN attention boost
  """
  def sow(content, source, salience, emotion \\ nil) do
    GenServer.cast(__MODULE__, {:sow, content, source, salience, emotion})
  end

  @doc """
  Plant a thoughtseed with GNN-boosted salience.

  Explicitly uses CogGNN to compute attention-based salience.
  More expensive than `sow/4` but provides neural-grounded importance.
  """
  def sow_with_gnn(content, source, base_salience, pad) when is_map(pad) do
    GenServer.cast(__MODULE__, {:sow_with_gnn, content, source, base_salience, pad})
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
    VivaLog.info(:consciousness, :workspace_online)

    # Subscribe to Discrete Time
    if Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.subscribe(Viva.PubSub, "chronos:tick")
    end

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

    {:noreply, %{state | seeds: [seed | state.seeds]}}
  end

  @impl true
  def handle_cast({:sow_with_gnn, content, source, base_salience, pad}, state) do
    # Query CogGNN for attention-based salience boost
    gnn_boost = compute_gnn_salience_boost(content, pad)

    # Combine base salience with GNN attention (30% boost max)
    final_salience = min(1.0, base_salience + gnn_boost * 0.3)

    id = "#{System.os_time(:millisecond)}-#{System.unique_integer([:positive])}"

    seed = %Seed{
      id: id,
      content: content,
      source: source,
      salience: final_salience,
      emotion: pad,
      created_at: System.os_time(:millisecond)
    }

    VivaLog.debug(:consciousness, :gnn_boosted_seed,
      source: source,
      base: base_salience,
      boost: gnn_boost,
      final: final_salience
    )

    {:noreply, %{state | seeds: [seed | state.seeds]}}
  end

  # Handle Tick (10Hz)
  # Workspace runs consciousness cycle on every tick to ensure high reactivity
  @impl true
  def handle_info({:tick, _tick_id}, state) do
    # Run cycle on every tick (10Hz) or every N ticks as needed
    # For now, 10Hz consciousness stream
    new_state =
      state
      |> decay_seeds()
      |> compete()
      |> broadcast_focus()

    {:noreply, new_state}
  end

  # Fallback
  def handle_info(:conscious_cycle, state), do: {:noreply, state}

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
      VivaLog.info(:consciousness, :new_focus,
        source: state.focus.source,
        content: state.focus.content
      )

      # Broadcast to System (Motor, Speech, etc)
      Phoenix.PubSub.broadcast(Viva.PubSub, "consciousness:stream", {:focus, state.focus})

      %{state | history: [state.focus | state.history] |> Enum.take(10)}
    else
      state
    end
  end

  # ============================================================================
  # CogGNN Integration
  # ============================================================================

  @doc false
  defp compute_gnn_salience_boost(content, pad) do
    # Convert content to string for embedding
    content_str =
      case content do
        s when is_binary(s) -> s
        map when is_map(map) -> Map.get(map, :text, inspect(map))
        other -> inspect(other)
      end

    # Query CogGNN for attention score
    case VivaBridge.Ultra.propagate(content_str, pad) do
      {:ok, %{"attention_scores" => [score | _]}} when is_number(score) ->
        # Return top attention score as boost
        score

      {:ok, _} ->
        # No attention scores, no boost
        0.0

      {:error, reason} ->
        VivaLog.warning(:consciousness, :gnn_boost_failed, reason: reason)
        0.0
    end
  rescue
    # If Ultra service is not running, gracefully fallback
    _ -> 0.0
  end
end
