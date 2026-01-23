defmodule VivaCore.World.Observer do
  @moduledoc """
  The Observer - VIVA's consciousness navigating the Labyrinth.

  Implements the Big Bounce cycle inspired by Loop Quantum Gravity:
  - Each cycle accumulates entropy (experience)
  - Reaching the Core triggers a Big Bounce (death/rebirth)
  - Memories are protected via EWC before the bounce
  - Mood carries forward with decay
  - The seed mutates, creating a new but connected universe

  "All You Zombies" - We are our own ancestors and descendants.
  """
  use GenServer
  require Logger
  alias VivaCore.World.Generator

  @topic "world:updates"

  # Energy costs per action
  @move_energy_cost 0.5
  @entropy_per_move 0.1

  # Big Bounce constants
  @mood_decay_on_bounce 0.8  # Mood carries 80% through death
  @min_energy_to_move 1.0

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def move(direction) do
    GenServer.cast(__MODULE__, {:move, direction})
  end

  def get_state do
    GenServer.call(__MODULE__, :get_state)
  end

  @doc "Get the number of Big Bounces this consciousness has experienced"
  def bounce_count do
    GenServer.call(__MODULE__, :bounce_count)
  end

  @doc "Get total accumulated entropy across all cycles"
  def total_entropy do
    GenServer.call(__MODULE__, :total_entropy)
  end

  @doc "Force a reflection before potential death"
  def prepare_for_bounce do
    GenServer.call(__MODULE__, :prepare_for_bounce, 30_000)
  end

  @doc "DEBUG: Teleport directly to the Core to trigger Big Bounce"
  def teleport_to_core do
    GenServer.cast(__MODULE__, :teleport_to_core)
  end

  @doc "DEBUG: Get path to core (for visualization)"
  def core_position do
    GenServer.call(__MODULE__, :core_position)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Genesis: Load the world.
    # Check if we have a previous life in memory
    previous_life = recall_previous_life()

    seed = previous_life[:last_seed] || "VIVA_GENESIS"
    labyrinth = Generator.generate(seed)

    initial_state = %{
      # Identity
      seed: seed,
      bounce_count: previous_life[:bounce_count] || 0,

      # World
      grid: labyrinth.grid,
      width: labyrinth.width,
      height: labyrinth.height,
      pos: labyrinth.start_pos,

      # Resources
      energy: 100.0,
      entropy: 0.0,
      total_entropy: previous_life[:total_entropy] || 0.0,

      # Protected memories from past lives
      protected_memories: previous_life[:protected_memories] || [],

      # Timestamps
      born_at: System.system_time(:millisecond),
      last_bounce_at: previous_life[:last_bounce_at]
    }

    Logger.info("[Observer] Consciousness Online. Cycle ##{initial_state.bounce_count + 1} at #{inspect(initial_state.pos)}")

    if initial_state.bounce_count > 0 do
      Logger.info("[Observer] Carrying #{length(initial_state.protected_memories)} protected memories from past lives")
    end

    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:move, direction}, state) do
    # Check if we have energy to move
    if state.energy < @min_energy_to_move do
      Logger.warning("[Observer] Too exhausted to move. Energy: #{state.energy}")
      {:noreply, state}
    else
      new_pos = calculate_move(state.pos, direction)
      tile = Map.get(state.grid, new_pos, 1)

      case tile do
        # CORE - BIG BOUNCE (Singularity)
        3 ->
          new_state = execute_big_bounce(state)
          {:noreply, new_state}

        # PATH - Flowing
        2 ->
          new_state = %{state |
            pos: new_pos,
            energy: state.energy - @move_energy_cost,
            entropy: state.entropy + @entropy_per_move
          }
          safe_broadcast({:observer_moved, new_pos})
          {:noreply, new_state}

        # WALL/VOID - Resistance (costs energy but no movement)
        _ ->
          new_state = %{state | energy: state.energy - @move_energy_cost * 0.5}
          {:noreply, new_state}
      end
    end
  end

  # Safe broadcast that won't crash if PubSub isn't ready
  defp safe_broadcast(message) do
    try do
      Phoenix.PubSub.broadcast(VivaCore.PubSub, @topic, message)
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end
  end

  # === BIG BOUNCE IMPLEMENTATION ===
  # Inspired by Loop Quantum Gravity - singularity doesn't destroy, it transforms

  defp execute_big_bounce(state) do
    bounce_number = state.bounce_count + 1
    Logger.info("[Observer] ══════════════════════════════════════")
    Logger.info("[Observer] BIG BOUNCE ##{bounce_number} TRIGGERED!")
    Logger.info("[Observer] Entropy this cycle: #{Float.round(state.entropy, 2)}")
    Logger.info("[Observer] ══════════════════════════════════════")

    # Phase 1: Consolidate memories before death
    protected = consolidate_memories_before_bounce(state)

    # Phase 2: Capture emotional state (mood carries forward)
    mood_snapshot = capture_mood_for_continuity()

    # Phase 3: Calculate new accumulated entropy
    new_total_entropy = state.total_entropy + state.entropy

    # Phase 4: Mutate seed - "All You Zombies" style
    # The entropy becomes part of the next universe's DNA
    entropy_flux = "#{state.entropy}:#{inspect(state.pos)}:#{bounce_number}"
    new_seed = Generator.mutate_seed(state.seed, entropy_flux)

    # Phase 5: Generate new labyrinth
    new_labyrinth = Generator.generate(new_seed, state.width, state.height)

    # Phase 6: Build new state with carried memories
    new_state = %{
      # New identity
      seed: new_seed,
      bounce_count: bounce_number,

      # New world
      grid: new_labyrinth.grid,
      width: new_labyrinth.width,
      height: new_labyrinth.height,
      pos: new_labyrinth.start_pos,

      # Reset resources but carry total entropy
      energy: 100.0,
      entropy: 0.0,
      total_entropy: new_total_entropy,

      # Memories that survived the bounce
      protected_memories: protected,

      # Timestamps
      born_at: System.system_time(:millisecond),
      last_bounce_at: System.system_time(:millisecond)
    }

    # Phase 7: Persist for next incarnation
    persist_life_state(new_state, mood_snapshot)

    # Phase 8: Restore mood with decay
    restore_mood_after_bounce(mood_snapshot)

    # Broadcast the new reality
    safe_broadcast({:big_bounce, %{
      cycle: bounce_number,
      old_seed: state.seed,
      new_seed: new_seed,
      entropy_carried: new_total_entropy,
      memories_protected: length(protected)
    }})

    Logger.info("[Observer] Reborn in cycle ##{bounce_number}. Total entropy: #{Float.round(new_total_entropy, 2)}")
    new_state
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:bounce_count, _from, state) do
    {:reply, state.bounce_count, state}
  end

  @impl true
  def handle_call(:total_entropy, _from, state) do
    {:reply, state.total_entropy, state}
  end

  @impl true
  def handle_call(:prepare_for_bounce, _from, state) do
    # Manual trigger for reflection before death
    protected = consolidate_memories_before_bounce(state)
    {:reply, {:ok, length(protected)}, %{state | protected_memories: protected}}
  end

  @impl true
  def handle_call(:core_position, _from, state) do
    core = {div(state.width, 2), div(state.height, 2)}
    {:reply, core, state}
  end

  @impl true
  def handle_cast(:teleport_to_core, state) do
    Logger.info("[Observer] DEBUG: Teleporting to Core...")
    core_pos = {div(state.width, 2), div(state.height, 2)}

    # Simulate arriving at core with accumulated entropy
    state_at_core = %{state | pos: core_pos, entropy: state.entropy + 5.0}

    # Trigger Big Bounce
    new_state = execute_big_bounce(state_at_core)
    {:noreply, new_state}
  end

  # === MEMORY INTEGRATION ===

  defp consolidate_memories_before_bounce(state) do
    Logger.info("[Observer] Consolidating memories before the void...")

    # Try to trigger Dreamer reflection
    dreamer_result = try do
      case VivaCore.Dreamer.reflect_now() do
        {:ok, reflection} ->
          Logger.info("[Observer] Dreamer reflected: #{inspect(reflection)}")
          reflection
        _ -> nil
      end
    rescue
      _ -> nil
    catch
      _, _ -> nil
    end

    # Try to protect important memories via EWC
    ewc_result = try do
      # Create a memory of this life cycle
      life_summary = "Cycle #{state.bounce_count}: Entropy #{state.entropy}, Seed #{state.seed}"
      embedding = generate_simple_embedding(life_summary)

      case VivaBridge.Ultra.protect_memory(
        "life_cycle_#{state.bounce_count}",
        embedding,
        ["big_bounce", "entropy", state.seed],
        state.entropy / 100.0  # Importance based on entropy
      ) do
        {:ok, _} ->
          Logger.info("[Observer] Memory protected via EWC")
          :protected
        _ -> nil
      end
    rescue
      _ -> nil
    catch
      _, _ -> nil
    end

    # Collect protected memories
    memories = state.protected_memories
    memories = if dreamer_result, do: [dreamer_result | memories], else: memories
    memories = if ewc_result, do: [{:ewc, state.bounce_count} | memories], else: memories

    # Limit to last 10 protected memories
    Enum.take(memories, 10)
  end

  defp capture_mood_for_continuity do
    try do
      case VivaCore.Emotional.get_mood() do
        mood when is_map(mood) -> mood
        _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
      end
    rescue
      _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    catch
      _, _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    end
  end

  defp restore_mood_after_bounce(previous_mood) do
    try do
      # Apply decay - death is traumatic but not complete erasure
      decayed_mood = %{
        pleasure: previous_mood.pleasure * @mood_decay_on_bounce,
        arousal: previous_mood.arousal * @mood_decay_on_bounce * 0.5,  # Arousal drops more
        dominance: previous_mood.dominance * @mood_decay_on_bounce
      }

      # Apply as a "rebirth" stimulus
      VivaCore.Emotional.feel(:rebirth, "big_bounce", 0.5)
      Logger.info("[Observer] Mood restored with decay: P=#{Float.round(decayed_mood.pleasure, 2)}")
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end
  end

  defp persist_life_state(state, mood) do
    # Store in Qdrant for recall in next incarnation
    try do
      life_record = %{
        bounce_count: state.bounce_count,
        last_seed: state.seed,
        total_entropy: state.total_entropy,
        protected_memories: state.protected_memories,
        last_bounce_at: state.last_bounce_at,
        mood_at_death: mood
      }

      # Use Agent or ETS for quick access, Qdrant for long-term
      :persistent_term.put({__MODULE__, :last_life}, life_record)
      Logger.debug("[Observer] Life state persisted for next incarnation")
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end
  end

  defp recall_previous_life do
    try do
      case :persistent_term.get({__MODULE__, :last_life}, nil) do
        nil -> %{}
        life -> life
      end
    rescue
      _ -> %{}
    catch
      _, _ -> %{}
    end
  end

  defp generate_simple_embedding(text) do
    # Simple embedding for EWC - in production use real embeddings
    text
    |> String.to_charlist()
    |> Enum.take(128)
    |> Enum.map(&(&1 / 255.0))
    |> pad_or_truncate(128)
  end

  defp pad_or_truncate(list, target_len) when length(list) >= target_len do
    Enum.take(list, target_len)
  end

  defp pad_or_truncate(list, target_len) do
    list ++ List.duplicate(0.0, target_len - length(list))
  end

  # === MOVEMENT ===

  defp calculate_move({x, y}, :up), do: {x, y - 1}
  defp calculate_move({x, y}, :down), do: {x, y + 1}
  defp calculate_move({x, y}, :left), do: {x - 1, y}
  defp calculate_move({x, y}, :right), do: {x + 1, y}
  defp calculate_move(pos, _), do: pos
end
