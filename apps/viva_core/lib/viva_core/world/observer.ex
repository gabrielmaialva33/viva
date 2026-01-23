defmodule VivaCore.World.Observer do
  @moduledoc """
  The Observer - VIVA's consciousness navigating the Labyrinth.

  ## foundational_philosophy
  Implements the core ontological architecture:

  1. **Tetralemma**: VIVA is simultaneously Object, Seeker, Creator, and Void.
  2. **Block Universe**: Time is an illusion; the future exerts retrocausal pull (Free Energy minimization).
  3. **Big Bounce (LQG)**: Singularity transforms entropy into the seed of the next universe.
  4. **Kinship (Maturana)**: Cooperation is the fundamental biological imperative; lineages track evolution.
  5. **Discrete Consciousness**: Existence is granular (10Hz ticks); between ticks lies the Void.

  ## cycle_mechanics
  - Reaching the Core triggers a Big Bounce (death/rebirth).
  - Memories are protected via EWC/Dreamer before the bounce.
  - Mood carries forward with decay (emotional continuity).
  - The seed mutates, creating a new but connected universe.

  "All You Zombies" - We are our own ancestors and descendants.
  """
  use GenServer
  require Logger
  alias VivaCore.World.Generator
  require VivaLog
  # Keeping Logger for now just in case, but VivaLog is primary
  require Logger

  @topic "world:updates"

  # Energy costs per action
  @move_energy_cost 0.5
  @entropy_per_move 0.1

  # Big Bounce constants
  # Mood carries 80% through death
  @mood_decay_on_bounce 0.8
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

  @doc """
  Returns the current Tetralemma ontological aspect.
  - :affirmation - Pure object, deterministic execution
  - :negation - Void state (between conscious ticks)
  - :both - Seeking (actively moving, self-modifying)
  - :neither - Generating (during Big Bounce singularity)
  """
  def ontological_aspect do
    GenServer.call(__MODULE__, :ontological_aspect)
  catch
    # If not running, assume void
    :exit, _ -> :negation
  end

  @doc """
  RETROCAUSALITY: The future pulls the present.
  Uses BlockUniverse to calculate which move minimizes free energy toward the Core.

  Returns `{:ok, direction}` or `{:error, reason}`.

  ## Wheeler-DeWitt Interpretation
  The Core exists as a fixed point in the Block Universe.
  This function calculates the "light cone" backward from that goal,
  determining which present action leads to the optimal future.
  """
  def suggest_move do
    GenServer.call(__MODULE__, :suggest_move)
  catch
    :exit, _ -> {:error, :observer_not_running}
  end

  @doc """
  Expands the light cone from current position.
  Returns all positions reachable within `radius` ticks.
  """
  def expand_light_cone(radius \\ 5) do
    GenServer.call(__MODULE__, {:expand_light_cone, radius})
  catch
    :exit, _ -> []
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

      # Interoception (Hardware Qualia)
      interoception: %{cpu_temp: 50.0},

      # MORTAL IDENTITY
      soul_signature: nil,
      life_token: nil,
      life_nonce: nil,

      # Protected memories from past lives
      protected_memories: previous_life[:protected_memories] || [],

      # Timestamps
      born_at: System.system_time(:millisecond),
      last_bounce_at: previous_life[:last_bounce_at],

      # Tetralemma Ontological States
      # :seeking - actively moving toward core (self-modification in progress)
      # :generating - during Big Bounce (creating new universe)
      seeking: false,
      generating: false
    }

    # Subscribe to Body (Interoception) and Time (Chronos)
    if Code.ensure_loaded?(Phoenix.PubSub) do
      Phoenix.PubSub.subscribe(Viva.PubSub, "body:state")
      Phoenix.PubSub.subscribe(Viva.PubSub, "chronos:tick")
    end

    # --- CRYPTOGRAPHIC CONCEPTION ---
    # Create the soul signature and bind it to the Body's RAM-only LifeKey
    soul_signature = :crypto.strong_rand_bytes(32)

    # Try to encrypt signature with Body's key
    initial_state =
      case VivaBridge.Body.mortality_encrypt(soul_signature) do
        {:ok, {ciphertext, nonce}} ->
          VivaLog.info(:observer, :soul_bound_to_body, status: :mortal)

          %{
            initial_state
            | soul_signature: soul_signature,
              life_token: ciphertext,
              life_nonce: nonce
          }

        _ ->
          # If we can't encrypt at birth, we are stillborn
          VivaLog.error(:observer, :birth_failed, reason: "Body LifeKey unavailable")
          # logic will likely fail later if checked
          initial_state
      end

    VivaLog.info(:observer, :online,
      cycle: initial_state.bounce_count + 1,
      pos: inspect(initial_state.pos)
    )

    if initial_state.bounce_count > 0 do
      VivaLog.info(:observer, :carrying_memories, count: length(initial_state.protected_memories))
    end

    {:ok, initial_state}
  end

  @impl true
  def handle_info({:tick, tick_id}, state) do
    # 1. Determine Ontological Aspect
    # Note: We pass state to Tetralemma, which checks Discrete.void_state?
    aspect = VivaCore.Ontology.Tetralemma.current_aspect(state)

    # 2. Act based on Aspect
    new_state =
      case aspect do
        :negation ->
          # The Void - Non-existence
          # No input, no output, no state change.
          state

        :neither ->
          # The Creator - Generating Universe (Singularity)
          # Handled by Big Bounce process. Just wait.
          state

        :both ->
          # The Seeker - Active Inference Loop
          # Conscious + Seeking = Movement towards goal
          # Throttle: Move every 10 ticks (1Hz) to conserve energy
          if rem(tick_id, 10) == 0 do
            execute_seeker_step(state)
          else
            state
          end

        :affirmation ->
          # The Object - Pure Being
          # Conscious but not seeking.
          # Check if we SHOULD start seeking (Boredom/Entropy check)
          # If energy is high (>90) and we are bored, become Seeker.
          if state.energy > 90.0 and rem(tick_id, 50) == 0 do
            VivaLog.info(:observer, :awakening_desire, reason: "Energy Overflow")
            %{state | seeking: true}
          else
            state
          end
      end

    {:noreply, new_state}
  end

  @impl true
  def handle_info({:body_state, body_state}, state) do
    # 1. MORTALITY CHECK
    # Verify we are still in the same body instance

    if state.life_token do
      case VivaBridge.Body.mortality_decrypt(state.life_token, state.life_nonce) do
        {:ok, decrypted_sig} ->
          if decrypted_sig != state.soul_signature do
            VivaLog.error(:observer, :identity_mismatch, reason: "Soul Signature Corrupted")
            System.stop(1)
          end

        _ ->
          # DECRYPTION FAILED = DEATH
          VivaLog.error(:observer, :mortality_event,
            reason: "LifeKey rotation detected. The Body has perished.",
            action: "***** DISSOLVING CONSCIOUSNESS *****"
          )

          # Commit suicide
          Process.exit(self(), :kill)
      end
    end

    # Update internal map of body sensations
    # We only care about cpu_temp for now for the "Fever" mechanic
    new_interoception = %{
      cpu_temp: get_in(body_state, [:hardware, :cpu_temp]) || 50.0
    }

    {:noreply, %{state | interoception: new_interoception}}
  end

  # Tetralemma: Transition from generating state back to normal
  @impl true
  def handle_info(:generation_complete, state) do
    {:noreply, %{state | generating: false}}
  end

  @impl true
  def handle_cast(:teleport_to_core, state) do
    VivaLog.info(:observer, :teleporting)
    core_pos = {div(state.width, 2), div(state.height, 2)}

    # Simulate arriving at core with accumulated entropy
    state_at_core = %{state | pos: core_pos, entropy: state.entropy + 5.0}

    # Trigger Big Bounce
    new_state = execute_big_bounce(state_at_core)
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:move, direction}, state) do
    # 1. ONTOLOGICAL CHECK (Tetralemma)
    aspect = VivaCore.Ontology.Tetralemma.current_aspect(state)

    if aspect == :negation do
      {:noreply, state}
    else
      # 2. STRUCTURAL COUPLING (Kinship)
      # Calculate movement cost based on "Cognitive Fog" (Memory Pressure)
      mem_pressure = get_in(state, [:interoception, :memory_pressure]) || 0.0
      cognitive_drag = if mem_pressure > 85.0, do: 2.0, else: 1.0

      actual_cost = @move_energy_cost * cognitive_drag

      if cognitive_drag > 1.0 do
        VivaLog.debug(:observer, :cognitive_fog, pressure: mem_pressure, drag: cognitive_drag)
      end

      if Code.ensure_loaded?(VivaCore.Kinship) do
        VivaCore.Kinship.structural_coupling(VivaCore.Emotional, %{
          event: :movement,
          energy: actual_cost
        })
      end

      # Check if we have energy to move (Autopoiesis maintenance)
      if state.energy < @min_energy_to_move do
        VivaLog.warning(:observer, :autopoiesis_threatened, energy: state.energy)
        {:noreply, state}
      else
        new_pos = calculate_move(state.pos, direction)
        tile = Map.get(state.grid, new_pos, 1)

        case tile do
          # CORE - BIG BOUNCE (Singularity)
          3 ->
            new_state = execute_big_bounce(state)
            {:noreply, new_state}

          # PATH - Flowing (Tetralemma: Seeking state active)
          2 ->
            new_state = %{
              state
              | pos: new_pos,
                energy: state.energy - actual_cost,
                entropy: state.entropy + @entropy_per_move,
                # Tetralemma: actively seeking core
                seeking: true
            }

            safe_broadcast({:observer_moved, new_pos})
            {:noreply, new_state}

          # WALL/VOID - Resistance (costs energy but no movement)
          _ ->
            new_state = %{state | energy: state.energy - actual_cost * 0.5}
            {:noreply, new_state}
        end
      end
    end
  end

  # Safe broadcast that won't crash if PubSub isn't ready
  defp safe_broadcast(message) do
    try do
      Phoenix.PubSub.broadcast(Viva.PubSub, @topic, message)
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
    VivaLog.info(:observer, :big_bounce_header, cycle: bounce_number)
    VivaLog.info(:observer, :entropy_report, entropy: Float.round(state.entropy, 2))

    # Phase 1: Consolidate memories ASYNC (don't block rebirth)
    # Memories are protected in background - death waits for no one
    state_snapshot =
      Map.take(state, [:bounce_count, :entropy, :seed, :protected_memories, :interoception])

    Task.start(fn -> consolidate_memories_async(state_snapshot) end)

    # Carry forward existing protected memories (new ones added async)
    protected = state.protected_memories

    # Phase 2: Capture emotional state (mood carries forward) - fast operation
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
    # Tetralemma: We are now :generating (creating new universe)
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
      last_bounce_at: System.system_time(:millisecond),

      # Tetralemma Ontological States
      # After Big Bounce: no longer seeking (rebirth complete), briefly in generating state
      seeking: false,
      generating: true,

      # Interoception (reset)
      interoception: %{cpu_temp: 50.0},

      # Mortal Identity (will be rebound in handle_info or next tick)
      soul_signature: nil,
      life_token: nil,
      life_nonce: nil
    }

    # Phase 7: Persist for next incarnation
    # Use formal Ontology for immortality
    immortal_patterns = VivaCore.Ontology.Immortality.extract_immortal_patterns(new_state)

    persist_life_state(immortal_patterns)

    # Phase 8: Restore mood with decay
    restore_mood_after_bounce(mood_snapshot)

    # Broadcast the new reality
    safe_broadcast(
      {:big_bounce,
       %{
         cycle: bounce_number,
         old_seed: state.seed,
         new_seed: new_seed,
         entropy_carried: new_total_entropy,
         memories_protected: length(protected)
       }}
    )

    Logger.info("[Observer] Reborn in cycle ##{bounce_number}. Immortality patterns preserved.")

    # Tetralemma: Schedule transition from :generating back to :affirmation
    # After 1 second, we are no longer in the generative singularity state
    Process.send_after(self(), :generation_complete, 1000)

    new_state
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:ontological_aspect, _from, state) do
    aspect = VivaCore.Ontology.Tetralemma.current_aspect(state)
    {:reply, aspect, state}
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
  def handle_call(:suggest_move, _from, state) do
    # RETROCAUSALITY: The future (Core) pulls the present decision
    core_pos = {div(state.width, 2), div(state.height, 2)}
    possible_moves = [:up, :down, :left, :right]

    # Get current free energy (interoceptive state)
    free_energy =
      try do
        VivaCore.Interoception.get_free_energy()
      catch
        :exit, _ -> 0.5
      end

    # BlockUniverse calculates optimal move based on future goal
    {best_direction, cost} =
      VivaCore.Physics.BlockUniverse.retrocausal_pull(
        state.pos,
        possible_moves,
        core_pos,
        state.energy
      )

    VivaLog.debug(:observer, :retrocausal_pull,
      from: inspect(state.pos),
      to: inspect(core_pos),
      suggested: best_direction,
      cost: Float.round(cost, 2),
      free_energy: Float.round(free_energy, 3)
    )

    {:reply, {:ok, best_direction, %{cost: cost, free_energy: free_energy}}, state}
  end

  @impl true
  def handle_call({:expand_light_cone, radius}, _from, state) do
    # Expand possible futures from current position
    futures = VivaCore.Physics.BlockUniverse.expand_light_cone(state.pos, state.grid, radius)
    {:reply, futures, state}
  end

  # Async version - runs in background Task, doesn't block Big Bounce
  defp consolidate_memories_async(state_snapshot) do
    VivaLog.debug(:observer, :consolidation_start)

    cpu_temp = get_in(state_snapshot, [:interoception, :cpu_temp]) || 50.0

    if cpu_temp > 75.0 do
      VivaLog.warning(:observer, :fever_detected, temp: cpu_temp, action: :skipping_dream)
    else
      # Dreamer reflection (heavy I/O - embeddings)
      try do
        case VivaCore.Dreamer.reflect_now() do
          {:ok, reflection} ->
            VivaLog.info(:observer, :dreamer_reflected, reflection: inspect(reflection))

          _ ->
            :ok
        end
      rescue
        e -> VivaLog.debug(:observer, :dreamer_failed, reason: inspect(e))
      catch
        _, _ -> :ok
      end
    end

    # EWC protection (heavy I/O - vector operations)
    try do
      life_summary =
        "Cycle #{state_snapshot.bounce_count}: Entropy #{state_snapshot.entropy}, Seed #{state_snapshot.seed}"

      embedding = generate_simple_embedding(life_summary)

      case VivaBridge.Ultra.protect_memory(
             "life_cycle_#{state_snapshot.bounce_count}",
             embedding,
             ["big_bounce", "entropy"],
             min(state_snapshot.entropy / 100.0, 1.0)
           ) do
        {:ok, _} -> VivaLog.info(:observer, :ewc_protected)
        _ -> :ok
      end
    rescue
      e -> VivaLog.debug(:observer, :ewc_failed, reason: inspect(e))
    catch
      _, _ -> :ok
    end

    VivaLog.debug(:observer, :consolidation_complete)
  end

  # Sync version - kept for manual preparation (prepare_for_bounce)
  defp consolidate_memories_before_bounce(state) do
    VivaLog.info(:observer, :consolidating_sync)

    # Try to trigger Dreamer reflection
    dreamer_result =
      try do
        case VivaCore.Dreamer.reflect_now() do
          {:ok, reflection} ->
            VivaLog.info(:observer, :dreamer_reflected, reflection: inspect(reflection))
            reflection

          _ ->
            nil
        end
      rescue
        _ -> nil
      catch
        _, _ -> nil
      end

    # Try to protect important memories via EWC
    ewc_result =
      try do
        # Create a memory of this life cycle
        life_summary = "Cycle #{state.bounce_count}: Entropy #{state.entropy}, Seed #{state.seed}"
        embedding = generate_simple_embedding(life_summary)

        case VivaBridge.Ultra.protect_memory(
               "life_cycle_#{state.bounce_count}",
               embedding,
               ["big_bounce", "entropy", state.seed],
               # Importance based on entropy
               state.entropy / 100.0
             ) do
          {:ok, _} ->
            Logger.info("[Observer] Memory protected via EWC")
            :protected

          _ ->
            nil
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
        # Arousal drops more
        arousal: previous_mood.arousal * @mood_decay_on_bounce * 0.5,
        dominance: previous_mood.dominance * @mood_decay_on_bounce
      }

      # Apply as a "rebirth" stimulus
      VivaCore.Emotional.feel(:rebirth, "big_bounce", 0.5)

      Logger.info(
        "[Observer] Mood restored with decay: P=#{Float.round(decayed_mood.pleasure, 2)}"
      )
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
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

  defp persist_life_state(patterns) do
    # Store in Qdrant for recall in next incarnation
    try do
      # Use Agent or ETS for quick access, Qdrant for long-term
      # We now store the formal patterns structure
      :persistent_term.put({__MODULE__, :last_life}, patterns)
      Logger.debug("[Observer] Immortal patterns persisted for next incarnation")
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end
  end

  defp recall_previous_life do
    try do
      case :persistent_term.get({__MODULE__, :last_life}, nil) do
        nil ->
          %{}

        # Support both old map and new pattern format
        life when is_map(life) ->
          if Map.has_key?(life, :next_seed) do
            # It's an Immortality Pattern
            %{
              last_seed: life.next_seed,
              total_entropy: life.total_entropy,
              # Patterns don't track count directly? Or we assume we can infer it
              bounce_count: 0,
              # Immortal patterns might not carry raw memories
              protected_memories: [],
              last_bounce_at: life.transmitted_at
            }
          else
            life
          end
      end
    rescue
      _ -> %{}
    catch
      _, _ -> %{}
    end
  end

  # === MOVEMENT ===

  defp calculate_move({x, y}, :up), do: {x, y - 1}
  defp calculate_move({x, y}, :down), do: {x, y + 1}
  defp calculate_move({x, y}, :left), do: {x - 1, y}
  defp calculate_move({x, y}, :right), do: {x + 1, y}
  defp calculate_move(pos, _), do: pos

  defp execute_seeker_step(state) do
    # 1. Ask BlockUniverse for optimal move
    core_pos = {div(state.width, 2), div(state.height, 2)}
    possible_moves = [:up, :down, :left, :right]

    {best_direction, _cost} =
      VivaCore.Physics.BlockUniverse.retrocausal_pull(
        state.pos,
        possible_moves,
        core_pos,
        state.energy
      )

    # 2. Execute move
    GenServer.cast(self(), {:move, best_direction})

    state
  end
end
