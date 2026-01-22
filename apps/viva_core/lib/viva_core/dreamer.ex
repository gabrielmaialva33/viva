defmodule VivaCore.Dreamer do
  @moduledoc """
  Dreamer GenServer - VIVA's reflection and consolidation system.

  Implements the reflection mechanism from generative agents (Park et al., 2023)
  adapted for VIVA's emotional architecture.

  ## Mathematical Foundations

  ### Retrieval Scoring (adapted from Park et al.)
  S(m, q) = w_r · D(m) + w_s · Sim(m, q) + w_i · I(m) + w_e · E(m)

  Where:
  - D(m) = e^(-age/τ) · (1 + min(0.5, log(1 + access_count)/κ))  # Temporal decay + spaced repetition
  - Sim(m, q) = cos_sim(emb_m, emb_q)                             # Semantic similarity
  - I(m) = importance ∈ [0, 1]                                    # Memory importance
  - E(m) = max(0, 1 - ||PAD_m - PAD_current|| / √12)              # Emotional resonance

  Weights: w_r=0.2, w_s=0.4, w_i=0.2, w_e=0.2

  ### Reflection Trigger (hybrid approach)
  Triggers when ANY condition is met:
  1. Importance accumulator reaches threshold (τ = 15.0)
  2. Time since last reflection exceeds limit (1 hour of activity)
  3. Sleep cycle initiated (deep reflection)

  ### Reflection Depth
  - depth=0: Direct memory (event)
  - depth=1: First-order reflection (insight about events)
  - depth=2: Second-order reflection (meta-cognition)

  ## Philosophy
  "Dreams are not noise - they are the soul reorganizing itself.
   In reflection, scattered experiences become coherent meaning."

  ## References
  - Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra
    of Human Behavior." arXiv:2304.03442
  - Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology.
  """

  use GenServer
  require Logger

  alias VivaCore.Memory
  alias VivaCore.Emotional

  # ============================================================================
  # Constants
  # ============================================================================

  # Reflection trigger threshold (sum of importance values)
  @importance_threshold 15.0

  # Maximum time between reflections (1 hour in seconds)
  @max_reflection_interval 3600

  # Decay scale for temporal scoring (1 week in seconds)
  @decay_scale 604_800

  # Scoring weights (sum = 1.0)
  @weight_recency 0.2
  @weight_similarity 0.4
  @weight_importance 0.2
  @weight_emotional 0.2

  # Number of focal points to generate per reflection
  @focal_point_count 3

  # Memories to retrieve per focal point
  @retrieval_limit 20

  # PAD space diagonal (√12 ≈ 3.464 for [-1,1]³ cube)
  @pad_diagonal 3.4641016151377544

  # Spaced repetition boost divisor
  @repetition_boost_divisor 10.0

  # Maximum boost from spaced repetition (caps at 50% boost)
  @max_repetition_boost 0.5

  # Maximum thoughts to keep in memory (prevents memory leak)
  @max_thoughts 1000

  # Maximum recent memory IDs to track
  @max_recent_memories 50

  # Recent memories to process for focal points
  @focal_point_memory_count 20

  # Recent thoughts for meta-reflection
  @meta_reflection_thought_count 10

  # Minimum thoughts required for meta-reflection
  @min_thoughts_for_meta 3

  # Sleep cycle iterations
  @sleep_cycle_iterations 3

  # Reflection check interval (5 minutes in ms)
  @reflection_check_interval 5 * 60 * 1000

  # Default importance/similarity values
  @default_importance 0.5
  @default_similarity 0.5
  @default_emotional_resonance 0.5

  # ============================================================================
  # Types
  # ============================================================================

  @typedoc "Focal point for reflection"
  @type focal_point :: %{
          question: String.t(),
          context: String.t(),
          timestamp: DateTime.t()
        }

  @typedoc "Generated reflection/insight"
  @type reflection :: %{
          insight: String.t(),
          evidence: [String.t()],
          depth: non_neg_integer(),
          importance: float(),
          focal_point: String.t(),
          timestamp: DateTime.t()
        }

  @typedoc "Dreamer status"
  @type status :: %{
          importance_accumulator: float(),
          threshold: float(),
          progress_percent: float(),
          last_reflection: DateTime.t(),
          seconds_since_reflection: integer(),
          reflection_count: non_neg_integer(),
          recent_memories_count: non_neg_integer(),
          thoughts_count: non_neg_integer(),
          total_insights_generated: non_neg_integer(),
          uptime_seconds: integer()
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Starts the Dreamer GenServer.

  ## Options
  - `:name` - Process name (default: __MODULE__)
  - `:memory` - Memory GenServer (default: VivaCore.Memory)
  - `:emotional` - Emotional GenServer (default: VivaCore.Emotional)

  ## Returns
  - `{:ok, pid}` on success
  - `{:error, reason}` on failure
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Notifies the Dreamer of a new memory storage.
  Called automatically by Memory GenServer after store.

  Accumulates importance and may trigger reflection.

  ## Parameters
  - `memory_id` - ID of the stored memory
  - `importance` - Importance value (0.0 to 1.0)
  - `server` - GenServer reference (default: __MODULE__)

  ## Returns
  - `:ok` (fire and forget)
  """
  @spec on_memory_stored(String.t(), float(), GenServer.server()) :: :ok
  def on_memory_stored(memory_id, importance, server \\ __MODULE__)
      when is_binary(memory_id) and is_number(importance) do
    GenServer.cast(server, {:memory_stored, memory_id, importance})
  end

  @doc """
  Forces a reflection cycle immediately.
  Useful for testing or manual triggering.

  ## Returns
  - `%{focal_points: [...], insights: [...], trigger: :manual}`
  """
  @spec reflect_now(GenServer.server()) :: map()
  def reflect_now(server \\ __MODULE__) do
    GenServer.call(server, :reflect_now, 60_000)
  end

  @doc """
  Initiates a deep reflection (sleep cycle).
  Performs multiple reflection iterations and higher-depth analysis.

  This operation runs asynchronously to avoid blocking the GenServer.

  ## Returns
  - `{:ok, ref}` - Reference to track the async operation
  """
  @spec sleep_cycle(GenServer.server()) :: {:ok, reference()}
  def sleep_cycle(server \\ __MODULE__) do
    GenServer.call(server, :sleep_cycle)
  end

  @doc """
  Returns the current state and statistics.

  ## Returns
  - Status map with accumulator, threshold, counts, etc.
  """
  @spec status(GenServer.server()) :: status()
  def status(server \\ __MODULE__) do
    GenServer.call(server, :status)
  end

  @doc """
  Returns recent reflections (thoughts).

  ## Parameters
  - `limit` - Maximum number of thoughts to return (default: 10)
  - `server` - GenServer reference

  ## Returns
  - List of thought maps
  """
  @spec recent_thoughts(non_neg_integer(), GenServer.server()) :: [reflection()]
  def recent_thoughts(limit \\ 10, server \\ __MODULE__) when is_integer(limit) and limit > 0 do
    GenServer.call(server, {:recent_thoughts, limit})
  end

  @doc """
  Retrieves memories with composite scoring.
  Uses the full scoring formula: recency + similarity + importance + emotional.

  ## Parameters
  - `query` - Search query string
  - `opts` - Options (`:limit` - max results)
  - `server` - GenServer reference

  ## Returns
  - List of memories with `:composite_score` field added
  """
  @spec retrieve_with_scoring(String.t(), keyword(), GenServer.server()) :: [map()]
  def retrieve_with_scoring(query, opts \\ [], server \\ __MODULE__) when is_binary(query) do
    GenServer.call(server, {:retrieve, query, opts}, 30_000)
  end

  @doc """
  Active Inference: Hallucinates a goal state (Target Prior).
  This represents where VIVA *wants* to be in the near future.

  The goal is not static; it shifts based on "whimsy" (random exploration)
  and "values" (preference for stable attractors).

  ## Parameters
  - `context` - Map with current context (e.g., current PAD)
  - `server` - GenServer reference

  ## Returns
  - Target PAD map %{pleasure: float, arousal: float, dominance: float}
  """
  def hallucinate_goal(context \\ %{}, server \\ __MODULE__) do
    GenServer.call(server, {:hallucinate_goal, context})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    memory_server = Keyword.get(opts, :memory, Memory)
    emotional_server = Keyword.get(opts, :emotional, Emotional)

    state = %{
      # Dependencies
      memory: memory_server,
      emotional: emotional_server,

      # Reflection trigger state
      importance_accumulator: 0.0,
      last_reflection: DateTime.utc_now(),
      reflection_count: 0,
      reflection_pending: false,

      # Recent memories (for focal point generation)
      recent_memory_ids: [],

      # Generated thoughts (limited to prevent memory leak)
      thoughts: [],

      # Statistics
      created_at: DateTime.utc_now(),
      total_insights_generated: 0,

      # Timer reference for cleanup
      timer_ref: nil,

      # Async sleep cycle tracking
      sleep_cycle_task: nil
    }

    Logger.info("[Dreamer] Dreamer neuron online. Ready to reflect.")

    # Schedule periodic check and store ref
    timer_ref = schedule_reflection_check()

    {:ok, %{state | timer_ref: timer_ref}}
  end

  @impl true
  def handle_cast({:memory_stored, memory_id, importance}, state) do
    # Validate importance
    importance = if is_number(importance), do: importance, else: @default_importance

    # Accumulate importance
    new_accumulator = state.importance_accumulator + importance

    # Track recent memories (with limit)
    recent = [memory_id | state.recent_memory_ids] |> Enum.take(@max_recent_memories)

    new_state = %{
      state
      | importance_accumulator: new_accumulator,
        recent_memory_ids: recent
    }

    # Check if threshold reached AND no reflection pending (prevents race condition)
    new_state =
      if new_accumulator >= @importance_threshold and not state.reflection_pending do
        Logger.info(
          "[Dreamer] Importance threshold reached (#{Float.round(new_accumulator, 2)}). Triggering reflection."
        )

        send(self(), :trigger_reflection)
        %{new_state | reflection_pending: true}
      else
        new_state
      end

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:reflect_now, _from, state) do
    {result, new_state} = do_reflection(state, :manual)
    {:reply, result, new_state}
  end

  @impl true
  def handle_call(:sleep_cycle, from, state) do
    Logger.info("[Dreamer] Sleep cycle initiated. Deep reflection starting asynchronously...")

    # Run sleep cycle in a separate task to avoid blocking
    task_ref = make_ref()

    task =
      Task.async(fn ->
        do_sleep_cycle(state, task_ref)
      end)

    # Reply immediately with the reference
    GenServer.reply(from, {:ok, task_ref})

    # Store task for tracking
    {:noreply, %{state | sleep_cycle_task: task}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    status = %{
      importance_accumulator: Float.round(state.importance_accumulator, 2),
      threshold: @importance_threshold,
      progress_percent:
        Float.round(state.importance_accumulator / @importance_threshold * 100, 1),
      last_reflection: state.last_reflection,
      seconds_since_reflection: DateTime.diff(DateTime.utc_now(), state.last_reflection),
      reflection_count: state.reflection_count,
      recent_memories_count: length(state.recent_memory_ids),
      thoughts_count: length(state.thoughts),
      total_insights_generated: state.total_insights_generated,
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at),
      reflection_pending: state.reflection_pending
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:recent_thoughts, limit}, _from, state) do
    thoughts = Enum.take(state.thoughts, limit)
    {:reply, thoughts, state}
  end

  @impl true
  def handle_call({:retrieve, query, opts}, _from, state) do
    result = do_retrieve(query, opts, state)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:hallucinate_goal, context}, _from, state) do
    # HOMEOSTASIS-BASED GOAL SELECTION (No RNG)
    #
    # Instead of random selection (70% positive, etc), we:
    # 1. Consult memory for what worked before (RAG)
    # 2. Calculate personal baseline from successful states
    # 3. Apply Yerkes-Dodson for optimal arousal
    # 4. Use small exploration noise only when stuck
    #
    # This makes VIVA seek what has ACTUALLY helped her, not random attractors.

    current_pad = %{
      pleasure: Map.get(context, :pleasure, 0.0),
      arousal: Map.get(context, :arousal, 0.0),
      dominance: Map.get(context, :dominance, 0.0)
    }

    # 1. Consult memory: "What states have made me feel good?"
    baseline = calculate_personal_baseline(state)

    # 2. Calculate target based on homeostasis + Yerkes-Dodson
    target = %{
      pleasure: baseline.pleasure,
      arousal: calculate_optimal_arousal(current_pad),
      dominance: baseline.dominance
    }

    # 3. Small exploration noise ONLY if we're stuck (low arousal + low pleasure)
    is_stuck = current_pad.pleasure < -0.2 and current_pad.arousal < 0.1
    exploration = if is_stuck, do: 0.08, else: 0.02

    final_target = %{
      pleasure: clamp(target.pleasure + (:rand.uniform() - 0.5) * exploration, -1.0, 1.0),
      arousal: clamp(target.arousal + (:rand.uniform() - 0.5) * exploration, -1.0, 1.0),
      dominance: clamp(target.dominance + (:rand.uniform() - 0.5) * exploration, -1.0, 1.0)
    }

    notify_hallucination(:homeostatic, final_target, baseline, is_stuck)

    {:reply, final_target, state}
  end

  # ============================================================================
  # Homeostatic Goal Calculation (Replaces RNG)
  # ============================================================================

  @doc false
  defp calculate_personal_baseline(state) do
    # Search for memories with positive emotional outcomes
    case safe_memory_search("estados positivos felicidade alívio sucesso", 10, state.memory) do
      {:ok, memories} when is_list(memories) and length(memories) > 0 ->
        # Extract PAD from successful memories
        pads =
          memories
          |> Enum.map(fn m ->
            emotion = get_memory_field(m, :emotion, nil)

            if emotion do
              %{
                pleasure: get_pad_value(emotion, :pleasure),
                arousal: get_pad_value(emotion, :arousal),
                dominance: get_pad_value(emotion, :dominance)
              }
            else
              nil
            end
          end)
          |> Enum.filter(&(&1 != nil))

        if Enum.empty?(pads) do
          default_baseline()
        else
          # Calculate mean of successful states
          n = length(pads)

          %{
            pleasure: Enum.sum(Enum.map(pads, & &1.pleasure)) / n,
            arousal: Enum.sum(Enum.map(pads, & &1.arousal)) / n,
            dominance: Enum.sum(Enum.map(pads, & &1.dominance)) / n
          }
        end

      _ ->
        # No memories yet, use biological default (slight positivity)
        default_baseline()
    end
  end

  defp default_baseline do
    # The "natural" state VIVA seeks when no memory exists
    # Slightly positive, calm, moderate agency
    %{pleasure: 0.2, arousal: 0.1, dominance: 0.1}
  end

  # Yerkes-Dodson Law: Optimal arousal depends on task complexity.
  # High dominance = can handle high arousal (challenging tasks)
  # Low dominance = needs low arousal (overwhelmed, needs calm)
  defp calculate_optimal_arousal(current_pad) do
    dominance = current_pad.dominance
    pleasure = current_pad.pleasure

    cond do
      # High dominance + positive → can be excited
      dominance > 0.3 and pleasure > 0 -> 0.4
      # High dominance + negative → needs activation to fix
      dominance > 0.3 and pleasure < 0 -> 0.3
      # Low dominance → needs calm to recover
      dominance < -0.3 -> 0.0
      # Negative pleasure → moderate arousal to change
      pleasure < -0.2 -> 0.2
      # Default: slight arousal (engagement without stress)
      true -> 0.15
    end
  end

  defp clamp(value, min, max) do
    value
    |> max(min)
    |> min(max)
  end

  defp notify_hallucination(goal_type, target, baseline, is_stuck) do
    stuck_marker = if is_stuck, do: " [EXPLORING]", else: ""

    Logger.debug(
      "[Dreamer] Hallucinated Goal: #{goal_type}#{stuck_marker} " <>
        "(Target P=#{Float.round(target.pleasure, 2)}, A=#{Float.round(target.arousal, 2)}, D=#{Float.round(target.dominance, 2)}) " <>
        "(Baseline P=#{Float.round(baseline.pleasure, 2)})"
    )
  end

  @impl true
  def handle_info(:trigger_reflection, state) do
    {_result, new_state} = do_reflection(state, :threshold)
    # Clear pending flag after reflection
    {:noreply, %{new_state | reflection_pending: false}}
  end

  @impl true
  def handle_info(:reflection_check, state) do
    timer_ref = schedule_reflection_check()

    # Check time-based trigger
    seconds_since = DateTime.diff(DateTime.utc_now(), state.last_reflection)

    new_state =
      if seconds_since >= @max_reflection_interval and
           length(state.recent_memory_ids) > 5 and
           not state.reflection_pending do
        Logger.info("[Dreamer] Time-based reflection trigger (#{seconds_since}s since last)")
        {_result, s} = do_reflection(state, :time)
        s
      else
        state
      end

    {:noreply, %{new_state | timer_ref: timer_ref}}
  end

  # Handle async task completion (sleep cycle)
  @impl true
  def handle_info({ref, {:sleep_cycle_complete, results, insights_count}}, state)
      when is_reference(ref) do
    # Demonitor and flush
    Process.demonitor(ref, [:flush])

    Logger.info("[Dreamer] Sleep cycle complete. Generated #{insights_count} insights.")

    # Update state with results from sleep cycle
    new_thoughts =
      (results.all_insights ++ state.thoughts)
      |> Enum.take(@max_thoughts)

    new_state = %{
      state
      | thoughts: new_thoughts,
        total_insights_generated: state.total_insights_generated + insights_count,
        sleep_cycle_task: nil,
        last_reflection: DateTime.utc_now(),
        reflection_count: state.reflection_count + @sleep_cycle_iterations + 1,
        importance_accumulator: 0.0,
        recent_memory_ids: []
    }

    {:noreply, new_state}
  end

  # Handle task failure
  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state)
      when is_reference(ref) do
    Logger.warning("[Dreamer] Async task failed: #{inspect(reason)}")
    {:noreply, %{state | sleep_cycle_task: nil}}
  end

  # Catch-all for unexpected messages
  @impl true
  def handle_info(msg, state) do
    Logger.warning("[Dreamer] Unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(reason, state) do
    # Cancel pending timer
    if state.timer_ref do
      Process.cancel_timer(state.timer_ref)
    end

    # Log termination
    Logger.info(
      "[Dreamer] Terminating: #{inspect(reason)}. #{length(state.thoughts)} thoughts in memory."
    )

    :ok
  end

  # ============================================================================
  # Private - Async Sleep Cycle
  # ============================================================================

  defp do_sleep_cycle(state, _task_ref) do
    # Multiple reflection iterations
    {all_insights, _final_state} =
      Enum.reduce(1..@sleep_cycle_iterations, {[], state}, fn iteration, {acc, s} ->
        Logger.info(
          "[Dreamer] Sleep reflection iteration #{iteration}/#{@sleep_cycle_iterations}"
        )

        case safe_reflection(s, :sleep) do
          {:ok, result, new_s} ->
            insights = Map.get(result, :insights, [])
            {insights ++ acc, new_s}

          {:error, reason} ->
            Logger.warning("[Dreamer] Sleep iteration #{iteration} failed: #{inspect(reason)}")
            {acc, s}
        end
      end)

    # Meta-reflection
    meta_insights =
      case safe_meta_reflection(state) do
        {:ok, result} -> Map.get(result, :meta_insights, [])
        {:error, _} -> []
      end

    # MEMORY CONSOLIDATION: Episodic → Semantic
    # Important episodic memories are promoted to long-term semantic storage
    consolidated_count = consolidate_memories(state)

    total_insights = length(all_insights) + length(meta_insights)

    results = %{
      all_insights: meta_insights ++ all_insights,
      iterations: @sleep_cycle_iterations,
      consolidated_memories: consolidated_count
    }

    {:sleep_cycle_complete, results, total_insights}
  end

  # ============================================================================
  # Private - Memory Consolidation (Episodic → Semantic)
  # ============================================================================

  @consolidation_threshold 0.7
  @consolidation_limit 20

  @doc false
  defp consolidate_memories(state) do
    Logger.info("[Dreamer] Starting memory consolidation (DRE Episodic → Semantic)...")

    # Calculate personal baseline for alignment check
    baseline_pad = calculate_personal_baseline(state)

    # Search for candidate episodic memories
    case safe_memory_search("", @consolidation_limit, state.memory) do
      {:ok, candidates} when is_list(candidates) ->
        # Filter memories using DRE Consolidation Score
        to_consolidate =
          candidates
          |> Enum.filter(fn m ->
            type = get_memory_field(m, :type, "generic")
            is_episodic = type == "episodic" or type == :episodic

            if is_episodic do
              # Extract parameters for DRE
              emotion =
                get_memory_field(m, :emotion, %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})

              memory_pad = %{
                pleasure: Map.get(emotion, :pleasure, 0.0),
                arousal: Map.get(emotion, :arousal, 0.0),
                dominance: Map.get(emotion, :dominance, 0.0)
              }

              importance = get_memory_field(m, :importance, 0.0)

              created_at =
                get_memory_field(m, :created_at, DateTime.utc_now() |> DateTime.to_iso8601())

              # Calculate age (robust)
              age_seconds =
                case DateTime.from_iso8601(created_at) do
                  {:ok, dt, _} -> DateTime.diff(DateTime.utc_now(), dt)
                  _ -> 0
                end

              access_count = get_memory_field(m, :access_count, 1)

              # DRE Score
              score =
                VivaCore.Mathematics.consolidation_score(
                  memory_pad,
                  baseline_pad,
                  importance,
                  age_seconds,
                  access_count
                )

              if score >= @consolidation_threshold do
                Logger.debug(
                  "[Dreamer] Consolidating Memory #{get_memory_field(m, :id, "?")} Score: #{Float.round(score, 3)}"
                )

                true
              else
                false
              end
            else
              false
            end
          end)
          |> Enum.take(@consolidation_limit)

        if Enum.empty?(to_consolidate) do
          Logger.debug("[Dreamer] No memories passed DRE threshold (#{@consolidation_threshold})")
          0
        else
          # Consolidate each memory
          consolidated =
            Enum.reduce(to_consolidate, 0, fn memory, count ->
              case consolidate_single_memory(memory, state) do
                :ok -> count + 1
                :error -> count
              end
            end)

          Logger.info("[Dreamer] Consolidated #{consolidated} memories via DRE")
          consolidated
        end

      {:error, reason} ->
        Logger.warning("[Dreamer] Consolidation search failed: #{inspect(reason)}")
        0
    end
  end

  defp consolidate_single_memory(memory, state) do
    id = get_memory_field(memory, :id, nil)
    content = get_memory_field(memory, :content, "")
    importance = get_memory_field(memory, :importance, 0.5)
    emotion = get_memory_field(memory, :emotion, nil)

    if id == nil or content == "" do
      :error
    else
      # Store as semantic (Qdrant - persistent)
      semantic_metadata = %{
        type: :semantic,
        # Slightly reduce importance for consolidated
        importance: importance * 0.9,
        emotion: emotion,
        consolidated_from: id,
        consolidated_at: DateTime.utc_now() |> DateTime.to_iso8601()
      }

      case safe_memory_store(content, semantic_metadata, state.memory) do
        {:ok, _new_id} ->
          # Optionally forget the episodic version to save space
          # For now, we keep both (episodic fades naturally via decay)
          Logger.debug("[Dreamer] Consolidated memory #{id} → semantic")
          :ok

        {:error, _reason} ->
          :error
      end
    end
  end

  defp safe_reflection(state, trigger_type) do
    try do
      {result, new_state} = do_reflection(state, trigger_type)
      {:ok, result, new_state}
    rescue
      e ->
        Logger.error("[Dreamer] Reflection failed: #{Exception.message(e)}")
        {:error, e}
    catch
      :exit, reason ->
        Logger.error("[Dreamer] Reflection exited: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp safe_meta_reflection(state) do
    try do
      {result, _new_state} = do_meta_reflection(state)
      {:ok, result}
    rescue
      e ->
        Logger.error("[Dreamer] Meta-reflection failed: #{Exception.message(e)}")
        {:error, e}
    catch
      :exit, reason ->
        Logger.error("[Dreamer] Meta-reflection exited: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # ============================================================================
  # Private - Reflection Process
  # ============================================================================

  defp do_reflection(state, trigger_type) do
    Logger.info("[Dreamer] Starting reflection (trigger: #{trigger_type})")

    # Step 1: Generate focal points from recent memories
    focal_points = generate_focal_points(state)

    if Enum.empty?(focal_points) do
      Logger.info("[Dreamer] No focal points generated. Skipping reflection.")
      result = %{focal_points: [], insights: [], trigger: trigger_type}
      {result, reset_accumulator(state)}
    else
      # Step 2: For each focal point, retrieve relevant memories
      retrieved =
        Enum.map(focal_points, fn fp ->
          memories = do_retrieve(fp.question, [limit: @retrieval_limit], state)
          {fp, memories}
        end)

      # Step 3: Generate insights from retrieved memories
      insights = generate_insights(retrieved, state)

      # Step 4: Store insights as thoughts
      new_state = store_thoughts(insights, state)

      result = %{
        focal_points: focal_points,
        insights: insights,
        trigger: trigger_type
      }

      Logger.info("[Dreamer] Reflection complete. Generated #{length(insights)} insights.")

      {result, reset_accumulator(new_state)}
    end
  end

  defp do_meta_reflection(state) do
    # Reflect on recent thoughts (depth=2)
    recent_thoughts = Enum.take(state.thoughts, @meta_reflection_thought_count)

    if length(recent_thoughts) < @min_thoughts_for_meta do
      {%{meta_insights: []}, state}
    else
      # Generate meta-question about the thoughts
      thought_summary =
        recent_thoughts
        |> Enum.map(&Map.get(&1, :insight, ""))
        |> Enum.join("; ")

      meta_question =
        "What patterns emerge from these reflections: #{String.slice(thought_summary, 0, 200)}?"

      # Retrieve memories related to the pattern
      memories = do_retrieve(meta_question, [limit: 10], state)

      # Generate meta-insight
      meta_insight = %{
        insight: synthesize_meta_insight(recent_thoughts, memories),
        evidence: recent_thoughts |> Enum.map(&Map.get(&1, :insight, "")) |> Enum.take(5),
        depth: 2,
        importance: 0.8,
        focal_point: meta_question,
        timestamp: DateTime.utc_now()
      }

      # Store meta-thought (with limit)
      new_thoughts =
        [meta_insight | state.thoughts]
        |> Enum.take(@max_thoughts)

      new_state = %{state | thoughts: new_thoughts}

      # Store in Memory as semantic knowledge (with error handling)
      safe_memory_store(
        meta_insight.insight,
        %{
          type: :semantic,
          importance: meta_insight.importance,
          metadata: %{depth: 2, source: :meta_reflection}
        },
        state.memory
      )

      {%{meta_insights: [meta_insight]}, new_state}
    end
  end

  defp generate_focal_points(state) do
    recent_ids = Enum.take(state.recent_memory_ids, @focal_point_memory_count)

    if Enum.empty?(recent_ids) do
      []
    else
      # Get recent memory contents with error handling
      recent_contents =
        recent_ids
        |> Enum.map(fn id -> safe_memory_get(id, state.memory) end)
        |> Enum.filter(&(&1 != nil))
        |> Enum.map(fn m -> get_memory_field(m, :content, "") end)
        |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))

      if Enum.empty?(recent_contents) do
        []
      else
        generate_focal_points_from_content(recent_contents)
      end
    end
  end

  defp generate_focal_points_from_content(contents) do
    contents
    |> Enum.take(@focal_point_count * 2)
    |> Enum.uniq_by(&extract_topic/1)
    |> Enum.take(@focal_point_count)
    |> Enum.map(fn content ->
      topic = extract_topic(content)

      %{
        question: "What have I learned about #{topic}?",
        context: String.slice(content, 0, 100),
        timestamp: DateTime.utc_now()
      }
    end)
  end

  defp extract_topic(content) when is_binary(content) do
    content
    |> String.split(~r/[.!?\n]/, parts: 2)
    |> List.first()
    |> Kernel.||("")
    |> String.slice(0, 50)
    |> String.trim()
  end

  defp extract_topic(_), do: ""

  defp generate_insights(retrieved, state) do
    Enum.flat_map(retrieved, fn {focal_point, memories} ->
      generate_insights_for_focal_point(focal_point, memories, state)
    end)
  end

  defp generate_insights_for_focal_point(focal_point, memories, state) do
    if Enum.empty?(memories) do
      []
    else
      # Recurrence: Dreamer affects Emotional state based on memory valence
      feedback = calculate_emotional_feedback(memories)

      if feedback do
        Logger.debug("[Dreamer] Insight triggered emotion: #{feedback}")
        Emotional.feel(feedback, "dreamer", 0.8, state.emotional)
      end

      memory_contents =
        Enum.map(memories, fn m ->
          get_memory_field(m, :content, "")
        end)

      insight = synthesize_insight(focal_point.question, memory_contents)

      [
        %{
          insight: insight,
          evidence: Enum.take(memory_contents, 3),
          depth: 1,
          importance: calculate_insight_importance(memories),
          focal_point: focal_point.question,
          timestamp: DateTime.utc_now()
        }
      ]
    end
  end

  defp calculate_emotional_feedback(memories) do
    total_pleasure =
      memories
      |> Enum.map(fn m ->
        emotion = get_memory_field(m, :emotion, nil)
        if emotion, do: get_pad_value(emotion, :pleasure), else: 0.0
      end)
      |> Enum.sum()

    avg_pleasure = total_pleasure / length(memories)

    cond do
      avg_pleasure > 0.1 -> :lucid_insight
      avg_pleasure < -0.1 -> :grim_realization
      true -> nil
    end
  end

  defp synthesize_insight(question, contents) do
    content_summary =
      contents
      |> Enum.take(3)
      |> Enum.map(&String.slice(to_string(&1), 0, 50))
      |> Enum.join(", ")

    "Reflecting on '#{String.slice(question, 0, 30)}...': Based on #{length(contents)} memories, " <>
      "I observe patterns related to: #{content_summary}..."
  end

  defp synthesize_meta_insight(thoughts, memories) do
    thought_count = length(thoughts)
    memory_count = length(memories)

    common_themes =
      thoughts
      |> Enum.map(&Map.get(&1, :focal_point, ""))
      |> Enum.take(3)
      |> Enum.join(", ")

    "Meta-reflection: After #{thought_count} reflections with #{memory_count} supporting memories, " <>
      "I notice recurring themes around: #{String.slice(common_themes, 0, 100)}. " <>
      "This suggests an emerging pattern in my understanding."
  end

  defp calculate_insight_importance(memories) do
    if Enum.empty?(memories) do
      @default_importance
    else
      avg =
        memories
        |> Enum.map(fn m ->
          get_memory_field(m, :importance, @default_importance)
        end)
        |> Enum.sum()
        |> Kernel./(length(memories))

      min(1.0, avg * 1.2)
    end
  end

  defp store_thoughts(insights, state) do
    # Store each insight as a thought memory (with error handling)
    Enum.each(insights, fn insight ->
      safe_memory_store(
        insight.insight,
        %{
          type: :semantic,
          importance: insight.importance,
          metadata: %{
            depth: insight.depth,
            source: :reflection,
            focal_point: insight.focal_point,
            evidence_count: length(insight.evidence)
          }
        },
        state.memory
      )
    end)

    # Update local state with limit to prevent memory leak
    new_thoughts =
      (insights ++ state.thoughts)
      |> Enum.take(@max_thoughts)

    %{
      state
      | thoughts: new_thoughts,
        total_insights_generated: state.total_insights_generated + length(insights)
    }
  end

  defp reset_accumulator(state) do
    %{
      state
      | importance_accumulator: 0.0,
        last_reflection: DateTime.utc_now(),
        reflection_count: state.reflection_count + 1,
        recent_memory_ids: []
    }
  end

  # ============================================================================
  # Private - Safe Memory Operations
  # ============================================================================

  defp safe_memory_get(id, memory_server) do
    try do
      Memory.get(id, memory_server)
    rescue
      e in [ArgumentError, RuntimeError] ->
        Logger.warning("[Dreamer] Memory.get failed for #{id}: #{Exception.message(e)}")
        nil
    catch
      :exit, {:noproc, _} ->
        Logger.warning("[Dreamer] Memory server not running")
        nil

      :exit, {:timeout, _} ->
        Logger.warning("[Dreamer] Memory.get timeout for #{id}")
        nil
    end
  end

  defp safe_memory_store(content, metadata, memory_server) do
    try do
      Memory.store(content, metadata, memory_server)
    rescue
      e in [ArgumentError, RuntimeError] ->
        Logger.warning("[Dreamer] Memory.store failed: #{Exception.message(e)}")
        {:error, e}
    catch
      :exit, {:noproc, _} ->
        Logger.warning("[Dreamer] Memory server not running")
        {:error, :noproc}

      :exit, {:timeout, _} ->
        Logger.warning("[Dreamer] Memory.store timeout")
        {:error, :timeout}
    end
  end

  # ============================================================================
  # Private - Retrieval with Composite Scoring
  # ============================================================================

  defp do_retrieve(query, opts, state) do
    limit = Keyword.get(opts, :limit, @retrieval_limit)

    # Get basic search results from Memory with error handling
    case safe_memory_search(query, limit * 2, state.memory) do
      {:ok, memories} when is_list(memories) ->
        # Get current emotional state for resonance calculation
        current_pad = get_current_pad(state)

        # Apply composite scoring
        memories
        |> Enum.map(fn m -> score_memory(m, query, current_pad) end)
        |> Enum.sort_by(fn {_m, score} -> -score end)
        |> Enum.take(limit)
        |> Enum.map(fn {m, score} -> Map.put(m, :composite_score, score) end)

      {:error, _reason} ->
        []
    end
  end

  defp safe_memory_search(query, limit, memory_server) do
    try do
      case Memory.search(query, [limit: limit], memory_server) do
        memories when is_list(memories) -> {:ok, memories}
        {:error, reason} -> {:error, reason}
        other -> {:ok, if(is_list(other), do: other, else: [])}
      end
    rescue
      e in [ArgumentError, RuntimeError] ->
        Logger.warning("[Dreamer] Memory.search failed: #{Exception.message(e)}")
        {:error, e}
    catch
      :exit, {:noproc, _} ->
        Logger.warning("[Dreamer] Memory server not running")
        {:error, :noproc}

      :exit, {:timeout, _} ->
        Logger.warning("[Dreamer] Memory.search timeout")
        {:error, :timeout}
    end
  end

  defp score_memory(memory, _query, current_pad) do
    # Extract values from memory (handle both atom and string keys)
    timestamp = get_memory_field(memory, :timestamp, 0)
    access_count = get_memory_field(memory, :access_count, 0)
    importance = get_memory_field(memory, :importance, @default_importance)
    similarity = get_memory_field(memory, :similarity, @default_similarity)
    emotion = get_memory_field(memory, :emotion, nil)

    # 1. Recency with spaced repetition (capped boost)
    recency = calculate_recency(timestamp, access_count)

    # 2. Similarity (already calculated by Memory.search)
    sim = if is_number(similarity), do: similarity, else: @default_similarity

    # 3. Importance
    imp = if is_number(importance), do: importance, else: @default_importance

    # 4. Emotional resonance (clamped to [0, 1])
    emotional = calculate_emotional_resonance(emotion, current_pad)

    # Composite score
    score =
      @weight_recency * recency +
        @weight_similarity * sim +
        @weight_importance * imp +
        @weight_emotional * emotional

    {memory, score}
  end

  defp get_memory_field(%{} = memory, field, default) when is_atom(field) do
    case Map.get(memory, field) do
      nil -> Map.get(memory, Atom.to_string(field), default)
      value -> value
    end
  end

  defp get_memory_field(_, _, default), do: default

  defp calculate_recency(timestamp, access_count) when is_integer(timestamp) do
    now = DateTime.utc_now() |> DateTime.to_unix()
    age = max(0, now - timestamp)

    # Base decay: e^(-age/τ)
    base_decay = :math.exp(-age / @decay_scale)

    # Spaced repetition boost: capped to prevent saturation at 1.0
    # boost = min(@max_repetition_boost, log(1 + access_count) / κ)
    access_count = if is_integer(access_count), do: access_count, else: 0
    raw_boost = :math.log(1 + access_count) / @repetition_boost_divisor
    capped_boost = min(@max_repetition_boost, raw_boost)

    # Final recency = base_decay * (1 + capped_boost)
    # This ensures recency never exceeds base_decay * 1.5
    base_decay * (1 + capped_boost)
  end

  defp calculate_recency(timestamp, access_count) when is_binary(timestamp) do
    case DateTime.from_iso8601(timestamp) do
      {:ok, dt, _} -> calculate_recency(DateTime.to_unix(dt), access_count)
      _ -> @default_similarity
    end
  end

  defp calculate_recency(%DateTime{} = dt, access_count) do
    calculate_recency(DateTime.to_unix(dt), access_count)
  end

  defp calculate_recency(_, _), do: @default_similarity

  defp calculate_emotional_resonance(nil, _current_pad), do: @default_emotional_resonance

  defp calculate_emotional_resonance(emotion, current_pad) when is_map(emotion) do
    # Extract PAD values
    p1 = get_pad_value(emotion, :pleasure)
    a1 = get_pad_value(emotion, :arousal)
    d1 = get_pad_value(emotion, :dominance)

    p2 = get_pad_value(current_pad, :pleasure)
    a2 = get_pad_value(current_pad, :arousal)
    d2 = get_pad_value(current_pad, :dominance)

    # Euclidean distance in PAD space
    distance = :math.sqrt(:math.pow(p1 - p2, 2) + :math.pow(a1 - a2, 2) + :math.pow(d1 - d2, 2))

    # Normalize: 1 - (distance / max_distance), clamped to [0, 1]
    # This ensures we never return negative values
    max(0.0, 1 - distance / @pad_diagonal)
  end

  defp calculate_emotional_resonance(_, _), do: @default_emotional_resonance

  defp get_pad_value(map, key) when is_map(map) and is_atom(key) do
    value = Map.get(map, key, Map.get(map, Atom.to_string(key), 0.0))
    if is_number(value), do: value, else: 0.0
  end

  defp get_current_pad(state) do
    try do
      case Emotional.get_state(state.emotional) do
        %{pleasure: _, arousal: _, dominance: _} = pad -> pad
        _ -> default_pad()
      end
    rescue
      e in [ArgumentError, RuntimeError] ->
        Logger.debug("[Dreamer] Failed to get PAD state: #{Exception.message(e)}")
        default_pad()
    catch
      :exit, {:noproc, _} ->
        Logger.debug("[Dreamer] Emotional server not running")
        default_pad()

      :exit, {:timeout, _} ->
        Logger.debug("[Dreamer] Emotional.get_state timeout")
        default_pad()
    end
  end

  defp default_pad, do: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

  # ============================================================================
  # Private - Scheduling
  # ============================================================================

  defp schedule_reflection_check do
    Process.send_after(self(), :reflection_check, @reflection_check_interval)
  end
end
