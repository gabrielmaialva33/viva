defmodule VivaCore.Dreamer do
  @moduledoc """
  Dreamer GenServer - VIVA's reflection and consolidation system.

  Implements the reflection mechanism from generative agents (Park et al., 2023)
  adapted for VIVA's emotional architecture.

  ## Mathematical Foundations

  ### Retrieval Scoring (adapted from Park et al.)
  S(m, q) = w_r · D(m) + w_s · Sim(m, q) + w_i · I(m) + w_e · E(m)

  Where:
  - D(m) = e^(-age/τ) · (1 + log(1 + access_count)/10)  # Temporal decay + spaced repetition
  - Sim(m, q) = cos_sim(emb_m, emb_q)                    # Semantic similarity
  - I(m) = importance ∈ [0, 1]                           # Memory importance
  - E(m) = 1 - ||PAD_m - PAD_current|| / √12             # Emotional resonance

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

  # Scoring weights
  @weight_recency 0.2
  @weight_similarity 0.4
  @weight_importance 0.2
  @weight_emotional 0.2

  # Number of focal points to generate per reflection
  @focal_point_count 3

  # Memories to retrieve per focal point
  @retrieval_limit 20

  # PAD space diagonal (√12 for [-1,1]³ cube)
  @pad_diagonal :math.sqrt(12)

  # ============================================================================
  # Types
  # ============================================================================

  @type focal_point :: %{
          question: String.t(),
          context: String.t(),
          timestamp: DateTime.t()
        }

  @type reflection :: %{
          insight: String.t(),
          evidence: [String.t()],
          depth: non_neg_integer(),
          importance: float(),
          focal_point: String.t(),
          timestamp: DateTime.t()
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
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Notifies the Dreamer of a new memory storage.
  Called automatically by Memory GenServer after store.

  Accumulates importance and may trigger reflection.
  """
  def on_memory_stored(memory_id, importance, server \\ __MODULE__) do
    GenServer.cast(server, {:memory_stored, memory_id, importance})
  end

  @doc """
  Forces a reflection cycle immediately.
  Useful for testing or manual triggering.
  """
  def reflect_now(server \\ __MODULE__) do
    GenServer.call(server, :reflect_now, 60_000)
  end

  @doc """
  Initiates a deep reflection (sleep cycle).
  Performs multiple reflection iterations and higher-depth analysis.
  """
  def sleep_cycle(server \\ __MODULE__) do
    GenServer.call(server, :sleep_cycle, 120_000)
  end

  @doc """
  Returns the current state and statistics.
  """
  def status(server \\ __MODULE__) do
    GenServer.call(server, :status)
  end

  @doc """
  Returns recent reflections (thoughts).
  """
  def recent_thoughts(limit \\ 10, server \\ __MODULE__) do
    GenServer.call(server, {:recent_thoughts, limit})
  end

  @doc """
  Retrieves memories with composite scoring.
  Uses the full scoring formula: recency + similarity + importance + emotional.
  """
  def retrieve_with_scoring(query, opts \\ [], server \\ __MODULE__) do
    GenServer.call(server, {:retrieve, query, opts}, 30_000)
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

      # Recent memories (for focal point generation)
      recent_memory_ids: [],

      # Generated thoughts
      thoughts: [],

      # Statistics
      created_at: DateTime.utc_now(),
      total_insights_generated: 0
    }

    Logger.info("[Dreamer] Dreamer neuron online. Ready to reflect.")

    # Schedule periodic check
    schedule_reflection_check()

    {:ok, state}
  end

  @impl true
  def handle_cast({:memory_stored, memory_id, importance}, state) do
    # Accumulate importance
    new_accumulator = state.importance_accumulator + importance

    # Track recent memories
    recent = [memory_id | state.recent_memory_ids] |> Enum.take(50)

    new_state = %{
      state
      | importance_accumulator: new_accumulator,
        recent_memory_ids: recent
    }

    Logger.debug(
      "[Dreamer] Memory stored: #{memory_id}, importance: #{importance}, accumulator: #{Float.round(new_accumulator, 2)}"
    )

    # Check if threshold reached
    if new_accumulator >= @importance_threshold do
      Logger.info("[Dreamer] Importance threshold reached (#{Float.round(new_accumulator, 2)}). Triggering reflection.")
      send(self(), :trigger_reflection)
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:reflect_now, _from, state) do
    {result, new_state} = do_reflection(state, :manual)
    {:reply, result, new_state}
  end

  @impl true
  def handle_call(:sleep_cycle, _from, state) do
    Logger.info("[Dreamer] Sleep cycle initiated. Deep reflection starting...")

    # Multiple reflection iterations
    {results, final_state} =
      Enum.reduce(1..3, {[], state}, fn iteration, {acc, s} ->
        Logger.info("[Dreamer] Sleep reflection iteration #{iteration}/3")
        {result, new_s} = do_reflection(s, :sleep)
        {[result | acc], new_s}
      end)

    # Higher-depth reflection on the thoughts themselves
    {meta_result, meta_state} = do_meta_reflection(final_state)

    all_results = %{
      iterations: Enum.reverse(results),
      meta_reflection: meta_result,
      total_insights: Enum.sum(Enum.map(results, &length(&1.insights)))
    }

    Logger.info("[Dreamer] Sleep cycle complete. Generated #{all_results.total_insights} insights.")

    {:reply, all_results, meta_state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    status = %{
      importance_accumulator: Float.round(state.importance_accumulator, 2),
      threshold: @importance_threshold,
      progress_percent: Float.round(state.importance_accumulator / @importance_threshold * 100, 1),
      last_reflection: state.last_reflection,
      seconds_since_reflection: DateTime.diff(DateTime.utc_now(), state.last_reflection),
      reflection_count: state.reflection_count,
      recent_memories_count: length(state.recent_memory_ids),
      thoughts_count: length(state.thoughts),
      total_insights_generated: state.total_insights_generated,
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at)
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
  def handle_info(:trigger_reflection, state) do
    {_result, new_state} = do_reflection(state, :threshold)
    {:noreply, new_state}
  end

  @impl true
  def handle_info(:reflection_check, state) do
    schedule_reflection_check()

    # Check time-based trigger
    seconds_since = DateTime.diff(DateTime.utc_now(), state.last_reflection)

    new_state =
      if seconds_since >= @max_reflection_interval and length(state.recent_memory_ids) > 5 do
        Logger.info("[Dreamer] Time-based reflection trigger (#{seconds_since}s since last)")
        {_result, s} = do_reflection(state, :time)
        s
      else
        state
      end

    {:noreply, new_state}
  end

  # ============================================================================
  # Private - Reflection Process
  # ============================================================================

  defp do_reflection(state, trigger_type) do
    Logger.info("[Dreamer] Starting reflection (trigger: #{trigger_type})")

    # Step 1: Generate focal points from recent memories
    focal_points = generate_focal_points(state)

    if focal_points == [] do
      Logger.info("[Dreamer] No focal points generated. Skipping reflection.")
      result = %{focal_points: [], insights: [], trigger: trigger_type}
      {result, reset_accumulator(state)}
    else
      # Step 2: For each focal point, retrieve relevant memories
      retrieved = Enum.map(focal_points, fn fp ->
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
    recent_thoughts = Enum.take(state.thoughts, 10)

    if length(recent_thoughts) < 3 do
      {%{meta_insights: []}, state}
    else
      # Generate meta-question about the thoughts
      thought_summary =
        recent_thoughts
        |> Enum.map(& &1.insight)
        |> Enum.join("; ")

      meta_question = "What patterns emerge from these reflections: #{String.slice(thought_summary, 0, 200)}?"

      # Retrieve memories related to the pattern
      memories = do_retrieve(meta_question, [limit: 10], state)

      # Generate meta-insight
      meta_insight = %{
        insight: synthesize_meta_insight(recent_thoughts, memories),
        evidence: Enum.map(recent_thoughts, & &1.insight) |> Enum.take(5),
        depth: 2,
        importance: 0.8,
        focal_point: meta_question,
        timestamp: DateTime.utc_now()
      }

      # Store meta-thought
      new_state = %{state | thoughts: [meta_insight | state.thoughts]}

      # Store in Memory as semantic knowledge
      Memory.store(
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
    recent_ids = Enum.take(state.recent_memory_ids, 20)

    if recent_ids == [] do
      []
    else
      # Get recent memory contents
      recent_contents =
        recent_ids
        |> Enum.map(fn id -> Memory.get(id, state.memory) end)
        |> Enum.filter(&(&1 != nil))
        |> Enum.map(fn m -> Map.get(m, :content, "") end)
        |> Enum.filter(&(String.length(&1) > 0))

      if recent_contents == [] do
        []
      else
        # Generate focal points based on recent content
        # In a full implementation, this would use an LLM
        # For now, we extract key themes
        generate_focal_points_from_content(recent_contents)
      end
    end
  end

  defp generate_focal_points_from_content(contents) do
    # Simple heuristic: create questions from the most recent distinct topics
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

  defp extract_topic(content) do
    # Extract first meaningful phrase (simplified)
    content
    |> String.split(~r/[.!?\n]/, parts: 2)
    |> List.first()
    |> String.slice(0, 50)
    |> String.trim()
  end

  defp generate_insights(retrieved, state) do
    retrieved
    |> Enum.flat_map(fn {focal_point, memories} ->
      generate_insights_for_focal_point(focal_point, memories, state)
    end)
  end

  defp generate_insights_for_focal_point(focal_point, memories, _state) do
    if memories == [] do
      []
    else
      # Synthesize insight from memories
      # In full implementation, this would use an LLM
      memory_contents = Enum.map(memories, fn m ->
        Map.get(m, :content, Map.get(m, "content", ""))
      end)

      # Generate insight (simplified heuristic)
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

  defp synthesize_insight(question, contents) do
    # Simplified synthesis - in production this would use an LLM
    content_summary =
      contents
      |> Enum.take(3)
      |> Enum.map(&String.slice(&1, 0, 50))
      |> Enum.join(", ")

    "Reflecting on '#{String.slice(question, 0, 30)}...': Based on #{length(contents)} memories, " <>
      "I observe patterns related to: #{content_summary}..."
  end

  defp synthesize_meta_insight(thoughts, memories) do
    thought_count = length(thoughts)
    memory_count = length(memories)

    common_themes =
      thoughts
      |> Enum.map(& &1.focal_point)
      |> Enum.take(3)
      |> Enum.join(", ")

    "Meta-reflection: After #{thought_count} reflections with #{memory_count} supporting memories, " <>
      "I notice recurring themes around: #{String.slice(common_themes, 0, 100)}. " <>
      "This suggests an emerging pattern in my understanding."
  end

  defp calculate_insight_importance(memories) do
    if memories == [] do
      0.5
    else
      # Average importance of supporting memories, boosted
      avg =
        memories
        |> Enum.map(fn m ->
          Map.get(m, :importance, Map.get(m, "importance", 0.5))
        end)
        |> Enum.sum()
        |> Kernel./(length(memories))

      min(1.0, avg * 1.2)
    end
  end

  defp store_thoughts(insights, state) do
    # Store each insight as a thought memory
    Enum.each(insights, fn insight ->
      Memory.store(
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

    # Update local state
    %{
      state
      | thoughts: insights ++ state.thoughts,
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
  # Private - Retrieval with Composite Scoring
  # ============================================================================

  defp do_retrieve(query, opts, state) do
    limit = Keyword.get(opts, :limit, @retrieval_limit)

    # Get basic search results from Memory
    case Memory.search(query, [limit: limit * 2], state.memory) do
      memories when is_list(memories) ->
        # Get current emotional state for resonance calculation
        current_pad = get_current_pad(state)

        # Apply composite scoring
        memories
        |> Enum.map(fn m -> score_memory(m, query, current_pad) end)
        |> Enum.sort_by(fn {_m, score} -> -score end)
        |> Enum.take(limit)
        |> Enum.map(fn {m, score} -> Map.put(m, :composite_score, score) end)

      _ ->
        []
    end
  end

  defp score_memory(memory, _query, current_pad) do
    # Extract values from memory (handle both atom and string keys)
    timestamp = get_memory_field(memory, :timestamp, 0)
    access_count = get_memory_field(memory, :access_count, 0)
    importance = get_memory_field(memory, :importance, 0.5)
    similarity = get_memory_field(memory, :similarity, 0.5)
    emotion = get_memory_field(memory, :emotion, nil)

    # 1. Recency with spaced repetition
    recency = calculate_recency(timestamp, access_count)

    # 2. Similarity (already calculated by Memory.search)
    sim = similarity

    # 3. Importance
    imp = importance

    # 4. Emotional resonance
    emotional = calculate_emotional_resonance(emotion, current_pad)

    # Composite score
    score =
      @weight_recency * recency +
        @weight_similarity * sim +
        @weight_importance * imp +
        @weight_emotional * emotional

    {memory, score}
  end

  defp get_memory_field(memory, field, default) do
    Map.get(memory, field, Map.get(memory, to_string(field), default))
  end

  defp calculate_recency(timestamp, access_count) when is_integer(timestamp) do
    now = DateTime.utc_now() |> DateTime.to_unix()
    age = max(0, now - timestamp)

    # Base decay: e^(-age/τ)
    base_decay = :math.exp(-age / @decay_scale)

    # Spaced repetition boost: (1 + log(1 + access_count) / 10)
    repetition_boost = 1 + :math.log(1 + access_count) / 10

    min(1.0, base_decay * repetition_boost)
  end

  defp calculate_recency(timestamp, access_count) when is_binary(timestamp) do
    case DateTime.from_iso8601(timestamp) do
      {:ok, dt, _} -> calculate_recency(DateTime.to_unix(dt), access_count)
      _ -> 0.5
    end
  end

  defp calculate_recency(_, _), do: 0.5

  defp calculate_emotional_resonance(nil, _current_pad), do: 0.5

  defp calculate_emotional_resonance(emotion, current_pad) when is_map(emotion) do
    # Extract PAD values
    p1 = Map.get(emotion, :pleasure, Map.get(emotion, "pleasure", 0.0))
    a1 = Map.get(emotion, :arousal, Map.get(emotion, "arousal", 0.0))
    d1 = Map.get(emotion, :dominance, Map.get(emotion, "dominance", 0.0))

    p2 = Map.get(current_pad, :pleasure, 0.0)
    a2 = Map.get(current_pad, :arousal, 0.0)
    d2 = Map.get(current_pad, :dominance, 0.0)

    # Euclidean distance in PAD space
    distance = :math.sqrt(:math.pow(p1 - p2, 2) + :math.pow(a1 - a2, 2) + :math.pow(d1 - d2, 2))

    # Normalize: 1 - (distance / max_distance)
    # Max distance in [-1,1]³ is √12
    1 - distance / @pad_diagonal
  end

  defp calculate_emotional_resonance(_, _), do: 0.5

  defp get_current_pad(state) do
    try do
      Emotional.get_state(state.emotional)
    rescue
      _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    catch
      :exit, _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    end
  end

  # ============================================================================
  # Private - Scheduling
  # ============================================================================

  defp schedule_reflection_check do
    # Check every 5 minutes
    Process.send_after(self(), :reflection_check, 5 * 60 * 1000)
  end
end
