defmodule VivaCore.Memory do
  @moduledoc """
  Memory GenServer - VIVA's episodic and semantic memory system.

  Implements:
  - Episodic memory: specific events with timestamp and emotion
  - Semantic memory: general knowledge and patterns
  - Temporal decay: older memories naturally fade (Ebbinghaus curve)
  - Spaced repetition: frequently accessed memories decay slower

  ## Architecture

  Uses Qdrant vector database with:
  - Cosine similarity for semantic search
  - exp_decay formula for temporal relevance
  - Payload filtering for memory type and emotion

  ## Memory Persistence Philosophy

  Semantic memories PERSIST even after VIVA's "death".
  This allows reincarnation: new VIVA inherits knowledge,
  but not the identity of the previous instance.

  ## Episode Structure

  Each memory contains:
  - content: the actual memory (text or structured)
  - type: :episodic | :semantic | :emotional | :procedural
  - importance: 0.0-1.0 (affects decay rate)
  - emotion: PAD state when memory was formed
  - timestamp: when memory was created
  - access_count: times recalled (spaced repetition)
  - last_accessed: for recency calculations
  """

  use GenServer
  require Logger

  alias VivaCore.Qdrant
  alias VivaCore.Embedder
  alias VivaBridge.Memory, as: NativeMemory

  @behaviour VivaCore.MemoryBackend

  # Decay parameters (based on Ebbinghaus research)
  @default_decay_scale 604_800  # 1 week in seconds
  # TODO: implement per-memory decay scaling based on importance

  # ============================================================================
  # Types
  # ============================================================================

  @type memory_type :: :episodic | :semantic | :emotional | :procedural | :generic
  @type pad_emotion :: %{pleasure: float(), arousal: float(), dominance: float()}

  @type episode :: %{
    id: String.t(),
    content: String.t(),
    type: memory_type(),
    importance: float(),
    emotion: pad_emotion() | nil,
    timestamp: DateTime.t(),
    access_count: non_neg_integer(),
    last_accessed: DateTime.t(),
    similarity: float() | nil
  }

  # ============================================================================
  # Public API
  # ============================================================================

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Stores an episodic memory.

  ## Options
  - type: :episodic (default) | :semantic | :emotional | :procedural
  - importance: 0.0-1.0 (default 0.5)
  - emotion: %{pleasure: f, arousal: f, dominance: f}

  ## Examples

      iex> VivaCore.Memory.store("Met Gabriel for the first time", %{
      ...>   type: :episodic,
      ...>   importance: 0.9,
      ...>   emotion: %{pleasure: 0.8, arousal: 0.6, dominance: 0.5}
      ...> })
      {:ok, "mem_abc123"}
  """
  @impl VivaCore.MemoryBackend
  def store(content, metadata \\ %{}, server \\ __MODULE__) do
    GenServer.call(server, {:store, content, metadata}, 30_000)
  end

  @doc """
  Searches memories by semantic similarity with temporal decay.

  Recent memories are boosted, old memories fade naturally.
  Frequently accessed memories (spaced repetition) decay slower.

  ## Options
  - limit: max results (default 10)
  - type: filter by memory type
  - min_importance: minimum importance threshold
  - decay_scale: seconds for 50% decay (default 1 week)

  ## Examples

      iex> VivaCore.Memory.search("Gabriel", limit: 5)
      [%{content: "Met Gabriel", similarity: 0.95, ...}]
  """
  @impl VivaCore.MemoryBackend
  def search(query, opts \\ [], server \\ __MODULE__) do
    GenServer.call(server, {:search, query, opts}, 30_000)
  end

  @doc """
  Recalls a specific memory by ID.
  Increments access_count (spaced repetition).
  """
  @impl VivaCore.MemoryBackend
  def get(id, server \\ __MODULE__) do
    GenServer.call(server, {:get, id})
  end

  @doc """
  Explicitly forgets a memory (immediate deletion).
  """
  @impl VivaCore.MemoryBackend
  def forget(id, server \\ __MODULE__) do
    GenServer.cast(server, {:forget, id})
  end

  @doc """
  Returns memory statistics.
  """
  def stats(server \\ __MODULE__) do
    GenServer.call(server, :stats)
  end

  @doc """
  Stores an experience (shorthand for episodic with emotion).

  ## Example

      iex> emotion = %{pleasure: 0.7, arousal: 0.5, dominance: 0.6}
      iex> VivaCore.Memory.experience("Gabriel praised my work", emotion, importance: 0.8)
  """
  def experience(content, emotion, opts \\ [], server \\ __MODULE__) do
    metadata = Keyword.merge([type: :episodic, emotion: emotion], opts) |> Map.new()
    store(content, metadata, server)
  end

  @doc """
  Stores a learned fact (semantic memory).
  """
  def learn(content, opts \\ [], server \\ __MODULE__) do
    metadata = Keyword.merge([type: :semantic, importance: 0.6], opts) |> Map.new()
    store(content, metadata, server)
  end

  @doc """
  Associates current emotional state with a memory pattern.
  """
  def emotional_imprint(content, pad_state, server \\ __MODULE__) do
    store(content, %{type: :emotional, emotion: pad_state, importance: 0.7}, server)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    config_backend = Application.get_env(:viva_core, :memory_backend, :qdrant)
    backend = Keyword.get(opts, :backend, config_backend)

    state = %{
      backend: backend,
      created_at: DateTime.utc_now(),
      store_count: 0,
      search_count: 0
    }

    # Initialize Qdrant collection
    case backend do
      :qdrant ->
        case Qdrant.ensure_collection() do
          :ok ->
            Logger.info("[Memory] Memory neuron online (backend: Qdrant)")
            {:ok, state}

          {:error, reason} ->
            Logger.warning("[Memory] Qdrant unavailable: #{inspect(reason)}, using in-memory fallback")
            {:ok, Map.merge(state, %{backend: :in_memory, memories: %{}, index: []})}
        end

      :in_memory ->
        Logger.info("[Memory] Memory neuron online (backend: in-memory)")
        {:ok, Map.merge(state, %{memories: %{}, index: []})}

      :rust_native ->
        # Default path relative to home
        home = System.user_home!()
        path = Application.get_env(:viva_core, :native_memory_path, Path.join(home, ".viva/memory"))
        case NativeMemory.init(path) do
          {:ok, resource} ->
            Logger.info("[Memory] Memory neuron online (backend: Rust Native - God Mode)")
            {:ok, Map.put(state, :resource, resource)}
          {:error, reason} ->
            Logger.error("[Memory] Failed to init native memory: #{inspect(reason)}")
            {:stop, reason}
        end
    end
  end

  @impl true
  def handle_call({:store, content, metadata}, _from, state) do
    id = generate_id()
    now = DateTime.utc_now()

    # Build payload
    payload = %{
      content: content,
      type: Atom.to_string(metadata[:type] || :generic),
      importance: metadata[:importance] || 0.5,
      emotion: metadata[:emotion],
      timestamp: DateTime.to_iso8601(now),
      access_count: 0,
      last_accessed: DateTime.to_iso8601(now)
    }

    result = case state.backend do
      :qdrant ->
        # Generate embedding
        case Embedder.embed(content) do
          {:ok, embedding} ->
            Qdrant.upsert_point(id, embedding, payload)

          {:error, reason} ->
            Logger.error("[Memory] Embedding failed: #{inspect(reason)}")
            {:error, :embedding_failed}
        end

      :in_memory ->
        entry = Map.merge(payload, %{id: id, embedding: Embedder.embed_hash(content)})
        new_memories = Map.put(state.memories, id, entry)
        new_index = [id | state.index]
        {:ok, id, %{state | memories: new_memories, index: new_index}}

      :rust_native ->
        case Embedder.embed(content) do
          {:ok, embedding} ->
            # Convert generic metadata to strings
            str_metadata = metadata
            |> Map.new(fn {k, v} -> {to_string(k), to_string(v)} end)
            |> Map.put("id", id)
            |> Map.put("type", payload.type)
            |> Map.put("importance", to_string(payload.importance))

            case NativeMemory.store(state.resource, content, embedding, str_metadata) do
              {:ok, _key} -> {:ok, id} # Return UUID to caller
              error -> error
            end
          error -> error
        end
    end

    case result do
      {:ok, id} ->
        Logger.debug("[Memory] Stored: #{String.slice(content, 0, 50)}...")
        # Notify Dreamer of new memory
        notify_dreamer(id, metadata[:importance] || 0.5)
        {:reply, {:ok, id}, %{state | store_count: state.store_count + 1}}

      {:ok, id, new_state} ->
        Logger.debug("[Memory] Stored (in-memory): #{String.slice(content, 0, 50)}...")
        # Notify Dreamer of new memory
        notify_dreamer(id, metadata[:importance] || 0.5)
        {:reply, {:ok, id}, %{new_state | store_count: new_state.store_count + 1}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:search, query, opts}, _from, state) do
    limit = Keyword.get(opts, :limit, 10)
    type_filter = Keyword.get(opts, :type)
    min_importance = Keyword.get(opts, :min_importance, 0.0)

    # Calculate decay scale based on importance
    base_decay = Keyword.get(opts, :decay_scale, @default_decay_scale)

    result = case state.backend do
      :qdrant ->
        case Embedder.embed(query) do
          {:ok, query_vector} ->
            # Build filter
            filter = build_filter(type_filter, min_importance)

            Qdrant.search_with_decay(query_vector,
              limit: limit,
              decay_scale: base_decay,
              filter: filter
            )

          {:error, _} ->
            {:ok, []}
        end

      :in_memory ->
        search_in_memory(state, query, limit)

      :rust_native ->
        case Embedder.embed(query) do
          {:ok, query_vector} ->
            case NativeMemory.search(state.resource, query_vector, limit) do
              {:ok, raw_results} ->
                # Map tuple results {id, content, score, importance} to map
                formatted = Enum.map(raw_results, fn {id, content, score, _imp} ->
                  %{
                    id: id,
                    content: content,
                    similarity: score, # Rust returns STDP-boosted score
                    # Minimal fields for basic display
                    timestamp: nil,
                    importance: 0.0 # TODO: fetch full meta if needed
                  }
                end)
                {:ok, formatted}
              error -> error
            end
          error -> error
        end
    end

    case result do
      {:ok, memories} ->
        {:reply, memories, %{state | search_count: state.search_count + 1}}

      {:error, reason} ->
        Logger.warning("[Memory] Search failed: #{inspect(reason)}")
        {:reply, [], state}
    end
  end

  @impl true
  def handle_call({:get, id}, _from, state) do
    result = case state.backend do
      :qdrant ->
        case Qdrant.get_point(id) do
          {:ok, memory} ->
            # Increment access count (spaced repetition)
            Qdrant.update_payload(id, %{
              access_count: (memory[:access_count] || 0) + 1,
              last_accessed: DateTime.utc_now() |> DateTime.to_iso8601()
            })
            {:ok, memory}

          error -> error
        end

      :in_memory ->
        case Map.get(state.memories, id) do
          nil -> {:error, :not_found}
          mem -> {:ok, mem}
        end
    end

    case result do
      {:ok, memory} -> {:reply, memory, state}
      {:error, _} -> {:reply, nil, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    base_stats = %{
      backend: state.backend,
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at),
      store_count: state.store_count,
      search_count: state.search_count
    }

    stats = case state.backend do
      :qdrant ->
        case Qdrant.stats() do
          {:ok, qdrant_stats} -> Map.merge(base_stats, qdrant_stats)
          _ -> base_stats
        end

      :in_memory ->
        Map.put(base_stats, :points_count, map_size(state.memories))

      :rust_native ->
        {episodic, semantic} = NativeMemory.stats(state.resource)
        base_stats
        |> Map.put(:episodic_count, episodic)
        |> Map.put(:semantic_count, semantic)
        |> Map.put(:points_count, episodic + semantic)
    end

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:forget, id}, state) do
    new_state = case state.backend do
      :qdrant ->
        Qdrant.delete_point(id)
        state

      :in_memory ->
        new_memories = Map.delete(state.memories, id)
        new_index = Enum.reject(state.index, &(&1 == id))
        %{state | memories: new_memories, index: new_index}
    end

    Logger.debug("[Memory] Forgot: #{id}")
    {:noreply, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp generate_id do
    # Qdrant requires UUID format
    UUID.uuid4()
  end

  # Notify Dreamer of new memory (non-blocking)
  defp notify_dreamer(memory_id, importance) do
    # Check if Dreamer is running before sending
    case Process.whereis(VivaCore.Dreamer) do
      nil ->
        :ok  # Dreamer not running, skip notification

      _pid ->
        # Don't block the Memory process - fire and forget
        spawn(fn ->
          try do
            VivaCore.Dreamer.on_memory_stored(memory_id, importance)
          rescue
            _ -> :ok  # Ignore errors
          catch
            :exit, _ -> :ok
          end
        end)
    end
  end

  defp build_filter(nil, min_importance) when min_importance <= 0, do: nil

  defp build_filter(type, min_importance) do
    conditions = []

    conditions = if type do
      [%{key: "type", match: %{value: Atom.to_string(type)}} | conditions]
    else
      conditions
    end

    conditions = if min_importance > 0 do
      [%{key: "importance", range: %{gte: min_importance}} | conditions]
    else
      conditions
    end

    if conditions == [] do
      nil
    else
      %{must: conditions}
    end
  end

  defp search_in_memory(state, query, limit) do
    query_vec = Embedder.embed_hash(query)

    results = state.memories
    |> Map.values()
    |> Enum.map(fn mem ->
      similarity = cosine_similarity(query_vec, mem[:embedding] || [])
      Map.put(mem, :similarity, similarity)
    end)
    |> Enum.sort_by(& &1[:similarity], :desc)
    |> Enum.take(limit)

    {:ok, results}
  end

  defp cosine_similarity([], _), do: 0.0
  defp cosine_similarity(_, []), do: 0.0
  defp cosine_similarity(a, b) when length(a) != length(b), do: 0.0
  defp cosine_similarity(a, b) do
    dot = Enum.zip(a, b) |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
    mag_a = :math.sqrt(Enum.reduce(a, 0, fn x, acc -> acc + x * x end))
    mag_b = :math.sqrt(Enum.reduce(b, 0, fn x, acc -> acc + x * x end))

    if mag_a > 0 and mag_b > 0 do
      dot / (mag_a * mag_b)
    else
      0.0
    end
  end
end

# ============================================================================
# Memory Backend Behaviour
# ============================================================================

defmodule VivaCore.MemoryBackend do
  @moduledoc """
  Behaviour for memory backends.

  Allows switching between in-memory (dev) and Qdrant (prod)
  without changing application code.
  """

  @callback store(content :: term(), metadata :: map()) :: {:ok, String.t()} | {:error, term()}
  @callback search(query :: String.t(), opts :: keyword()) :: list(map())
  @callback get(id :: String.t()) :: map() | nil
  @callback forget(id :: String.t()) :: :ok
end
