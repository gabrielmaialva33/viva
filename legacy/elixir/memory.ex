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
  import Bitwise
  require VivaLog

  alias VivaCore.Qdrant
  alias VivaCore.Embedder
  alias VivaBridge.Memory, as: NativeMemory

  @behaviour VivaCore.MemoryBackend

  # Decay parameters (based on Ebbinghaus research)
  # 1 week in seconds
  @default_decay_scale 604_800
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

  @doc """
  Async storage for logs (SporeLogger).
  Fire-and-forget to avoid blocking the Logger system.
  """
  def store_log(content, level, server \\ __MODULE__) do
    GenServer.cast(server, {:store_log, content, level})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    # HYBRID MODE: Episodic → Rust HNSW, Semantic/Emotional → Qdrant
    config_backend = Application.get_env(:viva_core, :memory_backend, :hybrid)
    backend = Keyword.get(opts, :backend, config_backend)

    state = %{
      backend: backend,
      created_at: DateTime.utc_now(),
      store_count: 0,
      search_count: 0,
      rust_ready: false,
      qdrant_ready: false
    }

    case backend do
      :hybrid ->
        # Initialize BOTH backends
        rust_ok = init_rust_backend()
        qdrant_ok = init_qdrant_backend()

        cond do
          rust_ok and qdrant_ok ->
            VivaLog.info(:memory, :neuron_online)
            {:ok, %{state | rust_ready: true, qdrant_ready: true}}

          rust_ok ->
            VivaLog.warning(:memory, :qdrant_unavailable_episodic)
            {:ok, %{state | rust_ready: true, qdrant_ready: false}}

          qdrant_ok ->
            VivaLog.warning(:memory, :rust_unavailable)
            {:ok, %{state | rust_ready: false, qdrant_ready: true}}

          true ->
            VivaLog.error(:memory, :both_backends_failed)
            {:ok, Map.merge(state, %{backend: :in_memory, memories: %{}, index: []})}
        end

      :qdrant ->
        case Qdrant.ensure_collection() do
          :ok ->
            VivaLog.info(:memory, :neuron_online_qdrant)
            {:ok, %{state | qdrant_ready: true}}

          {:error, reason} ->
            VivaLog.warning(:memory, :qdrant_unavailable, reason: inspect(reason))
            {:ok, Map.merge(state, %{backend: :in_memory, memories: %{}, index: []})}
        end

      :in_memory ->
        VivaLog.info(:memory, :neuron_online_inmemory)
        {:ok, Map.merge(state, %{memories: %{}, index: []})}

      :rust_native ->
        case NativeMemory.init() do
          :ok ->
            VivaLog.info(:memory, :neuron_online_rust_native)
            {:ok, %{state | rust_ready: true}}

          {:error, reason} ->
            VivaLog.error(:memory, :init_failed, reason: inspect(reason))
            {:stop, reason}
        end
    end
  end

  defp init_rust_backend do
    case NativeMemory.init() do
      :ok -> true
      {:error, _} -> false
    end
  rescue
    _ -> false
  end

  defp init_qdrant_backend do
    case Qdrant.ensure_collection() do
      :ok -> true
      {:error, _} -> false
    end
  rescue
    _ -> false
  end

  @impl true
  def handle_call({:store, content, metadata}, _from, state) do
    case do_store(content, metadata, state) do
      {:ok, id} ->
        VivaLog.debug(:memory, :stored, preview: String.slice(content, 0, 50))
        notify_dreamer(id, metadata[:importance] || 0.5)
        {:reply, {:ok, id}, %{state | store_count: state.store_count + 1}}

      {:ok, id, new_state_data} ->
        VivaLog.debug(:memory, :stored_inmemory, preview: String.slice(content, 0, 50))
        notify_dreamer(id, metadata[:importance] || 0.5)
        # Handle in-memory state update
        new_state = Map.merge(state, new_state_data)
        {:reply, {:ok, id}, %{new_state | store_count: new_state.store_count + 1}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:search, query, opts}, _from, state) do
    limit = Keyword.get(opts, :limit, 10)
    type_filter = Keyword.get(opts, :type)
    types = Keyword.get(opts, :types, [:episodic, :semantic])
    min_importance = Keyword.get(opts, :min_importance, 0.0)
    base_decay = Keyword.get(opts, :decay_scale, @default_decay_scale)

    result =
      case state.backend do
        :hybrid ->
          # HYBRID SEARCH: query both backends based on requested types
          search_hybrid(query, types, type_filter, limit, min_importance, base_decay, state)

        :qdrant ->
          search_qdrant(query, type_filter, limit, min_importance, base_decay)

        :in_memory ->
          search_in_memory(state, query, limit)

        :rust_native ->
          search_rust_native(query, limit)
      end

    case result do
      {:ok, memories} ->
        {:reply, memories, %{state | search_count: state.search_count + 1}}

      {:error, reason} ->
        VivaLog.warning(:memory, :search_failed, reason: inspect(reason))
        {:reply, [], state}
    end
  end

  @impl true
  def handle_call({:get, id}, _from, state) do
    result =
      case state.backend do
        :qdrant ->
          case Qdrant.get_point(id) do
            {:ok, memory} ->
              # Increment access count (spaced repetition)
              Qdrant.update_payload(id, %{
                access_count: (memory[:access_count] || 0) + 1,
                last_accessed: DateTime.utc_now() |> DateTime.to_iso8601()
              })

              {:ok, memory}

            error ->
              error
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

    stats =
      case state.backend do
        :hybrid ->
          # Combine stats from both backends
          hybrid_stats = %{
            rust_ready: state.rust_ready,
            qdrant_ready: state.qdrant_ready
          }

          qdrant_stats =
            if state.qdrant_ready do
              case Qdrant.stats() do
                {:ok, qs} -> %{qdrant_points: qs[:points_count] || 0}
                _ -> %{qdrant_points: :unknown}
              end
            else
              %{qdrant_points: :unavailable}
            end

          Map.merge(base_stats, Map.merge(hybrid_stats, qdrant_stats))

        :qdrant ->
          case Qdrant.stats() do
            {:ok, qdrant_stats} -> Map.merge(base_stats, qdrant_stats)
            _ -> base_stats
          end

        :in_memory ->
          Map.put(base_stats, :points_count, map_size(state.memories))

        :rust_native ->
          # NativeMemory doesn't expose stats yet - return base stats
          Map.put(base_stats, :points_count, :unknown)
      end

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:forget, id}, state) do
    new_state =
      case state.backend do
        :hybrid ->
          # Try to delete from both backends
          if state.qdrant_ready, do: Qdrant.delete_point(id)
          # Note: Rust HNSW doesn't support deletion yet
          state

        :qdrant ->
          Qdrant.delete_point(id)
          state

        :in_memory ->
          new_memories = Map.delete(state.memories, id)
          new_index = Enum.reject(state.index, &(&1 == id))
          %{state | memories: new_memories, index: new_index}

        :rust_native ->
          # Rust HNSW doesn't support deletion yet
          state
      end

    VivaLog.debug(:memory, :forgot, id: id)
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:store_log, content, level}, state) do
    # Treat logs as "System Pain" (Emotional Memory)
    # High importance for errors, medium for warnings
    importance = if level == :error, do: 0.9, else: 0.5

    # Generate metadata
    metadata = %{
      type: :system_log,
      level: level,
      importance: importance,
      # Pain/Distress
      emotion: %{pleasure: -0.5, arousal: 0.8, dominance: -0.5}
    }

    # Inline storage (blocking the memory process briefly, but ensures logs are captured)
    case do_store(content, metadata, state) do
      {:ok, id} ->
        # Logged successfully (don't verify, just accept)
        notify_dreamer(id, importance)
        {:noreply, %{state | store_count: state.store_count + 1}}

      {:ok, id, new_state_data} ->
        notify_dreamer(id, importance)
        new_state = Map.merge(state, new_state_data)
        {:noreply, %{new_state | store_count: new_state.store_count + 1}}

      {:error, _reason} ->
        # Failed to store log - nothing we can do, don't crash the memory
        {:noreply, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp generate_id do
    # Qdrant requires UUID format - generate v4 UUID using Erlang crypto
    <<a1::32, a2::16, a3::16, a4::16, a5::48>> = :crypto.strong_rand_bytes(16)
    # Set version (4) and variant bits
    a3_v4 = (a3 &&& 0x0FFF) ||| 0x4000
    a4_var = (a4 &&& 0x3FFF) ||| 0x8000

    :io_lib.format(
      "~8.16.0b-~4.16.0b-~4.16.0b-~4.16.0b-~12.16.0b",
      [a1, a2, a3_v4, a4_var, a5]
    )
    |> IO.iodata_to_binary()
  end

  # Notify Dreamer of new memory (non-blocking)
  defp notify_dreamer(memory_id, importance) do
    # Check if Dreamer is running before sending
    case Process.whereis(VivaCore.Dreamer) do
      nil ->
        # Dreamer not running, skip notification
        :ok

      _pid ->
        # Don't block the Memory process - fire and forget
        spawn(fn ->
          try do
            VivaCore.Dreamer.on_memory_stored(memory_id, importance)
          rescue
            # Ignore errors
            _ -> :ok
          catch
            :exit, _ -> :ok
          end
        end)
    end
  end

  defp build_filter(nil, min_importance) when min_importance <= 0, do: nil

  defp build_filter(type, min_importance) do
    conditions = []

    conditions =
      if type do
        [%{key: "type", match: %{value: Atom.to_string(type)}} | conditions]
      else
        conditions
      end

    conditions =
      if min_importance > 0 do
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

    results =
      state.memories
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

  # HYBRID SEARCH: Query both Rust and Qdrant based on memory types
  defp search_hybrid(query, types, type_filter, limit, min_importance, base_decay, state) do
    case Embedder.embed(query) do
      {:ok, query_vector} ->
        results = []

        # Episodic search (Rust HNSW - fast ~1ms)
        results =
          if :episodic in types and state.rust_ready do
            case search_rust_native_raw(query_vector, limit) do
              {:ok, episodic} -> results ++ episodic
              _ -> results
            end
          else
            results
          end

        # Semantic/Emotional search (Qdrant - persistent)
        qdrant_types = types -- [:episodic]

        results =
          if qdrant_types != [] and state.qdrant_ready do
            filter = build_hybrid_filter(qdrant_types, type_filter, min_importance)

            case Qdrant.search_with_decay(query_vector,
                   limit: limit,
                   decay_scale: base_decay,
                   filter: filter
                 ) do
              {:ok, semantic} -> results ++ semantic
              _ -> results
            end
          else
            results
          end

        # Merge, dedupe by id, sort by similarity
        final =
          results
          |> Enum.uniq_by(& &1[:id])
          |> Enum.sort_by(& &1[:similarity], :desc)
          |> Enum.take(limit)

        {:ok, final}

      {:error, _} ->
        {:ok, []}
    end
  end

  defp search_qdrant(query, type_filter, limit, min_importance, base_decay) do
    case Embedder.embed(query) do
      {:ok, query_vector} ->
        filter = build_filter(type_filter, min_importance)

        Qdrant.search_with_decay(query_vector,
          limit: limit,
          decay_scale: base_decay,
          filter: filter
        )

      {:error, _} ->
        {:ok, []}
    end
  end

  defp search_rust_native(query, limit) do
    case Embedder.embed(query) do
      {:ok, query_vector} ->
        search_rust_native_raw(query_vector, limit)

      {:error, _} ->
        {:ok, []}
    end
  end

  defp search_rust_native_raw(query_vector, limit) do
    raw_results = NativeMemory.search(query_vector, limit)

    formatted =
      Enum.map(raw_results, fn result ->
        case result do
          {id, content, score, imp} ->
            %{
              id: id,
              content: content,
              similarity: score,
              importance: imp,
              type: :episodic,
              timestamp: nil
            }

          %{} = map ->
            Map.merge(%{timestamp: nil, importance: 0.0, type: :episodic}, map)

          _ ->
            %{
              id: nil,
              content: inspect(result),
              similarity: 0.0,
              timestamp: nil,
              importance: 0.0,
              type: :episodic
            }
        end
      end)

    {:ok, formatted}
  end

  defp build_hybrid_filter(types, type_filter, min_importance) do
    conditions = []

    # Filter by specific types (for Qdrant - semantic/emotional/procedural)
    conditions =
      if types != [] do
        type_conditions =
          Enum.map(types, fn t ->
            %{key: "type", match: %{value: Atom.to_string(t)}}
          end)

        # OR condition for types
        [%{should: type_conditions} | conditions]
      else
        conditions
      end

    # Override with specific type if provided
    conditions =
      if type_filter do
        [%{key: "type", match: %{value: Atom.to_string(type_filter)}} | conditions]
      else
        conditions
      end

    conditions =
      if min_importance > 0 do
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

  defp do_store(content, metadata, state) do
    id = generate_id()
    now = DateTime.utc_now()
    mem_type = metadata[:type] || :generic

    # Build payload
    payload = %{
      content: content,
      type: Atom.to_string(mem_type),
      importance: metadata[:importance] || 0.5,
      emotion: metadata[:emotion],
      timestamp: DateTime.to_iso8601(now),
      access_count: 0,
      last_accessed: DateTime.to_iso8601(now)
    }

    case state.backend do
      :hybrid ->
        # HYBRID ROUTING: episodic → Rust, others → Qdrant
        case mem_type do
          :episodic when state.rust_ready ->
            store_rust_native(id, content, payload, metadata)

          type when type in [:semantic, :emotional, :procedural] and state.qdrant_ready ->
            store_qdrant(id, content, payload)

          :episodic when state.qdrant_ready ->
            # Fallback: Rust not ready, use Qdrant
            VivaLog.debug(:memory, :rust_fallback_qdrant)
            store_qdrant(id, content, payload)

          _ when state.rust_ready ->
            # Fallback: Qdrant not ready, use Rust
            VivaLog.debug(:memory, :qdrant_fallback_rust)
            store_rust_native(id, content, payload, metadata)

          _ ->
            {:error, :no_backend_available}
        end

      :qdrant ->
        store_qdrant(id, content, payload)

      :in_memory ->
        entry = Map.merge(payload, %{id: id, embedding: Embedder.embed_hash(content)})
        {:ok, id, %{memories: Map.put(state.memories, id, entry), index: [id | state.index]}}

      :rust_native ->
        store_rust_native(id, content, payload, metadata)
    end
  end

  # Store in Qdrant (semantic, emotional, procedural)
  defp store_qdrant(id, content, payload) do
    case Embedder.embed(content) do
      {:ok, embedding} ->
        Qdrant.upsert_point(id, embedding, payload)
        {:ok, id}

      {:error, reason} ->
        VivaLog.error(:memory, :embedding_failed, reason: inspect(reason))
        {:error, :embedding_failed}
    end
  end

  # Store in Rust HNSW (episodic - fast, ~1ms)
  defp store_rust_native(id, content, payload, metadata) do
    case Embedder.embed(content) do
      {:ok, embedding} ->
        # Rust expects PascalCase for memory_type
        rust_type = type_to_pascal(metadata[:type] || :episodic)

        native_meta = %{
          id: id,
          content: content,
          memory_type: rust_type,
          importance: payload.importance,
          emotion: format_emotion_for_rust(payload.emotion),
          timestamp: System.system_time(:second),
          access_count: 0,
          last_accessed: System.system_time(:second)
        }

        case NativeMemory.store(embedding, native_meta) do
          id_str when is_binary(id_str) -> {:ok, id}
          {:ok, _} -> {:ok, id}
          error -> {:error, error}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Convert atom to PascalCase string for Rust enum
  defp type_to_pascal(:episodic), do: "Episodic"
  defp type_to_pascal(:semantic), do: "Semantic"
  defp type_to_pascal(:emotional), do: "Emotional"
  defp type_to_pascal(:procedural), do: "Procedural"
  defp type_to_pascal(_), do: "Episodic"

  # Format emotion for Rust (nil-safe)
  defp format_emotion_for_rust(nil), do: nil

  defp format_emotion_for_rust(%{pleasure: p, arousal: a, dominance: d}) do
    %{pleasure: p, arousal: a, dominance: d}
  end

  defp format_emotion_for_rust(_), do: nil
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
