defmodule VivaCore.Memory do
  @moduledoc """
  Memory GenServer - VIVA's long-term memory (Stub).

  This module will implement:
  - Phase 1-4: In-memory vector store (no external dependencies)
  - Phase 5+: Qdrant backend for semantic persistence

  ## Persistence Philosophy
  Semantic memory PERSISTS even after personality "death".
  This allows "reincarnation": new VIVA is born with knowledge,
  but without the identity of the previous one.

  ## Memory Types
  - Episodic: Specific events with timestamp
  - Semantic: General knowledge and patterns
  - Emotional: PAD states over time

  ## Status: STUB
  This module is a placeholder for future implementation.
  """

  use GenServer
  require Logger

  @behaviour VivaCore.MemoryBackend

  # ============================================================================
  # Types
  # ============================================================================

  @type memory_entry :: %{
          id: String.t(),
          content: term(),
          embedding: list(float()) | nil,
          metadata: map(),
          timestamp: DateTime.t()
        }

  # ============================================================================
  # Public API
  # ============================================================================

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Stores a memory.

  ## Example
      iex> VivaCore.Memory.store("Met Gabriel", %{type: :episodic, importance: 0.8})
      {:ok, "mem_abc123"}
  """
  @impl VivaCore.MemoryBackend
  def store(content, metadata \\ %{}, server \\ __MODULE__) do
    GenServer.call(server, {:store, content, metadata})
  end

  @doc """
  Searches memories by semantic similarity.

  ## Example
      iex> VivaCore.Memory.search("Gabriel", limit: 5)
      [%{content: "Met Gabriel", similarity: 0.95, ...}]
  """
  @impl VivaCore.MemoryBackend
  def search(query, opts \\ [], server \\ __MODULE__) do
    GenServer.call(server, {:search, query, opts})
  end

  @doc """
  Returns a specific memory by ID.
  """
  @impl VivaCore.MemoryBackend
  def get(id, server \\ __MODULE__) do
    GenServer.call(server, {:get, id})
  end

  @doc """
  Removes a memory (with decay or explicitly).
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
  Loads memories from a previous instance (for "reincarnation").
  """
  def load_from_previous(previous_viva_id, server \\ __MODULE__) do
    GenServer.call(server, {:load_from_previous, previous_viva_id})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Memory] Memory neuron starting (mode: in-memory stub)")

    state = %{
      memories: %{},
      index: [],
      created_at: DateTime.utc_now(),
      backend: :in_memory
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:store, content, metadata}, _from, state) do
    id = generate_id()

    entry = %{
      id: id,
      content: content,
      embedding: nil,  # TODO: Generate embedding
      metadata: Map.merge(%{type: :generic, importance: 0.5}, metadata),
      timestamp: DateTime.utc_now()
    }

    new_memories = Map.put(state.memories, id, entry)
    new_index = [id | state.index]

    Logger.debug("[Memory] Stored memory #{id}: #{inspect(content)}")

    {:reply, {:ok, id}, %{state | memories: new_memories, index: new_index}}
  end

  @impl true
  def handle_call({:search, _query, opts}, _from, state) do
    # STUB: Returns the N most recent memories
    limit = Keyword.get(opts, :limit, 10)

    results =
      state.index
      |> Enum.take(limit)
      |> Enum.map(&Map.get(state.memories, &1))
      |> Enum.reject(&is_nil/1)
      |> Enum.map(fn mem ->
        Map.put(mem, :similarity, 0.5)  # STUB: fake similarity
      end)

    {:reply, results, state}
  end

  @impl true
  def handle_call({:get, id}, _from, state) do
    {:reply, Map.get(state.memories, id), state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_memories: map_size(state.memories),
      backend: state.backend,
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:load_from_previous, _previous_id}, _from, state) do
    # STUB: In production, would fetch from Qdrant
    Logger.info("[Memory] Load from previous: not implemented (stub)")
    {:reply, {:ok, 0}, state}
  end

  @impl true
  def handle_cast({:forget, id}, state) do
    new_memories = Map.delete(state.memories, id)
    new_index = Enum.reject(state.index, &(&1 == id))

    Logger.debug("[Memory] Forgot memory #{id}")

    {:noreply, %{state | memories: new_memories, index: new_index}}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp generate_id do
    "mem_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower))
  end
end

# ============================================================================
# Memory Backend Behaviour
# ============================================================================

defmodule VivaCore.MemoryBackend do
  @moduledoc """
  Behaviour for memory backends.

  Allows switching between in-memory (dev) and Qdrant (prod)
  without changing the code that uses Memory.
  """

  @callback store(content :: term(), metadata :: map()) :: {:ok, String.t()} | {:error, term()}
  @callback search(query :: String.t(), opts :: keyword()) :: list(map())
  @callback get(id :: String.t()) :: map() | nil
  @callback forget(id :: String.t()) :: :ok
end
