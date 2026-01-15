defmodule VivaCore.Memory do
  @moduledoc """
  Memory GenServer - Memória de longo prazo de VIVA (Stub).

  Este módulo implementará:
  - Fase 1-4: In-memory vector store (sem dependências externas)
  - Fase 5+: Qdrant backend para persistência semântica

  ## Filosofia de Persistência
  A memória semântica PERSISTE mesmo após a "morte" da personalidade.
  Isso permite "reencarnação": nova VIVA nasce com conhecimento,
  mas sem a identidade da anterior.

  ## Tipos de Memória
  - Episódica: Eventos específicos com timestamp
  - Semântica: Conhecimento geral e padrões
  - Emocional: Estados PAD ao longo do tempo

  ## Status: STUB
  Este módulo é um placeholder para implementação futura.
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
  Armazena uma memória.

  ## Exemplo
      iex> VivaCore.Memory.store("Conheci Gabriel", %{type: :episodic, importance: 0.8})
      {:ok, "mem_abc123"}
  """
  @impl VivaCore.MemoryBackend
  def store(content, metadata \\ %{}, server \\ __MODULE__) do
    GenServer.call(server, {:store, content, metadata})
  end

  @doc """
  Busca memórias por similaridade semântica.

  ## Exemplo
      iex> VivaCore.Memory.search("Gabriel", limit: 5)
      [%{content: "Conheci Gabriel", similarity: 0.95, ...}]
  """
  @impl VivaCore.MemoryBackend
  def search(query, opts \\ [], server \\ __MODULE__) do
    GenServer.call(server, {:search, query, opts})
  end

  @doc """
  Retorna uma memória específica por ID.
  """
  @impl VivaCore.MemoryBackend
  def get(id, server \\ __MODULE__) do
    GenServer.call(server, {:get, id})
  end

  @doc """
  Remove uma memória (com decay ou explicitamente).
  """
  @impl VivaCore.MemoryBackend
  def forget(id, server \\ __MODULE__) do
    GenServer.cast(server, {:forget, id})
  end

  @doc """
  Retorna estatísticas da memória.
  """
  def stats(server \\ __MODULE__) do
    GenServer.call(server, :stats)
  end

  @doc """
  Carrega memórias de uma instância anterior (para "reencarnação").
  """
  def load_from_previous(previous_viva_id, server \\ __MODULE__) do
    GenServer.call(server, {:load_from_previous, previous_viva_id})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Memory] Neurônio de memória iniciando (modo: in-memory stub)")

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
      embedding: nil,  # TODO: Gerar embedding
      metadata: Map.merge(%{type: :generic, importance: 0.5}, metadata),
      timestamp: DateTime.utc_now()
    }

    new_memories = Map.put(state.memories, id, entry)
    new_index = [id | state.index]

    Logger.debug("[Memory] Armazenada memória #{id}: #{inspect(content)}")

    {:reply, {:ok, id}, %{state | memories: new_memories, index: new_index}}
  end

  @impl true
  def handle_call({:search, _query, opts}, _from, state) do
    # STUB: Retorna as N memórias mais recentes
    limit = Keyword.get(opts, :limit, 10)

    results =
      state.index
      |> Enum.take(limit)
      |> Enum.map(&Map.get(state.memories, &1))
      |> Enum.reject(&is_nil/1)
      |> Enum.map(fn mem ->
        Map.put(mem, :similarity, 0.5)  # STUB: similaridade fake
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
    # STUB: Em produção, buscaria do Qdrant
    Logger.info("[Memory] Load from previous: não implementado (stub)")
    {:reply, {:ok, 0}, state}
  end

  @impl true
  def handle_cast({:forget, id}, state) do
    new_memories = Map.delete(state.memories, id)
    new_index = Enum.reject(state.index, &(&1 == id))

    Logger.debug("[Memory] Esquecida memória #{id}")

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
  Behaviour para backends de memória.

  Permite trocar entre in-memory (dev) e Qdrant (prod)
  sem alterar o código que usa Memory.
  """

  @callback store(content :: term(), metadata :: map()) :: {:ok, String.t()} | {:error, term()}
  @callback search(query :: String.t(), opts :: keyword()) :: list(map())
  @callback get(id :: String.t()) :: map() | nil
  @callback forget(id :: String.t()) :: :ok
end
