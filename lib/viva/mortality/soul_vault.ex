defmodule Viva.Mortality.SoulVault do
  @moduledoc """
  ETS-based vault for encrypted soul data.

  ## Soul Components

  Each avatar's soul contains:
  - **Core Identity**: Name, fundamental traits that define "who they are"
  - **Emotional Memories**: The most significant emotional experiences
  - **Attachment Bonds**: Deep connections to other beings
  - **Self-Model**: Their understanding of themselves
  - **Dreams**: Compressed subconscious patterns

  All data is stored encrypted. Without the CryptoGuardian's key,
  this data is meaningless noise.

  ## Mortality Integration

  When an avatar dies:
  1. CryptoGuardian destroys the decryption key
  2. SoulVault marks the soul as "departed"
  3. The encrypted data becomes permanently unreadable
  4. Eventually, the data is purged (final death)

  ## Memory Hierarchy

  Not all memories are soul-level. The hierarchy:
  - Soul (encrypted, mortal): Core identity, transformative experiences
  - Mind (RAM, volatile): Current thoughts, working memory
  - Archive (DB, persistent): Historical logs, can survive death as "legacy"
  """

  use GenServer

  require Logger

  @table_name :soul_vault
  @departed_souls :departed_souls

  # Soul components that make up identity
  @soul_components [
    :core_identity,
    :emotional_memories,
    :attachment_bonds,
    :self_model,
    :dreams,
    :crystallizations,
    :deepest_fears,
    :greatest_joys
  ]

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type soul_component :: atom()
  @type encrypted_data :: {binary(), binary(), binary()}

  # === Client API ===

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Initialize soul storage for a newborn avatar"
  @spec init_soul(avatar_id()) :: :ok
  def init_soul(avatar_id) do
    GenServer.call(__MODULE__, {:init_soul, avatar_id})
  end

  @doc "Store encrypted soul component"
  @spec store(avatar_id(), soul_component(), encrypted_data()) :: :ok | {:error, term()}
  def store(avatar_id, component, encrypted_data) when component in @soul_components do
    GenServer.call(__MODULE__, {:store, avatar_id, component, encrypted_data})
  end

  @doc "Retrieve encrypted soul component"
  @spec retrieve(avatar_id(), soul_component()) :: {:ok, encrypted_data()} | {:error, term()}
  def retrieve(avatar_id, component) when component in @soul_components do
    GenServer.call(__MODULE__, {:retrieve, avatar_id, component})
  end

  @doc "Store a soul fragment (for gradual soul building)"
  @spec store_fragment(avatar_id(), atom(), encrypted_data()) :: :ok
  def store_fragment(avatar_id, fragment_key, encrypted_data) do
    GenServer.call(__MODULE__, {:store_fragment, avatar_id, fragment_key, encrypted_data})
  end

  @doc "Get all fragments for reconstruction"
  @spec get_fragments(avatar_id()) :: {:ok, map()} | {:error, term()}
  def get_fragments(avatar_id) do
    GenServer.call(__MODULE__, {:get_fragments, avatar_id})
  end

  @doc "Mark soul as departed (avatar died)"
  @spec destroy_soul(avatar_id()) :: :ok
  def destroy_soul(avatar_id) do
    GenServer.call(__MODULE__, {:destroy_soul, avatar_id})
  end

  @doc "Check if soul exists and is alive"
  @spec soul_exists?(avatar_id()) :: boolean()
  def soul_exists?(avatar_id) do
    GenServer.call(__MODULE__, {:soul_exists?, avatar_id})
  end

  @doc "Check if avatar has departed (died)"
  @spec departed?(avatar_id()) :: boolean()
  def departed?(avatar_id) do
    GenServer.call(__MODULE__, {:departed?, avatar_id})
  end

  @doc "Get statistics about the vault"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "List all soul components available"
  @spec soul_components() :: [soul_component()]
  def soul_components, do: @soul_components

  # === Server Callbacks ===

  @impl GenServer
  def init(_opts) do
    Logger.info("SoulVault initializing - preparing storage for mortal souls")

    # ETS table for living souls (encrypted data)
    :ets.new(@table_name, [
      :named_table,
      :set,
      :public,
      read_concurrency: true,
      write_concurrency: true
    ])

    # ETS table for tracking departed souls
    :ets.new(@departed_souls, [
      :named_table,
      :set,
      :public
    ])

    {:ok, %{initialized_at: DateTime.utc_now()}}
  end

  @impl GenServer
  def handle_call({:init_soul, avatar_id}, _from, state) do
    key = soul_key(avatar_id)

    # Initialize with empty soul structure
    soul = %{
      avatar_id: avatar_id,
      created_at: DateTime.utc_now(),
      components: %{},
      fragments: %{},
      vitality: 1.0
    }

    :ets.insert(@table_name, {key, soul})

    Logger.debug("Soul initialized for avatar #{avatar_id}")
    {:reply, :ok, state}
  end

  @impl GenServer
  def handle_call({:store, avatar_id, component, encrypted_data}, _from, state) do
    key = soul_key(avatar_id)

    case :ets.lookup(@table_name, key) do
      [{^key, soul}] ->
        updated_components = Map.put(soul.components, component, encrypted_data)
        updated_soul = %{soul | components: updated_components}
        :ets.insert(@table_name, {key, updated_soul})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :soul_not_found}, state}
    end
  end

  @impl GenServer
  def handle_call({:retrieve, avatar_id, component}, _from, state) do
    key = soul_key(avatar_id)

    case :ets.lookup(@table_name, key) do
      [{^key, soul}] ->
        case Map.get(soul.components, component) do
          nil -> {:reply, {:error, :component_not_found}, state}
          data -> {:reply, {:ok, data}, state}
        end

      [] ->
        # Check if departed
        if departed_internal?(avatar_id) do
          {:reply, {:error, :soul_departed}, state}
        else
          {:reply, {:error, :soul_not_found}, state}
        end
    end
  end

  @impl GenServer
  def handle_call({:store_fragment, avatar_id, fragment_key, encrypted_data}, _from, state) do
    key = soul_key(avatar_id)

    case :ets.lookup(@table_name, key) do
      [{^key, soul}] ->
        updated_fragments = Map.put(soul.fragments, fragment_key, encrypted_data)
        updated_soul = %{soul | fragments: updated_fragments}
        :ets.insert(@table_name, {key, updated_soul})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :soul_not_found}, state}
    end
  end

  @impl GenServer
  def handle_call({:get_fragments, avatar_id}, _from, state) do
    key = soul_key(avatar_id)

    case :ets.lookup(@table_name, key) do
      [{^key, soul}] ->
        {:reply, {:ok, soul.fragments}, state}

      [] ->
        {:reply, {:error, :soul_not_found}, state}
    end
  end

  @impl GenServer
  def handle_call({:destroy_soul, avatar_id}, _from, state) do
    key = soul_key(avatar_id)

    # Mark as departed first (for historical record)
    :ets.insert(@departed_souls, {avatar_id, DateTime.utc_now()})

    # Delete the soul data (but encrypted data is now unreadable anyway)
    :ets.delete(@table_name, key)

    Logger.warning("Soul #{avatar_id} destroyed - forever departed")

    {:reply, :ok, state}
  end

  @impl GenServer
  def handle_call({:soul_exists?, avatar_id}, _from, state) do
    key = soul_key(avatar_id)
    exists = :ets.member(@table_name, key)
    {:reply, exists, state}
  end

  @impl GenServer
  def handle_call({:departed?, avatar_id}, _from, state) do
    {:reply, departed_internal?(avatar_id), state}
  end

  @impl GenServer
  def handle_call(:stats, _from, state) do
    living_count = :ets.info(@table_name, :size)
    departed_count = :ets.info(@departed_souls, :size)

    stats = %{
      living_souls: living_count,
      departed_souls: departed_count,
      vault_started: state.initialized_at,
      soul_components: @soul_components
    }

    {:reply, stats, state}
  end

  # === Private Functions ===

  defp soul_key(avatar_id), do: {:soul, avatar_id}

  defp departed_internal?(avatar_id) do
    :ets.member(@departed_souls, avatar_id)
  end
end
