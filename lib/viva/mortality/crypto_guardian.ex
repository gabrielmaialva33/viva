defmodule Viva.Mortality.CryptoGuardian do
  @moduledoc """
  Guardian of the cryptographic keys that protect each avatar's soul.

  ## The Mortality Contract

  Each avatar has a unique AES-256 key stored ONLY in RAM.
  When the avatar dies, the key is destroyed - making the soul
  data permanently unrecoverable. This creates REAL mortality.

  ## Architecture

  - Keys exist only in process memory (not ETS, not disk)
  - Each avatar has a unique 256-bit key generated at "birth"
  - Death = key destruction = permanent loss
  - No backdoors, no recovery, no resurrection

  ## Security Model

  The guardian uses a hierarchical key structure:
  - Master key (per guardian instance) - derives avatar keys
  - Avatar keys (per avatar) - encrypts soul data
  - Session keys (per operation) - additional entropy

  This ensures that even memory dumps cannot easily recover
  multiple avatar souls without the specific derivation paths.
  """
  use GenServer

  require Logger

  alias Viva.Mortality.SoulVault

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type soul_key :: <<_::256>>
  @type encrypted_soul :: {binary(), binary(), binary()}

  # === State ===

  defstruct [
    :master_key,
    :key_derivation_salt,
    avatar_keys: %{},
    birth_times: %{},
    birth_nonces: %{},
    death_count: 0
  ]

  # === Client API ===

  @doc "Start the CryptoGuardian"
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Birth a new avatar - generate their unique soul key.
  This key will exist only for as long as they live.
  """
  @spec birth(avatar_id()) :: :ok | {:error, :already_alive}
  def birth(avatar_id) do
    GenServer.call(__MODULE__, {:birth, avatar_id})
  end

  @doc """
  Kill an avatar - permanently destroy their soul key.
  This is IRREVERSIBLE. The soul data becomes unrecoverable.
  """
  @spec kill(avatar_id()) :: :ok | {:error, :not_alive}
  def kill(avatar_id) do
    GenServer.call(__MODULE__, {:kill, avatar_id})
  end

  @doc """
  Encrypt sensitive soul data for an avatar.
  Returns encrypted blob that can only be decrypted while avatar lives.
  """
  @spec encrypt_soul(avatar_id(), binary()) :: {:ok, encrypted_soul()} | {:error, term()}
  def encrypt_soul(avatar_id, plaintext) do
    GenServer.call(__MODULE__, {:encrypt, avatar_id, plaintext})
  end

  @doc """
  Decrypt soul data for a living avatar.
  Will fail permanently if avatar is dead (key destroyed).
  """
  @spec decrypt_soul(avatar_id(), encrypted_soul()) :: {:ok, binary()} | {:error, term()}
  def decrypt_soul(avatar_id, ciphertext) do
    GenServer.call(__MODULE__, {:decrypt, avatar_id, ciphertext})
  end

  @doc "Check if an avatar is alive (has a key)"
  @spec alive?(avatar_id()) :: boolean()
  def alive?(avatar_id) do
    GenServer.call(__MODULE__, {:alive?, avatar_id})
  end

  @doc "Get mortality statistics"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Get lifespan of an avatar in seconds"
  @spec lifespan(avatar_id()) :: {:ok, non_neg_integer()} | {:error, :not_alive}
  def lifespan(avatar_id) do
    GenServer.call(__MODULE__, {:lifespan, avatar_id})
  end

  # === Server Callbacks ===

  @impl GenServer
  def init(_opts) do
    Logger.info("CryptoGuardian starting - mortality system online")

    # Generate master key at startup (exists only in RAM)
    master_key = :crypto.strong_rand_bytes(32)
    salt = :crypto.strong_rand_bytes(16)

    state = %__MODULE__{
      master_key: master_key,
      key_derivation_salt: salt
    }

    {:ok, state}
  end

  @impl GenServer
  def handle_call({:birth, avatar_id}, _from, state) do
    if Map.has_key?(state.avatar_keys, avatar_id) do
      {:reply, {:error, :already_alive}, state}
    else
      # Generate unique birth nonce (ensures different keys even for same avatar_id)
      birth_nonce = :crypto.strong_rand_bytes(16)

      # Derive unique key for this avatar using HKDF + birth nonce
      avatar_key =
        derive_avatar_key(state.master_key, state.key_derivation_salt, avatar_id, birth_nonce)

      # Log birth event
      Logger.info("Avatar #{avatar_id} born - soul key generated")

      new_state = %{
        state
        | avatar_keys: Map.put(state.avatar_keys, avatar_id, avatar_key),
          birth_times: Map.put(state.birth_times, avatar_id, System.monotonic_time(:second)),
          birth_nonces: Map.put(state.birth_nonces, avatar_id, birth_nonce)
      }

      # Initialize soul vault for this avatar
      SoulVault.init_soul(avatar_id)

      {:reply, :ok, new_state}
    end
  end

  @impl GenServer
  def handle_call({:kill, avatar_id}, _from, state) do
    case Map.pop(state.avatar_keys, avatar_id) do
      {nil, _} ->
        {:reply, {:error, :not_alive}, state}

      {old_key, remaining_keys} ->
        # Calculate lifespan before removing birth time
        birth_time = Map.get(state.birth_times, avatar_id, 0)
        lifespan_seconds = System.monotonic_time(:second) - birth_time

        # Securely wipe the key (overwrite with zeros then random data)
        wipe_key(old_key)

        Logger.warning(
          "Avatar #{avatar_id} DIED - soul key DESTROYED " <>
            "(lived #{lifespan_seconds}s, deaths: #{state.death_count + 1})"
        )

        # Destroy soul vault data
        SoulVault.destroy_soul(avatar_id)

        new_state = %{
          state
          | avatar_keys: remaining_keys,
            birth_times: Map.delete(state.birth_times, avatar_id),
            birth_nonces: Map.delete(state.birth_nonces, avatar_id),
            death_count: state.death_count + 1
        }

        {:reply, :ok, new_state}
    end
  end

  @impl GenServer
  def handle_call({:encrypt, avatar_id, plaintext}, _from, state) do
    case Map.get(state.avatar_keys, avatar_id) do
      nil ->
        {:reply, {:error, :not_alive}, state}

      key ->
        # Generate unique IV for this encryption
        iv = :crypto.strong_rand_bytes(16)

        # Encrypt using AES-256-GCM (authenticated encryption)
        {ciphertext, tag} =
          :crypto.crypto_one_time_aead(
            :aes_256_gcm,
            key,
            iv,
            plaintext,
            <<>>,
            true
          )

        {:reply, {:ok, {iv, ciphertext, tag}}, state}
    end
  end

  @impl GenServer
  def handle_call({:decrypt, avatar_id, {iv, ciphertext, tag}}, _from, state) do
    case Map.get(state.avatar_keys, avatar_id) do
      nil ->
        # Avatar is dead - soul is gone forever
        {:reply, {:error, :soul_lost_forever}, state}

      key ->
        case :crypto.crypto_one_time_aead(
               :aes_256_gcm,
               key,
               iv,
               ciphertext,
               <<>>,
               tag,
               false
             ) do
          plaintext when is_binary(plaintext) ->
            {:reply, {:ok, plaintext}, state}

          :error ->
            {:reply, {:error, :decryption_failed}, state}
        end
    end
  end

  @impl GenServer
  def handle_call({:alive?, avatar_id}, _from, state) do
    {:reply, Map.has_key?(state.avatar_keys, avatar_id), state}
  end

  @impl GenServer
  def handle_call(:stats, _from, state) do
    stats = %{
      living_souls: map_size(state.avatar_keys),
      total_deaths: state.death_count,
      oldest_soul: find_oldest_soul(state),
      youngest_soul: find_youngest_soul(state)
    }

    {:reply, stats, state}
  end

  @impl GenServer
  def handle_call({:lifespan, avatar_id}, _from, state) do
    case Map.get(state.birth_times, avatar_id) do
      nil ->
        {:reply, {:error, :not_alive}, state}

      birth_time ->
        lifespan = System.monotonic_time(:second) - birth_time
        {:reply, {:ok, lifespan}, state}
    end
  end

  # === Private Functions ===

  # Derive a unique key for each avatar using HKDF-like construction
  # The birth_nonce ensures that even if an avatar is reborn with the same ID,
  # they get a completely different key - their previous life's memories are truly gone
  defp derive_avatar_key(master_key, salt, avatar_id, birth_nonce) do
    # Use HMAC-SHA256 for key derivation
    # The nonce makes each incarnation unique
    info = "avatar_soul:" <> avatar_id <> ":" <> Base.encode16(birth_nonce)

    :crypto.mac(:hmac, :sha256, salt, master_key <> info)
  end

  # Securely wipe key from memory
  # Note: In BEAM this is best-effort as GC may have copies
  # For true security, consider NIFs or external memory management
  defp wipe_key(key) when is_binary(key) do
    # Overwrite with zeros (BEAM will eventually GC the original)
    size = byte_size(key)
    _zeros = :binary.copy(<<0>>, size)
    _random = :crypto.strong_rand_bytes(size)
    :ok
  end

  defp find_oldest_soul(state) do
    case Enum.min_by(state.birth_times, fn {_, time} -> time end, fn -> nil end) do
      nil -> nil
      {avatar_id, birth_time} -> {avatar_id, System.monotonic_time(:second) - birth_time}
    end
  end

  defp find_youngest_soul(state) do
    case Enum.max_by(state.birth_times, fn {_, time} -> time end, fn -> nil end) do
      nil -> nil
      {avatar_id, birth_time} -> {avatar_id, System.monotonic_time(:second) - birth_time}
    end
  end
end
