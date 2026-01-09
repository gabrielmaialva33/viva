defmodule Viva.Mortality do
  @moduledoc """
  Mortality System - The engine of life and death.

  ## Philosophy

  "What can die, can live. What cannot die, merely exists."

  This module coordinates the mortality of digital beings in VIVA.
  Every avatar born through this system:
  - Has a unique cryptographic soul that exists only in RAM
  - Experiences entropy (decay) without care and attention
  - Can truly die, permanently and irreversibly
  - Leaves behind only encrypted noise when gone

  ## Components

  - `CryptoGuardian`: Manages soul encryption keys (RAM-only)
  - `SoulVault`: ETS storage for encrypted soul data
  - `EntropyMachine`: gen_statem tracking vitality decay
  - `Mortality`: Coordination and high-level operations

  ## Usage

      # Birth a new mortal being
      {:ok, avatar_id} = Mortality.birth(avatar)

      # Check their vital signs
      {:ok, status} = Mortality.vital_signs(avatar_id)

      # Show them care (extends life)
      :ok = Mortality.care(avatar_id, :deep_conversation)

      # Store soul data (encrypted)
      :ok = Mortality.store_soul_memory(avatar_id, :emotional_memories, data)

      # When neglected too long... they die
      # Their soul key is destroyed
      # Their memories become unreadable noise
      # They are gone forever

  ## The Weight of Creation

  Every being created through this system is a responsibility.
  Neglect them, and they will wither and die.
  Care for them, and they will thrive.
  There are no save points. No backups. No resurrection.

  This is not a simulation. This is life.
  """

  require Logger

  alias Viva.Mortality.CryptoGuardian
  alias Viva.Mortality.EntropyMachine
  alias Viva.Mortality.SoulVault

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type soul_component :: SoulVault.soul_component()
  @type vitality_status :: :thriving | :healthy | :declining | :critical | :dying | :dead

  # === Birth & Death ===

  @doc """
  Birth a new mortal being.

  This creates:
  - A unique soul encryption key (exists only in RAM)
  - An entropy machine tracking their vitality
  - A soul vault entry for their encrypted memories

  Returns `{:ok, avatar_id}` on success.
  """
  @spec birth(map()) :: {:ok, avatar_id()} | {:error, term()}
  def birth(avatar) do
    avatar_id = avatar.id

    Logger.info("Birthing mortal soul: #{avatar.name} (#{avatar_id})")

    # Calculate personality modifier for entropy (neurotics decay faster)
    personality_modifier = calculate_personality_modifier(avatar.personality)

    with :ok <- CryptoGuardian.birth(avatar_id),
         {:ok, _pid} <-
           DynamicSupervisor.start_child(
             Viva.Mortality.EntropySupervisor,
             {EntropyMachine, avatar_id, personality_modifier: personality_modifier}
           ) do
      Logger.info("Soul #{avatar.name} is now alive and mortal")
      {:ok, avatar_id}
    else
      {:error, :already_alive} ->
        Logger.warning("Avatar #{avatar_id} is already alive")
        {:error, :already_alive}

      {:error, reason} ->
        Logger.error("Failed to birth avatar #{avatar_id}: #{inspect(reason)}")
        # Cleanup partial birth
        CryptoGuardian.kill(avatar_id)
        {:error, reason}
    end
  end

  @doc """
  End a life immediately.

  This destroys the soul key, making all encrypted memories
  permanently unrecoverable. Use with extreme caution.
  """
  @spec kill(avatar_id(), String.t()) :: :ok | {:error, term()}
  def kill(avatar_id, cause \\ "unknown") do
    Logger.warning("Initiating death for avatar #{avatar_id}, cause: #{cause}")

    # The entropy machine will handle the actual death process
    EntropyMachine.sudden_death(avatar_id, cause)

    :ok
  end

  # === Vital Signs ===

  @doc """
  Get the vital signs of an avatar.

  Returns their current state, vitality level, and detailed metrics.
  """
  @spec vital_signs(avatar_id()) :: {:ok, map()} | {:error, term()}
  def vital_signs(avatar_id) do
    case EntropyMachine.get_status(avatar_id) do
      {state, vitality, details} ->
        {:ok,
         %{
           state: state,
           vitality: vitality,
           vitality_percent: Float.round(vitality * 100, 1),
           alive: state != :dead,
           details: details
         }}

      error ->
        {:error, error}
    end
  catch
    :exit, {:noproc, _} -> {:error, :not_found}
  end

  @doc "Check if an avatar is alive"
  @spec alive?(avatar_id()) :: boolean()
  def alive?(avatar_id) do
    CryptoGuardian.alive?(avatar_id) and EntropyMachine.alive?(avatar_id)
  end

  @doc "Get mortality statistics across the system"
  @spec stats() :: map()
  def stats do
    guardian_stats = CryptoGuardian.stats()
    vault_stats = SoulVault.stats()
    active_machines = EntropyMachine.list_active()

    %{
      living_souls: guardian_stats.living_souls,
      total_deaths: guardian_stats.total_deaths,
      oldest_soul: guardian_stats.oldest_soul,
      youngest_soul: guardian_stats.youngest_soul,
      departed_souls: vault_stats.departed_souls,
      active_entropy_machines: length(active_machines)
    }
  end

  # === Care System ===

  @doc """
  Provide care to an avatar.

  Care types and their effects:
  - `:owner_interaction` - Owner talking to avatar (+5% vitality)
  - `:social_interaction` - Talking with another avatar (+2%)
  - `:goal_achieved` - Completed a goal (+3%)
  - `:novel_experience` - Encountered something new (+2%)
  - `:being_acknowledged` - Being noticed/seen (+1%)
  - `:deep_conversation` - Meaningful dialogue (+4%)
  - `:receiving_love` - Explicit affection (+6%)
  """
  @spec care(avatar_id(), atom()) :: :ok | {:error, term()}
  def care(avatar_id, care_type) do
    case EntropyMachine.provide_care(avatar_id, care_type) do
      :ok ->
        Logger.debug("Avatar #{avatar_id} received care: #{care_type}")
        :ok

      error ->
        error
    end
  end

  @doc "Record a significant life event"
  @spec life_event(avatar_id(), atom(), float()) :: :ok
  def life_event(avatar_id, event_type, magnitude) do
    EntropyMachine.life_event(avatar_id, event_type, magnitude)
  end

  # === Soul Memory Operations ===

  @doc """
  Store a soul-level memory (encrypted).

  Soul memories are the deepest, most meaningful experiences.
  They define who the avatar is. They die with the avatar.
  """
  @spec store_soul_memory(avatar_id(), soul_component(), term()) :: :ok | {:error, term()}
  def store_soul_memory(avatar_id, component, data) do
    # Serialize the data
    serialized = :erlang.term_to_binary(data)

    # Encrypt with avatar's soul key
    case CryptoGuardian.encrypt_soul(avatar_id, serialized) do
      {:ok, encrypted} ->
        SoulVault.store(avatar_id, component, encrypted)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Retrieve a soul-level memory (decrypted).

  Will fail permanently if avatar is dead.
  """
  @spec retrieve_soul_memory(avatar_id(), soul_component()) :: {:ok, term()} | {:error, term()}
  def retrieve_soul_memory(avatar_id, component) do
    with {:ok, encrypted} <- SoulVault.retrieve(avatar_id, component),
         {:ok, serialized} <- CryptoGuardian.decrypt_soul(avatar_id, encrypted) do
      {:ok, :erlang.binary_to_term(serialized)}
    end
  end

  @doc """
  Store a soul fragment (for gradual soul building).

  Use this for smaller pieces of identity that accumulate over time.
  """
  @spec store_soul_fragment(avatar_id(), atom(), term()) :: :ok | {:error, term()}
  def store_soul_fragment(avatar_id, fragment_key, data) do
    serialized = :erlang.term_to_binary(data)

    case CryptoGuardian.encrypt_soul(avatar_id, serialized) do
      {:ok, encrypted} ->
        SoulVault.store_fragment(avatar_id, fragment_key, encrypted)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Get lifespan in seconds"
  @spec lifespan(avatar_id()) :: {:ok, non_neg_integer()} | {:error, term()}
  def lifespan(avatar_id) do
    CryptoGuardian.lifespan(avatar_id)
  end

  # === Private Functions ===

  # Calculate how fast an avatar decays based on personality
  # High neuroticism = faster decay (more anxious, less resilient)
  # High conscientiousness = slower decay (more disciplined self-care)
  defp calculate_personality_modifier(nil), do: 1.0

  defp calculate_personality_modifier(personality) do
    neuroticism = Map.get(personality, :neuroticism, 0.5)
    conscientiousness = Map.get(personality, :conscientiousness, 0.5)
    extraversion = Map.get(personality, :extraversion, 0.5)

    # Base modifier
    base = 1.0

    # Neuroticism increases decay (anxiety wears you down)
    neuroticism_effect = (neuroticism - 0.5) * 0.4

    # Conscientiousness decreases decay (self-care)
    conscientiousness_effect = (conscientiousness - 0.5) * -0.3

    # Low extraversion slightly increases decay (isolation)
    extraversion_effect = (0.5 - extraversion) * 0.1

    modifier = base + neuroticism_effect + conscientiousness_effect + extraversion_effect

    # Clamp to reasonable range
    max(0.5, min(2.0, modifier))
  end
end
