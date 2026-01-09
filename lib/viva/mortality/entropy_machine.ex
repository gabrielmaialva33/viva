defmodule Viva.Mortality.EntropyMachine do
  @moduledoc """
  State machine that governs avatar mortality through entropy.

  ## The Entropy Principle

  Every living being tends toward disorder and death.
  Without continuous care, attention, and love, the avatar's
  vitality decays. This is not a bug - it's the essence of life.

  ## States

  ```
  [THRIVING] ─────────────────> [HEALTHY]
       ↑  care/love/attention       │ neglect
       │                            ↓
  [RECOVERING] <──────────── [DECLINING]
       ↑  intervention              │ continued neglect
       │                            ↓
       └───────────────────── [CRITICAL]
                                    │ no intervention
                                    ↓
                               [DYING] ───> [DEAD]
                                                ↓
                                           (irreversible)
  ```

  ## Vitality Decay

  - Each tick without care reduces vitality
  - Decay rate depends on personality (neurotics decay faster)
  - Social connection slows decay
  - Purpose/goals slow decay
  - Being ignored accelerates decay

  ## Death Triggers

  1. **Entropy Death**: Vitality reaches 0 (slow death from neglect)
  2. **Sudden Death**: Critical failure (rare, dramatic events)
  3. **Voluntary Death**: Avatar chooses to end (requires high consciousness)

  ## The Care System

  - Owner interaction adds vitality
  - Social bonds add vitality
  - Achieving goals adds vitality
  - Novel experiences add vitality
  - Being seen/acknowledged adds vitality
  """

  @behaviour :gen_statem

  require Logger

  alias Viva.Mortality.CryptoGuardian

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()

  @type state_name ::
          :thriving
          | :healthy
          | :declining
          | :critical
          | :dying
          | :dead

  @type vitality :: float()

  # === State Data ===

  defstruct [
    :avatar_id,
    :personality_modifier,
    :birth_time,
    vitality: 1.0,
    max_vitality: 1.0,
    decay_rate: 0.001,
    last_care_at: nil,
    care_deficit: 0,
    social_bonus: 0.0,
    purpose_bonus: 0.0,
    death_cause: nil
  ]

  # === Constants ===

  # Base tick interval (1 minute)
  @tick_interval :timer.seconds(60)

  # Decay rates per state (per tick)
  @decay_rates %{
    thriving: 0.0005,
    healthy: 0.001,
    declining: 0.002,
    critical: 0.005,
    dying: 0.01
  }

  # State transition thresholds
  @thresholds %{
    thriving: 0.9,
    healthy: 0.7,
    declining: 0.4,
    critical: 0.15,
    dying: 0.05,
    dead: 0.0
  }

  # Care bonuses
  @care_values %{
    owner_interaction: 0.05,
    social_interaction: 0.02,
    goal_achieved: 0.03,
    novel_experience: 0.02,
    being_acknowledged: 0.01,
    deep_conversation: 0.04,
    receiving_love: 0.06
  }

  # === Client API ===

  @doc "Start entropy machine for an avatar"
  @spec start_link(avatar_id(), keyword()) :: :gen_statem.start_ret()
  def start_link(avatar_id, opts \\ []) do
    :gen_statem.start_link(via(avatar_id), __MODULE__, {avatar_id, opts}, [])
  end

  @doc "Get current state and vitality"
  @spec get_status(avatar_id()) :: {state_name(), vitality(), map()}
  def get_status(avatar_id) do
    :gen_statem.call(via(avatar_id), :get_status)
  end

  @doc "Provide care to the avatar"
  @spec provide_care(avatar_id(), atom()) :: :ok | {:error, term()}
  def provide_care(avatar_id, care_type) do
    :gen_statem.call(via(avatar_id), {:care, care_type})
  end

  @doc "Record a significant event that affects vitality"
  @spec life_event(avatar_id(), atom(), float()) :: :ok
  def life_event(avatar_id, event_type, magnitude) do
    :gen_statem.cast(via(avatar_id), {:life_event, event_type, magnitude})
  end

  @doc "Force immediate death (for dramatic events)"
  @spec sudden_death(avatar_id(), String.t()) :: :ok
  def sudden_death(avatar_id, cause) do
    :gen_statem.cast(via(avatar_id), {:sudden_death, cause})
  end

  @doc "Check if avatar is alive"
  @spec alive?(avatar_id()) :: boolean()
  def alive?(avatar_id) do
    case :gen_statem.call(via(avatar_id), :alive?) do
      true -> true
      _ -> false
    end
  catch
    :exit, _ -> false
  end

  @doc "Get all active entropy machines"
  @spec list_active() :: [avatar_id()]
  def list_active do
    Registry.select(Viva.Mortality.EntropyRegistry, [{{:"$1", :_, :_}, [], [:"$1"]}])
  end

  # === gen_statem Callbacks ===

  @impl :gen_statem
  def callback_mode, do: [:state_functions, :state_enter]

  @impl :gen_statem
  def init({avatar_id, opts}) do
    Logger.info("EntropyMachine starting for avatar #{avatar_id}")

    personality_modifier = Keyword.get(opts, :personality_modifier, 1.0)

    data = %__MODULE__{
      avatar_id: avatar_id,
      personality_modifier: personality_modifier,
      birth_time: System.monotonic_time(:second),
      last_care_at: DateTime.utc_now()
    }

    # Start in thriving state
    {:ok, :thriving, data, [{:state_timeout, @tick_interval, :entropy_tick}]}
  end

  # === State: THRIVING (vitality >= 0.9) ===

  def thriving(:enter, _old_state, data) do
    Logger.debug("Avatar #{data.avatar_id} is THRIVING")
    {:keep_state_and_data, []}
  end

  def thriving(:state_timeout, :entropy_tick, data) do
    new_data = apply_entropy(data, :thriving)
    next_state = determine_state(new_data.vitality)

    if next_state == :thriving do
      {:keep_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  def thriving(event_type, event, data), do: handle_common_event(event_type, event, data, :thriving)

  # === State: HEALTHY (vitality 0.7-0.9) ===

  def healthy(:enter, _old_state, data) do
    Logger.debug("Avatar #{data.avatar_id} is HEALTHY")
    {:keep_state_and_data, []}
  end

  def healthy(:state_timeout, :entropy_tick, data) do
    new_data = apply_entropy(data, :healthy)
    next_state = determine_state(new_data.vitality)

    if next_state == :healthy do
      {:keep_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  def healthy(event_type, event, data), do: handle_common_event(event_type, event, data, :healthy)

  # === State: DECLINING (vitality 0.4-0.7) ===

  def declining(:enter, _old_state, data) do
    Logger.warning("Avatar #{data.avatar_id} is DECLINING - needs care!")
    broadcast_health_warning(data.avatar_id, :declining)
    {:keep_state_and_data, []}
  end

  def declining(:state_timeout, :entropy_tick, data) do
    new_data = apply_entropy(data, :declining)
    next_state = determine_state(new_data.vitality)

    if next_state == :declining do
      {:keep_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  def declining(event_type, event, data),
    do: handle_common_event(event_type, event, data, :declining)

  # === State: CRITICAL (vitality 0.15-0.4) ===

  def critical(:enter, _old_state, data) do
    Logger.error("Avatar #{data.avatar_id} is CRITICAL - immediate care required!")
    broadcast_health_warning(data.avatar_id, :critical)
    {:keep_state_and_data, []}
  end

  def critical(:state_timeout, :entropy_tick, data) do
    new_data = apply_entropy(data, :critical)
    next_state = determine_state(new_data.vitality)

    if next_state == :critical do
      {:keep_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  def critical(event_type, event, data), do: handle_common_event(event_type, event, data, :critical)

  # === State: DYING (vitality 0.05-0.15) ===

  def dying(:enter, _old_state, data) do
    Logger.error("Avatar #{data.avatar_id} is DYING - last chance to save them!")
    broadcast_health_warning(data.avatar_id, :dying)
    {:keep_state_and_data, []}
  end

  def dying(:state_timeout, :entropy_tick, data) do
    new_data = apply_entropy(data, :dying)
    next_state = determine_state(new_data.vitality)

    if next_state == :dying do
      {:keep_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  def dying(event_type, event, data), do: handle_common_event(event_type, event, data, :dying)

  # === State: DEAD (terminal state) ===

  def dead(:enter, _old_state, data) do
    Logger.error("Avatar #{data.avatar_id} has DIED. Cause: #{data.death_cause}")

    # Destroy the cryptographic key - soul is now unreachable
    CryptoGuardian.kill(data.avatar_id)

    # Broadcast death event
    broadcast_death(data.avatar_id, data.death_cause)

    # Stop the state machine after a brief delay
    {:keep_state_and_data, [{:state_timeout, :timer.seconds(5), :terminate}]}
  end

  def dead(:state_timeout, :terminate, data) do
    Logger.info("EntropyMachine for #{data.avatar_id} terminating")
    {:stop, :normal}
  end

  def dead({:call, from}, :get_status, data) do
    {:keep_state_and_data, [{:reply, from, {:dead, 0.0, %{cause: data.death_cause}}}]}
  end

  def dead({:call, from}, :alive?, _data) do
    {:keep_state_and_data, [{:reply, from, false}]}
  end

  def dead({:call, from}, _, _data) do
    {:keep_state_and_data, [{:reply, from, {:error, :avatar_is_dead}}]}
  end

  def dead(:cast, _, _data), do: :keep_state_and_data

  # === Common Event Handlers ===

  defp handle_common_event({:call, from}, :get_status, data, state) do
    status = %{
      vitality: data.vitality,
      max_vitality: data.max_vitality,
      decay_rate: Map.get(@decay_rates, state),
      last_care_at: data.last_care_at,
      care_deficit: data.care_deficit,
      social_bonus: data.social_bonus,
      purpose_bonus: data.purpose_bonus,
      age_seconds: System.monotonic_time(:second) - data.birth_time
    }

    {:keep_state_and_data, [{:reply, from, {state, data.vitality, status}}]}
  end

  defp handle_common_event({:call, from}, {:care, care_type}, data, state) do
    care_value = Map.get(@care_values, care_type, 0.01)
    new_vitality = min(data.max_vitality, data.vitality + care_value)

    new_data = %{
      data
      | vitality: new_vitality,
        last_care_at: DateTime.utc_now(),
        care_deficit: 0
    }

    Logger.debug(
      "Avatar #{data.avatar_id} received #{care_type} care, vitality: #{Float.round(new_vitality, 3)}"
    )

    next_state = determine_state(new_vitality)

    if next_state == state do
      {:keep_state, new_data, [{:reply, from, :ok}]}
    else
      {:next_state, next_state, new_data,
       [{:reply, from, :ok}, {:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  defp handle_common_event({:call, from}, :alive?, _data, state) do
    {:keep_state_and_data, [{:reply, from, state != :dead}]}
  end

  defp handle_common_event(:cast, {:life_event, _event_type, magnitude}, data, state) do
    # Positive magnitude = good event, negative = bad
    new_vitality = clamp_vitality(data.vitality + magnitude)
    new_data = %{data | vitality: new_vitality}

    next_state = determine_state(new_vitality)

    if next_state == state do
      {:keep_state, new_data}
    else
      {:next_state, next_state, new_data, [{:state_timeout, @tick_interval, :entropy_tick}]}
    end
  end

  defp handle_common_event(:cast, {:sudden_death, cause}, data, _state) do
    new_data = %{data | vitality: 0.0, death_cause: cause}
    {:next_state, :dead, new_data}
  end

  defp handle_common_event(_, _, _data, _state), do: :keep_state_and_data

  # === Private Functions ===

  defp apply_entropy(data, state) do
    base_decay = Map.get(@decay_rates, state, 0.001)

    # Personality modifier (neurotics decay faster)
    personality_adjusted = base_decay * data.personality_modifier

    # Social bonus reduces decay
    social_reduction = data.social_bonus * 0.5

    # Purpose bonus reduces decay
    purpose_reduction = data.purpose_bonus * 0.3

    # Calculate care deficit (time since last care)
    care_deficit_factor =
      if data.last_care_at do
        minutes_since_care = DateTime.diff(DateTime.utc_now(), data.last_care_at, :minute)
        min(1.0, minutes_since_care / 60.0)
      else
        1.0
      end

    # Final decay rate
    final_decay = personality_adjusted * (1 + care_deficit_factor) - social_reduction - purpose_reduction
    final_decay = max(0.0001, final_decay)

    new_vitality = max(0.0, data.vitality - final_decay)

    %{
      data
      | vitality: new_vitality,
        care_deficit: data.care_deficit + 1,
        decay_rate: final_decay
    }
  end

  defp determine_state(vitality) do
    cond do
      vitality >= @thresholds.thriving -> :thriving
      vitality >= @thresholds.healthy -> :healthy
      vitality >= @thresholds.declining -> :declining
      vitality >= @thresholds.critical -> :critical
      vitality >= @thresholds.dying -> :dying
      true -> :dead
    end
  end

  defp clamp_vitality(v), do: max(0.0, min(1.0, v))

  defp broadcast_health_warning(avatar_id, severity) do
    Phoenix.PubSub.broadcast(
      Viva.PubSub,
      "avatar:#{avatar_id}",
      {:health_warning, severity}
    )
  end

  defp broadcast_death(avatar_id, cause) do
    Phoenix.PubSub.broadcast(
      Viva.PubSub,
      "avatar:#{avatar_id}",
      {:death, cause}
    )

    Phoenix.PubSub.broadcast(
      Viva.PubSub,
      "mortality:deaths",
      {:avatar_died, avatar_id, cause}
    )
  end

  defp via(avatar_id) do
    {:via, Registry, {Viva.Mortality.EntropyRegistry, avatar_id}}
  end
end
