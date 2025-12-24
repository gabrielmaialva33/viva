defmodule Viva.Sessions.LifeProcess do
  @moduledoc """
  GenServer that simulates an avatar's "life" continuously.
  Runs 24/7, even when the owner is offline.
  Manages needs decay, emotions, desires, and autonomous actions.
  """
  @behaviour Viva.Sessions.LifeProcessBehaviour

  use GenServer

  require Logger

  alias Phoenix.PubSub
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Systems.Allostasis
  alias Viva.Avatars.Systems.AttachmentBias
  alias Viva.Avatars.Systems.Biology
  alias Viva.Avatars.Systems.Consciousness
  alias Viva.Avatars.Systems.EmotionRegulation
  alias Viva.Avatars.Systems.Metacognition
  alias Viva.Avatars.Systems.Neurochemistry
  alias Viva.Avatars.Systems.Psychology
  alias Viva.Avatars.Systems.Senses
  alias Viva.Avatars.Systems.SomaticMarkers
  alias Viva.Sessions.AutonomousActions
  alias Viva.Sessions.DesireEngine
  alias Viva.Sessions.DreamProcessor
  alias Viva.Sessions.StimulusGathering
  alias Viva.Sessions.ThoughtEngine

  # === Struct ===

  defstruct [
    :avatar_id,
    :avatar,
    :state,
    :last_tick_at,
    :owner_online?,
    :current_conversation,
    :last_thought,
    tick_count: 0
  ]

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type process_state :: %__MODULE__{}

  # === Constants ===

  # 1 minute per tick
  @tick_interval :timer.seconds(60)
  # 1 real minute = 10 simulated minutes

  # Persist every 5 ticks (5 minutes with 60s tick interval)
  @persist_every_n_ticks 5

  # === Client API ===

  @spec start_link(avatar_id()) :: GenServer.on_start()
  def start_link(avatar_id) do
    GenServer.start_link(__MODULE__, avatar_id, name: via(avatar_id))
  end

  @spec get_state(avatar_id()) :: process_state()
  def get_state(avatar_id) do
    avatar_id
    |> via()
    |> GenServer.call(:get_state)
  end

  @spec owner_connected(avatar_id()) :: :ok
  def owner_connected(avatar_id) do
    avatar_id
    |> via()
    |> GenServer.cast(:owner_connected)
  end

  @spec owner_disconnected(avatar_id()) :: :ok
  def owner_disconnected(avatar_id) do
    avatar_id
    |> via()
    |> GenServer.cast(:owner_disconnected)
  end

  @spec trigger_thought(avatar_id()) :: :ok
  def trigger_thought(avatar_id) do
    avatar_id
    |> via()
    |> GenServer.cast(:trigger_thought)
  end

  @spec start_interaction(avatar_id(), avatar_id()) :: :ok
  def start_interaction(avatar_id, other_avatar_id) do
    avatar_id
    |> via()
    |> GenServer.call({:start_interaction, other_avatar_id})
  end

  @spec end_interaction(avatar_id()) :: :ok
  def end_interaction(avatar_id) do
    avatar_id
    |> via()
    |> GenServer.cast(:end_interaction)
  end

  @impl Viva.Sessions.LifeProcessBehaviour
  def set_thought(avatar_id, thought) do
    avatar_id
    |> via()
    |> GenServer.cast({:set_thought, thought})
  end

  # === Server Callbacks ===

  @impl GenServer
  def init(avatar_id) do
    Logger.info("Starting LifeProcess for avatar #{avatar_id}")

    case load_avatar(avatar_id) do
      {:ok, avatar} ->
        # Ensure all internal state components are initialized
        safe_internal_state =
          InternalState.ensure_integrity(avatar.internal_state, avatar.personality)

        state = %__MODULE__{
          avatar_id: avatar_id,
          avatar: %{avatar | internal_state: safe_internal_state},
          state: safe_internal_state,
          last_tick_at: DateTime.utc_now(),
          owner_online?: false,
          current_conversation: nil
        }

        schedule_tick()
        broadcast_status(state, :alive)

        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl GenServer
  def handle_call(:get_state, _, state) do
    {:reply, state, state}
  end

  @impl GenServer
  def handle_call({:start_interaction, other_avatar_id}, _, state) do
    # Trigger neurochemical burst for interaction
    new_bio = Neurochemistry.apply_effect(state.state.bio, :interaction_start)

    updated_internal = %{
      state.state
      | current_activity: :talking,
        interacting_with: other_avatar_id,
        bio: new_bio
    }

    new_state = %{state | current_conversation: other_avatar_id, state: updated_internal}
    {:reply, :ok, new_state}
  end

  @impl GenServer
  def handle_cast(:owner_connected, state) do
    Logger.debug("Owner connected for avatar #{state.avatar_id}")

    # Connect triggers a small dopamine spike (anticipation)
    new_bio = Neurochemistry.apply_effect(state.state.bio, :interaction_start)
    updated_internal = %{state.state | bio: new_bio}

    new_state =
      AutonomousActions.maybe_generate_greeting(%{
        state
        | owner_online?: true,
          state: updated_internal
      })

    {:noreply, new_state}
  end

  @impl GenServer
  def handle_cast(:owner_disconnected, state) do
    # Disconnect might feel like a "rejection" or just "interaction end" depending on context
    # For now, let's treat it as end of interaction
    new_bio = Neurochemistry.apply_effect(state.state.bio, :interaction_end)
    updated_internal = %{state.state | bio: new_bio}

    {:noreply, %{state | owner_online?: false, state: updated_internal}}
  end

  @impl GenServer
  def handle_cast(:trigger_thought, state) do
    {:noreply, ThoughtEngine.generate_thought(state)}
  end

  @impl GenServer
  def handle_cast(:end_interaction, state) do
    new_bio = Neurochemistry.apply_effect(state.state.bio, :interaction_end)

    updated_internal = %{
      state.state
      | current_activity: :idle,
        interacting_with: nil,
        bio: new_bio
    }

    new_state = %{state | current_conversation: nil, state: updated_internal}
    {:noreply, new_state}
  end

  @impl GenServer
  def handle_cast({:set_thought, thought}, state) do
    # Thinking consumes energy but provides dopamine
    new_bio = Neurochemistry.apply_effect(state.state.bio, :thought_generated)

    new_internal = %{
      state.state
      | current_thought: thought,
        bio: new_bio
    }

    broadcast_thought(state.avatar_id, thought)
    {:noreply, %{state | state: new_internal, last_thought: thought}}
  end

  @impl GenServer
  def handle_info(:tick, state) do
    tick_count = state.tick_count + 1
    avatar = state.avatar
    internal = state.state

    # 1. Gather environmental stimulus
    raw_stimulus = StimulusGathering.gather(state)

    # 2. ATTACHMENT BIAS: Filter stimulus through attachment style lens
    {stimulus, _interpretation} = AttachmentBias.interpret(raw_stimulus, avatar.personality)

    # 3. SENSES: Process perception through subjective filter
    {new_sensory, neuro_effects} =
      Senses.perceive(internal.sensory, stimulus, avatar.personality, internal.emotional)

    # 3.5. SOMATIC MARKERS: Recall body memories for this stimulus
    {updated_somatic, _somatic_bias} = SomaticMarkers.recall(internal.somatic, stimulus)

    # 4. Apply neurochemical effects from surprise/perception
    bio_with_perception = apply_neuro_effects(internal.bio, neuro_effects)

    # 5. Biological Tick (Hormones decay/build)
    # If interacting, apply ongoing interaction effects
    base_bio =
      if state.current_conversation do
        Neurochemistry.apply_effect(bio_with_perception, :interaction_ongoing)
      else
        bio_with_perception
      end

    new_bio = Biology.tick(base_bio, avatar.personality)

    # 6. ALLOSTASIS: Track chronic stress effects
    new_allostasis = Allostasis.tick(internal.allostasis, new_bio)

    # 7. Psychological Update (Translate hormones to PAD Vector)
    raw_emotional = Psychology.calculate_emotional_state(new_bio, avatar.personality)

    # 8. Apply allostatic dampening to emotions (chronic stress blunts emotional response)
    dampened_emotional = Allostasis.dampen_emotions(raw_emotional, new_allostasis)

    # 9. EMOTION REGULATION: Apply personality-based coping strategies
    {new_regulation, new_emotional, regulated_bio} =
      EmotionRegulation.regulate(
        internal.regulation,
        dampened_emotional,
        new_bio,
        avatar.personality
      )

    # 10. CONSCIOUSNESS: Integrate into unified experience
    new_consciousness =
      Consciousness.integrate(
        internal.consciousness,
        new_sensory,
        regulated_bio,
        new_emotional,
        internal.current_thought,
        avatar.personality
      )

    # 10.5. METACOGNITION: Process self-reflection and pattern detection
    {metacog_consciousness, _metacog_result} =
      Metacognition.process(
        new_consciousness,
        new_emotional,
        avatar.personality,
        tick_count
      )

    # 11. SOMATIC MARKERS: Maybe learn from intense experiences
    final_somatic =
      SomaticMarkers.maybe_learn(updated_somatic, stimulus, regulated_bio, new_emotional)

    # 12. Update Internal State with all new components
    new_internal = %{
      internal
      | bio: regulated_bio,
        emotional: new_emotional,
        sensory: new_sensory,
        consciousness: metacog_consciousness,
        allostasis: new_allostasis,
        regulation: new_regulation,
        somatic: final_somatic
    }

    # 13. Continue with high-level logic (now informed by experience)
    new_state =
      %{state | state: new_internal, tick_count: tick_count}
      |> maybe_sleep_and_reflect()
      |> maybe_develop_desire()
      |> ThoughtEngine.maybe_think()
      |> AutonomousActions.maybe_act()
      |> update_timestamp()

    maybe_persist_state(new_state, tick_count)
    schedule_tick()

    {:noreply, new_state}
  end

  # === Private Functions ===

  # Apply neurochemical effects from perception (surprise, hedonic responses)
  defp apply_neuro_effects(bio, effects) when is_list(effects) do
    Enum.reduce(effects, bio, fn effect, acc_bio ->
      Neurochemistry.apply_effect(acc_bio, effect)
    end)
  end

  defp apply_neuro_effects(bio, _), do: bio

  defp maybe_sleep_and_reflect(process_state) do
    internal = process_state.state

    # If exhausted or it's sleep time, go to sleep
    should_sleep = internal.bio.adenosine > 0.85 or sleep_time?(internal.bio)

    cond do
      internal.current_activity == :sleeping and internal.bio.adenosine > 0.1 ->
        # Continue sleeping, recover heavily
        new_bio = Neurochemistry.apply_effect(internal.bio, :deep_sleep_tick)
        new_internal = %{internal | bio: new_bio}
        %{process_state | state: new_internal}

      should_sleep and internal.current_activity != :sleeping ->
        # Fall asleep and trigger reflection
        Logger.info("Avatar #{process_state.avatar.name} is falling asleep. Triggering reflection.")
        DreamProcessor.trigger_dream_cycle(process_state)

        new_internal = %{internal | current_activity: :sleeping}
        %{process_state | state: new_internal}

      internal.current_activity == :sleeping and internal.bio.adenosine <= 0.1 ->
        # Wake up!
        Logger.info("Avatar #{process_state.avatar.name} is waking up refreshed.")
        new_internal = %{internal | current_activity: :idle}
        %{process_state | state: new_internal}

      true ->
        process_state
    end
  end

  defp sleep_time?(_) do
    # Simple check against configured sleep hour (mocked for now)
    false
  end

  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end

  defp via(avatar_id) do
    {:via, Registry, {Viva.Sessions.AvatarRegistry, avatar_id}}
  end

  defp load_avatar(avatar_id) do
    case Viva.Repo.get(Avatar, avatar_id) do
      nil -> {:error, :not_found}
      avatar -> {:ok, Viva.Repo.preload(avatar, [:memories])}
    end
  end

  # Develop desires based on fuzzy logic (Neurochemistry + Personality)
  # Delegates to DesireEngine module.
  defp maybe_develop_desire(process_state) do
    bio = process_state.state.bio
    emotional = process_state.state.emotional
    personality = process_state.avatar.personality
    somatic = process_state.state.somatic

    # Create somatic bias map for DesireEngine
    somatic_bias = %{
      bias: somatic.current_bias,
      signal: somatic.body_signal,
      markers_activated: if(somatic.last_marker_activation, do: 1, else: 0)
    }

    desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

    new_internal = %{process_state.state | current_desire: desire}
    %{process_state | state: new_internal}
  end

  defp update_timestamp(process_state) do
    new_internal = %{process_state.state | updated_at: DateTime.utc_now()}
    %{process_state | state: new_internal, last_tick_at: DateTime.utc_now()}
  end

  defp maybe_persist_state(process_state, tick_count) do
    if rem(tick_count, @persist_every_n_ticks) == 0 do
      Logger.debug("Persisting state for avatar #{process_state.avatar_id} at tick #{tick_count}")

      case Viva.Avatars.update_internal_state(process_state.avatar_id, process_state.state) do
        {:ok, _} ->
          :ok

        {:error, reason} ->
          Logger.error(
            "Failed to persist state for avatar #{process_state.avatar_id}: #{inspect(reason)}"
          )
      end
    end
  end

  defp broadcast_status(process_state, status) do
    PubSub.broadcast(Viva.PubSub, "avatar:#{process_state.avatar_id}", {:status, status})
  end

  defp broadcast_thought(avatar_id, thought) do
    PubSub.broadcast(Viva.PubSub, "avatar:#{avatar_id}", {:thought, thought})
  end
end
