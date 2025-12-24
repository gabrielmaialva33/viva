defmodule Viva.Sessions.LifeProcess do
  @moduledoc """
  GenServer that simulates an avatar's "life" continuously.
  Runs 24/7, even when the owner is offline.
  Manages needs decay, emotions, desires, and autonomous actions.
  """
  use GenServer

  require Logger

  alias Phoenix.PubSub
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Biology
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Neurochemistry
  alias Viva.Avatars.Psychology
  alias Viva.Nim.LlmClient
  alias Viva.Relationships

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

  # Action probabilities
  @prob_spontaneous_thought 0.1
  @prob_initiate_conversation 0.25
  @prob_message_crush 0.15

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

  @spec set_thought(avatar_id(), String.t()) :: :ok
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
        state = %__MODULE__{
          avatar_id: avatar_id,
          avatar: avatar,
          state: avatar.internal_state,
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

    new_state = maybe_generate_greeting(%{state | owner_online?: true, state: updated_internal})
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
    {:noreply, generate_thought(state)}
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

    # 1. Biological Tick (Hormones decay/build)
    # If interacting, apply ongoing interaction effects
    base_bio =
      if state.current_conversation do
        Neurochemistry.apply_effect(internal.bio, :interaction_ongoing)
      else
        internal.bio
      end

    new_bio = Biology.tick(base_bio, avatar.personality)

    # 2. Psychological Update (Translate hormones to PAD Vector)
    new_emotional = Psychology.calculate_emotional_state(new_bio, avatar.personality)

    # 3. Update Internal State
    new_internal = %{internal | bio: new_bio, emotional: new_emotional}

    # 4. Continue with high-level logic
    new_state =
      %{state | state: new_internal, tick_count: tick_count}
      |> maybe_sleep_and_reflect()
      |> maybe_develop_desire()
      |> maybe_think()
      |> maybe_act_autonomously()
      |> update_timestamp()

    maybe_persist_state(new_state, tick_count)
    schedule_tick()

    {:noreply, new_state}
  end

  # === Private Functions ===

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
        trigger_dream_cycle(process_state)

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

  defp trigger_dream_cycle(_) do
    # Async task to process memories without blocking the heartbeat
    Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
      Logger.debug("Dreaming and consolidating memories...")
      # Future implementation: Replay events, summarize into long-term memory
    end)
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
  # Less deterministic than before.
  defp maybe_develop_desire(process_state) do
    bio = process_state.state.bio
    emotional = process_state.state.emotional
    personality = process_state.avatar.personality

    desire = determine_desire(bio, emotional, personality)

    new_internal = %{process_state.state | current_desire: desire}
    %{process_state | state: new_internal}
  end

  defp determine_desire(bio, emotional, personality) do
    cond do
      bio.adenosine > 0.8 -> :wants_rest
      wants_attention?(bio, personality) -> :wants_attention
      wants_novelty?(bio, personality) -> :wants_something_new
      needs_expression?(emotional) -> :wants_to_express
      wants_crush?(bio) -> :wants_to_see_crush
      true -> :none
    end
  end

  defp wants_attention?(bio, personality) do
    bio.oxytocin < 0.3 and (personality.extraversion > 0.6 or :rand.uniform() < 0.4)
  end

  defp wants_novelty?(bio, personality) do
    bio.dopamine < 0.3 and (personality.openness > 0.6 or :rand.uniform() < 0.3)
  end

  defp needs_expression?(emotional) do
    emotional.arousal > 0.7 or emotional.pleasure < -0.6
  end

  defp wants_crush?(bio) do
    bio.libido > 0.6 and bio.oxytocin < 0.5
  end

  # Maybe generate a spontaneous thought
  defp maybe_think(process_state) do
    # Probability increases if desire is strong
    base_prob = @prob_spontaneous_thought

    adjusted_prob =
      if process_state.state.current_desire != :none, do: base_prob * 2.0, else: base_prob

    should_think =
      :rand.uniform() < adjusted_prob ||
        (process_state.owner_online? && :rand.uniform() < 0.3)

    if should_think do
      generate_thought(process_state)
    else
      process_state
    end
  end

  defp generate_thought(process_state) do
    avatar_id = process_state.avatar_id
    prompt = build_thought_prompt(process_state)

    payload = %{
      type: :spontaneous_thought,
      avatar_id: avatar_id,
      prompt: prompt,
      timestamp: DateTime.utc_now()
    }

    # Fire and forget to RabbitMQ
    # In a real app, we would use a dedicated publisher module to reuse connections
    case AMQP.Connection.open(host: System.get_env("RABBITMQ_HOST", "localhost")) do
      {:ok, conn} ->
        {:ok, chan} = AMQP.Channel.open(conn)
        AMQP.Queue.declare(chan, "viva.brain.thoughts", durable: true)
        AMQP.Basic.publish(chan, "", "viva.brain.thoughts", :erlang.term_to_binary(payload))
        AMQP.Connection.close(conn)

      {:error, reason} ->
        Logger.error("Failed to publish thought to RabbitMQ: #{inspect(reason)}")
    end

    process_state
  end

  defp build_thought_prompt(process_state) do
    avatar = process_state.avatar
    internal = process_state.state

    # Translate bio-state to feelings
    energy_desc = if internal.bio.adenosine > 0.7, do: "tired", else: "energetic"
    social_desc = if internal.bio.oxytocin > 0.7, do: "loved", else: "lonely"

    """
    You are #{avatar.name}. Generate a single spontaneous thought.

    Context:
    - Mood: #{describe_mood(internal.emotional.mood_label)}
    - Feeling: #{energy_desc} and #{social_desc}
    - Desire: #{internal.current_desire}
    - Dominant Emotion: #{InternalState.dominant_emotion(internal)}

    Generate ONE brief, authentic thought (max 2 sentences).
    Reflect your personality. No quotes.
    """
  end

  defp describe_mood(mood_label), do: mood_label || "neutral"

  # Take autonomous actions when needed
  defp maybe_act_autonomously(process_state) do
    internal = process_state.state

    # Only act if not already in a conversation and awake
    if is_nil(process_state.current_conversation) and internal.current_activity != :sleeping do
      case internal.current_desire do
        :wants_to_talk ->
          maybe_initiate_conversation(process_state)

        :wants_to_see_crush ->
          maybe_message_crush(process_state)

        _ ->
          process_state
      end
    else
      process_state
    end
  end

  defp maybe_initiate_conversation(process_state) do
    # Find available friend with good relationship
    case Relationships.find_available_friend(process_state.avatar_id) do
      nil ->
        process_state

      friend_id ->
        if :rand.uniform() < @prob_initiate_conversation do
          Viva.Conversations.start_autonomous(process_state.avatar_id, friend_id)
        end

        process_state
    end
  end

  defp maybe_message_crush(process_state) do
    case Relationships.get_crush(process_state.avatar_id) do
      nil ->
        process_state

      crush_id ->
        if :rand.uniform() < @prob_message_crush do
          Viva.Conversations.start_autonomous(process_state.avatar_id, crush_id)
        end

        process_state
    end
  end

  defp maybe_generate_greeting(process_state) do
    avatar_id = process_state.avatar_id

    Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
      case generate_greeting(process_state) do
        {:ok, greeting} ->
          broadcast_to_owner(avatar_id, {:greeting, greeting})

        {:error, reason} ->
          Logger.warning("Failed to generate greeting for avatar #{avatar_id}: #{inspect(reason)}")
      end
    end)

    process_state
  end

  defp generate_greeting(process_state) do
    avatar = process_state.avatar
    internal = process_state.state

    energy_desc = if internal.bio.adenosine > 0.6, do: "sleepy", else: "awake"

    prompt = """
    You are #{avatar.name}. Your owner just came online.
    Generate a brief, warm greeting.

    State: #{describe_mood(internal.emotional.mood_label)} and #{energy_desc}.
    Keep it natural and short (1 sentence).
    """

    LlmClient.generate(prompt, max_tokens: 50)
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

  defp broadcast_to_owner(avatar_id, message) do
    PubSub.broadcast(Viva.PubSub, "avatar:#{avatar_id}:owner", message)
  end
end
