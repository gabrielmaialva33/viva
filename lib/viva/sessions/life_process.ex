defmodule Viva.Sessions.LifeProcess do
  @moduledoc """
  GenServer that simulates an avatar's "life" continuously.
  Runs 24/7, even when the owner is offline.
  Manages needs decay, emotions, desires, and autonomous actions.
  """
  use GenServer
  require Logger

  alias Viva.Avatars.{Avatar, InternalState}
  alias Viva.Relationships
  alias Viva.Nim.LlmClient
  alias Phoenix.PubSub

  # 1 minute per tick
  @tick_interval :timer.seconds(60)
  # 1 real minute = 10 simulated minutes
  @time_scale 10

  # Persist every 5 ticks (5 minutes with 60s tick interval)
  @persist_every_n_ticks 5

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

  # === Client API ===

  def start_link(avatar_id) do
    GenServer.start_link(__MODULE__, avatar_id, name: via(avatar_id))
  end

  def get_state(avatar_id) do
    GenServer.call(via(avatar_id), :get_state)
  end

  def owner_connected(avatar_id) do
    GenServer.cast(via(avatar_id), :owner_connected)
  end

  def owner_disconnected(avatar_id) do
    GenServer.cast(via(avatar_id), :owner_disconnected)
  end

  def trigger_thought(avatar_id) do
    GenServer.cast(via(avatar_id), :trigger_thought)
  end

  def start_interaction(avatar_id, other_avatar_id) do
    GenServer.call(via(avatar_id), {:start_interaction, other_avatar_id})
  end

  def end_interaction(avatar_id) do
    GenServer.cast(via(avatar_id), :end_interaction)
  end

  # === Server Callbacks ===

  @impl true
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

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:start_interaction, other_avatar_id}, _from, state) do
    new_state = %{
      state
      | current_conversation: other_avatar_id,
        state: %{state.state | current_activity: :talking, interacting_with: other_avatar_id}
    }

    # Boost social need when interacting
    new_state = update_need(new_state, :social, 5.0)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_cast(:owner_connected, state) do
    Logger.debug("Owner connected for avatar #{state.avatar_id}")

    new_state =
      %{state | owner_online?: true}
      |> maybe_generate_greeting()

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:owner_disconnected, state) do
    {:noreply, %{state | owner_online?: false}}
  end

  @impl true
  def handle_cast(:trigger_thought, state) do
    {:noreply, generate_thought(state)}
  end

  @impl true
  def handle_cast(:end_interaction, state) do
    new_state = %{
      state
      | current_conversation: nil,
        state: %{state.state | current_activity: :idle, interacting_with: nil}
    }

    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:set_thought, thought}, state) do
    new_internal = %{state.state | current_thought: thought}
    {:noreply, %{state | state: new_internal, last_thought: thought}}
  end

  @impl true
  def handle_info(:tick, state) do
    tick_count = state.tick_count + 1

    new_state =
      state
      |> Map.put(:tick_count, tick_count)
      |> decay_needs()
      |> process_emotions()
      |> maybe_develop_desire()
      |> maybe_think()
      |> maybe_act_autonomously()
      |> update_timestamp()

    maybe_persist_state(new_state, tick_count)
    schedule_tick()

    {:noreply, new_state}
  end

  # === Private Functions ===

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

  # Decay needs over time
  defp decay_needs(process_state) do
    internal = process_state.state
    personality = process_state.avatar.personality

    # Extraverts lose social need faster
    social_decay = 0.5 + personality.extraversion * 0.5

    # High openness = need more stimulation
    stim_decay = 0.3 + personality.openness * 0.4

    new_internal = %{
      internal
      | energy: decay_value(internal.energy, 0.2),
        social: decay_value(internal.social, social_decay),
        stimulation: decay_value(internal.stimulation, stim_decay),
        comfort: decay_value(internal.comfort, 0.1)
    }

    %{process_state | state: new_internal}
  end

  defp decay_value(current, rate) do
    max(0.0, current - rate * @time_scale)
  end

  # Process and update emotions based on current state
  defp process_emotions(process_state) do
    internal = process_state.state
    emotions = internal.emotions

    # Loneliness increases when social need is low
    loneliness = calculate_loneliness(internal, process_state)

    # Joy correlates with overall wellbeing
    wellbeing = InternalState.wellbeing(internal)
    joy = emotions.joy * 0.9 + wellbeing * 0.1

    # Sadness inversely correlates with joy
    sadness = max(0, emotions.sadness * 0.95 - joy * 0.05)

    new_emotions = %{emotions | loneliness: loneliness, joy: joy, sadness: sadness}

    # Calculate new mood
    mood = calculate_mood(new_emotions)

    new_internal = %{internal | emotions: new_emotions, mood: mood}

    %{process_state | state: new_internal}
  end

  defp calculate_loneliness(internal, process_state) do
    base = 1.0 - internal.social / 100.0

    # Having close relationships reduces loneliness
    has_relationships = has_close_relationships?(process_state.avatar_id)
    relationship_factor = if has_relationships, do: -0.2, else: 0.1

    # Currently interacting reduces loneliness
    interaction_factor = if internal.interacting_with, do: -0.3, else: 0.0

    (base + relationship_factor + interaction_factor)
    |> max(0.0)
    |> min(1.0)
  end

  defp calculate_mood(emotions) do
    positive = emotions.joy + emotions.love + emotions.excitement + emotions.curiosity
    negative = emotions.sadness + emotions.anger + emotions.fear + emotions.loneliness

    ((positive - negative) / 4.0)
    |> max(-1.0)
    |> min(1.0)
  end

  defp has_close_relationships?(avatar_id) do
    Relationships.count_close_relationships(avatar_id) > 0
  end

  # Develop desires based on current needs
  defp maybe_develop_desire(process_state) do
    internal = process_state.state

    desire =
      cond do
        internal.social < 20 -> :wants_to_talk
        internal.energy < 15 -> :wants_rest
        internal.stimulation < 25 -> :wants_something_new
        internal.emotions.loneliness > 0.7 -> :wants_attention
        internal.emotions.love > 0.6 -> :wants_to_see_crush
        true -> :none
      end

    new_internal = %{internal | current_desire: desire}
    %{process_state | state: new_internal}
  end

  # Maybe generate a spontaneous thought
  defp maybe_think(process_state) do
    # 10% chance per tick, or 100% if owner is online and we have something to say
    should_think =
      :rand.uniform() < 0.1 ||
        (process_state.owner_online? && process_state.state.current_desire != :none)

    if should_think do
      generate_thought(process_state)
    else
      process_state
    end
  end

  defp generate_thought(process_state) do
    avatar_id = process_state.avatar_id

    Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
      case generate_thought_content(process_state) do
        {:ok, thought} ->
          broadcast_thought(avatar_id, thought)
          GenServer.cast(via(avatar_id), {:set_thought, thought})

        {:error, reason} ->
          Logger.warning("Failed to generate thought for avatar #{avatar_id}: #{inspect(reason)}")
      end
    end)

    process_state
  end

  defp generate_thought_content(process_state) do
    avatar = process_state.avatar
    internal = process_state.state

    prompt = """
    You are #{avatar.name}. Generate a single spontaneous thought based on your current state.

    Current mood: #{describe_mood(internal.mood)}
    Current desire: #{internal.current_desire}
    Dominant emotion: #{InternalState.dominant_emotion(internal)}
    Energy level: #{internal.energy}%

    Generate ONE brief, authentic thought (1-2 sentences max).
    It should feel natural and reflect your personality and current state.
    Don't use quotes around the thought.
    """

    LlmClient.generate(prompt, max_tokens: 100)
  end

  defp describe_mood(mood) when mood > 0.5, do: "very positive"
  defp describe_mood(mood) when mood > 0, do: "positive"
  defp describe_mood(mood) when mood > -0.5, do: "slightly negative"
  defp describe_mood(_), do: "negative"

  # Take autonomous actions when needed
  defp maybe_act_autonomously(process_state) do
    internal = process_state.state

    # Only act if not already in a conversation
    if is_nil(process_state.current_conversation) do
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
        # 30% chance to actually initiate
        if :rand.uniform() < 0.3 do
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
        if :rand.uniform() < 0.2 do
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

    prompt = """
    You are #{avatar.name}. Your owner just came online.
    Generate a brief, warm greeting that reflects your current mood and state.

    Current mood: #{describe_mood(internal.mood)}
    Time since last interaction: assume a few hours

    Keep it natural and short (1 sentence).
    """

    LlmClient.generate(prompt, max_tokens: 50)
  end

  defp update_need(process_state, need, delta) do
    internal = process_state.state
    current = Map.get(internal, need)
    new_value = min(100.0, max(0.0, current + delta))
    new_internal = Map.put(internal, need, new_value)
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

  defp broadcast_to_owner(avatar_id, message) do
    PubSub.broadcast(Viva.PubSub, "avatar:#{avatar_id}:owner", message)
  end
end
