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
  @prob_initiate_conversation 0.3
  @prob_message_crush 0.2

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
    # Update state and boost social need when interacting
    updated_state = %{
      state
      | current_conversation: other_avatar_id,
        state: %{state.state | current_activity: :talking, interacting_with: other_avatar_id}
    }

    new_state = update_need(updated_state, :social, 5.0)

    {:reply, :ok, new_state}
  end

  @impl GenServer
  def handle_cast(:owner_connected, state) do
    Logger.debug("Owner connected for avatar #{state.avatar_id}")

    new_state =
      state
      |> Map.put(:owner_online?, true)
      |> maybe_generate_greeting()

    {:noreply, new_state}
  end

  @impl GenServer
  def handle_cast(:owner_disconnected, state) do
    {:noreply, %{state | owner_online?: false}}
  end

  @impl GenServer
  def handle_cast(:trigger_thought, state) do
    {:noreply, generate_thought(state)}
  end

  @impl GenServer
  def handle_cast(:end_interaction, state) do
    new_state = %{
      state
      | current_conversation: nil,
        state: %{state.state | current_activity: :idle, interacting_with: nil}
    }

    {:noreply, new_state}
  end

  @impl GenServer
  def handle_cast({:set_thought, thought}, state) do
    new_internal = %{state.state | current_thought: thought}
    {:noreply, %{state | state: new_internal, last_thought: thought}}
  end

  @impl GenServer
  def handle_info(:tick, state) do
    tick_count = state.tick_count + 1
    avatar = state.avatar
    internal = state.state

    # 1. Biological Tick (Hormones decay/build)
    new_bio = Biology.tick(internal.bio, avatar.personality)

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
    should_sleep = internal.bio.adenosine > 0.9 or sleep_time?(internal.bio)

    cond do
      internal.current_activity == :sleeping and internal.bio.adenosine > 0.1 ->
        # Continue sleeping, recover energy
        new_bio = %{internal.bio | adenosine: max(0.0, internal.bio.adenosine - 0.05)}
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
      # 1. Fetch recent memories
      # 2. Call ReasoningClient.reflect_on_memories
      # 3. Save new insights as high-importance memories
      Logger.debug("Dreaming...")
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

  # Develop desires based on current needs
  defp maybe_develop_desire(process_state) do
    bio = process_state.state.bio
    emotional = process_state.state.emotional

    desire =
      cond do
        bio.oxytocin < 0.2 -> :wants_attention
        bio.dopamine < 0.2 -> :wants_something_new
        bio.adenosine > 0.8 -> :wants_rest
        emotional.pleasure < -0.5 -> :wants_to_express
        true -> :none
      end

    new_internal = %{process_state.state | current_desire: desire}
    %{process_state | state: new_internal}
  end

  # Maybe generate a spontaneous thought
  defp maybe_think(process_state) do
    # 10% chance per tick, or 100% if owner is online and we have something to say
    should_think =
      :rand.uniform() < @prob_spontaneous_thought ||
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
        # Chance to actually initiate
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
