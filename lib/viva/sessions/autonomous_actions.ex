defmodule Viva.Sessions.AutonomousActions do
  @moduledoc """
  Handles autonomous actions that avatars can take when not in conversation.
  Includes initiating conversations, messaging crushes, and generating greetings.
  Extracted from LifeProcess to reduce module dependencies.
  """

  require Logger

  alias Phoenix.PubSub
  alias Viva.AI.LLM.LlmClient
  alias Viva.Relationships

  # Action probabilities
  @prob_initiate_conversation 0.25
  @prob_message_crush 0.15

  @type process_state :: map()

  @doc """
  Execute autonomous actions based on current desire.
  Only acts if not in conversation and avatar is awake.
  """
  @spec maybe_act(process_state()) :: process_state()
  def maybe_act(process_state) do
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

  @doc """
  Generate and broadcast a greeting when owner connects.
  Runs asynchronously to avoid blocking the heartbeat.
  """
  @spec maybe_generate_greeting(process_state()) :: process_state()
  def maybe_generate_greeting(process_state) do
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

  # === Private Functions ===

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

  defp describe_mood(mood_label), do: mood_label || "neutral"

  defp broadcast_to_owner(avatar_id, message) do
    PubSub.broadcast(Viva.PubSub, "avatar:#{avatar_id}:owner", message)
  end
end
