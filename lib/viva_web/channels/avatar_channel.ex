defmodule VivaWeb.AvatarChannel do
  @moduledoc """
  Phoenix Channel for real-time avatar communication.
  Handles owner connection, avatar state updates, and conversations.
  """
  use Phoenix.Channel
  require Logger

  alias Phoenix.PubSub
  alias Viva.Conversations
  alias Viva.Sessions.LifeProcess
  alias Viva.Sessions.Supervisor

  @impl Phoenix.Channel
  def join("avatar:" <> avatar_id, _, socket) do
    # Verify ownership
    case verify_ownership(socket, avatar_id) do
      :ok ->
        # Ensure avatar life process is running
        {:ok, _} = Supervisor.start_avatar(avatar_id)

        # Notify avatar that owner connected
        LifeProcess.owner_connected(avatar_id)

        # Subscribe to avatar events
        PubSub.subscribe(Viva.PubSub, "avatar:#{avatar_id}")
        PubSub.subscribe(Viva.PubSub, "avatar:#{avatar_id}:owner")

        # Get current state
        state = LifeProcess.get_state(avatar_id)

        socket =
          socket
          |> assign(:avatar_id, avatar_id)
          |> assign(:current_conversation, nil)

        {:ok,
         %{
           status: "connected",
           avatar_state: serialize_state(state)
         }, socket}

      {:error, reason} ->
        {:error, %{reason: reason}}
    end
  end

  @impl Phoenix.Channel
  def join("conversation:" <> conversation_id, _, socket) do
    avatar_id = socket.assigns[:avatar_id]

    if avatar_id do
      PubSub.subscribe(Viva.PubSub, "conversation:#{conversation_id}")
      socket = assign(socket, :current_conversation, conversation_id)
      {:ok, socket}
    else
      {:error, %{reason: "must join avatar channel first"}}
    end
  end

  # === Incoming Messages ===

  @impl Phoenix.Channel
  def handle_in("get_state", _, socket) do
    state = LifeProcess.get_state(socket.assigns.avatar_id)
    {:reply, {:ok, serialize_state(state)}, socket}
  end

  @impl Phoenix.Channel
  def handle_in("trigger_thought", _, socket) do
    LifeProcess.trigger_thought(socket.assigns.avatar_id)
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_in("get_matches", payload, socket) do
    limit = Map.get(payload, "limit", 10)

    case Viva.Matching.Engine.find_matches(socket.assigns.avatar_id, limit: limit) do
      {:ok, matches} ->
        {:reply, {:ok, %{matches: serialize_matches(matches)}}, socket}

      {:error, reason} ->
        {:reply, {:error, %{reason: reason}}, socket}
    end
  end

  @impl Phoenix.Channel
  def handle_in("start_conversation", %{"with" => other_avatar_id}, socket) do
    avatar_id = socket.assigns.avatar_id

    case Conversations.start_interactive(avatar_id, other_avatar_id) do
      {:ok, conversation_id} ->
        socket = assign(socket, :current_conversation, conversation_id)
        {:reply, {:ok, %{conversation_id: conversation_id}}, socket}

      {:error, reason} ->
        {:reply, {:error, %{reason: reason}}, socket}
    end
  end

  @impl Phoenix.Channel
  def handle_in("send_message", %{"content" => content}, socket) do
    case socket.assigns.current_conversation do
      nil ->
        {:reply, {:error, %{reason: "not in a conversation"}}, socket}

      conversation_id ->
        avatar_id = socket.assigns.avatar_id
        Conversations.send_message(conversation_id, avatar_id, content)
        {:noreply, socket}
    end
  end

  @impl Phoenix.Channel
  def handle_in("end_conversation", _, socket) do
    case socket.assigns.current_conversation do
      nil ->
        {:reply, {:ok, %{}}, socket}

      conversation_id ->
        Conversations.end_conversation(conversation_id)
        socket = assign(socket, :current_conversation, nil)
        {:reply, {:ok, %{}}, socket}
    end
  end

  @impl Phoenix.Channel
  def handle_in("get_relationships", _, socket) do
    relationships = Viva.Relationships.list_for_avatar(socket.assigns.avatar_id)
    {:reply, {:ok, %{relationships: serialize_relationships(relationships)}}, socket}
  end

  @impl Phoenix.Channel
  def handle_in("get_memories", payload, socket) do
    limit = Map.get(payload, "limit", 20)
    type = Map.get(payload, "type")

    memories = Viva.Avatars.list_memories(socket.assigns.avatar_id, limit: limit, type: type)
    {:reply, {:ok, %{memories: serialize_memories(memories)}}, socket}
  end

  # === PubSub Messages ===

  @impl Phoenix.Channel
  def handle_info({:thought, thought}, socket) do
    push(socket, "thought", %{content: thought, timestamp: DateTime.utc_now()})
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:greeting, greeting}, socket) do
    push(socket, "greeting", %{content: greeting})
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:status, status}, socket) do
    push(socket, "status", %{status: status})
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:state_update, state}, socket) do
    push(socket, "state_update", serialize_state(state))
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:conversation_started, conv_info}, socket) do
    push(socket, "conversation_started", conv_info)
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:new_message, message}, socket) do
    push(socket, "new_message", serialize_message(message))
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:conversation_ended, conv_id}, socket) do
    push(socket, "conversation_ended", %{conversation_id: conv_id})

    socket =
      if socket.assigns.current_conversation == conv_id do
        assign(socket, :current_conversation, nil)
      else
        socket
      end

    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:relationship_update, relationship}, socket) do
    push(socket, "relationship_update", serialize_relationship(relationship))
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:match_found, match}, socket) do
    push(socket, "match_found", serialize_match(match))
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def terminate(_, socket) do
    if avatar_id = socket.assigns[:avatar_id] do
      LifeProcess.owner_disconnected(avatar_id)
    end

    :ok
  end

  # === Private Functions ===

  defp verify_ownership(socket, avatar_id) do
    user_id = socket.assigns[:user_id]

    if user_id do
      case Viva.Avatars.get_avatar(avatar_id) do
        nil ->
          {:error, "avatar not found"}

        %{user_id: ^user_id} ->
          :ok

        _ ->
          {:error, "not your avatar"}
      end
    else
      {:error, "not authenticated"}
    end
  end

  defp serialize_state(state) do
    %{
      mood: state.state.mood,
      energy: state.state.energy,
      social: state.state.social,
      stimulation: state.state.stimulation,
      current_activity: state.state.current_activity,
      current_desire: state.state.current_desire,
      current_thought: state.state.current_thought,
      emotions: Map.from_struct(state.state.emotions),
      interacting_with: state.state.interacting_with,
      owner_online: state.owner_online?
    }
  end

  defp serialize_matches(matches) do
    Enum.map(matches, &serialize_match/1)
  end

  defp serialize_match(match) do
    %{
      avatar: %{
        id: match.avatar.id,
        name: match.avatar.name,
        bio: match.avatar.bio,
        avatar_url: match.avatar.avatar_url,
        age: match.avatar.age
      },
      score: match.score,
      explanation: match.explanation
    }
  end

  defp serialize_relationships(relationships) do
    Enum.map(relationships, &serialize_relationship/1)
  end

  defp serialize_relationship(rel) do
    %{
      id: rel.id,
      other_avatar_id: other_avatar_id(rel),
      status: rel.status,
      familiarity: rel.familiarity,
      affection: rel.affection,
      attraction: rel.attraction,
      last_interaction_at: rel.last_interaction_at
    }
  end

  defp other_avatar_id(rel) do
    # This would need context about which avatar we're viewing from
    rel.avatar_b_id
  end

  defp serialize_memories(memories) do
    Enum.map(memories, fn m ->
      %{
        id: m.id,
        type: m.type,
        content: m.content,
        summary: m.summary,
        importance: m.importance,
        emotions_felt: m.emotions_felt,
        inserted_at: m.inserted_at
      }
    end)
  end

  defp serialize_message(message) do
    %{
      id: message.id,
      speaker_id: message.speaker_id,
      content: message.content,
      timestamp: message.timestamp,
      emotional_tone: message.emotional_tone
    }
  end
end
