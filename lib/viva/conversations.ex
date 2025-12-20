defmodule Viva.Conversations do
  @moduledoc """
  The Conversations context.
  Manages conversations and messages between avatars.
  """

  import Ecto.Query
  alias Viva.Repo
  alias Viva.Conversations.{Conversation, Message}

  # === Conversation CRUD ===

  def list_conversations(avatar_id, opts \\ []) do
    status = Keyword.get(opts, :status)
    type = Keyword.get(opts, :type)
    limit = Keyword.get(opts, :limit, 50)

    Conversation
    |> Conversation.involving(avatar_id)
    |> maybe_filter_conversation_status(status)
    |> maybe_filter_type(type)
    |> order_by([c], desc: c.started_at)
    |> limit(^limit)
    |> Repo.all()
  end

  def get_conversation(id), do: Repo.get(Conversation, id)

  def get_conversation!(id), do: Repo.get!(Conversation, id)

  def get_active_conversation(avatar_a_id, avatar_b_id) do
    Conversation
    |> Conversation.between(avatar_a_id, avatar_b_id)
    |> where([c], c.status == "active")
    |> Repo.one()
  end

  def start_conversation(avatar_a_id, avatar_b_id, opts \\ []) do
    type = Keyword.get(opts, :type, "interactive")
    topic = Keyword.get(opts, :topic)
    context = Keyword.get(opts, :context, %{})

    %Conversation{}
    |> Conversation.changeset(%{
      avatar_a_id: avatar_a_id,
      avatar_b_id: avatar_b_id,
      type: type,
      topic: topic,
      context: context,
      status: "active",
      started_at: DateTime.utc_now()
    })
    |> Repo.insert()
  end

  def end_conversation(conversation_id, opts \\ []) do
    analysis = Keyword.get(opts, :analysis)

    with conversation when not is_nil(conversation) <- get_conversation(conversation_id) do
      ended_at = DateTime.utc_now()
      duration = DateTime.diff(ended_at, conversation.started_at, :minute)

      conversation
      |> Conversation.changeset(%{
        status: "ended",
        ended_at: ended_at,
        duration_minutes: duration,
        analysis: analysis
      })
      |> Repo.update()
    end
  end

  def pause_conversation(conversation_id) do
    with conversation when not is_nil(conversation) <- get_conversation(conversation_id) do
      conversation
      |> Conversation.changeset(%{status: "paused"})
      |> Repo.update()
    end
  end

  def resume_conversation(conversation_id) do
    with conversation when not is_nil(conversation) <- get_conversation(conversation_id) do
      conversation
      |> Conversation.changeset(%{status: "active"})
      |> Repo.update()
    end
  end

  # === Messages ===

  def list_messages(conversation_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 100)
    after_timestamp = Keyword.get(opts, :after)

    Message
    |> where([m], m.conversation_id == ^conversation_id)
    |> maybe_filter_after(after_timestamp)
    |> order_by([m], asc: m.timestamp)
    |> limit(^limit)
    |> Repo.all()
  end

  def get_recent_messages(conversation_id, limit \\ 20) do
    Message
    |> where([m], m.conversation_id == ^conversation_id)
    |> order_by([m], desc: m.timestamp)
    |> limit(^limit)
    |> Repo.all()
    |> Enum.reverse()
  end

  def add_message(conversation_id, speaker_id, content, opts \\ []) do
    emotional_tone = Keyword.get(opts, :emotional_tone)
    emotions = Keyword.get(opts, :emotions, %{})
    content_type = Keyword.get(opts, :content_type, "text")
    audio_url = Keyword.get(opts, :audio_url)
    duration_seconds = Keyword.get(opts, :duration_seconds)

    # Insert message
    message_result =
      %Message{}
      |> Message.changeset(%{
        conversation_id: conversation_id,
        speaker_id: speaker_id,
        content: content,
        content_type: content_type,
        emotional_tone: emotional_tone,
        emotions: emotions,
        audio_url: audio_url,
        duration_seconds: duration_seconds,
        timestamp: DateTime.utc_now(),
        generated_at: System.monotonic_time(:millisecond)
      })
      |> Repo.insert()

    # Update conversation message count
    with {:ok, message} <- message_result do
      Conversation
      |> where([c], c.id == ^conversation_id)
      |> Repo.update_all(inc: [message_count: 1])

      {:ok, message}
    end
  end

  # === Autonomous Conversations ===

  def list_autonomous_conversations(opts \\ []) do
    status = Keyword.get(opts, :status, "active")
    limit = Keyword.get(opts, :limit, 50)

    Conversation
    |> where([c], c.type == "autonomous")
    |> maybe_filter_conversation_status(status)
    |> order_by([c], desc: c.started_at)
    |> limit(^limit)
    |> Repo.all()
  end

  def start_autonomous_conversation(avatar_a_id, avatar_b_id, context \\ %{}) do
    start_conversation(avatar_a_id, avatar_b_id,
      type: "autonomous",
      context: context
    )
  end

  # Aliases for compatibility
  def start_interactive(avatar_a_id, avatar_b_id) do
    start_conversation(avatar_a_id, avatar_b_id, type: "interactive")
  end

  def start_autonomous(avatar_a_id, avatar_b_id) do
    start_autonomous_conversation(avatar_a_id, avatar_b_id)
  end

  def send_message(conversation_id, speaker_id, content) do
    add_message(conversation_id, speaker_id, content)
  end

  # === Analytics ===

  def conversation_stats(avatar_id) do
    conversations = list_conversations(avatar_id)

    %{
      total: length(conversations),
      by_status: Enum.frequencies_by(conversations, & &1.status),
      by_type: Enum.frequencies_by(conversations, & &1.type),
      total_messages: Enum.reduce(conversations, 0, &(&1.message_count + &2)),
      avg_duration: avg_duration(conversations)
    }
  end

  def conversation_history(avatar_a_id, avatar_b_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)

    Conversation
    |> Conversation.between(avatar_a_id, avatar_b_id)
    |> order_by([c], desc: c.started_at)
    |> limit(^limit)
    |> Repo.all()
  end

  # === Private Helpers ===

  defp maybe_filter_conversation_status(query, nil), do: query
  defp maybe_filter_conversation_status(query, status), do: where(query, [c], c.status == ^status)

  defp maybe_filter_type(query, nil), do: query
  defp maybe_filter_type(query, type), do: where(query, [c], c.type == ^type)

  defp maybe_filter_after(query, nil), do: query
  defp maybe_filter_after(query, timestamp), do: where(query, [m], m.timestamp > ^timestamp)

  defp avg_duration([]), do: 0

  defp avg_duration(conversations) do
    with_duration = Enum.filter(conversations, & &1.duration_minutes)

    if length(with_duration) > 0 do
      sum = Enum.reduce(with_duration, 0, &(&1.duration_minutes + &2))
      sum / length(with_duration)
    else
      0
    end
  end
end
