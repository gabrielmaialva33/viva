defmodule Viva.ConversationsTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts.User
  alias Viva.Avatars.Avatar
  alias Viva.Conversations
  alias Viva.Conversations.Conversation
  alias Viva.Conversations.Message

  # Test fixtures
  defp create_user do
    %User{}
    |> User.registration_changeset(%{
      email: "test#{System.unique_integer([:positive])}@example.com",
      username: "user#{System.unique_integer([:positive])}",
      password: "SecurePass123"
    })
    |> Repo.insert!()
  end

  defp create_avatar(user) do
    %Avatar{}
    |> Avatar.changeset(%{
      user_id: user.id,
      name: "Avatar#{System.unique_integer([:positive])}",
      bio: "Test avatar",
      gender: :female,
      age: 25,
      personality: %{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }
    })
    |> Repo.insert!()
  end

  defp create_test_avatars do
    user = create_user()
    avatar_a = create_avatar(user)
    avatar_b = create_avatar(user)
    {avatar_a.id, avatar_b.id}
  end

  defp create_conversation do
    {avatar_a_id, avatar_b_id} = create_test_avatars()
    Conversations.start_conversation(avatar_a_id, avatar_b_id)
  end

  describe "start_conversation/3" do
    test "creates a new conversation between two avatars" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, %Conversation{} = conv} =
               Conversations.start_conversation(avatar_a_id, avatar_b_id)

      assert conv.avatar_a_id == avatar_a_id
      assert conv.avatar_b_id == avatar_b_id
      assert conv.status == "active"
      assert conv.type == "interactive"
      assert conv.started_at != nil
    end

    test "accepts type option" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, conv} =
               Conversations.start_conversation(avatar_a_id, avatar_b_id, type: "autonomous")

      assert conv.type == "autonomous"
    end

    test "accepts topic option" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, conv} =
               Conversations.start_conversation(avatar_a_id, avatar_b_id, topic: "Music")

      assert conv.topic == "Music"
    end

    test "accepts context option" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      context = %{location: "park", weather: "sunny"}

      assert {:ok, conv} =
               Conversations.start_conversation(avatar_a_id, avatar_b_id, context: context)

      assert conv.context == context
    end
  end

  describe "start_autonomous_conversation/3" do
    test "creates autonomous conversation" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, conv} = Conversations.start_autonomous_conversation(avatar_a_id, avatar_b_id)
      assert conv.type == "autonomous"
    end
  end

  describe "start_interactive/2" do
    test "creates interactive conversation" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, conv} = Conversations.start_interactive(avatar_a_id, avatar_b_id)
      assert conv.type == "interactive"
    end
  end

  describe "get_conversation/1 and get_conversation!/1" do
    test "returns conversation by id" do
      {:ok, conv} = create_conversation()

      assert Conversations.get_conversation(conv.id) == conv
      assert Conversations.get_conversation!(conv.id) == conv
    end

    test "get_conversation returns nil for non-existent id" do
      assert Conversations.get_conversation(Ecto.UUID.generate()) == nil
    end

    test "get_conversation! raises for non-existent id" do
      assert_raise Ecto.NoResultsError, fn ->
        Conversations.get_conversation!(Ecto.UUID.generate())
      end
    end
  end

  describe "get_active_conversation/2" do
    test "returns active conversation between avatars" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      {:ok, conv} = Conversations.start_conversation(avatar_a_id, avatar_b_id)

      assert Conversations.get_active_conversation(avatar_a_id, avatar_b_id).id == conv.id
      assert Conversations.get_active_conversation(avatar_b_id, avatar_a_id).id == conv.id
    end

    test "returns nil when no active conversation" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert Conversations.get_active_conversation(avatar_a_id, avatar_b_id) == nil
    end

    test "returns nil when conversation is ended" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      {:ok, conv} = Conversations.start_conversation(avatar_a_id, avatar_b_id)
      Conversations.end_conversation(conv.id)

      assert Conversations.get_active_conversation(avatar_a_id, avatar_b_id) == nil
    end
  end

  describe "end_conversation/2" do
    test "ends the conversation" do
      {:ok, conv} = create_conversation()

      assert {:ok, ended} = Conversations.end_conversation(conv.id)
      assert ended.status == "ended"
      assert ended.ended_at != nil
      assert ended.duration_minutes != nil
    end

    test "accepts analysis option" do
      {:ok, conv} = create_conversation()

      analysis = %{emotional_depth: 0.8, topics: ["music", "art"]}
      assert {:ok, ended} = Conversations.end_conversation(conv.id, analysis: analysis)
      assert ended.analysis == analysis
    end

    test "returns nil for non-existent conversation" do
      assert Conversations.end_conversation(Ecto.UUID.generate()) == nil
    end
  end

  describe "pause_conversation/1" do
    test "pauses the conversation" do
      {:ok, conv} = create_conversation()

      assert {:ok, paused} = Conversations.pause_conversation(conv.id)
      assert paused.status == "paused"
    end

    test "returns nil for non-existent conversation" do
      assert Conversations.pause_conversation(Ecto.UUID.generate()) == nil
    end
  end

  describe "resume_conversation/1" do
    test "resumes a paused conversation" do
      {:ok, conv} = create_conversation()
      {:ok, paused} = Conversations.pause_conversation(conv.id)

      assert {:ok, resumed} = Conversations.resume_conversation(paused.id)
      assert resumed.status == "active"
    end
  end

  describe "list_conversations/2" do
    test "lists conversations for avatar" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other = create_avatar(user)
      third = create_avatar(user)

      {:ok, conv1} = Conversations.start_conversation(avatar_id, other.id)
      {:ok, conv2} = Conversations.start_conversation(third.id, avatar_id)

      convs = Conversations.list_conversations(avatar_id)
      assert length(convs) == 2
      ids = Enum.map(convs, & &1.id)
      assert conv1.id in ids
      assert conv2.id in ids
    end

    test "filters by status" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, active} = Conversations.start_conversation(avatar_id, other1.id)
      {:ok, ended} = Conversations.start_conversation(avatar_id, other2.id)
      Conversations.end_conversation(ended.id)

      active_convs = Conversations.list_conversations(avatar_id, status: "active")
      assert length(active_convs) == 1
      assert hd(active_convs).id == active.id
    end

    test "filters by type" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, interactive} =
        Conversations.start_conversation(avatar_id, other1.id, type: "interactive")

      {:ok, _} =
        Conversations.start_conversation(avatar_id, other2.id, type: "autonomous")

      interactive_convs = Conversations.list_conversations(avatar_id, type: "interactive")
      assert length(interactive_convs) == 1
      assert hd(interactive_convs).id == interactive.id
    end

    test "respects limit" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id

      for _ <- 1..5 do
        other = create_avatar(user)
        Conversations.start_conversation(avatar_id, other.id)
      end

      convs = Conversations.list_conversations(avatar_id, limit: 3)
      assert length(convs) == 3
    end
  end

  describe "add_message/4" do
    test "adds message to conversation" do
      {:ok, conv} = create_conversation()

      assert {:ok, %Message{} = msg} =
               Conversations.add_message(conv.id, conv.avatar_a_id, "Hello!")

      assert msg.conversation_id == conv.id
      assert msg.speaker_id == conv.avatar_a_id
      assert msg.content == "Hello!"
      assert msg.content_type == "text"
      assert msg.timestamp != nil
    end

    test "accepts emotional_tone option" do
      {:ok, conv} = create_conversation()

      assert {:ok, msg} =
               Conversations.add_message(conv.id, conv.avatar_a_id, "Hello!",
                 emotional_tone: "happy"
               )

      assert msg.emotional_tone == "happy"
    end

    test "accepts emotions option" do
      {:ok, conv} = create_conversation()

      emotions = %{joy: 0.8, excitement: 0.6}

      assert {:ok, msg} =
               Conversations.add_message(conv.id, conv.avatar_a_id, "Hello!", emotions: emotions)

      assert msg.emotions == emotions
    end

    test "increments message_count" do
      {:ok, conv} = create_conversation()
      assert conv.message_count == 0

      Conversations.add_message(conv.id, conv.avatar_a_id, "One")
      Conversations.add_message(conv.id, conv.avatar_b_id, "Two")

      updated = Conversations.get_conversation!(conv.id)
      assert updated.message_count == 2
    end
  end

  describe "send_message/3" do
    test "is alias for add_message" do
      {:ok, conv} = create_conversation()

      assert {:ok, msg} = Conversations.send_message(conv.id, conv.avatar_a_id, "Hello!")
      assert msg.content == "Hello!"
    end
  end

  describe "list_messages/2" do
    test "lists messages in order" do
      {:ok, conv} = create_conversation()
      {:ok, msg1} = Conversations.add_message(conv.id, conv.avatar_a_id, "First")
      Process.sleep(10)
      {:ok, msg2} = Conversations.add_message(conv.id, conv.avatar_b_id, "Second")

      messages = Conversations.list_messages(conv.id)
      assert length(messages) == 2
      assert hd(messages).id == msg1.id
      assert List.last(messages).id == msg2.id
    end

    test "respects limit" do
      {:ok, conv} = create_conversation()

      for i <- 1..5 do
        Conversations.add_message(conv.id, conv.avatar_a_id, "Message #{i}")
      end

      messages = Conversations.list_messages(conv.id, limit: 3)
      assert length(messages) == 3
    end
  end

  describe "get_recent_messages/2" do
    test "returns messages in chronological order" do
      {:ok, conv} = create_conversation()
      {:ok, msg1} = Conversations.add_message(conv.id, conv.avatar_a_id, "First")
      # Sleep 1 second to ensure different timestamps (utc_datetime has second precision)
      Process.sleep(1000)
      {:ok, msg2} = Conversations.add_message(conv.id, conv.avatar_b_id, "Second")

      messages = Conversations.get_recent_messages(conv.id, 10)
      assert length(messages) == 2
      message_ids = Enum.map(messages, & &1.id)
      assert msg1.id in message_ids
      assert msg2.id in message_ids
      # First message should come before second in chronological order
      assert hd(messages).id == msg1.id
      assert List.last(messages).id == msg2.id
    end
  end

  describe "conversation_stats/1" do
    test "returns stats for avatar" do
      {:ok, conv} = create_conversation()
      avatar_id = conv.avatar_a_id
      Conversations.add_message(conv.id, avatar_id, "Hello")

      stats = Conversations.conversation_stats(avatar_id)
      assert stats.total >= 1
      assert is_map(stats.by_status)
      assert is_map(stats.by_type)
    end
  end

  describe "conversation_history/3" do
    test "returns history between two avatars" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      {:ok, conv1} = Conversations.start_conversation(avatar_a_id, avatar_b_id)
      Conversations.end_conversation(conv1.id)
      {:ok, _} = Conversations.start_conversation(avatar_a_id, avatar_b_id)

      history = Conversations.conversation_history(avatar_a_id, avatar_b_id)
      assert length(history) == 2
    end
  end

  describe "list_autonomous_conversations/1" do
    test "lists only autonomous conversations" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {avatar_c_id, avatar_d_id} = create_test_avatars()

      {:ok, _} =
        Conversations.start_conversation(avatar_a_id, avatar_b_id, type: "interactive")

      {:ok, autonomous} =
        Conversations.start_conversation(avatar_c_id, avatar_d_id, type: "autonomous")

      convs = Conversations.list_autonomous_conversations()
      ids = Enum.map(convs, & &1.id)
      assert autonomous.id in ids
    end
  end
end
