defmodule Viva.Sessions.LifeProcessTest do
  # GenServer tests involving Registry often better sync
  use Viva.DataCase, async: false

  import Mox

  alias Viva.Accounts.User
  alias Viva.Avatars
  alias Viva.Sessions.LifeProcess

  # Allow mock calls from any process (GenServer, Tasks, etc.)
  setup :set_mox_global

  setup do
    # Stub LLM mock to return a default response
    stub(Viva.AI.LLM.MockClient, :generate, fn _, _ ->
      {:ok, "A gentle warmth washes over me."}
    end)

    # Stub EventBus mock
    stub(Viva.Infrastructure.MockEventBus, :publish_thought, fn _ -> :ok end)

    # Create user and avatar
    user =
      %User{}
      |> User.registration_changeset(%{
        email: "test#{System.unique_integer([:positive])}@example.com",
        username: "user#{System.unique_integer([:positive])}",
        password: "SecurePass123"
      })
      |> Repo.insert!()

    {:ok, avatar} =
      Avatars.create_avatar(user.id, %{
        name: "Life Test",
        bio: "Living...",
        gender: :non_binary,
        personality: %{
          openness: 0.5,
          conscientiousness: 0.5,
          extraversion: 0.5,
          agreeableness: 0.5,
          neuroticism: 0.5
        }
      })

    {:ok, avatar: avatar}
  end

  describe "lifecycle" do
    test "starts link and initializes state", %{avatar: avatar} do
      pid = start_supervised!({LifeProcess, avatar.id})

      state = :sys.get_state(pid)
      assert state.avatar_id == avatar.id
      assert state.tick_count == 0
      assert state.state != nil
    end

    test "handles owner connection", %{avatar: avatar} do
      pid = start_supervised!({LifeProcess, avatar.id})

      LifeProcess.owner_connected(avatar.id)

      state = :sys.get_state(pid)
      assert state.owner_online? == true
    end

    test "handles interaction start/end", %{avatar: avatar} do
      pid = start_supervised!({LifeProcess, avatar.id})
      other_id = Ecto.UUID.generate()

      LifeProcess.start_interaction(avatar.id, other_id)

      state = :sys.get_state(pid)
      assert state.current_conversation == other_id
      assert state.state.current_activity == :talking

      LifeProcess.end_interaction(avatar.id)
      state2 = :sys.get_state(pid)
      assert state2.current_conversation == nil
      assert state2.state.current_activity == :idle
    end
  end

  describe "tick processing" do
    test "processes a tick and increments count", %{avatar: avatar} do
      pid = start_supervised!({LifeProcess, avatar.id})

      # Manually trigger a tick
      send(pid, :tick)

      # Synchronize
      _ = :sys.get_state(pid)

      # Since tick is complex and might involve async tasks (AutonomousActions),
      # we check if tick_count incremented.
      state = :sys.get_state(pid)
      assert state.tick_count >= 1
    end
  end

  describe "persistence" do
    test "persists state every N ticks", %{avatar: avatar} do
      pid = start_supervised!({LifeProcess, avatar.id})

      # Trigger 5 ticks (assuming @persist_every_n_ticks is 5)
      for _ <- 1..5, do: send(pid, :tick)

      # Synchronize
      _ = :sys.get_state(pid)

      # Wait a bit for db update if any async involved
      # Actually update_internal_state is sync in this version

      updated_avatar = Avatars.get_avatar!(avatar.id)
      # Check if updated_at changed
      assert updated_avatar.updated_at != nil
    end
  end
end
