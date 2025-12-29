defmodule Viva.Sessions.AutonomousActionsTest do
  use Viva.DataCase, async: false
  import Mox

  alias Viva.Avatars

  alias Viva.Conversations

  alias Viva.Relationships

  alias Viva.Sessions.AutonomousActions

  setup :verify_on_exit!

  setup do
    user =
      %Viva.Accounts.User{}
      |> Viva.Accounts.User.registration_changeset(%{
        email: "test#{System.unique_integer([:positive])}@example.com",
        username: "user#{System.unique_integer([:positive])}",
        password: "SecurePass123"
      })
      |> Repo.insert!()

    {:ok, a1} = Avatars.create_avatar(user.id, %{name: "Avatar 1", personality: %{}})

    {:ok, a2} = Avatars.create_avatar(user.id, %{name: "Avatar 2", personality: %{}})

    # Create relationship

    {:ok, _} = Relationships.create_relationship(a1.id, a2.id)

    rel = Relationships.get_relationship_between(a1.id, a2.id)

    Relationships.update_relationship(rel, %{status: :friends})

    initial_state = %{
      avatar_id: a1.id,
      avatar: a1,
      state: %{
        current_activity: :idle,
        current_desire: :none,
        bio: %{adenosine: 0.1},
        emotional: %{mood_label: "happy"}
      },
      current_conversation: nil
    }

    {:ok, a1: a1, a2: a2, state: initial_state}
  end

  describe "maybe_act/1" do
    test "initiates conversation when wants_to_talk", %{state: state, a2: _} do
      # Set desire

      state = put_in(state.state.current_desire, :wants_to_talk)

      # Probability is 0.25. We loop to ensure it triggers or we mock :rand.

      # Since we can't easily mock :rand in Elixir without extra libs,

      # we check if it calls Conversations.start_autonomous.

      # We check by looking at active conversations after many attempts

      for _ <- 1..50, do: AutonomousActions.maybe_act(state)

      convs = Conversations.list_conversations(state.avatar_id)

      refute Enum.empty?(convs)
    end

    test "ignores when already in conversation", %{state: base_state} do
      state_in_conv = %{base_state | current_conversation: Ecto.UUID.generate()}

      state_with_desire = put_in(state_in_conv.state.current_desire, :wants_to_talk)

      AutonomousActions.maybe_act(state_with_desire)

      assert Conversations.list_conversations(state_with_desire.avatar_id) == []
    end
  end

  describe "maybe_generate_greeting/1" do
    test "generates and broadcasts greeting", %{state: greeting_state} do
      # Mock LLM

      expect(Viva.AI.LLM.MockClient, :generate, fn _, _ ->
        {:ok, "Olá! Que bom te ver."}
      end)

      # Subscribe to PubSub

      Phoenix.PubSub.subscribe(Viva.PubSub, "avatar:#{greeting_state.avatar_id}:owner")

      AutonomousActions.maybe_generate_greeting(greeting_state)

      # Wait for async task and broadcast

      assert_receive {:greeting, "Olá! Que bom te ver."}, 1000
    end
  end
end
