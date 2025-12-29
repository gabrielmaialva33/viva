defmodule Viva.AvatarsTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts.User
  alias Viva.Avatars
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Memory

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

  defp valid_avatar_attrs do
    %{
      name: "Test Avatar",
      bio: "A test avatar",
      gender: :female,
      age: 25,
      personality: %{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }
    }
  end

  describe "list_avatars/1" do
    test "returns empty list when no avatars" do
      assert Avatars.list_avatars() == []
    end

    test "returns all avatars" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      avatars = Avatars.list_avatars()
      refute Enum.empty?(avatars)
      assert avatar.id in Enum.map(avatars, & &1.id)
    end

    test "filters by user_id" do
      user1 = create_user()
      user2 = create_user()

      {:ok, avatar1} = Avatars.create_avatar(user1.id, valid_avatar_attrs())
      {:ok, _} = Avatars.create_avatar(user2.id, valid_avatar_attrs())

      avatars = Avatars.list_avatars(user_id: user1.id)
      assert length(avatars) == 1
      assert hd(avatars).id == avatar1.id
    end

    test "filters by active status" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      # Default is active
      active_avatars = Avatars.list_avatars(active: true)
      assert avatar.id in Enum.map(active_avatars, & &1.id)

      # Deactivate the avatar
      {:ok, deactivated} = Avatars.update_avatar(avatar, %{is_active: false})
      assert deactivated.is_active == false

      inactive_avatars = Avatars.list_avatars(active: false)
      assert avatar.id in Enum.map(inactive_avatars, & &1.id)
    end
  end

  describe "list_active_avatar_ids/0" do
    test "returns IDs of active avatars" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      ids = Avatars.list_active_avatar_ids()
      assert avatar.id in ids
    end
  end

  describe "get_avatar/1 and get_avatar!/1" do
    test "returns avatar by id" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert Avatars.get_avatar(avatar.id).id == avatar.id
      assert Avatars.get_avatar!(avatar.id).id == avatar.id
    end
  end

  describe "create_avatar/2" do
    test "creates avatar with valid data" do
      user = create_user()

      assert {:ok, %Avatar{} = avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())
      assert avatar.name == "Test Avatar"
      assert avatar.user_id == user.id
    end
  end

  describe "update_internal_state/2" do
    test "updates internal_state JSONB field" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      new_state = Map.put(InternalState.new(), :current_thought, "Deep thoughts")

      assert {:ok, updated} = Avatars.update_internal_state(avatar.id, new_state)
      assert updated.internal_state.current_thought == "Deep thoughts"
    end
  end

  describe "mark_active/1" do
    test "updates last_active_at" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert {1, nil} = Avatars.mark_active(avatar.id)
    end
  end

  # Memory tests disabled due to pgvector dependency in environment
  # describe "memories" do ...
end
