defmodule Viva.AvatarsTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts.User
  alias Viva.Avatars
  alias Viva.Avatars.Avatar

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

    test "excludes inactive avatars" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())
      {:ok, _} = Avatars.update_avatar(avatar, %{is_active: false})

      ids = Avatars.list_active_avatar_ids()
      refute avatar.id in ids
    end
  end

  describe "get_avatar/1 and get_avatar!/1" do
    test "returns avatar by id" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert Avatars.get_avatar(avatar.id).id == avatar.id
      assert Avatars.get_avatar!(avatar.id).id == avatar.id
    end

    test "get_avatar returns nil for non-existent id" do
      assert Avatars.get_avatar(Ecto.UUID.generate()) == nil
    end

    test "get_avatar! raises for non-existent id" do
      assert_raise Ecto.NoResultsError, fn ->
        Avatars.get_avatar!(Ecto.UUID.generate())
      end
    end
  end

  describe "get_avatar_by_user/2" do
    test "returns avatar for matching user and avatar ids" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert Avatars.get_avatar_by_user(user.id, avatar.id).id == avatar.id
    end

    test "returns nil for mismatched user" do
      user1 = create_user()
      user2 = create_user()
      {:ok, avatar} = Avatars.create_avatar(user1.id, valid_avatar_attrs())

      assert Avatars.get_avatar_by_user(user2.id, avatar.id) == nil
    end
  end

  describe "create_avatar/2" do
    test "creates avatar with valid data" do
      user = create_user()

      assert {:ok, %Avatar{} = avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())
      assert avatar.name == "Test Avatar"
      assert avatar.user_id == user.id
      assert avatar.is_active == true
    end

    test "returns error with missing required fields" do
      user = create_user()

      # Missing name should fail
      invalid_attrs = %{
        bio: "Test",
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

      assert {:error, changeset} = Avatars.create_avatar(user.id, invalid_attrs)
      refute changeset.valid?
    end
  end

  describe "update_avatar/2" do
    test "updates avatar with valid data" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert {:ok, updated} = Avatars.update_avatar(avatar, %{name: "Updated Name"})
      assert updated.name == "Updated Name"
    end
  end

  describe "delete_avatar/1" do
    test "deletes the avatar" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      assert {:ok, %Avatar{}} = Avatars.delete_avatar(avatar)
      assert Avatars.get_avatar(avatar.id) == nil
    end
  end

  describe "generate_random_personality/0" do
    test "returns a Personality struct" do
      personality = Avatars.generate_random_personality()

      assert is_struct(personality)
      assert is_float(personality.openness)
      assert is_float(personality.conscientiousness)
      assert is_float(personality.extraversion)
      assert is_float(personality.agreeableness)
      assert is_float(personality.neuroticism)
    end

    test "personality values are within range" do
      personality = Avatars.generate_random_personality()

      assert personality.openness >= 0.0 and personality.openness <= 1.0
      assert personality.conscientiousness >= 0.0 and personality.conscientiousness <= 1.0
      assert personality.extraversion >= 0.0 and personality.extraversion <= 1.0
      assert personality.agreeableness >= 0.0 and personality.agreeableness <= 1.0
      assert personality.neuroticism >= 0.0 and personality.neuroticism <= 1.0
    end
  end

  describe "mark_active/1" do
    test "updates last_active_at" do
      user = create_user()
      {:ok, avatar} = Avatars.create_avatar(user.id, valid_avatar_attrs())

      {count, _} = Avatars.mark_active(avatar.id)
      assert count == 1

      updated = Avatars.get_avatar!(avatar.id)
      # Verify last_active_at is set and is recent
      assert updated.last_active_at != nil
    end
  end
end
