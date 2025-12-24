defmodule Viva.RelationshipsTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts.User
  alias Viva.Avatars.Avatar
  alias Viva.Relationships
  alias Viva.Relationships.Relationship

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

  defp create_relationship do
    {avatar_a_id, avatar_b_id} = create_test_avatars()
    Relationships.create_relationship(avatar_a_id, avatar_b_id)
  end

  describe "get_or_create_relationship/2" do
    test "creates new relationship when none exists" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, %Relationship{} = rel} =
               Relationships.get_or_create_relationship(avatar_a_id, avatar_b_id)

      assert rel.status == :strangers
      assert rel.familiarity == 0.0
      assert rel.trust == 0.5
      assert rel.affection == 0.0
      assert rel.attraction == 0.0
    end

    test "returns existing relationship" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      {:ok, rel1} = Relationships.get_or_create_relationship(avatar_a_id, avatar_b_id)
      {:ok, rel2} = Relationships.get_or_create_relationship(avatar_a_id, avatar_b_id)

      assert rel1.id == rel2.id
    end

    test "orders avatar IDs consistently" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      {:ok, rel1} = Relationships.get_or_create_relationship(avatar_a_id, avatar_b_id)
      {:ok, rel2} = Relationships.get_or_create_relationship(avatar_b_id, avatar_a_id)

      assert rel1.id == rel2.id
    end
  end

  describe "create_relationship/2" do
    test "is alias for get_or_create_relationship" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, %Relationship{}} = Relationships.create_relationship(avatar_a_id, avatar_b_id)
    end
  end

  describe "get_relationship/1 and get_relationship!/1" do
    test "returns relationship by id" do
      {:ok, rel} = create_relationship()

      assert Relationships.get_relationship(rel.id) == rel
      assert Relationships.get_relationship!(rel.id) == rel
    end

    test "get_relationship returns nil for non-existent id" do
      assert Relationships.get_relationship(Ecto.UUID.generate()) == nil
    end

    test "get_relationship! raises for non-existent id" do
      assert_raise Ecto.NoResultsError, fn ->
        Relationships.get_relationship!(Ecto.UUID.generate())
      end
    end
  end

  describe "get_relationship_between/2" do
    test "returns relationship between avatars" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {:ok, rel} = Relationships.create_relationship(avatar_a_id, avatar_b_id)

      assert Relationships.get_relationship_between(avatar_a_id, avatar_b_id).id == rel.id
      assert Relationships.get_relationship_between(avatar_b_id, avatar_a_id).id == rel.id
    end

    test "returns nil when no relationship exists" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      # Don't create a relationship between them
      assert Relationships.get_relationship_between(avatar_a_id, avatar_b_id) == nil
    end
  end

  describe "update_relationship/2" do
    test "updates relationship attributes" do
      {:ok, rel} = create_relationship()

      assert {:ok, updated} = Relationships.update_relationship(rel, %{familiarity: 0.5})
      assert updated.familiarity == 0.5
    end
  end

  describe "delete_relationship/1" do
    test "deletes the relationship" do
      {:ok, rel} = create_relationship()

      assert {:ok, %Relationship{}} = Relationships.delete_relationship(rel)
      assert Relationships.get_relationship(rel.id) == nil
    end
  end

  describe "list_relationships/2" do
    test "lists relationships for avatar" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, rel1} = Relationships.create_relationship(avatar_id, other1.id)
      {:ok, rel2} = Relationships.create_relationship(other2.id, avatar_id)

      rels = Relationships.list_relationships(avatar_id)
      assert length(rels) == 2
      ids = Enum.map(rels, & &1.id)
      assert rel1.id in ids
      assert rel2.id in ids
    end

    test "filters by status" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, strangers} = Relationships.create_relationship(avatar_id, other1.id)
      {:ok, friends} = Relationships.create_relationship(avatar_id, other2.id)
      Relationships.update_relationship(friends, %{status: :friends})

      stranger_rels = Relationships.list_relationships(avatar_id, status: :strangers)
      assert length(stranger_rels) == 1
      assert hd(stranger_rels).id == strangers.id
    end

    test "respects limit" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id

      for _ <- 1..5 do
        other = create_avatar(user)
        Relationships.create_relationship(avatar_id, other.id)
      end

      rels = Relationships.list_relationships(avatar_id, limit: 3)
      assert length(rels) == 3
    end
  end

  describe "record_interaction/3" do
    test "increments interaction count" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {:ok, rel} = Relationships.create_relationship(avatar_a_id, avatar_b_id)

      assert {:ok, updated} =
               Relationships.record_interaction(avatar_a_id, avatar_b_id, :conversation)

      assert updated.interaction_count == rel.interaction_count + 1
      assert updated.last_interaction_at != nil
    end

    test "creates relationship if it doesn't exist" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()

      assert {:ok, rel} = Relationships.record_interaction(avatar_a_id, avatar_b_id, :greeting)
      assert rel.interaction_count == 1
    end
  end

  describe "update_feelings/3" do
    test "updates feelings for specific avatar" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {:ok, rel} = Relationships.create_relationship(avatar_a_id, avatar_b_id)

      feelings = %{romantic_interest: 1.0, admiration: 0.8}
      {:ok, updated} = Relationships.update_feelings(rel, avatar_a_id, feelings)

      # The field updated depends on which avatar is making the update
      # Compare specific fields since one is a map and the other is a struct
      a_matches =
        updated.a_feelings.romantic_interest == 1.0 and updated.a_feelings.admiration == 0.8

      b_matches =
        updated.b_feelings.romantic_interest == 1.0 and updated.b_feelings.admiration == 0.8

      assert a_matches or b_matches
    end
  end

  describe "evolve_relationship/2" do
    test "updates relationship metrics" do
      {:ok, rel} = create_relationship()

      deltas = %{familiarity: 0.1, trust: 0.05, affection: 0.1}
      {:ok, updated} = Relationships.evolve_relationship(rel, deltas)

      assert updated.familiarity == 0.1
      assert updated.trust == 0.55
      assert updated.affection == 0.1
    end

    test "clamps values between 0 and 1" do
      {:ok, rel} = create_relationship()
      Relationships.update_relationship(rel, %{trust: 0.9})

      {:ok, updated} = Relationships.evolve_relationship(rel, %{trust: 0.5})
      assert updated.trust <= 1.0

      {:ok, updated2} = Relationships.evolve_relationship(rel, %{trust: -2.0})
      assert updated2.trust >= 0.0
    end

    test "updates status based on metrics" do
      {:ok, rel} = create_relationship()

      # Evolve to friends level
      {:ok, updated} =
        Relationships.evolve_relationship(rel, %{
          familiarity: 0.5,
          trust: 0.3,
          affection: 0.3
        })

      assert updated.status in [:acquaintances, :friends]
    end

    test "tracks unresolved conflicts" do
      {:ok, rel} = create_relationship()

      {:ok, updated} = Relationships.evolve_relationship(rel, %{conflict: 3})
      assert updated.unresolved_conflicts == 3

      {:ok, updated2} = Relationships.evolve_relationship(updated, %{conflict: -1})
      assert updated2.unresolved_conflicts == 2
    end
  end

  describe "list_potential_matches/2" do
    test "returns relationships with high attraction" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      high = create_avatar(user)
      low = create_avatar(user)

      {:ok, high_attraction} = Relationships.create_relationship(avatar_id, high.id)
      Relationships.update_relationship(high_attraction, %{attraction: 0.8, status: :friends})

      {:ok, low_attraction} = Relationships.create_relationship(avatar_id, low.id)
      Relationships.update_relationship(low_attraction, %{attraction: 0.3, status: :friends})

      matches = Relationships.list_potential_matches(avatar_id)
      ids = Enum.map(matches, & &1.id)
      assert high_attraction.id in ids
      refute low_attraction.id in ids
    end
  end

  describe "list_mutual_matches/1" do
    test "returns matched relationships" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, matched} = Relationships.create_relationship(avatar_id, other1.id)
      Relationships.update_relationship(matched, %{status: :matched})

      {:ok, _} = Relationships.create_relationship(avatar_id, other2.id)

      matches = Relationships.list_mutual_matches(avatar_id)
      assert length(matches) == 1
      assert hd(matches).id == matched.id
    end
  end

  describe "attempt_match/2" do
    test "creates matched status when both interested" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {:ok, rel} = Relationships.create_relationship(avatar_a_id, avatar_b_id)

      # Set up mutual interest (romantic_interest > 0.5 means interested)
      Relationships.update_feelings(rel, avatar_a_id, %{romantic_interest: 1.0})
      {:ok, updated_rel} = Relationships.get_or_create_relationship(avatar_a_id, avatar_b_id)
      Relationships.update_feelings(updated_rel, avatar_b_id, %{romantic_interest: 1.0})

      {:ok, matched} = Relationships.attempt_match(avatar_a_id, avatar_b_id)
      assert matched.status == :matched
      assert matched.matched_at != nil
    end

    test "records interest when only one avatar interested" do
      {avatar_a_id, avatar_b_id} = create_test_avatars()
      {:ok, _} = Relationships.create_relationship(avatar_a_id, avatar_b_id)

      {:ok, updated} = Relationships.attempt_match(avatar_a_id, avatar_b_id)
      assert updated.status != :matched
    end
  end

  describe "relationship_stats/1" do
    test "returns stats for avatar" do
      {:ok, rel} = create_relationship()
      avatar_id = rel.avatar_a_id

      stats = Relationships.relationship_stats(avatar_id)
      assert stats.total >= 1
      assert is_map(stats.by_status)
      assert is_float(stats.avg_familiarity) or stats.avg_familiarity == 0
    end
  end

  describe "count_close_relationships/1" do
    test "counts close relationships" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, close} = Relationships.create_relationship(avatar_id, other1.id)
      Relationships.update_relationship(close, %{status: :close_friends})

      {:ok, _} = Relationships.create_relationship(avatar_id, other2.id)

      count = Relationships.count_close_relationships(avatar_id)
      assert count == 1
    end
  end

  describe "list_for_avatar/1" do
    test "lists all relationships for avatar" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      other1 = create_avatar(user)
      other2 = create_avatar(user)

      {:ok, rel1} = Relationships.create_relationship(avatar_id, other1.id)
      {:ok, rel2} = Relationships.create_relationship(avatar_id, other2.id)

      rels = Relationships.list_for_avatar(avatar_id)
      ids = Enum.map(rels, & &1.id)
      assert rel1.id in ids
      assert rel2.id in ids
    end
  end

  describe "list_all/1" do
    test "lists all relationships" do
      {:ok, rel} = create_relationship()

      rels = Relationships.list_all()
      ids = Enum.map(rels, & &1.id)
      assert rel.id in ids
    end

    test "respects limit" do
      for _ <- 1..5 do
        create_relationship()
      end

      rels = Relationships.list_all(limit: 3)
      assert length(rels) <= 3
    end
  end

  describe "find_available_friend/1" do
    test "returns friend's avatar id" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      friend = create_avatar(user)
      friend_id = friend.id

      {:ok, rel} = Relationships.create_relationship(avatar_id, friend_id)
      Relationships.update_relationship(rel, %{status: :friends})

      assert Relationships.find_available_friend(avatar_id) == friend_id
    end

    test "returns nil when no friends" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      assert Relationships.find_available_friend(avatar_id) == nil
    end
  end

  describe "get_crush/1" do
    test "returns crush's avatar id" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      crush = create_avatar(user)
      crush_id = crush.id

      {:ok, rel} = Relationships.create_relationship(avatar_id, crush_id)
      Relationships.update_relationship(rel, %{status: :crush})

      assert Relationships.get_crush(avatar_id) == crush_id
    end

    test "returns nil when no crush" do
      user = create_user()
      avatar = create_avatar(user)
      avatar_id = avatar.id
      assert Relationships.get_crush(avatar_id) == nil
    end
  end
end
