defmodule Viva.Matching.EngineTest do
  use Viva.DataCase, async: false

  alias Viva.Avatars
  alias Viva.Avatars.Avatar
  alias Viva.Matching.Engine

  setup do
    # Clear cache before each test
    Engine.clear_cache()

    user =
      %Viva.Accounts.User{}
      |> Viva.Accounts.User.registration_changeset(%{
        email: "test#{System.unique_integer([:positive])}@example.com",
        username: "user#{System.unique_integer([:positive])}",
        password: "SecurePass123"
      })
      |> Repo.insert!()

    # Create two very compatible avatars
    {:ok, a1} =
      Avatars.create_avatar(user.id, %{
        name: "Compatible A",
        personality: %{
          openness: 0.9,
          conscientiousness: 0.9,
          extraversion: 0.9,
          agreeableness: 0.9,
          neuroticism: 0.1,
          enneagram_type: :type_9,
          interests: ["music", "art"],
          values: ["kindness"]
        }
      })

    {:ok, a2} =
      Avatars.create_avatar(user.id, %{
        name: "Compatible B",
        personality: %{
          openness: 0.8,
          conscientiousness: 0.8,
          extraversion: 0.8,
          agreeableness: 0.8,
          neuroticism: 0.2,
          enneagram_type: :type_9,
          interests: ["music", "art"],
          values: ["kindness"]
        }
      })

    # Create an incompatible one
    {:ok, a3} =
      Avatars.create_avatar(user.id, %{
        name: "Incompatible",
        personality: %{
          openness: 0.1,
          conscientiousness: 0.1,
          extraversion: 0.1,
          agreeableness: 0.1,
          neuroticism: 0.9,
          enneagram_type: :type_4,
          interests: ["sports"],
          values: ["money"]
        }
      })

    {:ok, a1: a1, a2: a2, a3: a3}
  end

  describe "calculate_compatibility/2" do
    test "returns high score for compatible avatars", %{a1: a1, a2: a2} do
      {:ok, score} = Engine.calculate_compatibility(a1.id, a2.id)
      assert score.total > 0.7
      assert score.personality > 0.7
      # Exact match
      assert score.interests == 1.0
    end

    test "returns low score for incompatible avatars", %{a1: a1, a3: a3} do
      {:ok, score} = Engine.calculate_compatibility(a1.id, a3.id)
      assert score.total < 0.5
    end
  end

  describe "find_matches/2" do
    test "returns sorted matches and uses cache", %{a1: a1, a2: a2, a3: a3} do
      # Initial call (cache miss)
      {:ok, matches} = Engine.find_matches(a1.id)
      assert length(matches) >= 2
      # Best match
      assert hd(matches).avatar.id == a2.id
      # Worst match
      assert List.last(matches).avatar.id == a3.id

      # Check stats
      stats = Engine.stats()
      assert stats.misses >= 1

      # Second call (cache hit)
      {:ok, _} = Engine.find_matches(a1.id)
      stats2 = Engine.stats()
      assert stats2.hits >= 1
    end

    test "respects limit and exclude options", %{a1: a1, a2: a2} do
      {:ok, matches} = Engine.find_matches(a1.id, limit: 1, exclude: [a2.id])
      # Since a2 is excluded, a1 vs a3 should be the only result
      assert length(matches) == 1
      refute hd(matches).avatar.id == a2.id
    end
  end

  describe "cache management" do
    test "invalidates entries", %{a1: a1} do
      Engine.find_matches(a1.id)
      # stats return a map with :size
      assert Engine.stats().size > 0

      Engine.invalidate(a1.id)
      # Wait for async deletion if any
      Process.sleep(100)
      assert Engine.stats().size == 0
    end

    test "clears all cache" do
      Engine.clear_cache()
      # Wait a bit
      Process.sleep(50)
      assert Engine.stats().size == 0
    end
  end
end
